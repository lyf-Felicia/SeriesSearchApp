#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
电视剧检索器 - 重构版本
支持多种检索模式：极速模式、深度模式、筛选模式
"""

import os
from typing import List, Optional, Dict, Literal
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.postprocessor import SentenceTransformerRerank
import logging

# 配置环境变量
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SAFETENSORS_FAST_GPU'] = '1'
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


class TVShowRetriever:
    """
    电视剧检索器 - 统一接口
    
    支持三种模式：
    1. ⚡ 极速模式 (Lightning): 纯向量检索，快速返回
    2. 🧠 深度模式 (Deep Thought): HyDE查询重写 + 双轨检索 + Rerank
    3. 🎯 筛选模式 (Filter): Metadata过滤 + 向量排序
    """
    
    def __init__(
        self,
        rich_text_index: VectorStoreIndex,
        basic_index: Optional[VectorStoreIndex] = None,
        similarity_top_k: int = 30,
        rerank_top_k: int = 10,
        use_hyde: bool = False,
        similarity_cutoff: float = 0.2
    ):
        """
        Args:
            rich_text_index: 富文本索引
            basic_index: 基础索引（可选）
            similarity_top_k: 初始检索数量
            rerank_top_k: 重排序后保留数量
            use_hyde: 是否使用HyDE查询重写
            similarity_cutoff: 相似度阈值
        """
        self.rich_text_index = rich_text_index
        self.basic_index = basic_index
        self.similarity_top_k = similarity_top_k
        self.rerank_top_k = rerank_top_k
        self.use_hyde = use_hyde
        self.similarity_cutoff = similarity_cutoff
        
        # 初始化Reranker（在检索器内部使用）
        self.reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",
            top_n=rerank_top_k
        )
        
        # 初始化LLM（用于HyDE）
        if use_hyde:
            if Settings.llm is None:
                Settings.llm = Ollama(
                    model="qwen2",
                    base_url="http://localhost:11434",
                    request_timeout=600.0
                )
    
    def _rewrite_query_hyde(self, query: str) -> str:
        """
        HyDE查询重写：将用户查询转换为假设性文档
        """
        if not self.use_hyde:
            return query
        
        try:
            prompt = f"""基于以下用户查询，生成一段假设性的电视剧描述文档，这段文档应该：
1. 包含用户查询中的关键元素
2. 扩展相关的剧情、人物、风格等细节
3. 用自然的中文描述

用户查询：{query}

假设性文档："""
            
            response = Settings.llm.complete(prompt)
            rewritten_query = str(response).strip()
            logger.info(f"HyDE查询重写: {query} -> {rewritten_query[:100]}...")
            return rewritten_query
        except Exception as e:
            logger.warning(f"HyDE查询重写失败: {e}，使用原始查询")
            return query
    
    def _retrieve_from_index(
        self,
        index: VectorStoreIndex,
        query_bundle: QueryBundle,
        filters: Optional[MetadataFilters] = None
    ) -> List[NodeWithScore]:
        """从单个索引检索"""
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.similarity_top_k,
            filters=filters
        )
        return retriever.retrieve(query_bundle)
    
    def _apply_rerank(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle
    ) -> List[NodeWithScore]:
        """应用重排序"""
        if not nodes:
            return nodes
        
        try:
            # Reranker需要Node对象列表
            node_list = [node.node for node in nodes]
            reranked_nodes = self.reranker.postprocess_nodes(
                node_list,
                query_bundle=query_bundle
            )
            
            # 转换回NodeWithScore格式，保持新的排序
            reranked_with_scores = []
            node_id_to_original = {n.node.node_id: n for n in nodes}
            
            for node in reranked_nodes:
                original_node = node_id_to_original.get(node.node_id)
                if original_node:
                    # 创建新的NodeWithScore，使用rerank后的顺序
                    from llama_index.core.schema import NodeWithScore
                    new_node = NodeWithScore(
                        node=node,
                        score=original_node.score if hasattr(original_node, 'score') and original_node.score else 0.5
                    )
                    reranked_with_scores.append(new_node)
            
            # 如果rerank成功，返回rerank结果；否则返回原始结果
            if reranked_with_scores:
                return reranked_with_scores[:self.rerank_top_k]
            else:
                # 如果rerank失败，返回原始结果的前top_k
                return nodes[:self.rerank_top_k]
        except Exception as e:
            logger.warning(f"重排序失败: {e}，返回原始结果")
            return nodes[:self.rerank_top_k]
    
    def _filter_by_similarity(
        self,
        nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """按相似度阈值过滤"""
        filtered = []
        for node in nodes:
            score = node.score if hasattr(node, 'score') and node.score else 0
            if score >= self.similarity_cutoff:
                filtered.append(node)
        return filtered
    
    def _merge_and_deduplicate(
        self,
        rich_nodes: List[NodeWithScore],
        basic_nodes: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """合并并去重"""
        seen_ids = set()
        merged = []
        
        # 富文本优先
        for node in rich_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                # 提升富文本优先级
                if hasattr(node, 'score') and node.score:
                    node.score = node.score * 1.1
                merged.append(node)
                seen_ids.add(node_id)
        
        # 添加基础索引结果
        for node in basic_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                merged.append(node)
                seen_ids.add(node_id)
        
        # 按分数排序
        merged.sort(key=lambda x: x.score if hasattr(x, 'score') and x.score else 0, reverse=True)
        return merged
    
    def _add_episode_context(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """为episode节点添加剧集上下文"""
        for node in nodes:
            if node.node.metadata.get('type') == 'episode':
                parent_title = node.node.metadata.get('parent_title', '未知剧集')
                ep_number = node.node.metadata.get('ep_number', '')
                ep_title = node.node.metadata.get('episode_title', '')
                original_text = node.node.text
                node.node.text = (
                    f"【来自剧集：{parent_title}，第{ep_number}集"
                    + (f"《{ep_title}》" if ep_title else "")
                    + "】\n" + original_text
                )
        return nodes
    
    def retrieve_lightning(
        self,
        query: str,
        scope: Literal["rich", "basic", "both"] = "both"
    ) -> List[NodeWithScore]:
        """
        ⚡ 极速模式：纯向量检索
        
        Args:
            query: 查询文本
            scope: 检索范围 ("rich", "basic", "both")
        
        Returns:
            检索到的节点列表（已按相似度排序）
        """
        query_bundle = QueryBundle(query)
        all_nodes = []
        
        # 从富文本索引检索
        if scope in ["rich", "both"]:
            rich_nodes = self._retrieve_from_index(self.rich_text_index, query_bundle)
            all_nodes.extend(rich_nodes)
        
        # 从基础索引检索
        if scope in ["basic", "both"] and self.basic_index:
            basic_nodes = self._retrieve_from_index(self.basic_index, query_bundle)
            all_nodes.extend(basic_nodes)
        
        # 合并去重（如果需要）
        if scope == "both" and self.basic_index and len(all_nodes) > 0:
            # 分离rich和basic节点
            rich_list = []
            basic_list = []
            for n in all_nodes:
                if n.node.metadata.get('index_type') == 'rich_text':
                    rich_list.append(n)
                else:
                    basic_list.append(n)
            if rich_list and basic_list:
                all_nodes = self._merge_and_deduplicate(rich_list, basic_list)
            elif rich_list:
                all_nodes = rich_list
            elif basic_list:
                all_nodes = basic_list
        
        # 相似度过滤（只过滤特别低的）
        all_nodes = self._filter_by_similarity(all_nodes)
        
        # 添加episode上下文
        all_nodes = self._add_episode_context(all_nodes)
        
        # 返回top 20（极速模式不需要rerank）
        return all_nodes[:20]
    
    def retrieve_deep(
        self,
        query: str,
        scope: Literal["rich", "basic", "both"] = "both"
    ) -> List[NodeWithScore]:
        """
        🧠 深度模式：HyDE查询重写 + 双轨检索 + Rerank
        
        Args:
            query: 查询文本
            scope: 检索范围
        
        Returns:
            重排序后的节点列表
        """
        # 1. HyDE查询重写（可选）
        rewritten_query = self._rewrite_query_hyde(query)
        query_bundle = QueryBundle(rewritten_query)
        
        # 2. 从两个索引检索
        rich_nodes = []
        basic_nodes = []
        
        if scope in ["rich", "both"]:
            rich_nodes = self._retrieve_from_index(self.rich_text_index, query_bundle)
        
        if scope in ["basic", "both"] and self.basic_index:
            basic_nodes = self._retrieve_from_index(self.basic_index, query_bundle)
        
        # 3. 合并去重
        if scope == "both" and self.basic_index:
            all_nodes = self._merge_and_deduplicate(rich_nodes, basic_nodes)
        else:
            all_nodes = rich_nodes if scope == "rich" else basic_nodes
        
        # 4. 相似度过滤（先不过滤，让rerank处理）
        # all_nodes = self._filter_by_similarity(all_nodes)
        
        # 5. Rerank重排序（使用原始查询）
        if all_nodes:
            original_query_bundle = QueryBundle(query)
            all_nodes = self._apply_rerank(all_nodes, original_query_bundle)
        else:
            logger.warning("检索结果为空")
        
        # 6. 相似度过滤（rerank后）
        all_nodes = self._filter_by_similarity(all_nodes)
        
        # 7. 添加episode上下文
        all_nodes = self._add_episode_context(all_nodes)
        
        return all_nodes
    
    def retrieve_filter(
        self,
        query: str,
        filters: Optional[Dict] = None,
        scope: Literal["rich", "basic", "both"] = "both"
    ) -> List[NodeWithScore]:
        """
        🎯 筛选模式：Metadata过滤 + 向量排序
        
        Args:
            query: 查询文本
            filters: 筛选条件，例如 {"year": "2022", "genre": "悬疑"}
            scope: 检索范围
        
        Returns:
            筛选后的节点列表
        """
        query_bundle = QueryBundle(query)
        
        # 构建Metadata过滤器
        metadata_filters = None
        if filters:
            filter_list = []
            for key, value in filters.items():
                filter_list.append({
                    "key": key,
                    "value": value,
                    "operator": "=="
                })
            if filter_list:
                metadata_filters = MetadataFilters(
                    filters=filter_list,
                    condition=FilterCondition.AND
                )
        
        # 检索
        all_nodes = []
        
        if scope in ["rich", "both"]:
            rich_nodes = self._retrieve_from_index(
                self.rich_text_index,
                query_bundle,
                filters=metadata_filters
            )
            all_nodes.extend(rich_nodes)
        
        if scope in ["basic", "both"] and self.basic_index:
            basic_nodes = self._retrieve_from_index(
                self.basic_index,
                query_bundle,
                filters=metadata_filters
            )
            all_nodes.extend(basic_nodes)
        
        # 合并去重（如果需要）
        if scope == "both" and self.basic_index and len(all_nodes) > 0:
            # 分离rich和basic节点
            rich_list = []
            basic_list = []
            for n in all_nodes:
                if n.node.metadata.get('index_type') == 'rich_text':
                    rich_list.append(n)
                else:
                    basic_list.append(n)
            if rich_list and basic_list:
                all_nodes = self._merge_and_deduplicate(rich_list, basic_list)
            elif rich_list:
                all_nodes = rich_list
            elif basic_list:
                all_nodes = basic_list
        
        # 相似度过滤（只过滤特别低的）
        all_nodes = self._filter_by_similarity(all_nodes)
        
        # 添加episode上下文
        all_nodes = self._add_episode_context(all_nodes)
        
        # 返回top 20（筛选模式不需要rerank）
        return all_nodes[:20]
    
    def retrieve(
        self,
        query: str,
        mode: Literal["lightning", "deep", "filter"] = "deep",
        scope: Literal["rich", "basic", "both"] = "both",
        filters: Optional[Dict] = None
    ) -> List[NodeWithScore]:
        """
        统一检索接口
        
        Args:
            query: 查询文本
            mode: 检索模式 ("lightning", "deep", "filter")
            scope: 检索范围
            filters: 筛选条件（仅filter模式使用）
        
        Returns:
            检索结果
        """
        if mode == "lightning":
            return self.retrieve_lightning(query, scope=scope)
        elif mode == "deep":
            return self.retrieve_deep(query, scope=scope)
        elif mode == "filter":
            return self.retrieve_filter(query, filters=filters, scope=scope)
        else:
            raise ValueError(f"未知的检索模式: {mode}")

