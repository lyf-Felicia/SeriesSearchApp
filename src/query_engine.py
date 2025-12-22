#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双轨制查询引擎
- 同时查询富文本索引和基础索引
- 合并结果并重排序
- 支持查询重写和重排序（使用 LlamaIndex 工具）
"""

import os
from typing import List, Optional
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore

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

from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from llama_index.core.postprocessor import SentenceTransformerRerank
import qdrant_client
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/query_engine.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_llamaindex():
    """配置LlamaIndex"""
    # 先设置 LLM，避免访问 Settings.llm 时触发默认解析
    try:
        Settings.llm = Ollama(
            model="qwen2",
            base_url="http://localhost:11434",
            request_timeout=600.0,
            context_window=8192,
            temperature=0.7
        )
    except Exception as e:
        logger.warning(f"初始化 Ollama LLM 失败: {e}，将使用默认设置")
    
    # 设置 Embedding 模型
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-zh-v1.5"
        )
    except Exception as e:
        logger.warning(f"初始化 Embedding 模型失败: {e}")


def load_index(collection_name: str, qdrant_path: str = "data/qdrant_data") -> Optional[VectorStoreIndex]:
    """
    加载索引
    
    注意：客户端会在索引对象生命周期内保持打开，不需要手动关闭
    """
    try:
        client = qdrant_client.QdrantClient(path=qdrant_path)
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )
        
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        return index
    except Exception as e:
        print(f"  ⚠ 加载索引 {collection_name} 失败: {e}")
        return None


class TypeSeparatedRetriever:
    """
    类型分离检索器
    分别检索摘要（series）和分集剧情（episode），避免混淆
    """
    
    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k_series: int = 5,
        similarity_top_k_episode: int = 5
    ):
        self.index = index
        self.similarity_top_k_series = similarity_top_k_series
        self.similarity_top_k_episode = similarity_top_k_episode
        
        # 创建摘要检索器（只检索 type="series"）
        series_filters = MetadataFilters(
            filters=[{
                "key": "type",
                "value": "series",
                "operator": "=="
            }],
            condition=FilterCondition.AND
        )
        self.series_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k_series,
            filters=series_filters
        )
        
        # 创建分集检索器（只检索 type="episode"）
        episode_filters = MetadataFilters(
            filters=[{
                "key": "type",
                "value": "episode",
                "operator": "=="
            }],
            condition=FilterCondition.AND
        )
        self.episode_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k_episode,
            filters=episode_filters
        )
    
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        分别检索摘要和分集，合并结果
        
        策略：
        - 摘要优先（series 类型）
        - 分集补充（episode 类型）
        - 保持分数排序
        """
        # 检索摘要（series）- 添加错误处理
        try:
            series_nodes = self.series_retriever.retrieve(query_bundle)
        except Exception as e:
            logger.error(f"检索摘要失败: {e}", exc_info=True)
            series_nodes = []  # 降级：返回空列表，继续处理
        
        # 检索分集（episode）- 添加错误处理
        try:
            episode_nodes = self.episode_retriever.retrieve(query_bundle)
        except Exception as e:
            logger.error(f"检索分集失败: {e}", exc_info=True)
            episode_nodes = []  # 降级：返回空列表，继续处理
        
        # 合并结果（摘要优先）
        merged_nodes = []
        
        # 先添加摘要结果（优先级更高）
        for node in series_nodes:
            try:
                # 给摘要结果加分（提升优先级）
                node.score = node.score * 1.2 if node.score else 1.0
                merged_nodes.append(node)
            except Exception as e:
                logger.warning(f"处理摘要节点时出错: {e}")
                continue  # 跳过有问题的节点
        
        # 再添加分集结果
        for node in episode_nodes:
            try:
                merged_nodes.append(node)
            except Exception as e:
                logger.warning(f"处理分集节点时出错: {e}")
                continue  # 跳过有问题的节点
        
        # 按分数排序
        try:
            merged_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        except Exception as e:
            logger.warning(f"排序结果时出错: {e}")
            # 如果排序失败，保持原顺序
        
        return merged_nodes


class DualTrackRetriever:
    """
    双轨制检索器
    同时从富文本索引和基础索引检索，并分离摘要和分集类型
    """
    
    def __init__(
        self,
        rich_text_index: VectorStoreIndex,
        basic_index: VectorStoreIndex,
        similarity_top_k_rich_series: int = 5,
        similarity_top_k_rich_episode: int = 5,
        similarity_top_k_basic_series: int = 3,
        similarity_top_k_basic_episode: int = 3
    ):
        # 富文本索引的类型分离检索器
        self.rich_text_retriever = TypeSeparatedRetriever(
            index=rich_text_index,
            similarity_top_k_series=similarity_top_k_rich_series,
            similarity_top_k_episode=similarity_top_k_rich_episode
        )
        
        # 基础索引的类型分离检索器
        self.basic_retriever = TypeSeparatedRetriever(
            index=basic_index,
            similarity_top_k_series=similarity_top_k_basic_series,
            similarity_top_k_episode=similarity_top_k_basic_episode
        )
    
    def retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        从两个索引检索并合并结果（已分离摘要和分集类型）
        
        策略：
        - 富文本索引优先（有更多信息）
        - 合并结果，去重
        - 保持分数排序
        - 为episode节点添加parent_title信息到文本中
        """
        # 从富文本索引检索（已分离类型）
        rich_nodes = self.rich_text_retriever.retrieve(query_bundle)
        
        # 从基础索引检索（已分离类型）
        basic_nodes = self.basic_retriever.retrieve(query_bundle)
        
        # 合并结果（富文本优先，去重）
        seen_ids = set()
        merged_nodes = []
        
        # 先添加富文本结果（优先级更高）
        for node in rich_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                # 给富文本结果加分（提升优先级）
                node.score = node.score * 1.1 if node.score else 1.0
                
                # 如果是episode类型，在文本前添加剧集信息
                if node.node.metadata.get('type') == 'episode':
                    parent_title = node.node.metadata.get('parent_title', '未知剧集')
                    ep_number = node.node.metadata.get('ep_number', '')
                    ep_title = node.node.metadata.get('episode_title', '')
                    # 在文本开头添加剧集信息
                    original_text = node.node.text
                    node.node.text = f"【来自剧集：{parent_title}，第{ep_number}集" + (f"《{ep_title}》" if ep_title else "") + "】\n" + original_text
                
                merged_nodes.append(node)
                seen_ids.add(node_id)
        
        # 再添加基础索引结果（如果未重复）
        for node in basic_nodes:
            node_id = node.node.node_id
            if node_id not in seen_ids:
                # 如果是episode类型，在文本前添加剧集信息
                if node.node.metadata.get('type') == 'episode':
                    parent_title = node.node.metadata.get('parent_title', '未知剧集')
                    ep_number = node.node.metadata.get('ep_number', '')
                    ep_title = node.node.metadata.get('episode_title', '')
                    original_text = node.node.text
                    node.node.text = f"【来自剧集：{parent_title}，第{ep_number}集" + (f"《{ep_title}》" if ep_title else "") + "】\n" + original_text
                
                merged_nodes.append(node)
                seen_ids.add(node_id)
        
        # 按分数排序
        merged_nodes.sort(key=lambda x: x.score or 0, reverse=True)
        
        return merged_nodes


def create_dual_track_query_engine(
    rich_text_index: VectorStoreIndex,
    basic_index: VectorStoreIndex,
    similarity_top_k_rich_series: int = 5,
    similarity_top_k_rich_episode: int = 5,
    similarity_top_k_basic_series: int = 3,
    similarity_top_k_basic_episode: int = 3,
    rerank_top_k: int = 5,
    use_rerank: bool = True,
    min_results: int = 5
):
    """
    创建双轨制查询引擎（分离摘要和分集类型）
    
    Args:
        rich_text_index: 富文本索引
        basic_index: 基础索引
        similarity_top_k_rich_series: 富文本索引摘要检索数量
        similarity_top_k_rich_episode: 富文本索引分集检索数量
        similarity_top_k_basic_series: 基础索引摘要检索数量
        similarity_top_k_basic_episode: 基础索引分集检索数量
        rerank_top_k: 重排序后保留的数量
        use_rerank: 是否使用重排序
    """
    # 创建双轨制检索器（已分离摘要和分集类型）
    dual_retriever = DualTrackRetriever(
        rich_text_index=rich_text_index,
        basic_index=basic_index,
        similarity_top_k_rich_series=similarity_top_k_rich_series,
        similarity_top_k_rich_episode=similarity_top_k_rich_episode,
        similarity_top_k_basic_series=similarity_top_k_basic_series,
        similarity_top_k_basic_episode=similarity_top_k_basic_episode
    )
    
    # 后处理器（重排序）
    postprocessors = []
    
    if use_rerank:
        # 使用 SentenceTransformer 重排序（本地模型）
        reranker = SentenceTransformerRerank(
            model="BAAI/bge-reranker-base",  # 中文重排序模型
            top_n=rerank_top_k
        )
        postprocessors.append(reranker)
        print(f"  ✓ 启用重排序: BAAI/bge-reranker-base (top_n={rerank_top_k})")
    
    # 相似度过滤（只过滤特别低的相关度，保留更多结果）
    postprocessors.append(SimilarityPostprocessor(similarity_cutoff=0.2))
    
    # 自定义响应合成器的prompt，区分series和episode，要求返回所有结果
    custom_prompt = PromptTemplate(
        """你是一个专业的电视剧推荐助手。根据用户查询，从提供的上下文信息中推荐最相关的电视剧。

重要说明：
1. 上下文信息包含两种类型：
   - **剧集摘要（series）**：包含剧名、主演、导演、剧情简介、人物与看点等信息
   - **分集剧情（episode）**：包含具体某一集的剧情内容，格式为"第X集《集名》\n剧情内容"
   
2. 推荐策略：
   - 如果检索到的是**剧集摘要**，直接推荐该剧集
   - 如果检索到的是**分集剧情**，必须说明这是来自《剧名》的第X集，并推荐该剧集
   - 分集剧情的metadata中包含"parent_title"字段，这是该分集所属的剧集名称

3. 推荐格式：
   - 对于剧集摘要：直接推荐剧名和相关信息
   - 对于分集剧情：格式为"《剧名》（根据第X集剧情推荐）"，然后说明推荐理由
   - **必须按照上下文提供的顺序，列出所有检索到的结果，不要只选择其中一个**

4. 推荐数量：
   - **必须返回所有提供的上下文中的剧集，按顺序列出**
   - 如果相关结果不足5个，请在回答末尾说明："注：由于相关度限制，只找到X个符合条件的剧集"

5. 相关度说明：
   - 如果返回的剧集少于5个，可能的原因包括：
     * 相似度阈值过滤：部分结果的相关度分数低于0.3，已被过滤
     * 检索结果有限：数据库中符合条件的内容较少

6. **重要：必须列出所有上下文中的剧集，不要遗漏任何一部。按照上下文顺序，为每个剧集提供简要介绍。**

上下文信息：
{context_str}

查询：{query_str}

请根据以上信息，为用户推荐所有相关的电视剧。必须列出上下文中的所有剧集，按顺序编号（1、2、3...）。如果推荐来自分集剧情，请明确说明对应的剧集名称和集数。
回答："""
    )
    
    # 创建响应合成器
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=custom_prompt
    )
    
    # 创建查询引擎（包装以检查结果数量）
    class ResultCountQueryEngine(RetrieverQueryEngine):
        def __init__(self, *args, min_results=5, **kwargs):
            super().__init__(*args, **kwargs)
            self.min_results = min_results
        
        def query(self, str_or_query_bundle):
            # 先检索节点
            if isinstance(str_or_query_bundle, str):
                from llama_index.core import QueryBundle
                query_bundle = QueryBundle(str_or_query_bundle)
            else:
                query_bundle = str_or_query_bundle
            
            # 获取检索结果
            nodes = self.retriever.retrieve(query_bundle)
            
            # 应用后处理器
            processed_nodes = nodes
            for postprocessor in self._node_postprocessors:
                processed_nodes = postprocessor.postprocess_nodes(
                    processed_nodes, 
                    query_bundle=query_bundle
                )
            
            # 检查结果数量并打印调试信息
            result_count = len(processed_nodes)
            logger.info(f"检索到 {result_count} 个节点（过滤后）")
            
            # 在prompt中明确要求返回所有结果
            original_query = query_bundle.query_str
            if result_count < self.min_results:
                query_bundle.query_str = f"{original_query}\n\n[系统提示：当前检索到{result_count}个相关结果。请按照相关度从高到低的顺序，列出所有{result_count}个结果，不要遗漏任何一部剧集。如果结果少于5个，请在回答末尾说明原因。]"
            else:
                query_bundle.query_str = f"{original_query}\n\n[系统提示：请按照相关度从高到低的顺序，列出所有{result_count}个检索结果，为每个结果编号（1、2、3...），不要遗漏任何一部剧集。]"
            
            # 调用父类的query方法
            return super().query(query_bundle)
    
    query_engine = ResultCountQueryEngine(
        retriever=dual_retriever,
        node_postprocessors=postprocessors,
        response_synthesizer=response_synthesizer,
        min_results=min_results
    )
    
    return query_engine


def main():
    """主函数：初始化双轨制查询引擎"""
    print("="*60)
    print("双轨制查询引擎初始化")
    print("="*60)
    
    # 1. 配置 LlamaIndex
    setup_llamaindex()
    print("\n✓ LlamaIndex 配置完成")
    
    # 2. 加载索引
    print("\n正在加载索引...")
    rich_text_index = load_index("tv_series_rich_text")
    basic_index = load_index("tv_series_basic")
    
    if not rich_text_index and not basic_index:
        print("❌ 错误：无法加载任何索引")
        return None
    
    if rich_text_index:
        print("  ✓ 富文本索引加载成功")
    if basic_index:
        print("  ✓ 基础索引加载成功")
    
    # 3. 创建查询引擎（分离摘要和分集类型）
    print("\n正在创建查询引擎（分离摘要和分集类型）...")
    query_engine = create_dual_track_query_engine(
        rich_text_index=rich_text_index or basic_index,  # 如果只有一个索引，使用它
        basic_index=basic_index or rich_text_index,
        similarity_top_k_rich_series=5,      # 富文本摘要检索数量
        similarity_top_k_rich_episode=5,     # 富文本分集检索数量
        similarity_top_k_basic_series=3,     # 基础摘要检索数量
        similarity_top_k_basic_episode=3,    # 基础分集检索数量
        rerank_top_k=5,
        use_rerank=True,
        min_results=5  # 期望的最小结果数量
    )
    
    print("\n✓ 查询引擎创建完成！")
    print("\n使用示例:")
    print("  response = query_engine.query('我想看男主是警察的甜宠剧')")
    print("  print(response)")
    
    return query_engine


if __name__ == '__main__':
    query_engine = main()
    
    # 测试查询
    if query_engine:
        print("\n" + "="*60)
        print("测试查询")
        print("="*60)
        test_query = "我想看奇幻爱情剧"
        print(f"\n查询: {test_query}")
        response = query_engine.query(test_query)
        print(f"\n结果:\n{response}")

