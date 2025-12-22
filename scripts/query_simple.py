#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的查询引擎 - 使用TVShowRetriever
"""

import os
from typing import List, Optional, Dict, Literal
from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.prompts import PromptTemplate

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
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
import logging

import importlib
tv_show_retriever_module = importlib.import_module('7_tv_show_retriever')
TVShowRetriever = tv_show_retriever_module.TVShowRetriever

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_engine.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_llamaindex():
    """配置LlamaIndex"""
    try:
        Settings.llm = Ollama(
            model="qwen2",
            base_url="http://localhost:11434",
            request_timeout=600.0,
            context_window=8192,
            temperature=0.7
        )
    except Exception as e:
        logger.warning(f"初始化 Ollama LLM 失败: {e}")
    
    try:
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-zh-v1.5"
        )
    except Exception as e:
        logger.warning(f"初始化 Embedding 模型失败: {e}")


def load_index(collection_name: str, qdrant_path: str = "data/qdrant_data") -> Optional[VectorStoreIndex]:
    """加载索引"""
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


class SimpleQueryEngine:
    """简化的查询引擎包装类"""
    
    def __init__(
        self,
        retriever: TVShowRetriever,
        mode: Literal["lightning", "deep", "filter"] = "deep"
    ):
        self.retriever = retriever
        self.mode = mode
        
        # 创建响应合成器
        custom_prompt = PromptTemplate(
            """你是一个专业的电视剧推荐助手。根据用户查询和提供的上下文信息，推荐最相关的电视剧。

重要说明：
1. 上下文信息包含两种类型：
   - **剧集摘要（series）**：包含剧名、主演、导演、剧情简介等信息
   - **分集剧情（episode）**：包含具体某一集的剧情内容，格式为"【来自剧集：XXX，第X集】..."
   
2. 推荐要求：
   - **必须列出所有上下文中的剧集，按顺序编号（1、2、3...）**
   - **不要遗漏任何一部剧集，必须返回所有结果**
   - 如果推荐来自分集剧情，请明确说明对应的剧集名称和集数

3. 推荐格式：
   - 对于剧集摘要：直接推荐剧名和相关信息
   - 对于分集剧情：格式为"《剧名》（根据第X集剧情推荐）"

4. **重要：必须返回所有上下文中的剧集，不要只选择其中一个。按照顺序列出所有结果。**

上下文信息：
{context_str}

查询：{query_str}

请按照相关度从高到低的顺序，列出所有检索到的剧集，为每个结果编号（1、2、3...）。必须返回所有结果，不要遗漏。
回答："""
        )
        
        self.response_synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.COMPACT,
            text_qa_template=custom_prompt
        )
    
    def query(
        self,
        query: str,
        scope: Literal["rich", "basic", "both"] = "both",
        filters: Optional[Dict] = None
    ):
        """执行查询"""
        # 检索节点
        nodes = self.retriever.retrieve(
            query=query,
            mode=self.mode,
            scope=scope,
            filters=filters
        )
        
        if not nodes:
            return "抱歉，没有找到相关的电视剧。"
        
        # 构建上下文
        context_str = "\n\n".join([
            f"【{i+1}】{node.node.text[:500]}..." 
            for i, node in enumerate(nodes)
        ])
        
        # 使用LLM生成回答
        query_bundle = QueryBundle(query)
        response = self.response_synthesizer.synthesize(
            query=query_bundle,
            nodes=[node.node for node in nodes],
            additional_source_nodes=[]
        )
        
        return response


def create_query_engine(
    rich_text_index: VectorStoreIndex,
    basic_index: Optional[VectorStoreIndex] = None,
    mode: Literal["lightning", "deep", "filter"] = "deep",
    use_hyde: bool = False
) -> SimpleQueryEngine:
    """
    创建查询引擎
    
    Args:
        rich_text_index: 富文本索引
        basic_index: 基础索引（可选）
        mode: 检索模式
        use_hyde: 是否使用HyDE（仅deep模式）
    """
    # 创建检索器
    retriever = TVShowRetriever(
        rich_text_index=rich_text_index,
        basic_index=basic_index,
        similarity_top_k=30,
        rerank_top_k=10,
        use_hyde=use_hyde if mode == "deep" else False,
        similarity_cutoff=0.2
    )
    
    # 创建查询引擎
    query_engine = SimpleQueryEngine(retriever, mode=mode)
    
    return query_engine


def main():
    """主函数"""
    print("="*60)
    print("简化查询引擎初始化")
    print("="*60)
    
    # 1. 配置LlamaIndex
    setup_llamaindex()
    print("\n✓ LlamaIndex 配置完成")
    
    # 2. 加载索引
    print("\n正在加载索引...")
    rich_text_index = load_index("tv_series_rich_text")
    basic_index = load_index("tv_series_basic")
    
    if not rich_text_index:
        print("❌ 错误：无法加载富文本索引")
        return None
    
    if rich_text_index:
        print("  ✓ 富文本索引加载成功")
    if basic_index:
        print("  ✓ 基础索引加载成功")
    
    # 3. 创建查询引擎（深度模式）
    print("\n正在创建查询引擎（深度模式）...")
    query_engine = create_query_engine(
        rich_text_index=rich_text_index,
        basic_index=basic_index,
        mode="deep",
        use_hyde=False  # 可以设置为True启用HyDE
    )
    
    print("\n✓ 查询引擎创建完成！")
    print("\n使用示例:")
    print("  response = query_engine.query('我想看悬疑烧脑的剧')")
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

