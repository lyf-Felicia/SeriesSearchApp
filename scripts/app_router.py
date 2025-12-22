#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自然语言路由器和应用接口
支持三种模式：极速模式、深度模式、筛选模式
"""

import os
import re
from typing import Literal, Optional, Dict, Tuple
from llama_index.core import VectorStoreIndex

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
import importlib
import logging

logger = logging.getLogger(__name__)

# 导入模块
tv_show_retriever_module = importlib.import_module('7_tv_show_retriever')
TVShowRetriever = tv_show_retriever_module.TVShowRetriever

qe_module = importlib.import_module('8_query_engine_simple')
load_index = qe_module.load_index
create_query_engine = qe_module.create_query_engine
SimpleQueryEngine = qe_module.SimpleQueryEngine


class LLMRouter:
    """
    简单的LLM路由器
    根据用户查询自动选择最合适的检索模式
    """
    
    def __init__(self):
        if Settings.llm is None:
            Settings.llm = Ollama(
                model="qwen2",
                base_url="http://localhost:11434",
                request_timeout=600.0
            )
    
    def route(self, query: str) -> Tuple[Literal["lightning", "deep", "filter"], Optional[Dict]]:
        """
        路由查询到合适的模式
        
        Returns:
            (mode, filters): 模式名称和可选的筛选条件
        """
        query_lower = query.lower()
        
        # 检查是否包含明确的筛选条件（年份、类型等）
        year_match = re.search(r'(\d{4})年', query)
        genre_match = re.search(r'(悬疑|爱情|奇幻|警匪|医疗|职场|家庭|甜宠|烧脑)', query)
        
        # 如果包含明确的筛选条件，使用筛选模式
        if year_match or genre_match:
            filters = {}
            if year_match:
                filters['year'] = year_match.group(1)
            if genre_match:
                filters['genre'] = genre_match.group(1)
            return ("filter", filters)
        
        # 检查是否是简单查询（短查询，常见关键词）
        simple_keywords = ['推荐', '想看', '找', '搜索', '查找']
        if len(query) < 20 and any(kw in query for kw in simple_keywords):
            return ("lightning", None)
        
        # 默认使用深度模式
        return ("deep", None)


class TVShowApp:
    """
    电视剧推荐应用
    统一接口，支持多种检索模式
    """
    
    def __init__(
        self,
        rich_text_index: VectorStoreIndex,
        basic_index: Optional[VectorStoreIndex] = None
    ):
        self.rich_text_index = rich_text_index
        self.basic_index = basic_index
        
        # 创建三种模式的查询引擎
        self.lightning_engine = create_query_engine(
            rich_text_index=rich_text_index,
            basic_index=basic_index,
            mode="lightning",
            use_hyde=False
        )
        
        self.deep_engine = create_query_engine(
            rich_text_index=rich_text_index,
            basic_index=basic_index,
            mode="deep",
            use_hyde=False  # 可以设置为True启用HyDE
        )
        
        self.filter_engine = create_query_engine(
            rich_text_index=rich_text_index,
            basic_index=basic_index,
            mode="filter",
            use_hyde=False
        )
        
        # 创建路由器
        self.router = LLMRouter()
    
    def query(
        self,
        query: str,
        mode: Optional[Literal["lightning", "deep", "filter"]] = None,
        scope: Literal["rich", "basic", "both"] = "both",
        filters: Optional[Dict] = None,
        auto_route: bool = True
    ):
        """
        执行查询
        
        Args:
            query: 查询文本
            mode: 检索模式（如果为None，则自动路由）
            scope: 检索范围
            filters: 筛选条件（仅filter模式使用）
            auto_route: 是否自动路由
        
        Returns:
            查询结果
        """
        # 自动路由
        if auto_route and mode is None:
            mode, detected_filters = self.router.route(query)
            if detected_filters:
                filters = detected_filters
        
        # 选择对应的查询引擎
        if mode == "lightning":
            engine = self.lightning_engine
        elif mode == "filter":
            engine = self.filter_engine
        else:  # deep or None
            engine = self.deep_engine
        
        # 执行查询
        return engine.query(query, scope=scope, filters=filters)
    
    def query_lightning(
        self,
        query: str,
        scope: Literal["rich", "basic", "both"] = "both"
    ):
        """⚡ 极速模式查询"""
        return self.lightning_engine.query(query, scope=scope)
    
    def query_deep(
        self,
        query: str,
        scope: Literal["rich", "basic", "both"] = "both"
    ):
        """🧠 深度模式查询"""
        return self.deep_engine.query(query, scope=scope)
    
    def query_filter(
        self,
        query: str,
        filters: Dict,
        scope: Literal["rich", "basic", "both"] = "both"
    ):
        """🎯 筛选模式查询"""
        return self.filter_engine.query(query, scope=scope, filters=filters)


def main():
    """主函数：初始化应用"""
    print("="*60)
    print("电视剧推荐系统 - 多模式检索")
    print("="*60)
    
    # 1. 配置LlamaIndex
    qe_module = importlib.import_module('8_query_engine_simple')
    setup_llamaindex = qe_module.setup_llamaindex
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
    
    # 3. 创建应用
    print("\n正在创建应用...")
    app = TVShowApp(rich_text_index, basic_index)
    print("  ✓ 应用创建成功")
    print("\n支持的模式：")
    print("  ⚡ 极速模式 (Lightning): 纯向量检索，快速返回")
    print("  🧠 深度模式 (Deep Thought): HyDE查询重写 + 双轨检索 + Rerank")
    print("  🎯 筛选模式 (Filter): Metadata过滤 + 向量排序")
    
    # 4. 测试查询
    print("\n" + "="*60)
    print("测试查询（自动路由）")
    print("="*60)
    
    test_queries = [
        "我想看奇幻爱情剧",  # 应该路由到deep模式
        "2022年悬疑剧",      # 应该路由到filter模式
        "推荐一些甜宠剧"     # 应该路由到lightning模式
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        mode, filters = app.router.route(query)
        print(f"  路由到: {mode} 模式" + (f", 筛选条件: {filters}" if filters else ""))
        try:
            response = app.query(query, auto_route=True)
            print(f"  结果: {str(response)[:200]}...")
        except Exception as e:
            print(f"  错误: {e}")
    
    return app


if __name__ == '__main__':
    app = main()

