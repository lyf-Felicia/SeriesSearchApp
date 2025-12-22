#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试检索功能，查看实际检索到的节点数量
"""

import sys
import importlib
import os
sys.path.insert(0, '.')

# 配置环境变量
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SAFETENSORS_FAST_GPU'] = '1'

# 导入模块（因为文件名以数字开头）
qe_module = importlib.import_module('6_query_engine_dual_track')
load_index = qe_module.load_index
create_dual_track_query_engine = qe_module.create_dual_track_query_engine
setup_llamaindex = qe_module.setup_llamaindex

from llama_index.core import QueryBundle

def main():
    print("="*60)
    print("测试检索功能")
    print("="*60)
    
    # 配置LlamaIndex
    print("\n0. 配置LlamaIndex...")
    setup_llamaindex()
    print("  ✓ 配置完成")
    
    # 加载索引
    print("\n1. 加载索引...")
    rich_index = load_index('tv_series_rich_text')
    if not rich_index:
        print("❌ 索引加载失败")
        return
    
    print("  ✓ 富文本索引加载成功")
    
    # 创建查询引擎
    print("\n2. 创建查询引擎...")
    query_engine = create_dual_track_query_engine(
        rich_text_index=rich_index,
        basic_index=rich_index,  # 暂时用同一个
        similarity_top_k_rich_series=5,
        similarity_top_k_rich_episode=5,
        similarity_top_k_basic_series=3,
        similarity_top_k_basic_episode=3,
        rerank_top_k=5,
        use_rerank=True
    )
    print("  ✓ 查询引擎创建成功")
    
    # 测试检索器（直接检索，不经过LLM）
    print("\n3. 测试检索器（直接检索，不经过LLM）...")
    query = "我想看悬疑烧脑的剧"
    print(f"   查询: {query}")
    
    retriever = query_engine.retriever
    query_bundle = QueryBundle(query)
    
    try:
        nodes = retriever.retrieve(query_bundle)
        print(f"\n   ✓ 检索成功，共检索到 {len(nodes)} 个节点")
        
        print("\n   检索结果详情:")
        for i, node in enumerate(nodes[:10], 1):
            title = node.node.metadata.get('title', '未知')
            node_type = node.node.metadata.get('type', '未知')
            score = node.score if hasattr(node, 'score') and node.score else 'N/A'
            print(f"   {i}. [{node_type}] {title} (score: {score})")
        
        # 测试完整查询（经过LLM）
        print("\n4. 测试完整查询（经过LLM生成回答）...")
        print("   注意：LLM会看到所有检索到的节点，但可能只选择最相关的来回答")
        response = query_engine.query(query)
        print(f"   回答长度: {len(str(response))} 字符")
        print(f"   回答内容预览: {str(response)[:300]}...")
        
        # 检查response中是否包含多个剧集
        response_str = str(response)
        import re
        # 查找所有可能的剧名（用《》标记的）
        titles = re.findall(r'《([^》]+)》', response_str)
        print(f"\n   回答中提到的剧集数量: {len(set(titles))}")
        if titles:
            print(f"   提到的剧集: {', '.join(set(titles))}")
        
    except Exception as e:
        print(f"   ✗ 检索失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

