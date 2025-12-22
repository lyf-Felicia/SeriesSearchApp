#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有核心模块是否正常工作
"""

import sys
import importlib

def test_module(module_name, functions=None):
    """测试模块导入和函数存在性"""
    try:
        module = importlib.import_module(module_name)
        print(f"✓ {module_name} 导入成功")
        if functions:
            for func_name in functions:
                if hasattr(module, func_name):
                    print(f"  ✓ {func_name} 存在")
                else:
                    print(f"  ✗ {func_name} 不存在")
                    return False
        return True
    except Exception as e:
        print(f"✗ {module_name} 导入失败: {e}")
        return False

def main():
    print("="*60)
    print("测试所有核心模块")
    print("="*60)
    print()
    
    modules_to_test = [
        ('4_load_data_dual_track', ['load_llm_summaries', 'build_dual_track_documents_generator']),
        ('5_build_index_dual_track', ['build_index_from_generator', 'load_checkpoint']),
        ('6_query_engine_dual_track', ['load_index', 'create_dual_track_query_engine', 'setup_llamaindex'])
    ]
    
    all_ok = True
    for module_name, functions in modules_to_test:
        if not test_module(module_name, functions):
            all_ok = False
        print()
    
    # 测试重排序功能
    print("测试重排序功能:")
    try:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        print("✓ SentenceTransformerRerank (重排序) 可用")
    except ImportError:
        try:
            from llama_index.core.postprocessor.sbert_rerank import SentenceTransformerRerank
            print("✓ SentenceTransformerRerank (重排序) 可用 (sbert_rerank路径)")
        except ImportError:
            print("✗ SentenceTransformerRerank (重排序) 不可用")
            all_ok = False
    print()
    
    # 测试核心依赖
    print("测试核心依赖:")
    dependencies = [
        ('llama_index.core', 'LlamaIndex Core'),
        ('llama_index.llms.ollama', 'Ollama LLM'),
        ('llama_index.embeddings.huggingface', 'HuggingFace Embeddings'),
        ('llama_index.vector_stores.qdrant', 'Qdrant Vector Store'),
        ('qdrant_client', 'Qdrant Client'),
        ('sentence_transformers', 'Sentence Transformers'),
    ]
    
    for dep_name, dep_desc in dependencies:
        try:
            importlib.import_module(dep_name)
            print(f"✓ {dep_desc}")
        except ImportError as e:
            print(f"✗ {dep_desc}: {e}")
            all_ok = False
    print()
    
    if all_ok:
        print("="*60)
        print("✓ 所有核心模块和依赖都可用！")
        print("="*60)
        return 0
    else:
        print("="*60)
        print("✗ 部分模块或依赖不可用")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())

