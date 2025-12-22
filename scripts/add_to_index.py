#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量添加富文本索引
将有新LLM摘要的剧集（从LLM_sum2.json）添加到富文本索引
这些剧集原本只在基础索引中，现在有了LLM摘要，需要添加到富文本索引
"""

import os
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

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

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import qdrant_client
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('add_to_rich_text_index.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 动态导入数据加载模块
_load_data_path = Path(__file__).parent / "4_load_data_dual_track.py"
spec = importlib.util.spec_from_file_location("load_data_dual_track", _load_data_path)
load_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data_module)

# 导入需要的函数
load_llm_summaries = load_data_module.load_llm_summaries
build_rich_text_document = load_data_module.build_rich_text_document
extract_character_profile = load_data_module.extract_character_profile
extract_character_keywords = load_data_module.extract_character_keywords


def setup_llamaindex():
    """配置LlamaIndex"""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh-v1.5"
    )
    print("✓ Embedding 模型已配置: BAAI/bge-large-zh-v1.5")


def load_series_from_db(series_ids: List[int], db_path: str = "data/database/final.db") -> Dict[int, Dict]:
    """
    从数据库加载指定series_id的剧集数据（包含episodes）
    
    Args:
        series_ids: 要加载的series_id列表
        db_path: 数据库路径
    
    Returns:
        dict: {series_id: series_data}
    """
    conn = None
    series_dict = {}
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 使用 IN 子句批量查询
        placeholders = ','.join('?' * len(series_ids))
        query = f'''
            SELECT 
                s.id as series_id,
                s.original_doc_id,
                s.title,
                s.summary,
                s.cast,
                s.director,
                s.year,
                s.region,
                s.genre,
                s.url as series_url,
                e.id as episode_id,
                e.ep_number,
                e.episode_title,
                e.content as episode_content,
                e.episode_url
            FROM series s
            LEFT JOIN episodes e ON s.id = e.series_id
            WHERE s.id IN ({placeholders})
            ORDER BY s.id, e.ep_number
        '''
        
        cursor.execute(query, series_ids)
        
        current_series = None
        current_series_id = None
        
        for row in cursor:
            series_id = row['series_id']
            
            # 如果遇到新的剧集，保存上一个
            if current_series_id is not None and series_id != current_series_id:
                series_dict[current_series_id] = current_series
                current_series = None
            
            # 初始化新剧集
            if current_series is None:
                current_series = {
                    'series_id': series_id,
                    'original_doc_id': row['original_doc_id'],
                    'title': row['title'],
                    'summary': row['summary'] or '',
                    'cast': row['cast'] or '',
                    'director': row['director'] or '',
                    'year': row['year'] or '',
                    'region': row['region'] or '',
                    'genre': row['genre'] or '',
                    'url': row['series_url'],
                    'episodes': []
                }
                current_series_id = series_id
            
            # 添加分集信息（如果有）
            if row['episode_id']:
                current_series['episodes'].append({
                    'ep_number': row['ep_number'],
                    'episode_title': row['episode_title'] or '',
                    'content': row['episode_content'] or '',
                    'episode_url': row['episode_url'] or ''
                })
        
        # 保存最后一个剧集
        if current_series is not None:
            series_dict[current_series_id] = current_series
        
        logger.info(f"从数据库加载了 {len(series_dict)} 部剧集")
        return series_dict
        
    except sqlite3.Error as e:
        logger.error(f"数据库操作失败: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载数据时发生未知错误: {e}", exc_info=True)
        raise
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"关闭数据库连接时出错: {e}")


def check_series_in_index(series_ids: List[int], index: VectorStoreIndex) -> set:
    """
    检查哪些series_id已经在富文本索引中存在
    
    Args:
        series_ids: 要检查的series_id列表
        index: 富文本索引
    
    Returns:
        set: 已存在的series_id集合
    """
    existing_ids = set()
    
    try:
        # 创建一个简单的检索器来查询
        retriever = index.as_retriever(similarity_top_k=10000)  # 获取大量结果
        
        # 尝试通过metadata过滤来查找（如果支持）
        # 由于Qdrant的限制，我们可能需要其他方法
        # 这里先返回空集合，后续可以根据实际情况调整
        logger.info("检查已存在的series_id（可能较慢）...")
        
        # 更简单的方法：尝试检索一些文档，检查metadata
        # 但为了效率，我们先假设都不存在，让程序继续
        # 如果node_id冲突，insert会处理（可能会覆盖或报错）
        
    except Exception as e:
        logger.warning(f"检查已存在series_id时出错: {e}")
    
    return existing_ids


def add_series_to_rich_text_index(
    llm_sum2_file: str = "data/llm_summaries.json",
    db_path: str = "data/database/final.db",
    qdrant_path: str = "data/qdrant_data",
    batch_size: int = 10
):
    """
    将有新LLM摘要的剧集添加到富文本索引
    
    Args:
        llm_sum2_file: LLM_sum2.json文件路径
        db_path: 数据库路径
        qdrant_path: Qdrant数据路径
        batch_size: 批处理大小（每批处理的剧集数）
    """
    print("="*60)
    print("增量添加富文本索引")
    print("="*60)
    
    # 1. 配置LlamaIndex
    setup_llamaindex()
    
    # 2. 加载LLM_sum2.json
    print(f"\n正在加载 {llm_sum2_file}...")
    llm_summaries = load_llm_summaries(llm_sum2_file)
    
    if not llm_summaries:
        print(f"❌ 错误: 无法加载 {llm_sum2_file}")
        return
    
    series_ids = [int(k) for k in llm_summaries.keys()]
    print(f"✓ 找到 {len(series_ids)} 部剧集需要添加到富文本索引")
    
    # 3. 从数据库加载剧集数据
    print(f"\n正在从数据库加载剧集数据...")
    series_data = load_series_from_db(series_ids, db_path)
    
    if not series_data:
        print("❌ 错误: 无法从数据库加载剧集数据")
        return
    
    missing_ids = set(series_ids) - set(series_data.keys())
    if missing_ids:
        print(f"⚠ 警告: 数据库中没有找到以下series_id: {list(missing_ids)[:10]}...")
    
    # 4. 连接到富文本索引
    print(f"\n正在连接富文本索引...")
    client = qdrant_client.QdrantClient(path=qdrant_path)
    rich_vector_store = QdrantVectorStore(
        client=client,
        collection_name="tv_series_rich_text"
    )
    rich_storage_context = StorageContext.from_defaults(
        vector_store=rich_vector_store
    )
    
    try:
        rich_index = VectorStoreIndex.from_vector_store(
            vector_store=rich_vector_store,
            storage_context=rich_storage_context
        )
        print("✓ 富文本索引连接成功")
    except Exception as e:
        print(f"❌ 错误: 无法加载富文本索引: {e}")
        logger.error(f"加载富文本索引失败: {e}", exc_info=True)
        return
    
    # 5. 批量添加文档
    print(f"\n开始批量添加文档（批处理大小: {batch_size}）...")
    print("="*60)
    
    total_added = 0
    total_episodes = 0
    failed_count = 0
    
    # 按批次处理
    for i in range(0, len(series_ids), batch_size):
        batch_ids = series_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(series_ids) + batch_size - 1) // batch_size
        
        print(f"\n处理批次 {batch_num}/{total_batches} (series_id: {batch_ids[0]} - {batch_ids[-1]})...")
        
        for series_id in batch_ids:
            if series_id not in series_data:
                logger.warning(f"series_id {series_id} 不在数据库中，跳过")
                continue
            
            if str(series_id) not in llm_summaries:
                logger.warning(f"series_id {series_id} 不在LLM_summaries中，跳过")
                continue
            
            try:
                series = series_data[series_id]
                llm_summary = llm_summaries[str(series_id)]
                
                # 构建富文本文档
                parent_doc, child_docs = build_rich_text_document(series, llm_summary)
                
                # 添加父文档
                rich_index.insert(parent_doc)
                total_added += 1
                
                # 添加子文档
                for child_doc in child_docs:
                    rich_index.insert(child_doc)
                    total_episodes += 1
                
                print(f"  ✓ [{series_id}] {series['title']} - 添加了 {len(child_docs)} 个分集文档")
                logger.info(f"成功添加 series_id={series_id}, title={series['title']}, episodes={len(child_docs)}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"添加 series_id={series_id} 失败: {e}", exc_info=True)
                print(f"  ✗ [{series_id}] 添加失败: {e}")
        
        # 每批次后显示进度
        print(f"  进度: {min(i+batch_size, len(series_ids))}/{len(series_ids)} 部剧集已处理")
    
    # 6. 完成
    print("\n" + "="*60)
    print("添加完成！")
    print("="*60)
    print(f"总计:")
    print(f"  - 成功添加剧集: {total_added} 部")
    print(f"  - 成功添加分集: {total_episodes} 个")
    print(f"  - 失败: {failed_count} 部")
    print(f"\n✓ 这些剧集现在可以在富文本索引中检索到")
    print("  注意: 基础索引中的对应内容保持不变，检索时会自动优先使用富文本索引的结果")
    
    logger.info(f"添加完成: 成功={total_added}, 失败={failed_count}, 分集={total_episodes}")


if __name__ == "__main__":
    add_series_to_rich_text_index(
        llm_sum2_file="data/llm_summaries.json",
        db_path="data/database/final.db",
        qdrant_path="data/qdrant_data",
        batch_size=10  # 可以根据实际情况调整
    )

