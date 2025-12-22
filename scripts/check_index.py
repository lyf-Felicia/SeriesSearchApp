#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查索引状态
查看富文本索引和基础索引的文档数量、集合信息等
"""

import qdrant_client
from pathlib import Path
import json


def check_index_status(qdrant_path: str = "data/qdrant_data"):
    """
    检查索引状态
    
    Args:
        qdrant_path: Qdrant数据路径
    """
    print("="*60)
    print("索引状态检查")
    print("="*60)
    
    client = None
    try:
        client = qdrant_client.QdrantClient(path=qdrant_path)
        
        # 获取所有集合
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        print(f"\n发现的集合: {collection_names}")
        print("\n" + "="*60)
        
        # 检查富文本索引
        if "tv_series_rich_text" in collection_names:
            print("\n【富文本索引】tv_series_rich_text")
            print("-" * 60)
            try:
                collection_info = client.get_collection("tv_series_rich_text")
                points_count = collection_info.points_count
                vectors_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else points_count
                
                print(f"  向量点数量: {points_count:,}")
                print(f"  向量数量: {vectors_count:,}")
                
                # 尝试获取一些统计信息
                if hasattr(collection_info, 'config'):
                    config = collection_info.config
                    if hasattr(config, 'params'):
                        params = config.params
                        if hasattr(params, 'vectors'):
                            vectors_config = params.vectors
                            if hasattr(vectors_config, 'size'):
                                print(f"  向量维度: {vectors_config.size}")
                
                # 尝试获取一些文档的metadata统计
                try:
                    # 获取前100个点来统计
                    scroll_result = client.scroll(
                        collection_name="tv_series_rich_text",
                        limit=100,
                        with_payload=True
                    )
                    
                    if scroll_result[0]:  # 如果有数据
                        series_count = 0
                        episode_count = 0
                        rich_text_count = 0
                        basic_count = 0
                        
                        for point in scroll_result[0]:
                            if point.payload:
                                doc_type = point.payload.get('type', '')
                                index_type = point.payload.get('index_type', '')
                                
                                if doc_type == 'series':
                                    series_count += 1
                                elif doc_type == 'episode':
                                    episode_count += 1
                                
                                if index_type == 'rich_text':
                                    rich_text_count += 1
                                elif index_type == 'basic':
                                    basic_count += 1
                        
                        print(f"\n  样本统计（前100个点）:")
                        print(f"    - 摘要文档 (type=series): {series_count}")
                        print(f"    - 分集文档 (type=episode): {episode_count}")
                        print(f"    - 富文本标记 (index_type=rich_text): {rich_text_count}")
                        print(f"    - 基础标记 (index_type=basic): {basic_count}")
                except Exception as e:
                    print(f"  ⚠ 无法获取详细统计: {e}")
                
            except Exception as e:
                print(f"  ❌ 获取集合信息失败: {e}")
        else:
            print("\n⚠ 富文本索引不存在: tv_series_rich_text")
        
        # 检查基础索引
        if "tv_series_basic" in collection_names:
            print("\n【基础索引】tv_series_basic")
            print("-" * 60)
            try:
                collection_info = client.get_collection("tv_series_basic")
                points_count = collection_info.points_count
                vectors_count = collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else points_count
                
                print(f"  向量点数量: {points_count:,}")
                print(f"  向量数量: {vectors_count:,}")
                
                # 尝试获取向量维度
                if hasattr(collection_info, 'config'):
                    config = collection_info.config
                    if hasattr(config, 'params'):
                        params = config.params
                        if hasattr(params, 'vectors'):
                            vectors_config = params.vectors
                            if hasattr(vectors_config, 'size'):
                                print(f"  向量维度: {vectors_config.size}")
                
                # 尝试获取一些文档的metadata统计
                try:
                    scroll_result = client.scroll(
                        collection_name="tv_series_basic",
                        limit=100,
                        with_payload=True
                    )
                    
                    if scroll_result[0]:
                        series_count = 0
                        episode_count = 0
                        
                        for point in scroll_result[0]:
                            if point.payload:
                                doc_type = point.payload.get('type', '')
                                
                                if doc_type == 'series':
                                    series_count += 1
                                elif doc_type == 'episode':
                                    episode_count += 1
                        
                        print(f"\n  样本统计（前100个点）:")
                        print(f"    - 摘要文档 (type=series): {series_count}")
                        print(f"    - 分集文档 (type=episode): {episode_count}")
                except Exception as e:
                    print(f"  ⚠ 无法获取详细统计: {e}")
                
            except Exception as e:
                print(f"  ❌ 获取集合信息失败: {e}")
        else:
            print("\n⚠ 基础索引不存在: tv_series_basic")
        
        # 检查checkpoint文件
        print("\n" + "="*60)
        print("【检查点文件状态】")
        print("-" * 60)
        
        checkpoint_files = [
            "tv_series_rich_text_checkpoint.json",
            "tv_series_basic_checkpoint.json"
        ]
        
        for checkpoint_file in checkpoint_files:
            checkpoint_path = Path(qdrant_path) / checkpoint_file
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint = json.load(f)
                    
                    doc_count = checkpoint.get('doc_count', 0)
                    processed_ids = checkpoint.get('processed_series_ids', [])
                    last_series_id = checkpoint.get('last_series_id')
                    
                    print(f"\n  {checkpoint_file}:")
                    print(f"    已处理文档数: {doc_count:,}")
                    print(f"    已处理剧集数: {len(processed_ids)}")
                    if last_series_id:
                        print(f"    最后处理的series_id: {last_series_id}")
                except Exception as e:
                    print(f"  ⚠ 无法读取 {checkpoint_file}: {e}")
            else:
                print(f"\n  {checkpoint_file}: 不存在（索引可能已完成构建）")
        
        print("\n" + "="*60)
        print("检查完成")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            try:
                client.close()
            except:
                pass


if __name__ == "__main__":
    check_index_status()

