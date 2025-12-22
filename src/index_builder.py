#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双轨制索引构建模块
- 富文本索引：有 LLM 摘要的 1100 部剧集
- 基础索引：其他剧集
- 使用不同的 Qdrant 集合存储
"""

from pathlib import Path
from typing import Optional, Dict, Set
import os

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
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import qdrant_client
import importlib.util
import logging
import json
from pathlib import Path

# 尝试导入 psutil（用于内存检测）
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

# 尝试导入 tqdm（如果可用，用于进度条）
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc="", unit="", total=None):
        return iterable

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/index_build.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 动态导入双轨制数据加载模块
_load_data_path = Path(__file__).parent / "data_loader.py"
spec = importlib.util.spec_from_file_location("load_data_dual_track", _load_data_path)
load_data_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_data_module)
build_dual_track_documents = load_data_module.build_dual_track_documents
build_dual_track_documents_generator = load_data_module.build_dual_track_documents_generator


def setup_llamaindex():
    """配置LlamaIndex（全本地化）"""
    print("正在配置LlamaIndex...")
    
    # 配置本地 Embedding (BGE中文)
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-zh-v1.5"
    )
    print("  ✓ Embedding: BAAI/bge-large-zh-v1.5")
    print("✓ LlamaIndex配置完成\n")


def save_checkpoint(
    collection_name: str, 
    doc_count: int, 
    processed_series_ids: Optional[Set[int]] = None,
    last_series_id: Optional[int] = None, 
    qdrant_path: str = "data/qdrant_data"
):
    """
    保存检查点（用于断点续传）
    
    改进：保存已处理的 series_id 集合，提高恢复精确度
    """
    checkpoint_file = Path(qdrant_path) / f"{collection_name}_checkpoint.json"
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            checkpoint_data = {
                'collection_name': collection_name,
                'doc_count': doc_count,
                'processed_series_ids': list(processed_series_ids) if processed_series_ids else [],
                'last_series_id': last_series_id,
                'timestamp': str(Path(qdrant_path).stat().st_mtime) if Path(qdrant_path).exists() else ''
            }
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"检查点已保存: {doc_count} 个文档, {len(processed_series_ids) if processed_series_ids else 0} 个剧集已处理")
    except Exception as e:
        logger.warning(f"保存检查点失败: {e}")


def load_checkpoint(collection_name: str, qdrant_path: str = "data/qdrant_data") -> Optional[Dict]:
    """
    加载检查点
    
    改进：返回包含 doc_count、processed_series_ids 和 last_series_id 的字典
    """
    checkpoint_file = Path(qdrant_path) / f"{collection_name}_checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                processed_ids = set(checkpoint.get('processed_series_ids', []))
                logger.info(
                    f"发现检查点: 已处理 {checkpoint.get('doc_count', 0)} 个文档, "
                    f"{len(processed_ids)} 个剧集, last_series_id={checkpoint.get('last_series_id')}"
                )
                # 将列表转换为集合
                checkpoint['processed_series_ids'] = processed_ids
                return checkpoint
        except Exception as e:
            logger.warning(f"加载检查点失败: {e}")
    return None


def get_optimal_batch_size(default_batch_size: Optional[int] = None) -> int:
    """
    根据可用内存动态计算最优批处理大小
    
    Args:
        default_batch_size: 如果提供，直接返回（不动态计算）
    
    Returns:
        int: 最优批处理大小
    """
    if default_batch_size is not None:
        return default_batch_size
    
    if not HAS_PSUTIL:
        return 1000  # 默认值
    
    try:
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            return 2000  # 大内存机器
        elif available_memory > 4:
            return 1000  # 中等内存
        else:
            return 500   # 小内存机器
    except Exception as e:
        logger.warning(f"无法获取内存信息，使用默认值: {e}")
        return 1000


def build_index_from_generator(
    document_generator,
    collection_name: str,
    qdrant_path: str = "data/qdrant_data",
    description: str = "",
    batch_size: Optional[int] = None,
    enable_checkpoint: bool = True
):
    """
    从生成器构建向量索引（流式处理，避免内存溢出）
    
    修复：添加检查点机制，支持断点续传
    
    Args:
        document_generator: Document对象生成器
        collection_name: Qdrant集合名称
        qdrant_path: Qdrant数据存储路径
        description: 索引描述（用于日志）
        batch_size: 批处理大小（每批处理的文档数，None表示自动计算）
        enable_checkpoint: 是否启用检查点（用于断点续传）
    
    Returns:
        VectorStoreIndex: 构建好的索引
    """
    # 动态计算最优批处理大小
    optimal_batch_size = get_optimal_batch_size(batch_size)
    
    print(f"\n{'='*60}")
    print(f"构建 {description}")
    print(f"{'='*60}")
    print(f"  集合名称: {collection_name}")
    print(f"  批处理大小: {optimal_batch_size} (自动计算)")
    if batch_size is None and HAS_PSUTIL:
        try:
            available_memory = psutil.virtual_memory().available / (1024**3)
            print(f"  可用内存: {available_memory:.1f} GB")
        except:
            pass
    
    # 检查是否有检查点
    checkpoint = None
    processed_series_ids = set()  # 已处理的 series_id 集合
    if enable_checkpoint:
        checkpoint = load_checkpoint(collection_name, qdrant_path)
        if checkpoint:
            checkpoint_count = checkpoint.get('doc_count', 0)
            processed_series_ids = checkpoint.get('processed_series_ids', set())
            last_series_id = checkpoint.get('last_series_id')
            print(f"  ⚠ 发现检查点: 已处理 {checkpoint_count} 个文档, {len(processed_series_ids)} 个剧集")
            if last_series_id:
                print(f"  最后处理的 series_id: {last_series_id}")
            response = input("  是否从检查点继续？(y/n，默认n): ").strip().lower()
            if response != 'y':
                checkpoint = None
                checkpoint_count = None
                processed_series_ids = set()
                last_series_id = None
            else:
                checkpoint_count = checkpoint.get('doc_count', 0)
                processed_series_ids = checkpoint.get('processed_series_ids', set())
                last_series_id = checkpoint.get('last_series_id')
        else:
            checkpoint_count = None
            last_series_id = None
    else:
        checkpoint_count = None
        last_series_id = None
    
    # 初始化Qdrant客户端（本地持久化）
    client = None
    try:
        client = qdrant_client.QdrantClient(path=qdrant_path)
    except Exception as e:
        logger.error(f"初始化 Qdrant 客户端失败: {e}", exc_info=True)
        print(f"❌ 错误: 初始化 Qdrant 客户端失败: {e}")
        return None
    
    try:
        # 创建向量存储
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )
        
        # 创建存储上下文
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # 流式构建索引（分批处理）
        print(f"\n正在构建向量索引（流式处理）...")
        print("  这可能需要一些时间，请耐心等待...")
        
        # 使用 tqdm 显示进度（如果可用）
        if HAS_TQDM:
            document_generator = tqdm(document_generator, desc="构建索引", unit="文档")
        
        doc_count = 0
        batch = []
        index = None
        skip_count = checkpoint_count or 0
        current_series_id = None
        current_series_ids_in_batch = set()  # 当前批次中的 series_id
        
        try:
            for doc in document_generator:
                # 获取文档的 series_id
                doc_series_id = None
                if hasattr(doc, 'metadata') and doc.metadata.get('series_id'):
                    doc_series_id = doc.metadata['series_id']
                
                # 如果从检查点恢复，使用 series_id 集合来跳过已处理的剧集
                if processed_series_ids and doc_series_id:
                    if doc_series_id in processed_series_ids:
                        # 跳过已处理的剧集的所有文档（包括父文档和子文档）
                        continue
                    # 如果遇到新的剧集，从 processed_series_ids 中移除（表示开始处理新剧集）
                    # 但这里不需要移除，因为我们用集合来判断
                
                # 如果使用文档计数跳过（向后兼容）
                if skip_count > 0:
                    skip_count -= 1
                    if doc_series_id:
                        current_series_id = doc_series_id
                    continue
                
                batch.append(doc)
                doc_count += 1
                
                # 记录当前处理的 series_id（用于检查点）
                if doc_series_id:
                    current_series_id = doc_series_id
                    current_series_ids_in_batch.add(doc_series_id)
                
                # 达到批处理大小时，批量添加到索引
                if len(batch) >= optimal_batch_size:
                    try:
                        if index is None:
                            # 第一批：创建索引
                            index = VectorStoreIndex.from_documents(
                                batch,
                                storage_context=storage_context,
                                show_progress=False
                            )
                            logger.info(f"索引创建成功，第一批 {len(batch)} 个文档")
                        else:
                            # 后续批次：增量添加（逐个插入，因为 insert() 可能不支持批量列表）
                            for doc in batch:
                                index.insert(doc)
                            logger.debug(f"增量添加 {len(batch)} 个文档")
                        
                        # 保存检查点（包含 processed_series_ids 和 last_series_id）
                        if enable_checkpoint:
                            # 更新已处理的 series_id 集合
                            processed_series_ids.update(current_series_ids_in_batch)
                            save_checkpoint(
                                collection_name, 
                                doc_count, 
                                processed_series_ids, 
                                current_series_id, 
                                qdrant_path
                            )
                            current_series_ids_in_batch.clear()  # 清空当前批次
                        
                        print(f"  已处理: {doc_count} 个文档...")
                        batch = []  # 清空批次，释放内存
                    except Exception as e:
                        logger.error(f"批处理失败 (doc_count={doc_count}): {e}", exc_info=True)
                        print(f"  ⚠ 批处理失败: {e}")
                        # 继续处理下一批，不中断整个流程
                        batch = []
                        continue
            
            # 处理最后一批
            if batch:
                try:
                    if index is None:
                        # 只有一批：直接创建
                        index = VectorStoreIndex.from_documents(
                            batch,
                            storage_context=storage_context,
                            show_progress=False
                        )
                    else:
                        # 最后一批：增量添加（逐个插入）
                        for doc in batch:
                            index.insert(doc)
                    
                    # 保存最终检查点（包含 processed_series_ids 和 last_series_id）
                    if enable_checkpoint:
                        # 更新已处理的 series_id 集合
                        processed_series_ids.update(current_series_ids_in_batch)
                        save_checkpoint(
                            collection_name, 
                            doc_count, 
                            processed_series_ids, 
                            current_series_id, 
                            qdrant_path
                        )
                except Exception as e:
                    logger.error(f"处理最后一批失败: {e}", exc_info=True)
                    print(f"  ⚠ 处理最后一批失败: {e}")
            
            if index is None:
                logger.warning(f"{description}: 没有文档，跳过")
                print(f"  ⚠ {description}: 没有文档，跳过")
                return None
            
            # 删除检查点（构建完成）
            if enable_checkpoint:
                checkpoint_file = Path(qdrant_path) / f"{collection_name}_checkpoint.json"
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                    logger.info("检查点已删除（构建完成）")
            
            logger.info(f"{description} 构建完成，共 {doc_count} 个文档")
            print(f"\n✓ {description} 构建完成！")
            print(f"  文档总数: {doc_count}")
            print(f"  索引已保存到: {qdrant_path}/{collection_name}")
            
            return index
            
        except KeyboardInterrupt:
            logger.warning("用户中断构建过程")
            print(f"\n⚠ 构建被用户中断")
            if enable_checkpoint and doc_count > 0:
                # 更新已处理的 series_id 集合
                processed_series_ids.update(current_series_ids_in_batch)
                save_checkpoint(
                    collection_name, 
                    doc_count, 
                    processed_series_ids, 
                    current_series_id, 
                    qdrant_path
                )
                print(f"  检查点已保存: {doc_count} 个文档, {len(processed_series_ids)} 个剧集")
                if current_series_id:
                    print(f"  最后处理的 series_id: {current_series_id}")
                print(f"  下次运行可以从检查点继续")
            return index
        except Exception as e:
            logger.error(f"构建索引失败: {e}", exc_info=True)
            print(f"\n❌ 构建索引失败: {e}")
            if enable_checkpoint and doc_count > 0:
                # 更新已处理的 series_id 集合
                processed_series_ids.update(current_series_ids_in_batch)
                save_checkpoint(
                    collection_name, 
                    doc_count, 
                    processed_series_ids, 
                    current_series_id, 
                    qdrant_path
                )
                print(f"  检查点已保存: {doc_count} 个文档, {len(processed_series_ids)} 个剧集")
                if current_series_id:
                    print(f"  最后处理的 series_id: {current_series_id}")
            import traceback
            traceback.print_exc()
            return None
    finally:
        # 确保 Qdrant 客户端关闭（避免锁定问题）
        if client is not None:
            try:
                # 显式关闭客户端连接
                if hasattr(client, 'close'):
                    client.close()
                # 等待一下确保资源释放
                import time
                time.sleep(0.5)
                del client
                logger.debug("Qdrant 客户端已关闭")
            except Exception as e:
                logger.warning(f"关闭 Qdrant 客户端时出错: {e}")


def build_index(
    documents,
    collection_name: str,
    qdrant_path: str = "data/qdrant_data",
    description: str = ""
):
    """
    构建向量索引（兼容接口，支持列表和生成器）
    
    Args:
        documents: Document对象列表或生成器
        collection_name: Qdrant集合名称
        qdrant_path: Qdrant数据存储路径
        description: 索引描述（用于日志）
    
    Returns:
        VectorStoreIndex: 构建好的索引
    """
    # 如果是生成器，使用流式处理
    if hasattr(documents, '__iter__') and not isinstance(documents, (list, tuple)):
        return build_index_from_generator(
            documents,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            description=description
        )
    
    # 如果是列表，检查是否为空
    if not documents:
        print(f"  ⚠ {description}: 没有文档，跳过")
        return None
    
    print(f"\n{'='*60}")
    print(f"构建 {description}")
    print(f"{'='*60}")
    print(f"  集合名称: {collection_name}")
    print(f"  文档数量: {len(documents)}")
    
    # 初始化Qdrant客户端（本地持久化）
    client = None
    try:
        client = qdrant_client.QdrantClient(path=qdrant_path)
        
        # 创建向量存储
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name
        )
        
        # 创建存储上下文
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        
        # 构建索引
        print(f"\n正在构建向量索引...")
        print("  这可能需要一些时间，请耐心等待...")
        
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True
        )
        
        print(f"\n✓ {description} 构建完成！")
        print(f"  索引已保存到: {qdrant_path}/{collection_name}")
        
        return index
    finally:
        # 确保 Qdrant 客户端关闭（避免锁定问题）
        if client is not None:
            try:
                if hasattr(client, 'close'):
                    client.close()
                import time
                time.sleep(0.5)  # 等待资源释放
                del client
            except Exception as e:
                logger.warning(f"关闭 Qdrant 客户端时出错: {e}")


def check_collection_exists(collection_name: str, qdrant_path: str = "data/qdrant_data") -> bool:
    """检查集合是否已存在"""
    try:
        client = qdrant_client.QdrantClient(path=qdrant_path)
        # 获取所有集合
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        exists = collection_name in collection_names
        if exists:
            # 获取集合信息
            try:
                collection_info = client.get_collection(collection_name)
                points_count = collection_info.points_count if hasattr(collection_info, 'points_count') else 0
                logger.info(f"集合 {collection_name} 已存在，包含 {points_count} 个向量点")
            except:
                pass
        client.close()
        return exists
    except Exception as e:
        logger.warning(f"检查集合是否存在时出错: {e}")
        return False


def main():
    """主函数：构建双轨制索引"""
    print("="*60)
    print("LlamaIndex 剧集推荐系统 - 双轨制索引构建")
    print("="*60)
    
    # 1. 配置LlamaIndex
    setup_llamaindex()
    
    # 2. 构建双轨制文档（使用生成器模式，流式处理）
    print("\n正在加载数据并构建文档（流式处理）...")
    
    # 优化：显式传递参数，避免使用默认值
    db_path = "data/database/final.db"
    llm_sum_file = "data/llm_summaries.json"
    
    # 3. 构建富文本索引（流式处理）
    print("\n构建富文本索引...")
    
    # 检查富文本索引是否已存在
    if check_collection_exists("tv_series_rich_text"):
        print("  ✓ 富文本索引已存在，跳过构建")
        print("  如需重新构建，请先删除集合")
        rich_text_index = None
    else:
        rich_text_index = build_index(
            documents=build_dual_track_documents_generator(
                db_path=db_path,
                llm_sum_file=llm_sum_file,
                filter_type="rich_text"
            ),
            collection_name="tv_series_rich_text",
            description="富文本索引（有 LLM 摘要的剧集）"
        )
        
        # 确保富文本索引的客户端完全关闭，避免锁定问题
        if rich_text_index is not None:
            import time
            print("\n等待资源释放...")
            time.sleep(2)  # 等待2秒确保客户端完全关闭
    
    # 4. 构建基础索引（流式处理）
    # 注意：虽然会重新加载数据库，但这是必要的，因为生成器只能迭代一次
    # 如果数据量很大，可以考虑使用缓存或分阶段处理
    print("\n构建基础索引...")
    basic_index = build_index(
        documents=build_dual_track_documents_generator(
            db_path=db_path,
            llm_sum_file=llm_sum_file,
            filter_type="basic"
        ),
        collection_name="tv_series_basic",
        description="基础索引（其他剧集）"
    )
    
    print("\n" + "="*60)
    print("双轨制索引构建完成！")
    print("="*60)
    print("\n下一步：运行查询引擎，支持双轨制检索")
    print("\n注意：由于生成器只能迭代一次，两个索引需要分别构建")
    print("     如果数据量很大，可以考虑优化为一次遍历同时构建两个索引")
    
    return rich_text_index, basic_index


if __name__ == '__main__':
    rich_index, basic_index = main()

