#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双轨制文档构建模块
- 富文本索引：有 LLM 摘要的 1100 部剧集（使用 LLM_sum.json）
- 基础索引：其他剧集（使用原始 summary）

核心优化策略：
1. 富文本索引使用 plot_summary（500字精炼摘要）而非 combined_text（避免重复和混淆）
2. 标签作为 metadata 用于过滤和增强检索
3. 人物与看点的关键信息提取为结构化 metadata
"""

import sqlite3
import json
import os
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from functools import lru_cache

# 尝试导入 tqdm（如果可用，用于进度条）
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 如果没有 tqdm，使用简单的包装器
    def tqdm(iterable, desc="", unit="", total=None):
        return iterable

# 预编译正则表达式（性能优化）
COMPILED_PATTERNS = {
    'occupation': [
        re.compile(r'职业身份[：:]\s*([^\n*]+)'),
        re.compile(r'职业身份[：:]\s*([^\n]+?)(?:\n|$)'),
        re.compile(r'职业[：:]\s*([^\n*]+)'),
    ],
    'trait': [
        re.compile(r'性格特征[：:]\s*([^\n*]+)'),
        re.compile(r'性格[：:]\s*([^\n*]+)'),
        re.compile(r'性格特征[：:]\s*([^\n]+?)(?:\n|$)'),
    ],
    'relationship': [
        re.compile(r'([^\n]+?)[与和][^\n]+?[：:]\s*([^\n]+)'),
        re.compile(r'([A-Za-z0-9\u4e00-\u9fa5]+)\s*与\s*([A-Za-z0-9\u4e00-\u9fa5]+)[：:]\s*([^\n]+)'),
    ],
    'conflict': [
        re.compile(r'核心冲突[^：:]*[：:]\s*\n\s*[-•·]\s*([^\n]+)'),
        re.compile(r'核心冲突[^：:]*[：:]\s*([^\n]+)'),
        re.compile(r'冲突[^：:]*[：:]\s*\n\s*[-•·]\s*([^\n]+)'),
    ]
}

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

from llama_index.core import Document
from llama_index.core.schema import RelatedNodeInfo, NodeRelationship
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# 配置 Embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-zh-v1.5")

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dual_track.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_llm_summaries(llm_sum_file: str = "data/llm_summaries.json") -> Dict:
    """
    加载 LLM 生成的摘要（添加错误处理和文件大小检查）
    
    Returns:
        dict: {series_id: summary_data}
    """
    if not Path(llm_sum_file).exists():
        logger.warning(f"{llm_sum_file} 不存在，将使用基础索引")
        print(f"⚠ 警告: {llm_sum_file} 不存在，将使用基础索引")
        return {}
    
    try:
        # 检查文件大小
        file_size = Path(llm_sum_file).stat().st_size / (1024 * 1024)  # MB
        if file_size > 500:  # 超过 500MB 警告
            logger.warning(f"llm_summaries.json 文件较大 ({file_size:.1f}MB)，加载可能需要一些时间")
        
        with open(llm_sum_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据格式
        if not isinstance(data, dict):
            logger.error(f"llm_summaries.json 格式错误：期望 dict，得到 {type(data)}")
            raise ValueError(f"llm_summaries.json 格式错误：期望 dict，得到 {type(data)}")
        
        logger.info(f"成功加载 LLM 摘要: {len(data)} 个条目")
        print(f"✓ 加载 LLM 摘要: {len(data)} 个条目")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败: {e}")
        print(f"❌ 错误: JSON 解析失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"加载 LLM 摘要失败: {e}", exc_info=True)
        print(f"❌ 错误: 加载 LLM 摘要失败: {e}")
        return {}


def extract_character_profile(combined_text: str) -> str:
    """
    从 combined_text 中提取"人物与看点"部分（去掉重复的剧情梗概）
    
    Returns:
        str: 人物与看点的完整文本，如果没有则返回空字符串
    """
    if not combined_text or "=== 人物与看点 ===" not in combined_text:
        return ""
    
    try:
        # 提取人物与看点部分（去掉"剧情梗概"部分）
        parts = combined_text.split("=== 人物与看点 ===")
        if len(parts) < 2:
            return ""
        
        character_section = parts[1].split("=== 剧情梗概 ===")[0].strip()
        return character_section
    except Exception as e:
        logger.warning(f"提取人物与看点失败: {e}", exc_info=True)
        print(f"  ⚠ 提取人物与看点失败: {e}")
        return ""


def extract_character_keywords(combined_text: str) -> Dict[str, List[str]]:
    """
    从 combined_text 的"人物与看点"部分提取关键信息（用于 metadata）
    
    改进的正则表达式，更健壮：
    - 支持中英文冒号、全角半角
    - 支持多种空格和换行格式
    - 容错性更强
    
    Returns:
        dict: {
            'occupations': [职业列表],
            'character_traits': [性格特征列表],
            'relationships': [关系模式列表],
            'conflicts': [核心冲突列表]
        }
    """
    result = {
        'occupations': [],
        'character_traits': [],
        'relationships': [],
        'conflicts': []
    }
    
    if not combined_text or "=== 人物与看点 ===" not in combined_text:
        return result
    
    try:
        # 提取人物与看点部分
        parts = combined_text.split("=== 人物与看点 ===")
        if len(parts) < 2:
            return result
        
        character_section = parts[1].split("=== 剧情梗概 ===")[0]
        
        # 使用预编译的正则表达式（性能优化）
        # 改进的职业身份提取（支持多种格式）
        for pattern in COMPILED_PATTERNS['occupation']:
            occupations = pattern.findall(character_section)
            if occupations:
                result['occupations'] = [o.strip() for o in occupations if o.strip()]
                break
        
        # 改进的性格特征提取
        for pattern in COMPILED_PATTERNS['trait']:
            traits = pattern.findall(character_section)
            if traits:
                result['character_traits'] = [t.strip() for t in traits if t.strip()]
                break
        
        # 改进的关系模式提取（更宽松的匹配）
        for pattern in COMPILED_PATTERNS['relationship']:
            relationships = pattern.findall(character_section)
            if relationships:
                # 处理不同的匹配组格式
                if isinstance(relationships[0], tuple):
                    if len(relationships[0]) == 3:
                        result['relationships'] = [f"{r[0]}与{r[1]}：{r[2]}" for r in relationships]
                    else:
                        result['relationships'] = [r.strip() if isinstance(r, str) else ' '.join(r).strip() for r in relationships]
                else:
                    result['relationships'] = [r.strip() for r in relationships if r.strip()]
                break
        
        # 改进的核心冲突提取（支持多种列表格式）
        for pattern in COMPILED_PATTERNS['conflict']:
            conflicts = pattern.findall(character_section)
            if conflicts:
                result['conflicts'] = [c.strip() for c in conflicts if c.strip()]
                break
        
    except Exception as e:
        print(f"  ⚠ 提取人物关键词失败: {e}")
    
    return result


def load_series_with_episodes_generator(db_path: str):
    """
    从SQLite数据库流式加载剧集及其分集数据（生成器模式，避免内存溢出）
    
    修复：确保数据库连接在 finally 块中关闭，避免连接泄漏
    
    Yields:
        Dict: 单个剧集数据（包含episodes列表）
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # 使用迭代器模式，不一次性加载所有数据
        cursor.execute('''
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
            WHERE s.title IS NOT NULL
            ORDER BY s.id, e.ep_number
        ''')
        
        current_series = None
        current_series_id = None
        
        # 流式处理，逐行读取
        for row in cursor:
            try:
                series_id = row['series_id']
                
                # 如果遇到新的剧集，先 yield 上一个剧集
                if current_series_id is not None and series_id != current_series_id:
                    yield current_series
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
            except Exception as e:
                logger.error(f"处理数据库行时出错: {e}", exc_info=True)
                continue  # 跳过有问题的行，继续处理
        
        # yield 最后一个剧集
        if current_series is not None:
            yield current_series
            
    except sqlite3.Error as e:
        logger.error(f"数据库操作失败: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"加载数据时发生未知错误: {e}", exc_info=True)
        raise
    finally:
        # 确保数据库连接关闭
        if conn is not None:
            try:
                conn.close()
                logger.debug("数据库连接已关闭")
            except Exception as e:
                logger.error(f"关闭数据库连接时出错: {e}", exc_info=True)


def build_rich_text_document(
    series: Dict,
    llm_summary: Dict
) -> Tuple[Document, List[Document]]:
    """
    构建富文本索引的文档（有 LLM 摘要的剧集）
    
    策略：
    - 使用 plot_summary + 人物与看点（去掉重复部分）作为主要文本
    - 人物与看点的完整内容比 tags 更精准，要充分使用
    - 标签作为 metadata（用于过滤）
    - 人物与看点的关键信息也提取为 metadata（双重利用）
    
    Returns:
        tuple: (parent_doc, child_docs)
    """
    series_id = series['series_id']
    
    # 使用 plot_summary 作为剧情摘要
    plot_summary = llm_summary.get('plot_summary', series.get('summary', '') or '暂无剧情简介')
    
    # 从 combined_text 提取人物与看点部分（去掉重复的剧情梗概）
    combined_text = llm_summary.get('combined_text', '')
    character_profile = extract_character_profile(combined_text)
    
    # 构建父节点文本（充分利用人物与看点信息）
    parent_content_parts = []
    parent_content_parts.append(f"剧名：{series['title']}")
    
    if series['cast']:
        parent_content_parts.append(f"主演：{series['cast']}")
    if series['director']:
        parent_content_parts.append(f"导演：{series['director']}")
    if series['year']:
        parent_content_parts.append(f"年份：{series['year']}")
    if series['region']:
        parent_content_parts.append(f"地区：{series['region']}")
    if series['genre']:
        parent_content_parts.append(f"类型：{series['genre']}")
    
    # 添加人物与看点（如果存在，这部分信息比 tags 更精准）
    if character_profile:
        parent_content_parts.append(f"\n人物与看点：\n{character_profile}")
    
    # 添加剧情摘要
    parent_content_parts.append(f"\n剧情摘要：\n{plot_summary}")
    parent_text = "\n".join(parent_content_parts)
    
    # 构建 metadata（最大化利用 LLM 生成的信息）
    doc_metadata = {
        "series_id": series_id,
        "doc_id": series['original_doc_id'],
        "title": series['title'],
        "year": series['year'] or '',
        "region": series['region'] or '',
        "genre": series['genre'] or '',
        "url": series['url'],
        "type": "series",
        "index_type": "rich_text",  # 标记为富文本索引
        "episode_count": len(series['episodes']),
        "has_llm_summary": True
    }
    
    # 添加标签到 metadata（用于过滤和增强检索）
    if llm_summary.get('tags'):
        doc_metadata['tags'] = ','.join(llm_summary['tags'])
    if llm_summary.get('occupation_tags'):
        doc_metadata['occupation_tags'] = ','.join(llm_summary['occupation_tags'])
    if llm_summary.get('character_tags'):
        doc_metadata['character_tags'] = ','.join(llm_summary['character_tags'])
    if llm_summary.get('style_tags'):
        doc_metadata['style_tags'] = ','.join(llm_summary['style_tags'])
    
    # 从 combined_text 提取人物与看点的关键信息（用于 metadata 过滤）
    # 注意：人物与看点的完整文本已经加入到主文本中，这里提取的是结构化关键词用于过滤
    if combined_text:
        char_keywords = extract_character_keywords(combined_text)
        if char_keywords['occupations']:
            doc_metadata['extracted_occupations'] = ','.join(char_keywords['occupations'][:5])  # 限制数量
        if char_keywords['character_traits']:
            doc_metadata['extracted_traits'] = ','.join(char_keywords['character_traits'][:5])
        if char_keywords['relationships']:
            doc_metadata['extracted_relationships'] = ','.join(char_keywords['relationships'][:3])
        if char_keywords['conflicts']:
            doc_metadata['extracted_conflicts'] = ','.join(char_keywords['conflicts'][:3])
    
    # 为父文档生成唯一的 node_id
    parent_node_id = f"series_{series_id}_{series['original_doc_id']}"
    
    parent_doc = Document(
        text=parent_text,
        metadata=doc_metadata,
        excluded_embed_metadata_keys=["series_id", "doc_id", "url", "index_type", "has_llm_summary"],
        id_=parent_node_id  # 设置父文档的 node_id
    )
    
    # 构建子节点（分集）- 充分利用分集剧情，并建立父子关系
    child_docs = []
    episodes_with_content = 0
    for ep in series['episodes']:
        if not ep.get('content'):
            continue
        
        episodes_with_content += 1
        
        # 构建分集文本（充分利用分集内容）
        ep_text_parts = []
        ep_text_parts.append(f"第{ep['ep_number']}集")
        if ep.get('episode_title'):
            ep_text_parts.append(f"《{ep['episode_title']}》")
        ep_text_parts.append(f"\n{ep['content']}")
        
        ep_text = "".join(ep_text_parts)
        
        # 为子文档生成唯一的 node_id
        child_node_id = f"episode_{series_id}_{ep['ep_number']}"
        
        # 建立父子关系：子文档指向父文档
        relationships = {
            NodeRelationship.PARENT: RelatedNodeInfo(
                node_id=parent_node_id,
                node_type="series",
                metadata={"title": series['title']}
            )
        }
        
        # 每个分集作为独立的 Document，会被单独索引和检索，但建立了父子关系
        child_doc = Document(
            text=ep_text,
            metadata={
                "series_id": series_id,
                "ep_number": ep['ep_number'],
                "episode_title": ep['episode_title'] or '',
                "episode_url": ep['episode_url'],
                "type": "episode",
                "parent_title": series['title'],
                "parent_doc_id": series['original_doc_id'],
                "index_type": "rich_text"
            },
            excluded_embed_metadata_keys=["series_id", "episode_url", "parent_doc_id", "index_type"],
            id_=child_node_id,
            relationships=relationships  # 建立父子关系
        )
        child_docs.append(child_doc)
    
    return parent_doc, child_docs


def build_basic_document(series: Dict) -> Tuple[Document, List[Document]]:
    """
    构建基础索引的文档（没有 LLM 摘要的剧集）
    
    Returns:
        tuple: (parent_doc, child_docs)
    """
    series_id = series['series_id']
    
    # 使用原始 summary
    summary = series.get('summary', '') or '暂无剧情简介'
    
    # 构建父节点文本
    parent_content_parts = []
    parent_content_parts.append(f"剧名：{series['title']}")
    
    if series['cast']:
        parent_content_parts.append(f"主演：{series['cast']}")
    if series['director']:
        parent_content_parts.append(f"导演：{series['director']}")
    if series['year']:
        parent_content_parts.append(f"年份：{series['year']}")
    if series['region']:
        parent_content_parts.append(f"地区：{series['region']}")
    if series['genre']:
        parent_content_parts.append(f"类型：{series['genre']}")
    
    parent_content_parts.append(f"\n剧情摘要：\n{summary}")
    parent_text = "\n".join(parent_content_parts)
    
    # 构建 metadata
    doc_metadata = {
        "series_id": series_id,
        "doc_id": series['original_doc_id'],
        "title": series['title'],
        "year": series['year'] or '',
        "region": series['region'] or '',
        "genre": series['genre'] or '',
        "url": series['url'],
        "type": "series",
        "index_type": "basic",  # 标记为基础索引
        "episode_count": len(series['episodes']),
        "has_llm_summary": False
    }
    
    # 为父文档生成唯一的 node_id
    parent_node_id = f"series_{series_id}_{series['original_doc_id']}"
    
    parent_doc = Document(
        text=parent_text,
        metadata=doc_metadata,
        excluded_embed_metadata_keys=["series_id", "doc_id", "url", "index_type", "has_llm_summary"],
        id_=parent_node_id  # 设置父文档的 node_id
    )
    
    # 构建子节点（分集）- 充分利用分集剧情，并建立父子关系
    child_docs = []
    for ep in series['episodes']:
        if not ep.get('content'):
            continue
        
        # 构建分集文本（充分利用分集内容）
        ep_text_parts = []
        ep_text_parts.append(f"第{ep['ep_number']}集")
        if ep.get('episode_title'):
            ep_text_parts.append(f"《{ep['episode_title']}》")
        ep_text_parts.append(f"\n{ep['content']}")
        
        ep_text = "".join(ep_text_parts)
        
        # 为子文档生成唯一的 node_id
        child_node_id = f"episode_{series_id}_{ep['ep_number']}"
        
        # 建立父子关系：子文档指向父文档
        relationships = {
            NodeRelationship.PARENT: RelatedNodeInfo(
                node_id=parent_node_id,
                node_type="series",
                metadata={"title": series['title']}
            )
        }
        
        # 每个分集作为独立的 Document，会被单独索引和检索，但建立了父子关系
        child_doc = Document(
            text=ep_text,
            metadata={
                "series_id": series_id,
                "ep_number": ep['ep_number'],
                "episode_title": ep['episode_title'] or '',
                "episode_url": ep['episode_url'],
                "type": "episode",
                "parent_title": series['title'],
                "parent_doc_id": series['original_doc_id'],
                "index_type": "basic"
            },
            excluded_embed_metadata_keys=["series_id", "episode_url", "parent_doc_id", "index_type"],
            id_=child_node_id,
            relationships=relationships  # 建立父子关系
        )
        child_docs.append(child_doc)
    
    return parent_doc, child_docs


def build_dual_track_documents_generator(
    db_path: str = "data/database/final.db",
    llm_sum_file: str = "data/llm_summaries.json",
    filter_type: Optional[str] = None
):
    """
    构建双轨制文档（生成器模式，避免内存溢出）
    
    优化：一次遍历数据库，同时生成两种类型的文档，避免重复加载
    
    Args:
        db_path: 数据库路径
        llm_sum_file: LLM摘要文件路径
        filter_type: 过滤类型，None表示不过滤，"rich_text"或"basic"表示只生成指定类型
    
    Yields:
        Document: Document 对象（如果 filter_type 指定，只返回该类型）
    """
    print("="*60)
    print("双轨制文档构建模块（流式处理）")
    print("="*60)
    
    # 1. 加载 LLM 摘要
    print("\n正在加载 LLM 摘要...")
    llm_summaries = load_llm_summaries(llm_sum_file)
    llm_series_ids = {int(k) for k in llm_summaries.keys()}
    print(f"  ✓ 有 LLM 摘要的剧集: {len(llm_series_ids)} 部")
    
    # 2. 流式处理数据库数据（生成器模式）
    print("\n正在流式处理数据库数据...")
    series_generator = load_series_with_episodes_generator(db_path)
    
    # 统计信息（使用列表存储，因为生成器只能迭代一次）
    stats = {
        'rich_text_count': 0,
        'basic_count': 0,
        'rich_text_episodes': 0,
        'basic_episodes': 0,
        'extraction_stats': {
            'character_profile_success': 0,
            'character_profile_failed': 0,
            'keywords_success': 0,
            'keywords_failed': 0
        }
    }
    
    print("\n正在构建文档（流式处理）...")
    # 使用 tqdm 显示进度条（如果可用）
    if HAS_TQDM:
        series_generator = tqdm(series_generator, desc="处理剧集", unit="部")
    
    for idx, series in enumerate(series_generator, 1):
        if not HAS_TQDM and idx % 100 == 0:
            print(f"  处理进度: {idx} 部剧集...")
        
        series_id = series['series_id']
        
        if series_id in llm_series_ids:
            # 富文本索引（修复：使用 .get() 方法，添加空值检查）
            llm_summary = llm_summaries.get(str(series_id))
            if not llm_summary:
                logger.warning(f"series_id {series_id} 在 llm_series_ids 中但 llm_summaries 中不存在，降级到基础索引")
                # 降级到基础索引
                parent_doc, child_docs = build_basic_document(series)
                if filter_type is None or filter_type == "basic":
                    yield parent_doc
                stats['basic_count'] += 1
                for child_doc in child_docs:
                    if filter_type is None or filter_type == "basic":
                        yield child_doc
                    stats['basic_episodes'] += 1
                continue
            
            try:
                # 统计人物与看点提取成功率
                combined_text = llm_summary.get('combined_text', '')
                character_profile = extract_character_profile(combined_text)
                if character_profile:
                    stats['extraction_stats']['character_profile_success'] += 1
                else:
                    stats['extraction_stats']['character_profile_failed'] += 1
                
                # 统计关键词提取成功率
                char_keywords = extract_character_keywords(combined_text)
                if any(char_keywords.values()):
                    stats['extraction_stats']['keywords_success'] += 1
                else:
                    stats['extraction_stats']['keywords_failed'] += 1
                
                parent_doc, child_docs = build_rich_text_document(series, llm_summary)
            except Exception as e:
                logger.error(f"构建富文本文档失败 (series_id={series_id}): {e}", exc_info=True)
                # 降级到基础索引
                parent_doc, child_docs = build_basic_document(series)
                if filter_type is None or filter_type == "basic":
                    yield parent_doc
                stats['basic_count'] += 1
                for child_doc in child_docs:
                    if filter_type is None or filter_type == "basic":
                        yield child_doc
                    stats['basic_episodes'] += 1
                continue
            
            # yield 父节点（如果不过滤或过滤类型匹配）
            if filter_type is None or filter_type == "rich_text":
                yield parent_doc
            stats['rich_text_count'] += 1
            
            # yield 子节点（分集）
            for child_doc in child_docs:
                if filter_type is None or filter_type == "rich_text":
                    yield child_doc
                stats['rich_text_episodes'] += 1
        else:
            # 基础索引
            parent_doc, child_docs = build_basic_document(series)
            
            # yield 父节点（如果不过滤或过滤类型匹配）
            if filter_type is None or filter_type == "basic":
                yield parent_doc
            stats['basic_count'] += 1
            
            # yield 子节点（分集）
            for child_doc in child_docs:
                if filter_type is None or filter_type == "basic":
                    yield child_doc
                stats['basic_episodes'] += 1
    
    # 输出统计信息（只在不过滤时输出，避免重复）
    if filter_type is None:
        print(f"\n✓ 文档构建完成:")
        print(f"  - 富文本索引: {stats['rich_text_count']} 部剧集, {stats['rich_text_episodes']} 个分集")
        print(f"  - 基础索引: {stats['basic_count']} 部剧集, {stats['basic_episodes']} 个分集")
        print(f"  - 总计: {stats['rich_text_count'] + stats['basic_count']} 部剧集, {stats['rich_text_episodes'] + stats['basic_episodes']} 个分集")
        
        # 输出提取统计
        total_rich = stats['rich_text_count']
        if total_rich > 0:
            profile_success_rate = stats['extraction_stats']['character_profile_success'] / total_rich * 100
            keywords_success_rate = stats['extraction_stats']['keywords_success'] / total_rich * 100
            print(f"\n📊 提取统计（富文本索引）:")
            print(f"  - 人物与看点提取成功率: {profile_success_rate:.1f}% ({stats['extraction_stats']['character_profile_success']}/{total_rich})")
            print(f"  - 关键词提取成功率: {keywords_success_rate:.1f}% ({stats['extraction_stats']['keywords_success']}/{total_rich})")
            if stats['extraction_stats']['character_profile_failed'] > 0:
                print(f"  ⚠ 警告: {stats['extraction_stats']['character_profile_failed']} 个条目的人物与看点提取失败")


def build_dual_track_documents(
    db_path: str = "data/database/final.db",
    llm_sum_file: str = "data/llm_summaries.json"
) -> Tuple[List[Document], List[Document]]:
    """
    构建双轨制文档（兼容接口，内部使用生成器模式）
    
    注意：此函数会加载所有文档到内存，仅用于兼容旧接口
    推荐直接使用 build_dual_track_documents_generator 进行流式处理
    
    Returns:
        tuple: (rich_text_documents, basic_documents)
    """
    rich_text_documents = []
    basic_documents = []
    
    # 使用生成器模式，流式处理
    for doc in build_dual_track_documents_generator(db_path, llm_sum_file, filter_type="rich_text"):
        rich_text_documents.append(doc)
    
    for doc in build_dual_track_documents_generator(db_path, llm_sum_file, filter_type="basic"):
        basic_documents.append(doc)
    
    return rich_text_documents, basic_documents


def main():
    """主函数"""
    db_path = "data/database/final.db"
    
    if not Path(db_path).exists():
        print(f"❌ 错误：数据库文件不存在: {db_path}")
        return [], []
    
    rich_text_docs, basic_docs = build_dual_track_documents(db_path)
    
    return rich_text_docs, basic_docs


if __name__ == '__main__':
    rich_docs, basic_docs = main()
    print(f"\n总计: {len(rich_docs) + len(basic_docs)} 个文档")

