#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从JSON文件中读取URL，爬取页面内容，提取结构化数据并存入SQLite数据库
"""

import json
import sqlite3
import re
import time
import logging
import threading
import queue
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed


def normalize_value(value: Optional[str], check_zero: bool = False) -> Optional[str]:
    """
    将空字符串、"暂无"等转换为None
    
    Args:
        value: 原始值（可以是None或字符串）
        check_zero: 是否检查"0"值（用于年份字段）
    
    Returns:
        Optional[str]: 规范化后的值，空值或"暂无"返回None，如果check_zero为True且值为"0"也返回None
        如果只包含空白字符（包括空行、空格等），也返回None
    """
    if not value:
        return None
    value_str = value.strip()
    if not value_str or value_str == "暂无":
        return None
    # 检查是否只包含空白字符（包括换行符、制表符等）
    if not re.sub(r'\s+', '', value_str):  # 移除所有空白字符后为空
        return None
    if check_zero and value_str == "0":
        return None
    return value_str


def parse_drama_page(html_content: str, url: Optional[str] = None) -> Dict:
    """
    解析剧集页面，提取结构化数据
    
    Args:
        html_content (str): 页面 HTML 内容
        url (str, optional): 当前页面的完整 URL
    
    Returns:
        dict: 包含 title, summary, cast, director, year, region, genre, url
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. title - 从 <title> 标签中获取，以"剧情介绍"为分界点截取前面的部分
    title_tag = soup.find('title')
    title = ""
    if title_tag:
        title_text = title_tag.get_text()
        title = title_text.split('剧情介绍')[0].strip()
    title = normalize_value(title)
    
    # 2. summary - 从 class 为 "m_jq" 的 div 元素中提取
    plot_div = soup.find('div', class_='m_jq')
    summary = plot_div.get_text(strip=True) if plot_div else ""
    # 清理全角空格
    summary = summary.replace('　', ' ').strip()
    summary = normalize_value(summary)
    
    # 3. cast - 找到包含"主演："的 em 标签，在其父元素中找出所有 a 链接
    cast = ""
    em_lead = soup.find('em', string=re.compile(r'^\s*主\s*演\s*：?\s*$'))
    if em_lead and em_lead.parent:
        parent_p = em_lead.parent
        actor_links = parent_p.find_all('a', href=True)
        actors = []
        for link in actor_links:
            text = link.get_text(strip=True)
            if '更多' not in text and text != '»':
                actors.append(text)
        cast = ", ".join(actors)
    cast = normalize_value(cast)
    
    # Helper function - 提取字段（导演、年份、地区、类型）
    def extract_field(label_pattern: str) -> str:
        """
        提取字段值
        
        Args:
            label_pattern: 标签文本的正则表达式模式（如'导\s*演'，允许中间有空格）
        
        Returns:
            str: 提取的字段值
        """
        # 构建正则表达式：允许标签文本中间有任意数量的空格
        # 例如：'导\s*演' -> 匹配"导"和"演"之间可以有任意数量的空格
        pattern = rf'^\s*{label_pattern}\s*：?\s*$'
        em = soup.find('em', string=re.compile(pattern))
        if em and em.parent:
            full_text = em.parent.get_text()
            # 移除em标签的文本，得到剩余的值
            em_text = em.get_text()
            value = full_text.replace(em_text, '', 1).strip()
            return value
        return ""
    
    director = normalize_value(extract_field(r'导\s*演'))
    year = normalize_value(extract_field(r'年\s*份'), check_zero=True)  # 年份为0时也返回None
    region = normalize_value(extract_field(r'地\s*区'))
    genre = normalize_value(extract_field(r'类\s*型'))
    
    return {
        "title": title,
        "summary": summary,
        "cast": cast,
        "director": director,
        "year": year,
        "region": region,
        "genre": genre,
        "url": url or ""
    }


def load_urls_from_json(json_file: Path) -> List[str]:
    """
    从JSON文件中加载URL列表
    
    Args:
        json_file: JSON文件路径
    
    Returns:
        List[str]: URL列表
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('urls', [])
    except Exception as e:
        print(f"读取JSON文件失败 {json_file}: {e}")
        return []


def load_urls_from_txt(txt_file: Path) -> List[str]:
    """
    从文本文件中加载URL列表（每行一个URL）
    
    Args:
        txt_file: 文本文件路径
    
    Returns:
        List[str]: URL列表
    """
    try:
        urls = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url:
                    urls.append(url)
        return urls
    except Exception as e:
        print(f"读取文本文件失败 {txt_file}: {e}")
        return []


def load_processed_urls(progress_file: Path, db_path: str) -> set:
    """
    从进度文件和数据库中加载已处理的URL集合（双重检查，确保一致性）
    
    Args:
        progress_file: 进度文件路径
        db_path: 数据库文件路径
    
    Returns:
        set: 已处理的URL集合
    """
    processed_urls = set()
    
    # 1. 从进度文件加载
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    url = line.strip()
                    if url:
                        processed_urls.add(url)
            print(f"从进度文件加载了 {len(processed_urls)} 个URL")
        except Exception as e:
            print(f"读取进度文件失败: {e}")
    
    # 2. 从数据库加载已存在的URL（作为最终权威来源）
    try:
        conn = sqlite3.connect(db_path, timeout=30.0)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT url FROM series_metadata WHERE url IS NOT NULL')
            db_urls = {row[0] for row in cursor.fetchall()}
            processed_urls.update(db_urls)  # 合并数据库中的URL
            print(f"从数据库加载了 {len(db_urls)} 个已存在的URL")
        finally:
            conn.close()
    except Exception as e:
        print(f"从数据库加载URL失败: {e}")
    
    print(f"总共加载了 {len(processed_urls)} 个已处理的URL（进度文件+数据库）")
    return processed_urls


def save_processed_url(progress_file: Path, url: str, lock: threading.Lock):
    """
    将已处理的URL追加到进度文件（线程安全）
    
    Args:
        progress_file: 进度文件路径
        url: 已处理的URL
        lock: 线程锁
    """
    try:
        with lock:
            with open(progress_file, 'a', encoding='utf-8') as f:
                f.write(url + '\n')
    except Exception as e:
        print(f"保存进度失败: {e}")


def fetch_html(url: str, max_retries: int = 3, timeout: int = 10) -> Optional[str]:
    """
    爬取URL的HTML内容
    
    Args:
        url: 目标URL
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
    
    Returns:
        Optional[str]: HTML内容，失败返回None
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  重试 {attempt + 1}/{max_retries}: {url}")
                time.sleep(2 ** attempt)  # 指数退避
            else:
                print(f"  爬取失败（已重试{max_retries}次）: {url} - {e}")
                return None
    
    return None


def init_database(db_path: str) -> sqlite3.Connection:
    """
    初始化数据库，创建表
    
    Args:
        db_path: 数据库文件路径
    
    Returns:
        sqlite3.Connection: 数据库连接
    """
    # 设置timeout以支持多线程并发访问
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    # 创建表（使用series_metadata作为表名）
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS series_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT UNIQUE,
            title TEXT,
            summary TEXT,
            cast TEXT,
            director TEXT,
            year TEXT,
            region TEXT,
            genre TEXT,
            url TEXT UNIQUE
        )
    ''')
    
    conn.commit()
    return conn


def get_max_doc_id_from_dir(html_dir: Path) -> int:
    """
    从目录中获取最大的doc_id数字（格式：a0, a1, a2...）
    
    Args:
        html_dir: HTML文件保存目录
    
    Returns:
        int: 最大的数字，如果目录为空则返回-1
    """
    max_num = -1
    if html_dir.exists():
        for file in html_dir.glob('a*.html'):
            try:
                # 提取文件名中的数字（去掉'a'前缀和.html后缀）
                stem = file.stem  # 例如 "a0"
                if stem.startswith('a'):
                    num = int(stem[1:])
                    max_num = max(max_num, num)
            except ValueError:
                continue
    return max_num


def save_html_file(html_dir: Path, doc_id: str, html_content: str) -> bool:
    """
    保存HTML文件（如果文件已存在则不覆盖）
    
    Args:
        html_dir: HTML文件保存目录
        doc_id: 文档ID（格式：a0, a1, a2...）
        html_content: HTML内容
    
    Returns:
        bool: 是否保存成功（如果文件已存在则返回False，表示未保存）
    """
    try:
        # 确保目录存在
        html_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存文件（文件名格式：a0.html, a1.html...）
        html_file = html_dir / f'{doc_id}.html'
        
        # 检查文件是否已存在，如果存在则不覆盖
        if html_file.exists():
            return False  # 文件已存在，不覆盖
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return True
    except Exception as e:
        print(f"  保存HTML文件失败: {e}")
        return False


def count_none_fields(drama_data: Dict) -> Tuple[int, List[str]]:
    """
    统计数据中None字段的数量和名称
    
    Args:
        drama_data: 剧集数据字典
    
    Returns:
        tuple: (None数量, None字段名称列表)
    """
    none_fields = []
    fields_to_check = ['title', 'summary', 'cast', 'director', 'year', 'region', 'genre']
    
    for field in fields_to_check:
        if drama_data.get(field) is None:
            none_fields.append(field)
    
    return len(none_fields), none_fields


def log_incomplete_record(logger: logging.Logger, url: str, none_fields: List[str]):
    """
    记录不完整的数据到日志
    
    Args:
        logger: 日志记录器
        url: URL地址
        none_fields: None字段列表
    """
    logger.warning(f"URL: {url}")
    logger.warning(f"缺失字段: {', '.join(none_fields)}")
    logger.warning("-" * 80)


def check_url_exists(db_path: str, url: str) -> bool:
    """
    检查URL是否已存在于数据库中（线程安全）
    
    Args:
        db_path: 数据库文件路径
        url: 要检查的URL
    
    Returns:
        bool: URL是否存在
    """
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM series_metadata WHERE url = ?', (url,))
        result = cursor.fetchone()
        return result is not None
    finally:
        conn.close()


def get_doc_id_by_url(db_path: str, url: str) -> Optional[str]:
    """
    根据URL获取数据库中已存在的doc_id（线程安全）
    
    Args:
        db_path: 数据库文件路径
        url: URL地址
    
    Returns:
        Optional[str]: 如果URL存在则返回对应的doc_id，否则返回None
    """
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT doc_id FROM series_metadata WHERE url = ?', (url,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()


def batch_check_urls_exist(db_path: str, urls: List[str]) -> set:
    """
    批量检查URL是否已存在于数据库中（线程安全）
    
    Args:
        db_path: 数据库文件路径
        urls: URL列表
    
    Returns:
        set: 已存在的URL集合
    """
    if not urls:
        return set()
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    try:
        cursor = conn.cursor()
        # 使用IN子句批量查询
        placeholders = ','.join('?' * len(urls))
        cursor.execute(f'SELECT url FROM series_metadata WHERE url IN ({placeholders})', urls)
        existing_urls = {row[0] for row in cursor.fetchall()}
        return existing_urls
    finally:
        conn.close()


def batch_insert_dramas(db_path: str, drama_list: List[Dict]) -> Tuple[int, int, List[str]]:
    """
    批量插入剧集数据到数据库（在插入前先检查URL和doc_id是否已存在）
    
    Args:
        db_path: 数据库文件路径
        drama_list: 剧集数据字典列表（包含doc_id）
    
    Returns:
        tuple: (成功插入数量, 跳过数量, 成功插入的URL列表)
    """
    if not drama_list:
        return 0, 0, []
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    success_count = 0
    skip_count = 0
    success_urls = []  # 成功插入的URL列表
    
    try:
        cursor = conn.cursor()
        
        # 批量检查URL和doc_id是否已存在
        urls = [d.get('url', '') for d in drama_list]
        doc_ids = [d.get('doc_id') for d in drama_list if d.get('doc_id')]
        
        existing_urls = set()
        existing_doc_ids = set()
        
        if urls:
            placeholders = ','.join('?' * len(urls))
            cursor.execute(f'SELECT url FROM series_metadata WHERE url IN ({placeholders})', urls)
            existing_urls = {row[0] for row in cursor.fetchall()}
        
        if doc_ids:
            placeholders = ','.join('?' * len(doc_ids))
            cursor.execute(f'SELECT doc_id FROM series_metadata WHERE doc_id IN ({placeholders})', doc_ids)
            existing_doc_ids = {row[0] for row in cursor.fetchall()}
        
        # 过滤出需要插入的数据（URL和doc_id都不存在）
        to_insert = []
        for drama_data in drama_list:
            url = drama_data.get('url', '')
            doc_id = drama_data.get('doc_id')
            
            if url in existing_urls or (doc_id and doc_id in existing_doc_ids):
                skip_count += 1
                continue
            
            to_insert.append(drama_data)
        
        if not to_insert:
            return success_count, skip_count, []
        
        # 准备批量插入数据
        values = []
        for drama_data in to_insert:
            values.append((
                drama_data.get('doc_id'),
                drama_data.get('title') or '',
                drama_data.get('summary') or '',
                drama_data.get('cast') or '',
                drama_data.get('director') or '',
                drama_data.get('year') or '',
                drama_data.get('region') or '',
                drama_data.get('genre') or '',
                drama_data.get('url', '')
            ))
        
        # 批量插入（使用事务确保原子性）
        try:
            cursor.executemany('''
                INSERT INTO series_metadata (doc_id, title, summary, cast, director, year, region, genre, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', values)
            conn.commit()
            success_count = len(to_insert)
            # 记录成功插入的URL
            success_urls = [d.get('url', '') for d in to_insert]
        except sqlite3.IntegrityError:
            # 如果仍然有唯一性冲突（可能是并发插入导致的），逐个插入
            conn.rollback()
            for drama_data in to_insert:
                try:
                    cursor.execute('''
                        INSERT INTO series_metadata (doc_id, title, summary, cast, director, year, region, genre, url)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        drama_data.get('doc_id'),
                        drama_data.get('title') or '',
                        drama_data.get('summary') or '',
                        drama_data.get('cast') or '',
                        drama_data.get('director') or '',
                        drama_data.get('year') or '',
                        drama_data.get('region') or '',
                        drama_data.get('genre') or '',
                        drama_data.get('url', '')
                    ))
                    conn.commit()
                    success_count += 1
                    success_urls.append(drama_data.get('url', ''))
                except sqlite3.IntegrityError:
                    # 唯一性冲突，跳过（可能是并发插入导致的）
                    skip_count += 1
                    conn.rollback()
                except Exception as e:
                    print(f"  批量插入时发生错误: {e}")
                    skip_count += 1
                    conn.rollback()
    except Exception as e:
        print(f"  批量插入数据库失败: {e}")
        conn.rollback()
    finally:
        conn.close()
    
    return success_count, skip_count, success_urls


def batch_insert_worker(
    db_path: str,
    data_queue: queue.Queue,
    batch_size: int,
    batch_timeout: float,
    stop_event: threading.Event,
    stats_lock: threading.Lock,
    stats: Dict[str, int],
    progress_file: Path,
    progress_lock: threading.Lock
):
    """
    批量插入工作线程（后台线程）
    
    Args:
        db_path: 数据库文件路径
        data_queue: 数据队列
        batch_size: 批量插入的大小
        batch_timeout: 批量插入的超时时间（秒）
        stop_event: 停止事件
        stats_lock: 统计信息锁
        stats: 统计信息字典
        progress_file: 进度文件路径
        progress_lock: 进度文件写入锁
    """
    batch = []
    batch_urls = []  # 记录批次中的URL，用于成功后保存进度
    last_insert_time = time.time()
    
    def insert_batch():
        """执行批量插入并保存进度"""
        nonlocal batch, batch_urls, last_insert_time
        if not batch:
            return
        
        success, skip, success_urls = batch_insert_dramas(db_path, batch)
        with stats_lock:
            stats['success'] += success
            stats['skip'] += skip
        
        # 批量插入成功后，保存成功插入的URL到进度文件
        # 只保存成功插入的URL（确保进度文件与数据库一致）
        if success_urls:
            with progress_lock:
                try:
                    with open(progress_file, 'a', encoding='utf-8') as f:
                        for url in success_urls:
                            f.write(url + '\n')
                except Exception as e:
                    print(f"  保存进度文件失败: {e}")
        
        print(f"  [批量插入] 成功 {success}, 跳过 {skip}")
        batch = []
        batch_urls = []
        last_insert_time = time.time()
    
    while not stop_event.is_set() or not data_queue.empty():
        try:
            # 从队列获取数据，设置超时避免无限等待
            try:
                drama_data = data_queue.get(timeout=1.0)
            except queue.Empty:
                # 如果队列为空，检查是否需要刷新批次
                if batch and (time.time() - last_insert_time) >= batch_timeout:
                    insert_batch()
                # 如果stop_event已设置且队列为空，退出循环
                if stop_event.is_set() and data_queue.empty():
                    break
                continue
            
            # 将数据添加到批次
            batch.append(drama_data)
            batch_urls.append(drama_data.get('url', ''))
            
            # 如果批次达到指定大小，执行批量插入
            if len(batch) >= batch_size:
                insert_batch()
            
            data_queue.task_done()
            
        except Exception as e:
            print(f"  批量插入工作线程错误: {e}")
            import traceback
            traceback.print_exc()
    
    # 处理剩余的批次（确保所有数据都被插入）
    if batch:
        print(f"  处理剩余的 {len(batch)} 条数据...")
        insert_batch()
    
    # 确保所有队列任务都完成
    data_queue.join()
    print("  批量插入工作线程完成")


def setup_logger(log_file: Path) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        log_file: 日志文件路径
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger('incomplete_records')
    logger.setLevel(logging.WARNING)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.WARNING)
    
    # 格式化
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger


def process_single_url(
    url: str,
    db_path: str,
    html_dir: Path,
    progress_file: Path,
    logger: logging.Logger,
    doc_id_lock: threading.Lock,
    progress_lock: threading.Lock,
    stats_lock: threading.Lock,
    next_doc_id_ref: list,  # 使用列表来传递可变引用
    stats: Dict[str, int],  # 统计信息字典
    data_queue: queue.Queue,  # 数据队列
    allowed_urls_set: set,  # URL白名单集合，确保只处理允许的URL
    skip_none_check: bool = False  # 是否跳过None字段数量检查（用于重新爬取）
) -> None:
    """
    处理单个URL（线程安全）
    
    Args:
        url: 要处理的URL
        db_path: 数据库文件路径
        html_dir: HTML文件保存目录
        progress_file: 进度文件路径
        logger: 日志记录器
        doc_id_lock: doc_id分配锁
        progress_lock: 进度文件写入锁
        stats_lock: 统计信息更新锁
        next_doc_id_ref: 下一个doc_id的引用（列表形式，用于传递可变引用）
        stats: 统计信息字典
        data_queue: 数据队列
        allowed_urls_set: URL白名单集合，确保只处理允许的URL
        skip_none_check: 是否跳过None字段数量检查（用于重新爬取incomplete_urls.txt中的URL）
    """
    try:
        # 验证URL是否在白名单中（安全措施，防止处理意外的URL）
        if url not in allowed_urls_set:
            print(f"  ⚠ 拒绝处理：URL不在白名单中: {url}")
            with stats_lock:
                stats['skip'] += 1
            return
        
        # 检查URL是否已存在于数据库（双重检查，避免重复爬取）
        existing_doc_id = get_doc_id_by_url(db_path, url)
        if existing_doc_id:
            with stats_lock:
                stats['skip'] += 1
            print(f"  ⊘ 跳过（URL已存在于数据库，doc_id: {existing_doc_id}）: {url}")
            save_processed_url(progress_file, url, progress_lock)
            return
        
        # 爬取HTML
        html_content = fetch_html(url)
        if html_content is None:
            with stats_lock:
                stats['error'] += 1
            print(f"  ✗ 爬取失败: {url}")
            save_processed_url(progress_file, url, progress_lock)
            return
        
        # 线程安全地分配doc_id（从目录中最大数字+1开始）
        with doc_id_lock:
            # 获取目录中最大的doc_id数字
            max_num = get_max_doc_id_from_dir(html_dir)
            # 从最大数字+1开始分配
            current_num = max_num + 1
            doc_id = f'a{current_num}'
            # 更新next_doc_id_ref指向下一个doc_id
            next_doc_id_ref[0] = f'a{current_num + 1}'
        
        # 保存HTML文件（如果文件已存在，说明并发冲突，递增doc_id重试）
        max_retries = 5  # 最多重试5次
        retry_count = 0
        html_saved = False
        while retry_count < max_retries and not html_saved:
            if save_html_file(html_dir, doc_id, html_content):
                # 保存成功
                html_saved = True
            else:
                # 文件已存在（并发冲突），递增doc_id并重试
                retry_count += 1
                with doc_id_lock:
                    current_num = int(doc_id[1:])
                    doc_id = f'a{current_num + 1}'
                    next_doc_id_ref[0] = f'a{current_num + 2}'
        
        if not html_saved:
            print(f"  错误: 无法找到可用的doc_id来保存HTML文件（重试{max_retries}次）: {url}")
            with stats_lock:
                stats['error'] += 1
            save_processed_url(progress_file, url, progress_lock)
            return
        
        # 解析页面
        drama_data = parse_drama_page(html_content, url=url)
        drama_data['doc_id'] = doc_id
        
        # 检查None字段数量（如果skip_none_check为True，则跳过此检查）
        if not skip_none_check:
            none_count, none_fields = count_none_fields(drama_data)
            
            if none_count > 1:
                # None数量超过1，记录到日志，不存入数据库
                with stats_lock:
                    stats['incomplete'] += 1
                log_incomplete_record(logger, url, none_fields)
                print(f"  ⚠ 跳过（缺失字段超过1个: {', '.join(none_fields)}）: {url}")
                save_processed_url(progress_file, url, progress_lock)
                return
        
        # 再次检查URL是否已存在（防止并发插入）
        if check_url_exists(db_path, url):
            with stats_lock:
                stats['skip'] += 1
            print(f"  ⊘ 跳过（URL已存在，并发检测）: {url}")
            save_processed_url(progress_file, url, progress_lock)
            return
        
        # 将数据放入队列，由批量插入线程处理
        data_queue.put(drama_data)
        title = drama_data.get('title') or 'N/A'
        print(f"  ✓ 已加入队列: {title} (doc_id: {doc_id})")
        
        # 注意：进度文件会在批量插入成功后由 batch_insert_worker 保存
        # 这里不立即保存，确保进度文件与数据库一致
        
    except Exception as e:
        print(f"  处理URL时发生错误 {url}: {e}")
        with stats_lock:
            stats['error'] += 1
        save_processed_url(progress_file, url, progress_lock)


def main():
    """主函数"""
    # 检查是否要重新爬取incomplete_urls.txt中的URL
    incomplete_urls_file = Path(__file__).parent / 'incomplete_urls.txt'
    retry_incomplete = incomplete_urls_file.exists()
    
    if retry_incomplete:
        print("="*50)
        print("检测到 incomplete_urls.txt 文件，将重新爬取其中的URL")
        print("本次爬取将忽略None字段数量检查，直接存入数据库")
        print("="*50 + "\n")
    
    # JSON文件路径
    base_dir = Path(__file__).parent / '1'
    json_files = [
        base_dir / 'show_1大陆' / 'urls_show_1.json',
        base_dir / 'show_4韩美' / 'urls_show_4.json',
        base_dir / 'show_50香港' / 'urls_show_50.json',
        base_dir / 'show_51网络' / 'urls_show_51.json',
    ]
    
    # 初始化数据库
    db_path = Path(__file__).parent / 'series.db'
    conn = init_database(str(db_path))
    conn.close()  # 关闭初始连接，每个线程会创建自己的连接
    print(f"数据库已初始化: {db_path}")
    
    # 初始化日志
    log_file = Path(__file__).parent / 'incomplete_records.log'
    logger = setup_logger(log_file)
    print(f"日志文件: {log_file}")
    
    # 初始化进度文件
    progress_file = Path(__file__).parent / 'crawl_progress.txt'
    processed_urls = load_processed_urls(progress_file, str(db_path))
    
    # 初始化HTML保存目录
    html_dir = Path(__file__).parent / 'html_files'
    html_dir.mkdir(parents=True, exist_ok=True)
    print(f"HTML文件保存目录: {html_dir}")
    
    # 获取下一个doc_id（从目录中最大数字+1）
    max_num = get_max_doc_id_from_dir(html_dir)
    next_doc_id = f'a{max_num + 1}'
    print(f"下一个doc_id: {next_doc_id} (目录中最大数字: {max_num})")
    
    # 收集所有URL
    all_urls = []
    if retry_incomplete:
        # 如果存在incomplete_urls.txt，只处理其中的URL（不再从JSON文件加载）
        incomplete_urls = load_urls_from_txt(incomplete_urls_file)
        all_urls.extend(incomplete_urls)
        print(f"从 incomplete_urls.txt 加载了 {len(incomplete_urls)} 个URL")
        print("⚠️  重新爬取模式：只处理 incomplete_urls.txt 中的URL，不会爬取其他URL")
    else:
        # 否则从JSON文件加载
        for json_file in json_files:
            if json_file.exists():
                urls = load_urls_from_json(json_file)
                all_urls.extend(urls)
                print(f"从 {json_file.name} 加载了 {len(urls)} 个URL")
            else:
                print(f"警告: 文件不存在 {json_file}")
    
    # 创建URL白名单集合（用于验证，确保只处理允许的URL）
    if retry_incomplete:
        # 重新爬取模式：白名单只包含 incomplete_urls.txt 中的URL
        allowed_urls_set = set(all_urls)
        print(f"URL白名单总数: {len(allowed_urls_set)} 个URL（仅来自 incomplete_urls.txt）")
    else:
        # 正常模式：只处理JSON文件中的URL
        allowed_urls_set = set(all_urls)
        print(f"URL白名单总数: {len(allowed_urls_set)} 个URL（来自JSON文件）")
    
    # 过滤掉已处理的URL（对于重新爬取模式，仍然需要检查，避免重复爬取）
    remaining_urls = [url for url in all_urls if url not in processed_urls]
    skipped_count = len(all_urls) - len(remaining_urls)
    
    if skipped_count > 0:
        print(f"\n已跳过 {skipped_count} 个已处理的URL（从进度文件恢复）")
    print(f"剩余需要爬取 {len(remaining_urls)} 个URL")
    
    # 设置线程数（可以根据需要调整）
    max_workers = 10
    print(f"使用 {max_workers} 个线程进行并发爬取")
    
    # 批量插入配置
    batch_size = 100  # 批量插入大小
    batch_timeout = 5.0  # 批量插入超时时间（秒）
    print(f"批量插入配置: 批次大小={batch_size}, 超时={batch_timeout}秒\n")
    
    # 创建线程锁
    doc_id_lock = threading.Lock()
    progress_lock = threading.Lock()
    stats_lock = threading.Lock()
    
    # 使用列表来传递可变引用（用于doc_id递增）
    next_doc_id_ref = [next_doc_id]
    
    # 统计信息
    stats = {
        'success': 0,
        'skip': 0,
        'error': 0,
        'incomplete': 0
    }
    
    # 创建数据队列（线程安全）
    data_queue = queue.Queue()
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 启动批量插入工作线程
    batch_thread = threading.Thread(
        target=batch_insert_worker,
        args=(
            str(db_path),
            data_queue,
            batch_size,
            batch_timeout,
            stop_event,
            stats_lock,
            stats,
            progress_file,
            progress_lock
        ),
        daemon=True
    )
    batch_thread.start()
    print("批量插入工作线程已启动\n")
    
    # 使用线程池并发处理
    total = len(remaining_urls)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = {
                executor.submit(
                    process_single_url,
                    url,
                    str(db_path),
                    html_dir,
                    progress_file,
                    logger,
                    doc_id_lock,
                    progress_lock,
                    stats_lock,
                    next_doc_id_ref,
                    stats,
                    data_queue,
                    allowed_urls_set,  # 传递URL白名单
                    skip_none_check=retry_incomplete  # 如果是重新爬取模式，跳过None检查
                ): url for url in remaining_urls
            }
            
            # 处理完成的任务
            completed = 0
            for future in as_completed(futures):
                completed += 1
                url = futures[future]
                try:
                    future.result()  # 获取结果，如果有异常会抛出
                except Exception as e:
                    print(f"处理URL时发生未捕获的异常 {url}: {e}")
                
                # 显示进度
                if completed % 10 == 0 or completed == total:
                    print(f"\n进度: {completed}/{total} ({completed*100//total}%)\n")
    finally:
        # 等待所有数据被放入队列
        print("\n等待所有数据放入队列...")
        # 等待所有爬取任务完成，确保所有数据都已放入队列
        
        # 停止批量插入线程（设置停止事件）
        print("通知批量插入线程停止...")
        stop_event.set()
        
        # 等待批量插入线程处理完所有数据并结束
        print("等待批量插入完成...")
        batch_thread.join(timeout=60.0)  # 增加超时时间到60秒
        
        if batch_thread.is_alive():
            print("警告: 批量插入线程未能在超时时间内结束，可能还有数据未处理")
        else:
            print("批量插入线程已结束，所有数据已处理完成")
    
    # 打印统计信息
    print("\n" + "="*50)
    print("爬取完成！")
    print(f"本次处理: {total}")
    print(f"成功: {stats['success']}")
    print(f"跳过: {stats['skip']} (URL已存在)")
    print(f"不完整: {stats['incomplete']} (缺失字段>1，已记录到日志)")
    print(f"失败: {stats['error']}")
    if skipped_count > 0:
        print(f"从进度恢复: {skipped_count} 个URL")
    print("="*50)
    if stats['incomplete'] > 0:
        print(f"\n不完整记录已保存到: {log_file}")
    print(f"进度已保存到: {progress_file}")


if __name__ == '__main__':
    main()

