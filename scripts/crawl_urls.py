import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from collections import defaultdict, deque
import time
import random
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class URLCrawler:
    def __init__(self, base_url="https://www.jutingyuan.com"):
        self.base_url = base_url
        self.visited_urls = set()
        self.target_urls = defaultdict(set)
        
        # 创建Session来保持会话和Cookie
        self.session = requests.Session()
        
        # 定义URL模式（支持www和非www）
        self.patterns = {
            'show_1': re.compile(r'https://(www\.)?jutingyuan\.com/show/1-\d+\.html'),
            'story': re.compile(r'https://(www\.)?jutingyuan\.com/story/'),
            'actor': re.compile(r'https://(www\.)?jutingyuan\.com/actor/'),
            'show_5': re.compile(r'https://(www\.)?jutingyuan\.com/show/5-\d+\.html'),
            'show_4': re.compile(r'https://(www\.)?jutingyuan\.com/show/4-\d+\.html'),
            'show_50': re.compile(r'https://(www\.)?jutingyuan\.com/show/50-\d+\.html'),
            'show_51': re.compile(r'https://(www\.)?jutingyuan\.com/show/51-\d+\.html'),
            'show_17': re.compile(r'https://(www\.)?jutingyuan\.com/show/17-\d+\.html')
        }
        
        # 定义起始URL映射
        self.start_urls = {
            'show_1': 'https://www.jutingyuan.com/class/1---0.html',
            'story': 'https://www.jutingyuan.com/class/1---0.html',
            'actor': 'https://www.jutingyuan.com/class/1---0.html',
            'show_5': 'https://www.jutingyuan.com/class/5---0.html',
            'show_4': 'https://www.jutingyuan.com/class/4---0.html',
            'show_50': 'https://www.jutingyuan.com/class/50---0.html',
            'show_51': 'https://www.jutingyuan.com/class/51---0.html'
        }
        
        # 文件名映射
        self.filenames = {
            'show_1': 'urls_show_1.txt',
            'story': 'urls_story.txt',
            'actor': 'urls_actor.txt',
            'show_5': 'urls_show_5.txt',
            'show_4': 'urls_show_4.txt',
            'show_50': 'urls_show_50.txt',
            'show_51': 'urls_show_51.txt',
            'show_17': 'urls_show_17.txt'
        }
        
        # 记录story与show_17的对应关系 {story_url: [show_17_url1, show_17_url2, ...]}
        # 使用这种格式更节省空间，因为一个story通常对应多个show_17
        self.story_to_show_17 = defaultdict(list)
        
        # 存储上一个访问的URL，用于设置Referer
        self.last_url = None
        
        # 进度文件路径（按类型分别保存，避免单个文件过大）
        self.progress_visited_file = 'crawl_progress_visited.txt'  # 所有已访问URL（用于去重）
        # 按类型分别保存目标URL的JSON文件（避免单个文件过大）
        self.progress_targets_files = {
            pattern_name: f'crawl_progress_targets_{pattern_name}.json'
            for pattern_name in self.patterns.keys()
        }
        # story与show_17的对应关系文件（一个story对应多个show_17，节省空间）
        self.progress_story_mapping_file = 'crawl_progress_story_to_show_17.json'
        # 按类型分别保存已访问的URL（可选，用于更细粒度的恢复）
        self.progress_visited_by_type = {
            pattern_name: f'crawl_progress_visited_{pattern_name}.txt'
            for pattern_name in self.patterns.keys()
        }
        
        # 线程锁，用于多线程安全
        self.lock = Lock()
        
        # 加载之前的进度
        self.load_progress()
    
    def get_browser_headers(self, referer=None):
        """生成完整的浏览器请求头，模拟真实浏览器"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'same-origin' if referer else 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'DNT': '1',
        }
        
        # 如果有Referer，添加它
        if referer:
            headers['Referer'] = referer
        
        return headers
    
    def load_progress(self):
        """从文件加载之前的爬取进度（按类型分别加载）"""
        # 加载已访问的URL
        if os.path.exists(self.progress_visited_file):
            try:
                with open(self.progress_visited_file, 'r', encoding='utf-8') as f:
                    visited_count = 0
                    for line in f:
                        url = line.strip()
                        if url:
                            self.visited_urls.add(url)
                            visited_count += 1
                print(f"已加载 {visited_count} 个之前访问过的URL")
            except Exception as e:
                print(f"加载已访问URL时出错: {e}")
        
        # 按类型分别加载目标URL（避免单个文件过大）
        total_targets = 0
        for pattern_name, filename in self.progress_targets_files.items():
            if os.path.exists(filename):
                try:
                    with open(filename, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            # 如果是列表格式（旧格式兼容）
                            self.target_urls[pattern_name] = set(data)
                        elif isinstance(data, dict) and 'urls' in data:
                            # 如果是新格式，包含urls字段
                            self.target_urls[pattern_name] = set(data['urls'])
                        else:
                            self.target_urls[pattern_name] = set()
                        total_targets += len(self.target_urls[pattern_name])
                except Exception as e:
                    print(f"加载目标URL [{pattern_name}] 时出错: {e}")
        
        if total_targets > 0:
            print(f"已加载 {total_targets} 个之前找到的目标URL（按类型分别加载）")
        
        # 加载story与show_17的对应关系（一个story对应多个show_17）
        if os.path.exists(self.progress_story_mapping_file):
            try:
                with open(self.progress_story_mapping_file, 'r', encoding='utf-8') as f:
                    loaded_mapping = json.load(f)
                    # 转换为defaultdict(list)格式
                    for story_url, show_17_list in loaded_mapping.items():
                        self.story_to_show_17[story_url] = show_17_list
                    total_show_17 = sum(len(show_17_list) for show_17_list in self.story_to_show_17.values())
                    print(f"已加载 {len(self.story_to_show_17)} 个story对应的 {total_show_17} 个show_17关系")
            except Exception as e:
                print(f"加载story对应关系时出错: {e}")
    
    def save_progress(self, async_save=False):
        """保存当前爬取进度到文件（按类型分别保存，避免单个文件过大）
        Args:
            async_save: 是否异步保存（不等待完成）
        """
        try:
            # 使用锁保护，避免并发写入冲突
            with self.lock:
                visited_urls_copy = self.visited_urls.copy()
                target_urls_copy = {k: v.copy() for k, v in self.target_urls.items()}
            
            # 保存已访问的URL（主文件，用于快速去重检查）
            # 如果文件过大，可以考虑只保存最近访问的URL或使用数据库
            with open(self.progress_visited_file, 'w', encoding='utf-8') as f:
                # 分批写入，每批10000个，避免内存占用过大
                visited_list = sorted(visited_urls_copy)
                for i in range(0, len(visited_list), 10000):
                    batch = visited_list[i:i+10000]
                    f.write('\n'.join(batch) + '\n')
            
            # 按类型分别保存已访问的URL（可选，用于更细粒度的恢复）
            # 这样可以避免单个文件过大，也方便按类型恢复
            visited_by_type = defaultdict(set)
            for url in visited_urls_copy:
                pattern_match = self.matches_pattern(url)
                if pattern_match:
                    visited_by_type[pattern_match].add(url)
            
            for pattern_name, urls_set in visited_by_type.items():
                if urls_set:  # 只保存非空集合
                    filename = self.progress_visited_by_type[pattern_name]
                    with open(filename, 'w', encoding='utf-8') as f:
                        for url in sorted(urls_set):
                            f.write(url + '\n')
            
            # 按类型分别保存目标URL（避免单个文件过大）
            for pattern_name, urls_set in target_urls_copy.items():
                filename = self.progress_targets_files[pattern_name]
                data = {
                    'pattern': pattern_name,
                    'urls': sorted(list(urls_set)),
                    'count': len(urls_set)
                }
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 保存story与show_17的对应关系（一个story对应多个show_17，节省空间）
            with self.lock:
                # 转换为普通dict，因为defaultdict不能直接序列化
                story_mapping_copy = dict(self.story_to_show_17)
            
            with open(self.progress_story_mapping_file, 'w', encoding='utf-8') as f:
                json.dump(story_mapping_copy, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存进度时出错: {e}")
            return False
    
    def random_delay(self, min_delay=0.1, max_delay=0.3):
        """随机延迟，模拟人类浏览行为（优化后的延迟时间）"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def normalize_url(self, url):
        """规范化URL，统一使用www域名"""
        parsed = urlparse(url)
        # 只处理jutingyuan.com域名（非www版本）
        if parsed.netloc == 'jutingyuan.com':
            # 将非www域名转换为www域名
            parsed = parsed._replace(netloc='www.jutingyuan.com')
            url = parsed.geturl()
        # 对于相对URL或其他域名，保持不变
        return url
    
    def is_valid_url(self, url):
        """检查URL是否属于目标域名"""
        parsed = urlparse(url)
        # 支持www和非www域名
        return parsed.netloc in ['jutingyuan.com', 'www.jutingyuan.com'] or parsed.netloc == ''
    
    def matches_pattern(self, url):
        """检查URL是否匹配任一目标模式"""
        # 规范化URL
        url = self.normalize_url(url)
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(url):
                return pattern_name
        return None
    
    def extract_urls_from_page(self, html_content, current_url):
        """从HTML内容中提取所有链接"""
        soup = BeautifulSoup(html_content, 'html.parser')
        urls = set()
        
        # 提取所有a标签的href
        for link in soup.find_all('a', href=True):
            href = link['href']
            # 转换为绝对URL
            absolute_url = urljoin(current_url, href)
            # 移除fragment
            absolute_url = absolute_url.split('#')[0]
            
            # 规范化URL（统一使用www域名）
            absolute_url = self.normalize_url(absolute_url)
            
            # 检查是否属于目标域名
            if self.is_valid_url(absolute_url):
                urls.add(absolute_url)
        
        return urls
    
    def crawl_single_url(self, url, session, source_url=None):
        """爬取单个URL（用于多线程，确保线程安全）
        Args:
            url: 要爬取的URL
            session: 请求会话
            source_url: 来源URL（用于记录show_17与story的对应关系）
        """
        url = self.normalize_url(url)
        source_url = self.normalize_url(source_url) if source_url else None
        
        # 检查是否已访问（使用锁保护，确保原子性）
        with self.lock:
            if url in self.visited_urls:
                return None, []
            # 立即标记为"正在处理"，避免其他线程重复处理
            # 注意：这里不添加到visited_urls，因为如果请求失败，我们仍然希望标记为已访问
        
        # 检查是否匹配目标模式
        pattern_match = self.matches_pattern(url)
        if pattern_match:
            with self.lock:
                # 双重检查，确保不重复添加
                if url not in self.target_urls[pattern_match]:
                    self.target_urls[pattern_match].add(url)
                    print(f"找到目标URL [{pattern_match}]: {url}")
                    
                    # 如果找到show_17类型的URL，且来源是story类型，记录对应关系
                    # 使用 {story_url: [show_17_url1, show_17_url2, ...]} 格式，节省空间
                    if pattern_match == 'show_17' and source_url:
                        source_pattern = self.matches_pattern(source_url)
                        if source_pattern == 'story':
                            # 使用列表存储，一个story对应多个show_17
                            if url not in self.story_to_show_17[source_url]:
                                self.story_to_show_17[source_url].append(url)
                                count = len(self.story_to_show_17[source_url])
                                print(f"  记录对应关系: story ({source_url}) -> show_17 [{count}个]")
        
        try:
            headers = self.get_browser_headers(referer=self.base_url)
            response = session.get(
                url,
                headers=headers,
                timeout=10,  # 减少超时时间
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            # 标记为已访问（使用锁保护）
            with self.lock:
                # 再次检查，防止在请求过程中被其他线程标记
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                else:
                    # 如果已经被其他线程处理过，返回空结果
                    return None, []
            
            # 提取链接
            new_urls = self.extract_urls_from_page(response.text, url)
            
            # 如果当前URL是story类型，标记为来源URL，用于后续找到的show_17
            current_source = url if pattern_match == 'story' else source_url
            
            return url, new_urls, current_source
            
        except Exception as e:
            # 标记为已访问（即使失败也标记，避免重复尝试）
            with self.lock:
                self.visited_urls.add(url)
            return None, [], None
    
    def crawl(self, start_url, max_pages=None, delay=0.1, save_interval=100, max_workers=5):
        """爬取网站（优化版，支持多线程）
        Args:
            start_url: 起始URL
            max_pages: 最大爬取页面数，None表示无限制
            delay: 延迟时间（秒），已优化为更小的值
            save_interval: 每爬取多少页保存一次进度
            max_workers: 最大并发线程数
        """
        # 规范化起始URL
        start_url = self.normalize_url(start_url)
        
        # 如果起始URL已经访问过，跳过
        if start_url in self.visited_urls:
            print(f"起始URL已访问过，跳过: {start_url}")
            return
        
        # 使用deque提高性能，使用set跟踪待访问URL避免重复
        to_visit = deque([start_url])
        to_visit_set = {start_url}  # 用于快速检查URL是否已在待访问队列中
        pages_crawled = 0
        last_save_count = 0
        last_print_count = 0
        
        print(f"开始爬取，起始URL: {start_url}")
        print(f"并发线程数: {max_workers}")
        if max_pages is None:
            print("最大爬取页面数: 无限制（将爬取所有可访问的URL）")
        else:
            print(f"最大爬取页面数: {max_pages}")
        print(f"进度保存间隔: 每 {save_interval} 页保存一次")
        
        # 为每个线程创建独立的Session
        sessions = [requests.Session() for _ in range(max_workers)]
        session_index = 0
        
        while to_visit:
            # 如果设置了max_pages限制，则检查是否达到限制
            if max_pages is not None and pages_crawled >= max_pages:
                print(f"已达到最大爬取页面数限制: {max_pages}")
                break
            
            # 批量处理URL（最多处理max_workers个）
            batch_urls = []
            with self.lock:
                while len(batch_urls) < max_workers and to_visit:
                    url = to_visit.popleft()
                    url = self.normalize_url(url)
                    # 从待访问集合中移除
                    to_visit_set.discard(url)
                    # 检查是否已访问过
                    if url not in self.visited_urls:
                        batch_urls.append(url)
            
            if not batch_urls:
                # 如果没有新URL，等待一下
                time.sleep(0.1)
                continue
            
            # 使用线程池并发爬取
            # 使用字典跟踪每个URL的来源URL（用于记录show_17与story的对应关系）
            url_sources = {}  # {url: source_url}
            with ThreadPoolExecutor(max_workers=min(len(batch_urls), max_workers)) as executor:
                futures = {}
                for i, url in enumerate(batch_urls):
                    session = sessions[i % len(sessions)]
                    # 获取该URL的来源URL（如果之前有记录）
                    source_url = url_sources.get(url)
                    future = executor.submit(self.crawl_single_url, url, session, source_url)
                    futures[future] = url
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=15)
                        if len(result) == 3:
                            crawled_url, new_urls, current_source = result
                        else:
                            # 兼容旧版本（返回2个值）
                            crawled_url, new_urls = result
                            current_source = None
                            
                        if crawled_url:
                            pages_crawled += 1
                            
                            # 如果当前URL是story类型，记录为后续URL的来源
                            crawled_pattern = self.matches_pattern(crawled_url)
                            if crawled_pattern == 'story':
                                current_source = crawled_url
                            
                            # 添加新URL到待访问列表（使用锁和集合确保不重复）
                            for new_url in new_urls:
                                new_url = self.normalize_url(new_url)
                                # 检查是否已访问过或已在待访问队列中
                                with self.lock:
                                    if (new_url not in self.visited_urls and 
                                        new_url not in to_visit_set and 
                                        self.is_valid_url(new_url)):
                                        to_visit.append(new_url)
                                        to_visit_set.add(new_url)  # 添加到集合中
                                        # 如果来源是story类型，记录来源关系
                                        if current_source and self.matches_pattern(current_source) == 'story':
                                            url_sources[new_url] = current_source
                            
                            # 减少打印频率（每10页打印一次）
                            if pages_crawled - last_print_count >= 10:
                                with self.lock:
                                    queue_size = len(to_visit)
                                print(f"已爬取 {pages_crawled} 页，待访问队列: {queue_size} 个URL")
                                last_print_count = pages_crawled
                            
                            # 定期保存进度
                            if pages_crawled - last_save_count >= save_interval:
                                if self.save_progress():
                                    print(f"已保存进度（已爬取 {pages_crawled} 页，已访问 {len(self.visited_urls)} 个URL）")
                                last_save_count = pages_crawled
                            
                            # 短暂延迟，避免过于频繁
                            if delay > 0:
                                time.sleep(delay)
                    except Exception as e:
                        pass  # 错误已在crawl_single_url中处理
        
        # 最后保存一次进度
        self.save_progress()
        
        print(f"\n爬取完成！共爬取 {pages_crawled} 页")
        print(f"找到的目标URL统计:")
        for pattern_name, urls in self.target_urls.items():
            print(f"  {pattern_name}: {len(urls)} 个")
        print(f"进度已保存到文件")
    
    def validate_url(self, url, timeout=15):
        """验证URL是否可以正常访问，返回(是否有效, 错误信息)"""
        try:
            # 规范化URL
            url = self.normalize_url(url)
            
            # 生成完整的浏览器请求头
            headers = self.get_browser_headers(referer=self.base_url)
            
            # 使用Session发送请求，保持Cookie和会话
            response = self.session.get(
                url, 
                headers=headers, 
                timeout=timeout, 
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            return True, None
        except requests.exceptions.Timeout:
            error_msg = f"超时错误 (Timeout): 请求超过 {timeout} 秒"
            return False, error_msg
        except requests.exceptions.ConnectionError as e:
            error_msg = f"连接错误 (ConnectionError): {str(e)}"
            return False, error_msg
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "未知"
            error_msg = f"HTTP错误 (HTTP {status_code}): {str(e)}"
            return False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"请求异常 (RequestException): {str(e)}"
            return False, error_msg
        except Exception as e:
            error_msg = f"未知错误: {type(e).__name__}: {str(e)}"
            return False, error_msg
    
    def validate_url_batch(self, urls, max_workers=10):
        """批量验证URL（使用多线程加速）"""
        valid_urls = []
        invalid_count = 0
        
        def validate_single(url):
            is_valid, error_msg = self.validate_url(url, timeout=5)  # 减少超时时间
            return url, is_valid, error_msg
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(validate_single, url): url for url in urls}
            
            for future in as_completed(futures):
                try:
                    url, is_valid, error_msg = future.result()
                    if is_valid:
                        valid_urls.append(url)
                    else:
                        invalid_count += 1
                except Exception as e:
                    invalid_count += 1
        
        return valid_urls, invalid_count
    
    def save_urls(self, skip_validation=False):
        """验证并保存URL到文件，按类型分别保存到不同文件，避免单个文件过大
        Args:
            skip_validation: 是否跳过验证（直接保存所有URL，用于快速保存）
        """
        if skip_validation:
            print("\n跳过验证，直接保存所有URL（按类型分别保存）...")
            for pattern_name, urls in self.target_urls.items():
                if not urls:  # 跳过空集合
                    continue
                filename = self.filenames[pattern_name]
                # 分批写入，避免大文件一次性写入内存
                with open(filename, 'w', encoding='utf-8') as f:
                    sorted_urls = sorted(urls)
                    # 每批10000个URL写入一次
                    for i in range(0, len(sorted_urls), 10000):
                        batch = sorted_urls[i:i+10000]
                        f.write('\n'.join(batch) + '\n')
                print(f"[{pattern_name}] 已保存 {len(urls)} 个URL到 {filename}")
            
            # 保存story与show_17的对应关系（一个story对应多个show_17，节省空间）
            if self.story_to_show_17:
                mapping_filename = 'story_to_show_17_mapping.json'
                # 转换为普通dict以便序列化
                story_mapping = dict(self.story_to_show_17)
                with open(mapping_filename, 'w', encoding='utf-8') as f:
                    json.dump(story_mapping, f, ensure_ascii=False, indent=2)
                total_show_17 = sum(len(show_17_list) for show_17_list in story_mapping.values())
                print(f"已保存 {len(story_mapping)} 个story对应的 {total_show_17} 个show_17关系到: {mapping_filename}")
            return
        
        print("\n开始验证URL有效性（使用多线程加速，按类型分别保存）...")
        
        for pattern_name, urls in self.target_urls.items():
            if not urls:  # 跳过空集合
                continue
                
            filename = self.filenames[pattern_name]
            
            print(f"\n验证 [{pattern_name}] 类URL (共 {len(urls)} 个)...")
            
            # 批量验证
            valid_urls, invalid_count = self.validate_url_batch(list(urls), max_workers=10)
            
            # 保存有效的URL（分批写入，避免大文件一次性写入内存）
            with open(filename, 'w', encoding='utf-8') as f:
                sorted_valid_urls = sorted(valid_urls)
                # 每批10000个URL写入一次
                for i in range(0, len(sorted_valid_urls), 10000):
                    batch = sorted_valid_urls[i:i+10000]
                    f.write('\n'.join(batch) + '\n')
            
            print(f"\n[{pattern_name}] 验证完成:")
            print(f"  有效URL: {len(valid_urls)} 个")
            print(f"  失效URL: {invalid_count} 个")
            print(f"  已保存到: {filename}")
        
        # 保存story与show_17的对应关系（一个story对应多个show_17，节省空间）
        if self.story_to_show_17:
            mapping_filename = 'story_to_show_17_mapping.json'
            # 转换为普通dict以便序列化
            story_mapping = dict(self.story_to_show_17)
            with open(mapping_filename, 'w', encoding='utf-8') as f:
                json.dump(story_mapping, f, ensure_ascii=False, indent=2)
            total_show_17 = sum(len(show_17_list) for show_17_list in story_mapping.values())
            print(f"\n已保存 {len(story_mapping)} 个story对应的 {total_show_17} 个show_17关系到: {mapping_filename}")


def main():
    crawler = URLCrawler()
    
    # 优化参数：减少延迟，增加并发，提高保存间隔
    crawl_delay = 0.1  # 延迟从1秒降低到0.1秒
    save_interval = 200  # 保存间隔从50页增加到200页
    max_workers = 8  # 并发线程数
    
    # 第一阶段：从 class/1---0.html 开始爬取，寻找 show_1、story、actor
    print("=" * 60)
    print("第一阶段：爬取 show/1-、story/、actor/ 类型的URL")
    print("=" * 60)
    start_url_1 = "https://www.jutingyuan.com/class/1---0.html"
    crawler.crawl(start_url_1, max_pages=None, delay=crawl_delay, save_interval=save_interval, max_workers=max_workers)
    
    # 第二阶段：从 class/5---0.html 开始爬取，寻找 show_5（注意：此阶段不会包含story和actor）
    print("\n" + "=" * 60)
    print("第二阶段：爬取 show/5- 类型的URL（不包含story和actor）")
    print("=" * 60)
    start_url_5 = "https://www.jutingyuan.com/class/5---0.html"
    crawler.crawl(start_url_5, max_pages=None, delay=crawl_delay, save_interval=save_interval, max_workers=max_workers)
    
    # 第三阶段：从 class/4---0.html 开始爬取，寻找 show_4、story、actor
    print("\n" + "=" * 60)
    print("第三阶段：爬取 show/4-、story/、actor/ 类型的URL")
    print("=" * 60)
    start_url_4 = "https://www.jutingyuan.com/class/4---0.html"
    crawler.crawl(start_url_4, max_pages=None, delay=crawl_delay, save_interval=save_interval, max_workers=max_workers)
    
    # 第四阶段：从 class/50---0.html 开始爬取，寻找 show_50、story、actor
    print("\n" + "=" * 60)
    print("第四阶段：爬取 show/50-、story/、actor/ 类型的URL")
    print("=" * 60)
    start_url_50 = "https://www.jutingyuan.com/class/50---0.html"
    crawler.crawl(start_url_50, max_pages=None, delay=crawl_delay, save_interval=save_interval, max_workers=max_workers)
    
    # 第五阶段：从 class/51---0.html 开始爬取，寻找 show_51、story、actor
    print("\n" + "=" * 60)
    print("第五阶段：爬取 show/51-、story/、actor/ 类型的URL")
    print("=" * 60)
    start_url_51 = "https://www.jutingyuan.com/class/51---0.html"
    crawler.crawl(start_url_51, max_pages=None, delay=crawl_delay, save_interval=save_interval, max_workers=max_workers)
    
    # 最后保存一次进度
    crawler.save_progress()
    
    # 保存结果（可以选择跳过验证以加快速度）
    print("\n" + "=" * 60)
    print("开始保存URL")
    print("=" * 60)
    # 如果只需要快速保存，可以设置 skip_validation=True
    # 如果需要验证URL有效性，设置 skip_validation=False（会慢一些）
    crawler.save_urls(skip_validation=False)
    
    print("\n所有URL已保存完成！")
    print(f"爬取进度已保存，下次运行将自动从中断点恢复")


if __name__ == "__main__":
    main()

