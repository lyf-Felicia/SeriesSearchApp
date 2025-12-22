import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from collections import defaultdict
import time
import random
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

class RetryURLCrawler:
    def __init__(self, source_dir="try1", base_url="https://www.jutingyuan.com"):
        self.source_dir = source_dir
        self.base_url = base_url
        self.visited_urls = set()
        # 按show类型和pattern类型分别存储URL {show_type: {pattern: set(urls)}}
        self.target_urls_by_source = defaultdict(lambda: defaultdict(set))
        # 也保留一个总的集合用于快速查找
        self.target_urls = defaultdict(set)
        
        # 创建Session来保持会话和Cookie
        self.session = requests.Session()
        
        # 定义URL模式（只关注story和actor）
        self.patterns = {
            'story': re.compile(r'https://(www\.)?jutingyuan\.com/story/'),
            'actor': re.compile(r'https://(www\.)?jutingyuan\.com/actor/')
        }
        
        # 要处理的show类型
        self.source_types = ['show_1', 'show_4', 'show_50', 'show_51']
        
        # 进度文件路径
        self.progress_visited_file = 'retry_crawl_progress_visited.txt'
        # 按show类型分别保存进度文件
        self.progress_targets_files = {}
        for show_type in self.source_types:
            for pattern_name in self.patterns.keys():
                key = f'{show_type}_{pattern_name}'
                self.progress_targets_files[key] = f'retry_crawl_progress_targets_{show_type}_{pattern_name}.json'
        
        # 线程锁
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
        
        if referer:
            headers['Referer'] = referer
        
        return headers
    
    def load_progress(self):
        """从文件加载之前的爬取进度"""
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
        
        # 按show类型和pattern类型分别加载目标URL
        total_targets = 0
        for key, filename in self.progress_targets_files.items():
            if os.path.exists(filename):
                try:
                    # key格式: show_1_story 或 show_4_actor
                    # 解析show_type和pattern_name
                    if key.startswith('show_'):
                        # 找到最后一个下划线的位置
                        parts = key.rsplit('_', 1)
                        if len(parts) == 2:
                            show_type = parts[0]  # show_1, show_4等
                            pattern_name = parts[1]  # story或actor
                            
                            with open(filename, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, list):
                                    urls = data
                                elif isinstance(data, dict) and 'urls' in data:
                                    urls = data['urls']
                                else:
                                    urls = []
                                
                                for url in urls:
                                    self.target_urls_by_source[show_type][pattern_name].add(url)
                                    self.target_urls[pattern_name].add(url)
                                total_targets += len(urls)
                except Exception as e:
                    print(f"加载目标URL [{key}] 时出错: {e}")
        
        if total_targets > 0:
            print(f"已加载 {total_targets} 个之前找到的目标URL（按show类型分别加载）")
    
    def save_progress(self):
        """保存当前爬取进度到文件（按show类型分别保存）"""
        try:
            with self.lock:
                visited_urls_copy = self.visited_urls.copy()
                target_urls_by_source_copy = {
                    show_type: {pattern: urls.copy() for pattern, urls in patterns.items()}
                    for show_type, patterns in self.target_urls_by_source.items()
                }
            
            # 保存已访问的URL
            with open(self.progress_visited_file, 'w', encoding='utf-8') as f:
                visited_list = sorted(visited_urls_copy)
                for i in range(0, len(visited_list), 10000):
                    batch = visited_list[i:i+10000]
                    f.write('\n'.join(batch) + '\n')
            
            # 按show类型和pattern类型分别保存目标URL
            for show_type, patterns in target_urls_by_source_copy.items():
                for pattern_name, urls_set in patterns.items():
                    if urls_set:  # 只保存非空集合
                        key = f'{show_type}_{pattern_name}'
                        filename = self.progress_targets_files.get(key)
                        if filename:
                            data = {
                                'show_type': show_type,
                                'pattern': pattern_name,
                                'urls': sorted(list(urls_set)),
                                'count': len(urls_set)
                            }
                            with open(filename, 'w', encoding='utf-8') as f:
                                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"保存进度时出错: {e}")
            return False
    
    def normalize_url(self, url):
        """规范化URL，统一使用www域名"""
        parsed = urlparse(url)
        if parsed.netloc == 'jutingyuan.com':
            parsed = parsed._replace(netloc='www.jutingyuan.com')
            url = parsed.geturl()
        return url
    
    def is_valid_url(self, url):
        """检查URL是否属于目标域名"""
        parsed = urlparse(url)
        return parsed.netloc in ['jutingyuan.com', 'www.jutingyuan.com'] or parsed.netloc == ''
    
    def matches_pattern(self, url):
        """检查URL是否匹配目标模式（只检查story和actor）"""
        url = self.normalize_url(url)
        for pattern_name, pattern in self.patterns.items():
            if pattern.match(url):
                return pattern_name
        return None
    
    def extract_urls_from_page(self, html_content, current_url):
        """从HTML内容中提取所有链接"""
        soup = BeautifulSoup(html_content, 'html.parser')
        urls = set()
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(current_url, href)
            absolute_url = absolute_url.split('#')[0]
            absolute_url = self.normalize_url(absolute_url)
            
            if self.is_valid_url(absolute_url):
                urls.add(absolute_url)
        
        return urls
    
    def load_source_urls(self):
        """从try1目录加载show_1, show_4, show_50, show_51的URL（自动去重）
        返回: [(url, show_type), ...] 列表，包含URL和对应的show类型
        """
        source_urls_list = []  # 列表，保持URL和show类型的对应关系
        source_urls_set = set()  # 用于去重
        
        for show_type in self.source_types:
            # 尝试从进度文件加载
            progress_file = os.path.join(self.source_dir, f'crawl_progress_targets_{show_type}.json')
            if os.path.exists(progress_file):
                try:
                    with open(progress_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            urls = data
                        elif isinstance(data, dict) and 'urls' in data:
                            urls = data['urls']
                        else:
                            urls = []
                        
                        loaded_count = 0
                        for url in urls:
                            url = self.normalize_url(url)
                            if url not in source_urls_set:
                                source_urls_set.add(url)
                                source_urls_list.append((url, show_type))
                                loaded_count += 1
                        print(f"从 {progress_file} 加载了 {loaded_count} 个 {show_type} URL（去重后）")
                except Exception as e:
                    print(f"加载 {progress_file} 时出错: {e}")
            
            # 也尝试从txt文件加载（如果存在）
            txt_file = os.path.join(self.source_dir, f'urls_{show_type}.txt')
            if os.path.exists(txt_file):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        urls = [line.strip() for line in f if line.strip()]
                        loaded_count = 0
                        for url in urls:
                            url = self.normalize_url(url)
                            if url not in source_urls_set:
                                source_urls_set.add(url)
                                source_urls_list.append((url, show_type))
                                loaded_count += 1
                        print(f"从 {txt_file} 加载了 {loaded_count} 个 {show_type} URL（去重后）")
                except Exception as e:
                    print(f"加载 {txt_file} 时出错: {e}")
        
        return source_urls_list
    
    def crawl_single_url(self, url, session, source_show_type=None):
        """爬取单个URL，只提取直接链接到的story和actor URL（不加入队列）
        Args:
            url: 要爬取的URL
            session: 请求会话
            source_show_type: 来源show类型（show_1, show_4, show_50, show_51）
        """
        url = self.normalize_url(url)
        
        # 检查是否已访问
        with self.lock:
            if url in self.visited_urls:
                return None, []
        
        try:
            headers = self.get_browser_headers(referer=self.base_url)
            response = session.get(
                url,
                headers=headers,
                timeout=10,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            # 标记为已访问
            with self.lock:
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                else:
                    return None, []
            
            # 提取链接
            all_urls = self.extract_urls_from_page(response.text, url)
            
            # 只筛选出story和actor类型的URL，并记录来源show类型
            target_urls = []
            for extracted_url in all_urls:
                pattern_match = self.matches_pattern(extracted_url)
                if pattern_match and source_show_type:
                    with self.lock:
                        # 添加到总集合
                        if extracted_url not in self.target_urls[pattern_match]:
                            self.target_urls[pattern_match].add(extracted_url)
                        # 按来源show类型分别存储
                        if extracted_url not in self.target_urls_by_source[source_show_type][pattern_match]:
                            self.target_urls_by_source[source_show_type][pattern_match].add(extracted_url)
                            target_urls.append((pattern_match, extracted_url))
            
            return url, target_urls
            
        except Exception as e:
            with self.lock:
                self.visited_urls.add(url)
            return None, []
    
    def crawl(self, max_workers=8, delay=0.1, save_interval=200):
        """从源URL列表爬取，只提取直接链接到的story和actor URL"""
        # 加载源URL（包含show类型信息）
        source_urls_with_type = self.load_source_urls()
        
        if not source_urls_with_type:
            print("没有找到源URL，请检查try1目录中的文件")
            return
        
        print(f"共加载 {len(source_urls_with_type)} 个源URL（已去重）")
        print(f"开始爬取，只提取直接链接到的story和actor URL")
        print(f"并发线程数: {max_workers}")
        print(f"进度保存间隔: 每 {save_interval} 页保存一次")
        
        # 为每个线程创建独立的Session
        sessions = [requests.Session() for _ in range(max_workers)]
        
        pages_crawled = 0
        last_save_count = 0
        last_print_count = 0
        
        # 使用集合跟踪正在处理的URL，避免重复提交
        processing_urls = set()
        
        # 使用线程池并发爬取
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i, (url, show_type) in enumerate(source_urls_with_type):
                # 使用锁保护，确保线程安全
                with self.lock:
                    # 检查是否已访问或正在处理
                    if url in self.visited_urls or url in processing_urls:
                        continue
                    # 标记为正在处理
                    processing_urls.add(url)
                
                session = sessions[i % len(sessions)]
                future = executor.submit(self.crawl_single_url, url, session, show_type)
                futures[future] = (url, show_type)
            
            # 收集结果
            for future in as_completed(futures):
                url, show_type = futures[future]
                try:
                    crawled_url, target_urls = future.result(timeout=15)
                    
                    # 从正在处理集合中移除
                    with self.lock:
                        processing_urls.discard(url)
                    
                    if crawled_url:
                        pages_crawled += 1
                        
                        # 打印找到的目标URL
                        for pattern_match, target_url in target_urls:
                            print(f"找到目标URL [{pattern_match}] (来自{show_type}): {target_url}")
                        
                        # 减少打印频率
                        if pages_crawled - last_print_count >= 10:
                            with self.lock:
                                story_count = len(self.target_urls['story'])
                                actor_count = len(self.target_urls['actor'])
                            print(f"已爬取 {pages_crawled} 页，找到 story: {story_count} 个，actor: {actor_count} 个")
                            last_print_count = pages_crawled
                        
                        # 定期保存进度
                        if pages_crawled - last_save_count >= save_interval:
                            if self.save_progress():
                                with self.lock:
                                    story_count = len(self.target_urls['story'])
                                    actor_count = len(self.target_urls['actor'])
                                print(f"已保存进度（已爬取 {pages_crawled} 页，story: {story_count} 个，actor: {actor_count} 个）")
                            last_save_count = pages_crawled
                        
                        # 短暂延迟
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        # 即使失败也要从正在处理集合中移除
                        with self.lock:
                            processing_urls.discard(url)
                            
                except Exception as e:
                    # 确保从正在处理集合中移除
                    with self.lock:
                        processing_urls.discard(url)
                    pass
        
        # 最后保存一次进度
        self.save_progress()
        
        print(f"\n爬取完成！共爬取 {pages_crawled} 页")
        print(f"找到的目标URL统计（按show类型分别统计）:")
        for show_type in self.source_types:
            story_count = len(self.target_urls_by_source[show_type]['story'])
            actor_count = len(self.target_urls_by_source[show_type]['actor'])
            if story_count > 0 or actor_count > 0:
                print(f"  {show_type}:")
                print(f"    story: {story_count} 个")
                print(f"    actor: {actor_count} 个")
        print(f"\n总计:")
        for pattern_name, urls in self.target_urls.items():
            print(f"  {pattern_name}: {len(urls)} 个")
        print(f"进度已保存到文件")
    
    def save_urls(self, skip_validation=False):
        """保存URL到文件（按show类型分别保存）"""
        if skip_validation:
            print("\n跳过验证，直接保存所有URL（按show类型分别保存）...")
            # 按show类型和pattern类型分别保存
            for show_type in self.source_types:
                for pattern_name in self.patterns.keys():
                    urls = self.target_urls_by_source[show_type][pattern_name]
                    if not urls:
                        continue
                    filename = f'urls_{pattern_name}_from_{show_type}.txt'
                    with open(filename, 'w', encoding='utf-8') as f:
                        sorted_urls = sorted(urls)
                        for i in range(0, len(sorted_urls), 10000):
                            batch = sorted_urls[i:i+10000]
                            f.write('\n'.join(batch) + '\n')
                    print(f"[{show_type} -> {pattern_name}] 已保存 {len(urls)} 个URL到 {filename}")
            return
        
        print("\n开始验证URL有效性（使用多线程加速，按show类型分别保存）...")
        
        def validate_single(url):
            try:
                url = self.normalize_url(url)
                headers = self.get_browser_headers(referer=self.base_url)
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=5,
                    allow_redirects=True,
                    verify=True
                )
                response.raise_for_status()
                return url, True, None
            except Exception as e:
                return url, False, str(e)
        
        # 按show类型和pattern类型分别验证和保存
        for show_type in self.source_types:
            for pattern_name in self.patterns.keys():
                urls = list(self.target_urls_by_source[show_type][pattern_name])
                if not urls:
                    continue
                
                filename = f'urls_{pattern_name}_from_{show_type}.txt'
                print(f"\n验证 [{show_type} -> {pattern_name}] 类URL (共 {len(urls)} 个)...")
                
                # 批量验证
                valid_urls = []
                invalid_count = 0
                
                with ThreadPoolExecutor(max_workers=10) as executor:
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
                
                # 保存有效的URL
                with open(filename, 'w', encoding='utf-8') as f:
                    sorted_valid_urls = sorted(valid_urls)
                    for i in range(0, len(sorted_valid_urls), 10000):
                        batch = sorted_valid_urls[i:i+10000]
                        f.write('\n'.join(batch) + '\n')
                
                print(f"\n[{show_type} -> {pattern_name}] 验证完成:")
                print(f"  有效URL: {len(valid_urls)} 个")
                print(f"  失效URL: {invalid_count} 个")
                print(f"  已保存到: {filename}")


def main():
    crawler = RetryURLCrawler(source_dir="try1")
    
    # 优化参数
    crawl_delay = 0.1
    save_interval = 200
    max_workers = 8
    
    print("=" * 60)
    print("重新爬取story和actor类型的URL")
    print("从show_1, show_4, show_50, show_51中提取直接链接到的URL")
    print("=" * 60)
    
    # 开始爬取
    crawler.crawl(max_workers=max_workers, delay=crawl_delay, save_interval=save_interval)
    
    # 保存结果
    print("\n" + "=" * 60)
    print("开始保存URL")
    print("=" * 60)
    crawler.save_urls(skip_validation=False)
    
    print("\n所有URL已保存完成！")
    print(f"爬取进度已保存，下次运行将自动从中断点恢复")


if __name__ == "__main__":
    main()

