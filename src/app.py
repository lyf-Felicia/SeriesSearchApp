import streamlit as st
import requests
import re
import random
import json
import sqlite3
import time
import html
import os
from typing import List, Dict, Any, Generator
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from qdrant_client import QdrantClient
from openai import OpenAI
import urllib.request
import urllib.error
import zipfile

def download_data_from_releases():
    # 从 secrets 读取配置，如果没有则使用默认值
    repo = st.secrets.get("GITHUB_REPO", "lyf-Felicia/SeriesSearchApp")
    tag = st.secrets.get("RELEASE_TAG", "1.0")
    # 使用正确的 GitHub Release URL 格式
    release_base = f"https://github.com/{repo}/releases/download/{tag}"
    
    # 显示调试信息
    st.info(f"📦 从 GitHub Release 下载数据\n- 仓库: {repo}\n- 标签: {tag}\n- 基础URL: {release_base}")
    
    os.makedirs("data/database", exist_ok=True)
    os.makedirs("data/qdrant_data", exist_ok=True)
    
    files = {
        "data/llm_summaries.json": f"{release_base}/llm_summaries.json",
        "data/database/final.db": f"{release_base}/final.db",
        "data/qdrant_data.zip": f"{release_base}/qdrant_data.zip"
    }
    
    download_failed = False
    
    for local_path, url in files.items():
        # 优化判断逻辑：如果文件已存在且大小 > 1KB，跳过下载
        # 对于 zip 文件，检查解压后的目录是否存在
        if local_path.endswith('.zip'):
            if os.path.exists("data/qdrant_data") and os.path.exists("data/qdrant_data/meta.json"):
                st.success(f"✓ {os.path.basename(local_path)} 已存在，跳过下载")
                continue
        elif os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            st.success(f"✓ {os.path.basename(local_path)} 已存在，跳过下载")
            continue
            
        try:
            with st.spinner(f"正在下载 {os.path.basename(local_path)}..."):
                # 显示实际下载 URL（用于调试）
                st.text(f"下载URL: {url}")
                
                # 使用自定义 Header 模拟浏览器，防止被 GitHub 拦截
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                urllib.request.urlretrieve(url, local_path)
                
                # 校验：如果下载的文件太小（可能是下载到了报错页面），抛出异常
                if os.path.getsize(local_path) < 100:
                    with open(local_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    st.error(f"❌ 下载的文件内容异常（可能下载到了错误页面）\nURL: {url}\n内容预览: {content[:200]}")
                    download_failed = True
                    if os.path.exists(local_path):
                        os.remove(local_path)
                    continue

                if local_path.endswith('.zip'):
                    with zipfile.ZipFile(local_path, 'r') as zip_ref:
                        zip_ref.extractall("data/")
                    os.remove(local_path)
                    st.success(f"✓ {os.path.basename(local_path)} 下载并解压成功")
                else:
                    st.success(f"✓ {os.path.basename(local_path)} 下载成功")
        except urllib.error.HTTPError as e:
            st.error(f"❌ 下载失败 {os.path.basename(local_path)}: HTTP {e.code} {e.reason}\nURL: {url}\n\n请检查：\n1. Release 标签是否正确（当前: {tag}）\n2. 文件名是否正确\n3. Release 是否已发布")
            download_failed = True
        except Exception as e:
            st.error(f"❌ 下载失败 {os.path.basename(local_path)}: {str(e)}\nURL: {url}")
            download_failed = True
    
    # 如果下载失败，停止应用执行
    if download_failed:
        st.error("⚠️ 数据文件下载失败，应用无法继续运行。请检查 GitHub Release 配置。")
        st.stop()
    
    # 验证所有必需的文件是否存在且可读
    required_files = {
        "数据库文件": "data/database/final.db",
        "LLM摘要文件": "data/llm_summaries.json",
        "Qdrant数据目录": "data/qdrant_data"
    }
    
    missing_files = []
    for name, path in required_files.items():
        if os.path.isdir(path):
            # 对于目录，检查是否有内容
            if not os.listdir(path):
                missing_files.append(f"{name} ({path}) - 目录为空")
            elif path == "data/qdrant_data" and not os.path.exists(os.path.join(path, "meta.json")):
                missing_files.append(f"{name} ({path}) - 缺少 meta.json 文件")
        elif not os.path.exists(path):
            missing_files.append(f"{name} ({path}) - 文件不存在")
        elif os.path.getsize(path) < 100:
            missing_files.append(f"{name} ({path}) - 文件大小异常（可能损坏）")
    
    if missing_files:
        st.error("⚠️ 数据文件验证失败：\n" + "\n".join(f"- {f}" for f in missing_files))
        st.stop()
    
    # 等待一小段时间确保文件系统完全同步
    time.sleep(0.5)

download_data_from_releases()

# ================= 🟢 配置区域 =================
# 优先从 Streamlit secrets 读取，如果没有则使用默认值
LLM_API_KEY = st.secrets.get("LLM_API_KEY", "")
LLM_BASE_URL = st.secrets.get("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_MODEL_NAME = st.secrets.get("LLM_MODEL_NAME", "qwen-max")
QDRANT_PATH = st.secrets.get("QDRANT_PATH", "data/qdrant_data")
EMBEDDING_MODEL_PATH = st.secrets.get("EMBEDDING_MODEL_PATH", "BAAI/bge-large-zh-v1.5")
DB_PATH = st.secrets.get("DB_PATH", "data/database/final.db")

# ==============================================================================
# 1. 辅助函数：清理文本中的HTML标签
# ==============================================================================
def clean_html_tags(text):
    """清理文本中的所有HTML标签，只保留纯文本"""
    if not text:
        return ""
    # 移除所有HTML标签
    text = re.sub(r'<[^>]+>', '', str(text))
    # 转义剩余的HTML特殊字符
    text = html.escape(text)
    return text

def _render_turn_content(turn):
    """渲染单轮对话的内容（用户查询、AI推荐、剧集列表）"""
    # 显示用户查询（右侧对齐）
    with st.container():
        col_user_empty, col_user_content = st.columns([3, 7])
        with col_user_content:
            # 清理并转义用户查询文本
            user_text = clean_html_tags(turn.get('query', ''))
            st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin: 0.75rem 0;">
                <div style="background: linear-gradient(135deg, #4f46e5, #6366f1); color: white; padding: 0.6rem 0.875rem; border-radius: 16px; border-bottom-right-radius: 4px; box-shadow: 0 2px 6px rgba(79, 70, 229, 0.2); max-width: 55%; word-wrap: break-word; line-height: 1.4; white-space: pre-wrap; font-size: 0.95em;">
                    {user_text}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # 显示AI推荐（左侧对齐）
    if turn.get('recommendation'):
        with st.container():
            col_ai_content, col_ai_empty = st.columns([7, 3])
            with col_ai_content:
                # 清理并转义AI推荐文本
                ai_text = clean_html_tags(turn.get('recommendation', ''))
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 0.75rem 0;">
                    <div style="background: #f8fafc; padding: 0.6rem 0.875rem; border-radius: 16px; border: 1px solid #e2e8f0; border-bottom-left-radius: 4px; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06); max-width: 55%; word-wrap: break-word; line-height: 1.4; white-space: pre-wrap; font-size: 0.95em;">
                        {ai_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # 显示剧集列表（左侧对齐）
    if turn.get('results'):
        st.markdown('<div style="padding: 0 1rem;">', unsafe_allow_html=True)
        for i, r in enumerate(turn['results']):
            with st.container(border=True):
                col_img, col_txt = st.columns([1, 3])
                with col_img:
                    st.image(fetch_poster_url(r['title']))
                with col_txt:
                    score = r.get('score', 0)
                    title = r.get('title', '未知')
                    year = r.get('year', '未知')
                    genre = r.get('genre', '未知')
                    region = r.get('region', '未知')
                    
                    # 基本信息（始终显示）
                    st.markdown(f"### 《{title}》 <span style='color:grey;font-size:0.8em'>匹配度:{score:.2f}</span>", unsafe_allow_html=True)
                    st.markdown(f"<span style='color:#64748b;font-size:0.9em'>{year} · {genre} · {region}</span>", unsafe_allow_html=True)
                    
                    # 详细信息（可展开）
                    with st.expander("查看详情", expanded=False):
                        # 完整简介
                        display_text = r.get('display_text', '')
                        if display_text:
                            st.markdown("**简介：**")
                            st.write(display_text)
                        
                        # 高能剧情命中
                        if r.get('matched_episodes'):
                            st.markdown("**高能剧情命中：**")
                            for ep in r['matched_episodes']:
                                st.success(f"第{ep['ep_number']}集: {ep['content_snippet']}")
                        
                        # 其他信息
                        st.markdown("**详细信息：**")
                        col_info1, col_info2 = st.columns(2)
                        with col_info1:
                            st.text(f"年份: {year}")
                            st.text(f"类型: {genre}")
                        with col_info2:
                            st.text(f"地区: {region}")
                            st.text(f"匹配度: {score:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 2. 辅助函数：必应实时搜图
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_poster_url(query_title):
    """
    通过必应搜索获取海报，验证图片可访问性
    """
    fallback_images = [
        "https://images.unsplash.com/photo-1536440136628-849c177e76a1?auto=format&fit=crop&w=500&q=60",
        "https://images.unsplash.com/photo-1485846234645-a62644f84728?auto=format&fit=crop&w=500&q=60",
    ]
    
    keyword = f"电视剧 {query_title} 海报"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }

    def is_image_accessible(url):
        """验证图片URL是否可访问"""
        try:
            response = requests.head(url, headers=headers, timeout=2, allow_redirects=True)
            return response.status_code == 200
        except:
            return False

    try:
        url = "https://cn.bing.com/images/search"
        params = {"q": keyword, "first": 1} 
        response = requests.get(url, params=params, headers=headers, timeout=3)
        
        if response.status_code == 200:
            html = response.text
            
            # 1. 优先找 turl (缩略图)
            pattern_thumb = r'turl&quot;:&quot;(https://tse[^&]+?)&quot;'
            matches = re.findall(pattern_thumb, html)
            
            # 验证前5个缩略图
            for match in matches[:5]:
                if is_image_accessible(match):
                    return match
            
            # 2. 尝试原图 murl
            pattern_full = r'murl&quot;:&quot;(http[^&]+?)&quot;'
            matches_full = re.findall(pattern_full, html)
            
            for match in matches_full[:3]:
                if is_image_accessible(match):
                    return match

    except Exception as e:
        pass

    return random.choice(fallback_images)

# ==============================================================================
# 2. 后端逻辑 SmartTVRetriever（修改版：支持多选）
# ==============================================================================
class SmartTVRetriever:
    def __init__(self):
        # 验证文件路径
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"数据库文件不存在: {DB_PATH}\n请确保数据文件已正确下载。")
        
        if not os.path.exists(QDRANT_PATH):
            raise FileNotFoundError(f"Qdrant 数据目录不存在: {QDRANT_PATH}\n请确保数据文件已正确下载并解压。")
        
        if not os.path.exists("data/qdrant_data/meta.json"):
            raise FileNotFoundError(f"Qdrant 数据不完整: {QDRANT_PATH}/meta.json 不存在\n请确保 qdrant_data.zip 已正确解压。")
        
        try:
            print(f"正在加载本地 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
            self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_PATH, trust_remote_code=True)
            Settings.embed_model = self.embed_model
        except Exception as e:
            st.error(f"Embedding 模型加载失败: {e}")
            raise

        try:
            print(f"正在连接 Qdrant 向量数据库: {QDRANT_PATH} ...")
            self.client = QdrantClient(path=QDRANT_PATH)
            self.rich_index = self._load_index("tv_series_rich_text")
            self.basic_index = self._load_index("tv_series_basic")
        except Exception as e:
            st.error(f"Qdrant 向量数据库加载失败: {e}\n路径: {QDRANT_PATH}")
            raise
        
        try:
            print(f"正在连接 SQL 数据库: {DB_PATH} ...")
            # 验证数据库文件可读
            if not os.access(DB_PATH, os.R_OK):
                raise PermissionError(f"数据库文件不可读: {DB_PATH}")
            self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # 测试数据库连接
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            cursor.fetchone()
        except sqlite3.Error as e:
            st.error(f"SQL 数据库连接失败: {e}\n路径: {DB_PATH}")
            raise
        except Exception as e:
            st.error(f"数据库文件访问失败: {e}\n路径: {DB_PATH}")
            raise

        try:
            self.llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        except Exception as e:
            st.error(f"LLM 客户端初始化失败: {e}")
            raise

    def _load_index(self, collection_name: str):
        vector_store = QdrantVectorStore(client=self.client, collection_name=collection_name)
        return VectorStoreIndex.from_vector_store(vector_store=vector_store)

    def filter_search(self, years: List[str] = None, genres: List[str] = None, 
                     regions: List[str] = None, limit: int = 10) -> List[Dict]:
        """修改为支持多选的筛选函数"""
        cursor = self.conn.cursor()
        sql = "SELECT * FROM series WHERE 1=1"
        params = []
        
        # 年份多选处理
        if years and len(years) > 0:
            year_conditions = []
            for year in years:
                if year == "更早":
                    year_conditions.append("CAST(year AS INTEGER) < 2018")
                else:
                    year_conditions.append("year = ?")
                    params.append(year)
            if year_conditions:
                sql += f" AND ({' OR '.join(year_conditions)})"
        
        # 地区多选处理 - 处理"中国大陆"和"大陆"的映射
        if regions and len(regions) > 0:
            region_conditions_list = []
            for r in regions:
                if r == "中国大陆":
                    # "中国大陆"同时匹配"中国大陆"和"大陆"
                    region_conditions_list.append("(region LIKE ? OR region LIKE ?)")
                    params.append("%中国大陆%")
                    params.append("%大陆%")
                else:
                    region_conditions_list.append("region LIKE ?")
                    params.append(f"%{r}%")
            
            if region_conditions_list:
                region_conditions = " OR ".join(region_conditions_list)
                sql += f" AND ({region_conditions})"
        
        # 类型多选处理
        if genres and len(genres) > 0:
            genre_conditions = " OR ".join(["genre LIKE ?" for _ in genres])
            sql += f" AND ({genre_conditions})"
            params.extend([f"%{g}%" for g in genres])
            
        sql += " LIMIT ?"
        params.append(limit)
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                res_dict = {
                    "series_id": row['id'],
                    "title": row['title'],
                    "year": row['year'],
                    "genre": row['genre'],
                    "region": row['region'],
                    "source_type": "SQL",
                    "score": 1.0,
                    "actors": row['cast'] if 'cast' in row.keys() else "暂无演员信息",
                    "description": row['summary'] if 'summary' in row.keys() else "暂无剧情简介"
                }
                results.append(res_dict)
            return results
        except sqlite3.Error as e:
            print(f"SQL Error: {e}")
            return []

    def _llm_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """利用大模型对初筛结果进行精排"""
        if not candidates:
            return []

        items_text = ""
        for i, res in enumerate(candidates):
            items_text += f"ID: {i} | 标题: 《{res['title']}》 | 简介: {res['display_text'][:200]}\n"

        rerank_prompt = f"""你是一个专业的影视推荐官。请根据用户的需求，对候选剧集进行相关性打分。

用户需求："{query}"

候选列表：
{items_text}

任务要求：
1. 严格根据用户需求与剧集内容的相关度打分（0-10分）。
2. 只返回 JSON 格式，包含一个数组，每个对象包含 id 和 score。
3. 如果剧集完全符合人设（如用户要看"医生"，该剧主角确实是医生），给 9-10 分。
4. 如果只是背景提到或不相关，给 0-3 分。

输出示例：
[
  {{"id": 0, "score": 9.5}},
  {{"id": 1, "score": 4.0}}
]"""

        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "你是一个严格的评分机器，只输出JSON数据。"},
                    {"role": "user", "content": rerank_prompt}
                ],
                response_format={ "type": "json_object" if "qwen" in LLM_MODEL_NAME else "text" },
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            
            scores_data = json.loads(content)
            if isinstance(scores_data, dict):
                scores_list = scores_data.get("results", scores_data.get("scores", list(scores_data.values())[0]))
            else:
                scores_list = scores_data

            for item in scores_list:
                idx = int(item['id'])
                if idx < len(candidates):
                    candidates[idx]['rerank_score'] = float(item['score'])
            
            candidates.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            print(f"Rerank Error: {e}")
            return candidates[:top_k]

    def semantic_search(self, user_query: str, top_k: int = 5) -> Dict:
        intent_data = self._classify_intent(user_query)
        recall_top_k = 15 
        
        retriever_rich = self.rich_index.as_retriever(similarity_top_k=recall_top_k)
        retriever_basic = self.basic_index.as_retriever(similarity_top_k=recall_top_k)
        
        nodes_rich = retriever_rich.retrieve(user_query)
        nodes_basic = retriever_basic.retrieve(user_query)

        candidates = self._merge_and_rank_results(nodes_rich, nodes_basic, user_query)
        st.toast("🚀 正在调用大模型进行精准排序...", icon="🧠")
        final_results = self._llm_rerank(user_query, candidates, top_k)

        return {
            "query_analysis": intent_data,
            "results": final_results,
            "user_query": user_query
        }

    def _classify_intent(self, query: str) -> Dict:
        prompt = f"""你是一个影视搜索专家，负责将用户查询解析为搜索参数。
    
    【任务】
    从用户输入中提取以下字段并返回JSON：
    1. intent: 
       - "PERSONA": 用户提到了特定职业、身份、人设（如：医生、霸总、单亲妈妈）。
       - "SCENE": 用户提到了具体情节或名场面（如：跳崖、雨中分手、误会）。
       - "THEME": 模糊的题材、风格或情绪（如：甜宠、虐心、爽剧）。
    2. keywords: 核心关键词列表。
    3. occupation: 提取出的具体职业或身份标签（若无则为空列表）。
    
    【示例】
    输入："想看男主是医生的甜宠剧"
    返回：{{"intent": "PERSONA", "keywords": ["医生", "甜宠"], "occupation": ["医生"]}}
    
    输入："男女主在雨中分手的名场面"
    返回：{{"intent": "SCENE", "keywords": ["雨中分手", "分手"], "occupation": []}}
    
    输入："{query}"
    """
    
        try:
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" },
                temperature=0.1 
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
        
            if parsed.get("occupation") and len(parsed["occupation"]) > 0:
                parsed["intent"] = "PERSONA"
            
            print(f"优化后的意图识别: {parsed}")
            return parsed
        except Exception as e:
            return {"intent": "THEME", "keywords": [query], "occupation": []}

    def _merge_and_rank_results(self, nodes_rich, nodes_basic, query):
        series_map = {}
        def process(nodes, src, boost=0.0):
            for node in nodes:
                m = node.metadata
                sid = m['series_id']
                score = node.score + boost
                
                full_text = node.text if len(node.text) > 200 else node.text
                
                if sid not in series_map:
                    series_map[sid] = {
                        "series_id": sid,
                        "title": m.get('title') or m.get('parent_title'),
                        "score": score,
                        "source_type": src,
                        "hit_type": m['type'],
                        "matched_episodes": [],
                        "display_text": full_text,
                        "year": m.get('year', '未知'),
                        "genre": m.get('genre', '未知'),
                        "region": m.get('region', '未知')
                    }
                else:
                    if score > series_map[sid]["score"]:
                        series_map[sid]["score"] = score
                    if len(full_text) > len(series_map[sid]["display_text"]):
                        series_map[sid]["display_text"] = full_text
                
                cur = series_map[sid]
                if m['type'] == 'episode':
                    cur['matched_episodes'].append({
                        "ep_number": m['ep_number'],
                        "content_snippet": node.text[:150] + "..."
                    })
        
        process(nodes_rich, "Rich", 0.1)
        process(nodes_basic, "Basic", 0.0)
        
        final_list = list(series_map.values())
        final_list.sort(key=lambda x: x['score'], reverse=True)
        return final_list

    def generate_recommendation_stream(self, query, results) -> Generator[str, None, None]:
        """流式生成器：完美过滤思考过程"""
        if not results:
            yield "未找到相关剧集，请尝试换个描述。"
            return

        ctx = "\n".join([f"- 《{r['title']}》: {r['display_text'][:150]}" for r in results[:3]])
        # 如果query中包含对话历史上下文（格式：之前问过:xxx | 现在问:yyy），则提取并格式化
        if " | 现在问: " in query:
            parts = query.split(" | 现在问: ")
            history_part = parts[0] if len(parts) > 1 else ""
            current_query = parts[1] if len(parts) > 1 else query
            if history_part:
                prompt = f"对话历史：{history_part}\n\n用户现在问：{current_query}\n\n推荐以下剧集：\n{ctx}\n\n要求：基于对话历史理解用户意图，直接以朋友语气推荐，严禁使用 <think> 标签，严禁输出思考过程，第一句话就进入主题。300字左右。"
            else:
                prompt = f"用户搜：{current_query}\n推荐以下剧集：\n{ctx}\n\n要求：直接以朋友语气推荐，严禁使用 <think> 标签，严禁输出思考过程，第一句话就进入主题。300字左右。"
        else:
            prompt = f"用户搜：{query}\n推荐以下剧集：\n{ctx}\n\n要求：直接以朋友语气推荐，严禁使用 <think> 标签，严禁输出思考过程，第一句话就进入主题。300字左右。"

        try:
            stream = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "system", "content": "你是一个直接输出结果的助手，不废话，不思考。"},
                          {"role": "user", "content": prompt}],
                stream=True
            )

            is_thinking = False
            full_buffer = ""

            for chunk in stream:
                if not chunk.choices: continue
                content = chunk.choices[0].delta.content or ""
                full_buffer += content

                if "<think>" in full_buffer and "</think>" not in full_buffer:
                    is_thinking = True
                    continue
                
                if "</think>" in full_buffer:
                    full_buffer = full_buffer.split("</think>")[-1]
                    is_thinking = False
                    continue

                if not is_thinking and content:
                    if len(full_buffer) < 5 and (content.strip() in ["好的", "首先", "为您"]):
                        continue
                    yield content

        except Exception as e:
            yield f"推荐生成出错: {e}"

# ==============================================================================
# 3. Streamlit 前端（修改版：支持多选）
# ==============================================================================
st.set_page_config(page_title="智能电视剧搜索引擎", page_icon="📺", layout="wide")

st.markdown("""
<style>
    /* 全局样式优化 - 清新渐变背景 */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f4f8 25%, #fff5f5 50%, #f0f9ff 75%, #faf5ff 100%);
        background-size: 400% 400%;
        animation: gradientShift 25s ease infinite;
        position: relative;
        z-index: 1;
    }
    
    /* 背景装饰图案 */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 15% 20%, rgba(147, 197, 253, 0.15) 0%, transparent 40%),
            radial-gradient(circle at 85% 60%, rgba(251, 191, 36, 0.12) 0%, transparent 40%),
            radial-gradient(circle at 50% 85%, rgba(196, 181, 253, 0.1) 0%, transparent 45%),
            radial-gradient(circle at 70% 15%, rgba(167, 243, 208, 0.12) 0%, transparent 40%);
        pointer-events: none;
        z-index: -1;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* 主内容区域背景 */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(255, 255, 255, 0.8) inset;
        backdrop-filter: blur(20px);
        margin-top: 2rem;
        margin-bottom: 2rem;
        position: relative;
        z-index: 10;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* 字体优化 */
    html, body, [class*="css"], .stMarkdown, .stText {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', 'Helvetica Neue', Arial, sans-serif !important;
        letter-spacing: 0.3px;
        font-weight: 400;
    }
    
    /* 文字大小和行高优化 */
    .stMarkdown p, .stText, p, li, span {
        font-size: 16px !important;
        line-height: 1.8 !important;
        color: #2d3748 !important;
        word-spacing: 1px;
    }
    
    /* 标题样式优化 - 紫色渐变 */
    h1 {
        background: linear-gradient(135deg, #4f46e5 0%, #6366f1 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800 !important;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
        font-size: 2.5rem !important;
        position: relative;
    }
    
    h1::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #4f46e5, #6366f1);
        border-radius: 2px;
    }
    
    h2 {
        color: #1e293b !important;
        font-weight: 700 !important;
        letter-spacing: -0.3px;
        margin-top: 1.5rem;
        position: relative;
        padding-left: 1rem;
    }
    
    h2::before {
        content: '';
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        width: 4px;
        height: 24px;
        background: linear-gradient(180deg, #4f46e5, #6366f1);
        border-radius: 2px;
    }
    
    h3, h4 {
        color: #334155 !important;
        font-weight: 600 !important;
    }
    
    /* 按钮样式 - 白色背景蓝紫色文字 */
    .stButton > button {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.875rem 2.25rem !important;
        border-radius: 12px !important;
        background: #ffffff !important;
        border: 2px solid #4f46e5 !important;
        box-shadow: 0 2px 8px rgba(79, 70, 229, 0.15) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.3px;
        color: #4f46e5 !important;
        text-shadow: none !important;
    }
    
    .stButton > button:hover {
        background: #f8fafc !important;
        border-color: #6366f1 !important;
        color: #6366f1 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        background: #f1f5f9 !important;
        border-color: #4338ca !important;
        color: #4338ca !important;
    }
    
    /* Pills 样式 - 选中状态红色边框 */
    div[data-testid="stPills"] button {
        font-size: 14px !important;
        padding: 8px 18px !important;
        border-radius: 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        background: #f8fafc !important;
        color: #475569 !important;
        border: 1.5px solid #e2e8f0 !important;
    }
    
    div[data-testid="stPills"] button:hover:not([aria-pressed="true"]) {
        background: #f1f5f9 !important;
        border-color: #cbd5e1 !important;
        color: #334155 !important;
    }
    
    div[data-testid="stPills"] button[aria-pressed="true"] {
        background: #4f46e5 !important;
        color: #ffffff !important;
        border-color: #4f46e5 !important;
        border-width: 2px !important;
        box-shadow: 0 2px 6px rgba(79, 70, 229, 0.25) !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stPills"] button[aria-pressed="true"]:hover {
        background: #4338ca !important;
        border-color: #4338ca !important;
        color: #ffffff !important;
    }
    
    /* Tab 样式优化 - 去掉下划线 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: none !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px 12px 0 0 !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        color: #64748b !important;
        background: transparent !important;
        border-bottom: none !important;
    }
    
    /* 移除所有可能的红色元素和下划线 */
    .stTabs [data-baseweb="tab"][aria-selected="true"],
    .stTabs [aria-selected="true"] {
        background: rgba(79, 70, 229, 0.08) !important;
        color: #4f46e5 !important;
        font-weight: 600 !important;
        border-bottom: none !important;
        box-shadow: none !important;
        border-color: transparent !important;
    }
    
    /* 移除所有伪元素的红色 */
    .stTabs [data-baseweb="tab"]::before,
    .stTabs [data-baseweb="tab"]::after,
    .stTabs [data-baseweb="tab"][aria-selected="true"]::before,
    .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
        background: none !important;
        border: none !important;
        border-color: transparent !important;
        display: none !important;
    }
    
    /* 覆盖内部元素的红色 */
    .stTabs [data-baseweb="tab"] * {
        border-color: inherit !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #4f46e5 !important;
        background: rgba(79, 70, 229, 0.05) !important;
    }
    
    /* 强制覆盖任何红色样式 */
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div {
        color: #4f46e5 !important;
        border-color: #4f46e5 !important;
    }
    
    /* 容器边框美化 */
    [data-testid="stHorizontalBlock"] > div[data-testid="column"],
    [data-baseweb="card"] {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 16px !important;
        padding: 1.25rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.3s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
    }
    
    /* 输入框优化 */
    .stTextArea textarea, .stTextInput input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        font-size: 15px !important;
        line-height: 1.7 !important;
        padding: 14px 16px !important;
        background: rgba(255, 255, 255, 0.95) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #4f46e5 !important;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1) !important;
        outline: none !important;
    }
    
    /* 图片圆角 */
    .stImage img {
        border-radius: 16px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 分隔线 */
    hr {
        margin: 2.5rem 0 !important;
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #e2e8f0, #cbd5e1, #e2e8f0, transparent) !important;
    }
    
    /* Caption 样式 */
    .stCaption {
        color: #64748b !important;
        font-size: 14px !important;
        font-style: normal !important;
        font-weight: 500 !important;
    }
    
    /* 确保所有内容可见 */
    .main, .main > div, .main > div > div,
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], [data-testid="stToolbar"] {
        position: relative !important;
        z-index: 10 !important;
    }
</style>
""", unsafe_allow_html=True)

# 美化标题
st.markdown("""
<div style="text-align: center; padding: 1rem 0 2rem 0;">
    <h1 style="margin-bottom: 0.5rem;">智能电视剧搜索引擎</h1>
    <p style="font-size: 1.1rem; color: #64748b; margin: 0.5rem 0; font-weight: 500;">
        结合 SQL 传统检索与 LLM 语义理解的新一代检索系统
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; color: #94a3b8; font-size: 0.9rem;">
        <span>智能推荐</span>
        <span>•</span>
        <span>精准匹配</span>
        <span>•</span>
        <span>语义理解</span>
    </div>
</div>
""", unsafe_allow_html=True)
st.divider()

@st.cache_resource
def load_retriever():
    try:
        # 显示加载进度
        with st.spinner("正在初始化后端系统..."):
            retriever = SmartTVRetriever()
            st.success("✓ 后端系统加载成功")
            return retriever
    except FileNotFoundError as e:
        st.error(f"❌ 文件未找到: {e}\n\n请检查：\n1. 数据文件是否已正确下载\n2. 文件路径是否正确")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None
    except Exception as e:
        st.error(f"❌ 后端加载失败: {e}")
        import traceback
        with st.expander("查看详细错误信息"):
            st.code(traceback.format_exc())
        return None

retriever = load_retriever()
if not retriever: st.stop()

tab1, tab2 = st.tabs(["传统筛选", "智能搜索"])

# --- Tab 1: 传统筛选（修改版：支持多选）---
with tab1:
    st.subheader("精准多维筛选")
    st.caption("支持多选！点击标签即可选中/取消，选择「全部」将清除其他选项")
    
    YEAR_OPTIONS = ["全部", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "更早"]
    GENRE_OPTIONS = ["全部", "古装", "爱情", "悬疑", "动作", "剧情", "喜剧", "奇幻", "武侠", "青春", "战争", "校园", "励志", "革命", "乡村", "警匪", "恐怖", "冒险", "惊悚", "神话魔幻", "言情"]
    REGION_OPTIONS = ["全部", "中国大陆", "中国香港", "美国", "韩国", "日本", "英国"]

    c1, c2 = st.columns([3, 1])
    with c1:
        # 使用 selection_mode="multi" 支持多选，默认选中"全部"
        s_years = st.pills("年份（可多选）", YEAR_OPTIONS, selection_mode="multi", default=["全部"], key="py")
        s_genres = st.pills("类型（可多选）", GENRE_OPTIONS, selection_mode="multi", default=["全部"], key="pg")
        s_regions = st.pills("地区（可多选）", REGION_OPTIONS, selection_mode="multi", default=["全部"], key="pr")
    with c2:
        limit = st.slider("显示数量", 1, 50, 10, key="limit")
        st.write("")
        btn_filter = st.button("立即筛选", type="primary")

    if btn_filter:
        st.divider()
        
        # 处理"全部"逻辑：如果选了"全部"，则忽略该维度的其他选项
        query_years = None
        if s_years and "全部" not in s_years:
            query_years = list(s_years)
        
        query_genres = None
        if s_genres and "全部" not in s_genres:
            query_genres = list(s_genres)
        
        query_regions = None
        if s_regions and "全部" not in s_regions:
            query_regions = list(s_regions)

        # 显示当前筛选条件
        filter_info = []
        if query_years:
            filter_info.append(f"年份: {', '.join(query_years)}")
        if query_genres:
            filter_info.append(f"类型: {', '.join(query_genres)}")
        if query_regions:
            filter_info.append(f"地区: {', '.join(query_regions)}")
        
        if filter_info:
            st.info(f"当前筛选条件：{' | '.join(filter_info)}")
        else:
            st.info("当前筛选条件：全部")

        with st.spinner("正在检索..."):
            results = retriever.filter_search(query_years, query_genres, query_regions, limit)
            if results:
                st.success(f"找到 {len(results)} 部作品")
                for r in results:
                    with st.container(border=True):
                        col_img, col_txt = st.columns([1, 4])
                        with col_img:
                            url = fetch_poster_url(r['title'])
                            st.image(url)
                        with col_txt:
                            st.markdown(f"### 《{r['title']}》")
                            st.markdown(f"**年份:** `{r['year']}` | **地区:** `{r['region']}` | **类型:** `{r['genre']}`")
                            
                            actors = r.get('actors', '暂无演员信息')
                            desc = r.get('description', '暂无简介')
                            short_desc = desc[:120] + "..." if len(desc) > 120 else desc
                            
                            st.markdown(f"**主演:** {actors}")
                            st.markdown(f"**简介:** {short_desc}")
                            
                            with st.expander("查看完整简介"):
                                st.write(desc)
            else:
                st.warning("未找到匹配剧集，请尝试调整筛选条件。")

# --- Tab 2: 智能搜索（多轮对话版本）---
with tab2:
    st.subheader("语义理解搜索")
    
    # 初始化对话会话管理（类似ChatGPT的设计）
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "conversation_counter" not in st.session_state:
        st.session_state.conversation_counter = 0
    if "scroll_to_top" not in st.session_state:
        st.session_state.scroll_to_top = False
    
    # 自动滚动到顶部
    if st.session_state.scroll_to_top:
        st.markdown("""
        <script>
            window.parent.scrollTo({ top: 0, behavior: 'smooth' });
        </script>
        """, unsafe_allow_html=True)
        st.session_state.scroll_to_top = False
    
    # 获取当前对话的所有轮次
    current_turns = []
    if st.session_state.current_conversation_id and st.session_state.current_conversation_id in st.session_state.conversations:
        current_turns = st.session_state.conversations[st.session_state.current_conversation_id]
    
    # 显示当前对话的所有轮次（类似ChatGPT的对话流）
    if current_turns:
        for idx, turn in enumerate(current_turns):
            is_last_turn = (idx == len(current_turns) - 1)  # 判断是否是最后一轮（最新的一轮）
            query_preview = clean_html_tags(turn.get('query', ''))[:30]  # 预览文本用于expander标签
            
            # 如果不是最后一轮，使用expander包裹（默认折叠）
            if not is_last_turn:
                with st.expander(f"📝 第{idx+1}轮对话: {query_preview}...", expanded=False):
                    _render_turn_content(turn)
            else:
                # 最后一轮直接显示（展开）
                _render_turn_content(turn)
    else:
        # 首次进入，显示欢迎信息
        st.markdown("""
        <div style="text-align: center; padding: 3rem 1rem; color: #64748b;">
            <h3 style="color: #1e293b;">开始新的搜索对话</h3>
            <p>描述你想看的剧集，例如："男主是医生的现代剧"、"想看看甜宠剧"等</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 输入区域（固定在底部，类似ChatGPT）
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # 新对话按钮
    if st.button("➕ 新对话", help="开始全新的对话会话", key="new_conversation_btn"):
        # 创建新对话会话（清空当前，开始全新对话）
        st.session_state.conversation_counter += 1
        st.session_state.current_conversation_id = f"conv_{st.session_state.conversation_counter}"
        st.session_state.conversations[st.session_state.current_conversation_id] = []
        st.session_state.scroll_to_top = True
        st.rerun()
    
    # 输入框和发送按钮
    with st.container():
        col_input1, col_input2, col_input3 = st.columns([6, 1.5, 1])
        with col_input1:
            query = st.text_area(
                "", 
                placeholder="描述你的需求...",
                height=60,
                key="semantic_query_input",
                label_visibility="collapsed"
            )
        with col_input2:
            top_k = st.number_input("推荐数量", 1, 20, 3, key="semantic_top_k", label_visibility="visible")
        with col_input3:
            st.write("")  # 添加空白行对齐
            st.write("")  # 添加空白行对齐
            btn_search = st.button("发送", type="primary", key="send_search_btn")
    
    # 处理搜索（在当前对话中添加新轮次）
    if btn_search and query:
        try:
            # 如果没有当前对话，创建新对话
            if not st.session_state.current_conversation_id:
                st.session_state.conversation_counter += 1
                st.session_state.current_conversation_id = f"conv_{st.session_state.conversation_counter}"
                st.session_state.conversations[st.session_state.current_conversation_id] = []
            
            # 构建增强的查询（结合当前对话的历史上下文）
            enhanced_query = query
            conversation_context = ""  # 用于推荐生成的上下文
            current_turns = st.session_state.conversations.get(st.session_state.current_conversation_id, [])
            if current_turns:
                # 构建完整的对话历史上下文（包含之前的所有轮次）
                context_parts = []
                for prev_turn in current_turns[-3:]:  # 取最近3轮作为上下文
                    prev_query = prev_turn.get('query', '')
                    if prev_query:
                        context_parts.append(f"之前问过: {prev_query}")
                if context_parts:
                    conversation_context = " | ".join(context_parts)
                    # 用于搜索的增强查询（简单拼接最近的查询）
                    previous_queries = [turn['query'] for turn in current_turns[-2:]]
                    context = " ".join(previous_queries)
                    enhanced_query = f"{context} {query}"
            
            with st.spinner("AI 正在分析您的需求..."):
                res = retriever.semantic_search(enhanced_query, top_k)
            
            # 生成推荐语（使用带上下文的查询）
            recommendation_text = ""
            try:
                # 如果有对话历史，将上下文信息传递给推荐生成
                query_for_recommendation = query
                if conversation_context:
                    query_for_recommendation = f"{conversation_context} | 现在问: {query}"
                
                for chunk in retriever.generate_recommendation_stream(query_for_recommendation, res.get("results", [])):
                    recommendation_text += chunk
            except:
                recommendation_text = "根据您的搜索，为您找到了以下相关剧集。"
            
            # 清理推荐文本中的HTML标签
            recommendation_text = clean_html_tags(recommendation_text)
            # 清理查询文本
            query_clean = clean_html_tags(query)
            
            # 在当前对话中添加新轮次
            turn_data = {
                "query": query_clean,
                "recommendation": recommendation_text,
                "results": res.get("results", [])
            }
            st.session_state.conversations[st.session_state.current_conversation_id].append(turn_data)
            
            # 标记需要滚动到顶部
            st.session_state.scroll_to_top = True
            
            # 自动刷新显示
            st.rerun()
            
        except Exception as e:
            st.error(f"搜索过程出错: {str(e)}")
            import traceback
            with st.expander("查看详细错误信息"):
                st.code(traceback.format_exc())