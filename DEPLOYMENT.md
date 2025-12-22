# Streamlit Cloud 部署指南

## 📦 需要推送到 GitHub 的文件

### 必需文件（核心应用）

```
SeriesSearchApp/
├── src/
│   └── app.py              ✅ 主应用文件（必需）
├── requirements.txt        ✅ Python 依赖（必需）
├── .streamlit/
│   ├── config.toml         ✅ Streamlit 配置（必需）
│   └── secrets.toml.example ✅ 配置模板（推荐）
├── README.md               ✅ 项目文档（推荐）
└── .gitignore              ✅ Git 配置（必需）
```

### 可选文件（根据需要）

```
SeriesSearchApp/
├── src/
│   ├── data_loader.py      ⚠️ 如果 app.py 需要则保留
│   ├── query_engine.py     ⚠️ 如果 app.py 需要则保留
│   └── index_builder.py   ⚠️ 如果需要在云端构建索引则保留
├── scripts/                ⚠️ 辅助脚本（可选）
└── tests/                  ⚠️ 测试文件（可选）
```

## 🚫 不推送的文件（已在 .gitignore 中）

- `data/` - 数据文件（数据库、向量数据库、JSON 文件）
- `logs/` - 日志文件
- `venv/` - 虚拟环境
- `.streamlit/secrets.toml` - 包含真实密钥的文件

## 📝 部署步骤

### 1. 初始化 Git 仓库

```bash
cd "/Users/lyfialiu/Desktop/个性化选修课/智能信息检索导论/SeriesSearchApp"
git init
git add README.md .gitignore requirements.txt .streamlit/ src/app.py
# 如果 app.py 引用了其他模块，也要添加
git commit -m "Initial commit for Streamlit Cloud"
```

### 2. 创建 GitHub 仓库并推送

```bash
git remote add origin <your-github-repo-url>
git branch -M main
git push -u origin main
```

### 3. 在 Streamlit Cloud 配置

1. 访问 https://streamlit.io/cloud
2. 用 GitHub 账号登录
3. 点击 "New app"
4. 选择仓库：`SeriesSearchApp`
5. 主文件路径：`src/app.py`
6. 在 "Secrets" 中添加：
   ```
   LLM_API_KEY=your-api-key
   LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   LLM_MODEL_NAME=qwen-max
   QDRANT_PATH=data/qdrant_data
   EMBEDDING_MODEL_PATH=BAAI/bge-large-zh-v1.5
   DB_PATH=data/database/final.db
   ```

### 4. 数据文件处理

由于数据文件太大无法推送到 GitHub，有以下方案：

#### 方案 A：使用 Git LFS（适合中等大小文件）

```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "data/llm_summaries.json"
git lfs track "data/database/final.db"

# 添加到仓库
git add .gitattributes
git add data/llm_summaries.json data/database/final.db
```

#### 方案 B：云存储下载（推荐）

修改 `src/app.py`，在应用启动时从云存储下载数据：

```python
import os
import urllib.request

def download_data_if_needed():
    """如果数据文件不存在，从云存储下载"""
    data_files = {
        "data/llm_summaries.json": "https://your-storage.com/llm_summaries.json",
        "data/database/final.db": "https://your-storage.com/final.db",
    }
    
    for local_path, url in data_files.items():
        if not os.path.exists(local_path):
            print(f"下载 {local_path}...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(url, local_path)
```

#### 方案 C：在 Streamlit Cloud 上构建索引

如果数据文件太大，可以在应用首次启动时提示用户等待索引构建。

## ✅ 检查清单

- [ ] `src/app.py` 已更新为使用 `st.secrets`
- [ ] `requirements.txt` 包含所有依赖
- [ ] `.streamlit/config.toml` 已配置
- [ ] `.gitignore` 已正确配置
- [ ] 数据文件处理方案已确定
- [ ] GitHub 仓库已创建并推送
- [ ] Streamlit Cloud Secrets 已配置

