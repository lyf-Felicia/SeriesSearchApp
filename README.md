# 智能剧集推荐系统

基于 LlamaIndex 和向量检索的智能剧集推荐系统，支持自然语言查询和个性化推荐。

## 🚀 快速开始

### 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据文件（放在 data/ 目录下）
# - data/database/final.db
# - data/qdrant_data/ (向量数据库)
# - data/llm_summaries.json (可选)

# 3. 运行应用
streamlit run src/app.py
```

### Streamlit Cloud 部署

详细部署指南请参考 [DEPLOYMENT.md](DEPLOYMENT.md)

## 📁 项目结构

```
SeriesSearchApp/
├── src/
│   ├── app.py              # Streamlit Web 应用（主文件）
│   ├── data_loader.py      # 数据加载模块
│   ├── index_builder.py    # 索引构建模块
│   └── query_engine.py     # 查询引擎模块
├── scripts/                 # 辅助脚本
├── tests/                  # 测试文件
├── data/                   # 数据文件目录（不推送到 Git）
├── logs/                   # 日志文件目录（不推送到 Git）
├── requirements.txt        # Python 依赖
├── .streamlit/            # Streamlit 配置
│   ├── config.toml
│   └── secrets.toml.example
├── README.md              # 项目说明
├── DEPLOYMENT.md          # 部署指南
└── .gitignore             # Git 忽略文件
```

## ⚙️ 配置

### 本地配置

复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml` 并填入配置：

```toml
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-max"
QDRANT_PATH = "data/qdrant_data"
EMBEDDING_MODEL_PATH = "BAAI/bge-large-zh-v1.5"
DB_PATH = "data/database/final.db"
```

### Streamlit Cloud 配置

在 Streamlit Cloud 的 Secrets 中配置上述变量。

## 📝 主要功能

- 🔍 **智能检索**：基于自然语言查询，从数千部剧集中精准匹配
- 📊 **双轨制索引**：区分有 LLM 摘要的剧集和基础剧集
- 🎬 **层次化检索**：支持剧集摘要和分集剧情的联合检索
- 💬 **对话式推荐**：基于 Streamlit 的交互式 Web 界面

## 🛠️ 技术栈

- **Streamlit** - Web 应用框架
- **LlamaIndex** - LLM 应用框架
- **Qdrant** - 向量数据库
- **BGE** - 中文嵌入模型
- **SQLite** - 关系数据库

## 📄 许可证

本项目为课程设计项目，仅供学习使用。
