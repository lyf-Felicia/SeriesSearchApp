# 智能剧集推荐系统

基于大语言模型和向量检索技术的智能剧集推荐系统，支持自然语言查询、语义检索和个性化推荐。

## 目录

- [项目概述](#项目概述)
- [技术架构](#技术架构)
- [核心特性](#核心特性)
- [系统设计](#系统设计)
- [快速开始](#快速开始)
- [部署指南](#部署指南)
- [技术栈](#技术栈)

---

## 项目概述

### 背景与问题

传统推荐系统存在以下局限性：

1. **语义理解不足**：依赖关键词精确匹配，无法理解语义相似性和用户意图
2. **内容理解浅层**：仅基于元数据（标题、类型）推荐，无法理解剧情内容、人物关系等深层信息
3. **检索策略单一**：所有内容使用相同检索策略，无法区分内容质量差异
4. **推荐缺乏解释性**：无法解释推荐理由，用户难以判断相关性

### 解决方案

本系统采用**双轨制索引架构**和**LLM增强检索**技术，实现：

- 基于向量相似度的语义检索
- LLM 驱动的意图理解和查询优化
- 多阶段检索与重排序机制
- 层次化文档结构支持精准匹配

---

## 技术架构

### 系统架构

```
┌─────────────────────────────────────────┐
│          Presentation Layer              │
│         (Streamlit Web UI)              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Application Layer                 │
│  ┌────────────────────────────────────┐ │
│  │   Query Processing                 │ │
│  │   - Intent Classification          │ │
│  │   - Query Understanding            │ │
│  │   - Result Generation              │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Retrieval Layer                   │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Vector Search│  │ LLM Rerank  │    │
│  │ - Rich Index │  │ - Scoring   │    │
│  │ - Basic Index│  │ - Ranking    │    │
│  └──────────────┘  └──────────────┘    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Data Layer                        │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Qdrant       │  │ SQLite       │    │
│  │ Vector Store │  │ Metadata DB  │    │
│  └──────────────┘  └──────────────┘    │
└──────────────────────────────────────────┘
```

### 技术选型

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| **Web框架** | Streamlit | 快速构建交互式Web应用 |
| **LLM框架** | LlamaIndex | 文档管理和检索框架 |
| **向量数据库** | Qdrant | 高性能向量相似度搜索 |
| **嵌入模型** | BGE-large-zh-v1.5 | 1024维中文语义向量 |
| **LLM API** | 阿里云DashScope | 兼容OpenAI格式的API |
| **关系数据库** | SQLite | 轻量级元数据存储 |

---

## 核心特性

### 1. 双轨制索引架构

**设计理念**：根据内容质量采用差异化索引策略

- **富文本索引**（Rich Text Index）
  - 覆盖范围：1100 部有 LLM 生成摘要的剧集
  - 文档内容：精炼的 `plot_summary`（~500字）
  - 元数据：结构化标签（tags, occupation_tags, character_tags, style_tags）
  - 检索策略：语义检索 + 元数据过滤

- **基础索引**（Basic Index）
  - 覆盖范围：其余剧集（~1397 部）
  - 文档内容：原始 `summary`
  - 元数据：基础字段（title, year, region, genre）
  - 检索策略：语义检索

**优势**：
- 充分利用高质量内容提升检索精度
- 保证全量覆盖，不遗漏任何剧集
- 根据内容质量动态调整检索权重

### 2. 层次化文档结构

**设计理念**：使用父-子文档关系建立知识图谱

- **父文档**（Parent Document）：剧集级别
  - 包含：整体剧情摘要、人物关系、主题标签
  - 用途：匹配剧集整体主题和风格

- **子文档**（Child Document）：分集级别
  - 包含：具体分集剧情、名场面描述
  - 用途：精准匹配具体情节和场景

**实现**：基于 LlamaIndex `NodeRelationship.PARENT` 建立层次关系

**优势**：
- 支持"名场面"级别的精准检索
- 检索到分集时自动获取剧集上下文
- 避免信息孤岛，建立完整知识图谱

### 3. 多阶段检索与重排序

**检索流程**：

```
用户查询
    ↓
意图分类 (LLM)
    ↓
向量检索 (BGE)
    ├─→ 富文本索引检索 (Top-K=15)
    └─→ 基础索引检索 (Top-K=15)
    ↓
结果合并与去重
    ↓
LLM 重排序 (Top-K=5)
    ↓
最终结果
```

**技术细节**：

1. **向量检索阶段**
   - 使用 BGE-large-zh-v1.5 进行语义编码
   - 分别从双轨制索引检索，合并候选集
   - 召回 Top-K 候选（K=15）

2. **重排序阶段**
   - 使用 LLM 对候选结果进行相关性评分
   - 考虑查询意图与内容的深度匹配度
   - 输出最终 Top-K 结果（K=5）

**优势**：
- 兼顾召回率和精确率
- LLM 能够理解复杂的匹配逻辑
- 提供更精准的推荐结果

### 4. LLM 增强的意图理解

**意图分类**：

| 意图类型 | 说明 | 示例 |
|---------|------|------|
| **PERSONA** | 职业、身份、人设 | "医生"、"霸总"、"单亲妈妈" |
| **SCENE** | 具体情节、名场面 | "雨中分手"、"跳崖"、"误会" |
| **THEME** | 题材、风格、情绪 | "甜宠"、"悬疑"、"虐心" |

**实现**：
- 使用 LLM 进行意图分类和关键词提取
- 根据意图类型调整检索策略
- 提取职业标签用于元数据过滤

### 5. 性能优化

- **流式处理**：使用生成器模式，内存占用 < 500MB
- **动态批处理**：根据可用内存自动调整批处理大小
- **检查点机制**：支持中断恢复，避免重复计算
- **预编译正则**：提升文本处理性能

---

## 系统设计

### 数据模型

**剧集文档结构**：
```python
{
    "series_id": int,
    "title": str,
    "text": str,  # plot_summary 或 summary
    "metadata": {
        "type": "series" | "episode",
        "index_type": "rich_text" | "basic",
        "has_llm_summary": bool,
        "year": str,
        "genre": str,
        "region": str,
        "tags": List[str],
        "occupation_tags": List[str],
        "character_tags": List[str],
        "style_tags": List[str]
    },
    "relationships": {
        "PARENT": RelatedNodeInfo  # 仅子文档有
    }
}
```

### 检索算法

**相似度计算**：
- 使用余弦相似度（Cosine Similarity）
- 向量维度：1024（BGE-large-zh-v1.5）
- 相似度阈值：动态调整

**结果合并策略**：
- 按 `series_id` 去重
- 保留最高相似度分数
- 合并来自不同索引的元数据

**重排序算法**：
- 使用 LLM 对候选结果进行相关性评分（0-10分）
- 考虑用户意图与内容的匹配度
- 根据评分重新排序

### 查询处理流程

1. **查询解析**：LLM 进行意图分类和关键词提取
2. **向量检索**：从双轨制索引中检索候选结果
3. **结果合并**：合并摘要和分集检索结果，去重
4. **重排序**：LLM 对候选结果进行精准评分
5. **结果生成**：生成推荐说明和可视化结果

---

## 快速开始

### 环境要求

- Python 3.9+
- 8GB+ RAM（推荐 16GB）
- 10GB+ 磁盘空间

### 安装步骤

```bash
# 1. 克隆仓库
git clone <your-repo-url>
cd SeriesSearchApp

# 2. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据文件
# - data/database/final.db
# - data/llm_summaries.json

# 5. 构建索引
python3 src/index_builder.py

# 6. 运行应用
streamlit run src/app.py
```

### 配置说明

复制 `.streamlit/secrets.toml.example` 为 `.streamlit/secrets.toml` 并配置：

```toml
LLM_API_KEY = "your-api-key"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-max"
QDRANT_PATH = "data/qdrant_data"
EMBEDDING_MODEL_PATH = "BAAI/bge-large-zh-v1.5"
DB_PATH = "data/database/final.db"
```

---

## 部署指南

### Streamlit Cloud 部署

详细部署指南请参考 [DEPLOYMENT.md](DEPLOYMENT.md)

**快速部署步骤**：
1. 推送代码到 GitHub 仓库
2. 访问 https://streamlit.io/cloud
3. 连接 GitHub 仓库
4. 配置 Secrets（API 密钥等）
5. 部署完成

---

## 技术栈

### 核心框架
- **Streamlit** 1.50+ - Web 应用框架
- **LlamaIndex** 0.10+ - LLM 应用框架
- **Qdrant** 1.7+ - 向量数据库

### 模型与 API
- **BAAI/bge-large-zh-v1.5** - 中文嵌入模型（1024维）
- **阿里云 DashScope** - LLM API（兼容 OpenAI）
- **Sentence Transformers** 2.2+ - 重排序模型

### 数据存储
- **SQLite** - 关系数据库
- **Qdrant** - 向量数据库（本地 SQLite 存储）

### 工具库
- **OpenAI** 1.0+ - API 客户端
- **BeautifulSoup4** 4.11+ - HTML 解析
- **Requests** 2.28+ - HTTP 请求

---

## 项目结构

```
SeriesSearchApp/
├── src/                    # 核心源代码
│   ├── app.py             # Streamlit 应用
│   ├── data_loader.py     # 数据加载模块
│   ├── index_builder.py   # 索引构建模块
│   └── query_engine.py    # 查询引擎模块
├── scripts/               # 辅助脚本
├── tests/                 # 测试文件
├── data/                  # 数据文件（不推送到 Git）
├── logs/                  # 日志文件（不推送到 Git）
├── requirements.txt       # Python 依赖
├── .streamlit/           # Streamlit 配置
└── README.md             # 项目文档
```

---

## 常见问题

**Q: 如何添加新的 LLM 摘要？**  
A: 更新 `data/llm_summaries.json`，然后重新运行 `src/index_builder.py`

**Q: 索引构建中断怎么办？**  
A: 系统支持检查点恢复，下次运行会自动检测并询问是否继续

**Q: 如何调整检索参数？**  
A: 在 `src/app.py` 的 `semantic_search` 方法中修改 `recall_top_k` 和 `top_k` 参数

**Q: 内存不足怎么办？**  
A: 系统已优化为流式处理（< 500MB），如仍不足可减小批处理大小

---

## 许可证

本项目为课程设计项目，仅供学习使用。
