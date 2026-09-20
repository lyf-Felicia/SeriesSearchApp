# SeriesSearch

**Hybrid discovery for Chinese TV series, combining deterministic metadata filters with LLM-assisted semantic retrieval.**

[![CI](https://github.com/lyf-Felicia/SeriesSearchApp/actions/workflows/ci.yml/badge.svg)](https://github.com/lyf-Felicia/SeriesSearchApp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0f766e.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab.svg)](https://www.python.org/)

SeriesSearch is a Streamlit retrieval product with two complementary experiences: precise year/genre/region filtering over SQLite, and natural-language discovery over rich and basic Qdrant indexes. The semantic path uses an OpenAI-compatible LLM for intent extraction, candidate reranking, and recommendation explanations.

![SeriesSearch architecture](docs/architecture.svg)

## Product Capabilities

| Experience | Implementation | Best for |
|---|---|---|
| Structured filters | Parameterized SQLite queries with multi-select filters | Known year, genre, or region constraints |
| Natural-language search | BGE embeddings over two Qdrant collections | Persona, scene, theme, and plot-level requests |
| Candidate reranking | LLM relevance scoring after dual-index recall | Resolving nuanced intent across similar candidates |
| Recommendation explanation | Streaming LLM response grounded in retrieved candidates | Understanding why each result matches |
| Conversation context | Recent turns retained in Streamlit session state | Refining a search within one browser session |

### Semantic Request Path

1. Classify the query as `PERSONA`, `SCENE`, or `THEME` and extract keywords.
2. Retrieve 15 candidates independently from the rich-text and basic Qdrant indexes.
3. Merge candidates by `series_id` while retaining episode-level evidence.
4. Ask the configured LLM to score relevance and return the final top results.
5. Stream a recommendation explanation to the interface.

The structured filter path is intentionally independent of the LLM and vector store.

## Architecture

### Online application

- `src/app.py` owns the Streamlit interface, session state, semantic orchestration, and result presentation.
- `src/filter_search.py` owns parameterized SQLite filtering.
- `src/release_assets.py` owns the trusted asset boundary: fixed repository/tag, HTTPS host allowlist, byte-size and SHA-256 verification, bounded ZIP extraction, and atomic replacement.
- Local Qdrant stores `tv_series_rich_text` and `tv_series_basic`; SQLite serves deterministic filters.

### Offline data path

`src/data_loader.py` converts series and episode records into LlamaIndex documents. `src/index_builder.py` embeds them with `BAAI/bge-large-zh-v1.5` and builds the two Qdrant collections with batching and checkpoints.

### Experimental code

The following modules are research prototypes and are **not connected to the production Streamlit entry point**:

- `scripts/retriever.py`: Lightning/Deep/Filter modes, optional HyDE, and a local BGE reranker.
- `scripts/app_router.py`: rule-based routing that still references legacy modules.
- `src/query_engine.py`: an alternate Ollama/LlamaIndex query prototype.

## Repository Layout

```text
SeriesSearchApp/
├── src/
│   ├── app.py                 # production Streamlit entry point
│   ├── filter_search.py       # testable SQLite filter service
│   ├── release_assets.py      # verified Release download/extraction
│   ├── data_loader.py         # SQLite/summary to document pipeline
│   ├── index_builder.py       # offline dual-index builder
│   └── query_engine.py        # experimental Ollama path
├── scripts/                   # collection, preparation, and experiments
├── tests/                     # offline security and SQL contract tests
├── docs/architecture.svg      # product and data-supply architecture
├── .github/workflows/ci.yml   # compile, pytest, and Gitleaks checks
├── DATA_POLICY.md             # third-party data and model boundaries
└── SECURITY.md                # disclosure and secret-handling policy
```

## Setup

### Prerequisites

- Python 3.11
- At least 6 GB of free disk space for downloaded assets, extraction, and model cache
- An OpenAI-compatible API endpoint and key
- Network access to GitHub Releases and the configured embedding-model source

```bash
git clone https://github.com/lyf-Felicia/SeriesSearchApp.git
cd SeriesSearchApp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Set `LLM_API_KEY` in `.streamlit/secrets.toml`. The file is ignored by Git.

```toml
LLM_API_KEY = "your-provider-key"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-max"
QDRANT_PATH = "data/qdrant_data"
EMBEDDING_MODEL_PATH = "BAAI/bge-large-zh-v1.5"
DB_PATH = "data/database/final.db"
```

## Run

```bash
streamlit run src/app.py
```

On the first start, the application downloads approximately 2.55 GB of versioned data assets from GitHub Release `1.0`. It verifies exact sizes and SHA-256 digests before installing them. The embedding model is fetched separately if it is not already cached.

## Verification

The offline suite does not require the large Release assets or an API key.

```bash
python -W error -m compileall -q src scripts
python -m pytest -q
```

Coverage currently focuses on the highest-risk local boundaries: SQL filter behavior and injection resistance, Release hash validation, streamed atomic downloads, ZIP path traversal, symlink rejection, and validated directory replacement. LLM and Qdrant integration tests remain roadmap work.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for resource requirements and a release checklist. The code is structured for Streamlit hosting, but a successful production deployment has not yet been verified in CI because startup requires multi-gigabyte assets, model download, and a live LLM credential.

## Security and Data Use

- Secrets belong only in Streamlit Secrets or another deployment secret manager.
- Search text is sent to the configured LLM provider. Review its retention policy before deployment.
- The MIT license applies to source code, not third-party metadata, plot text, images, model weights, or Release assets.
- Review [SECURITY.md](SECURITY.md) and [DATA_POLICY.md](DATA_POLICY.md) before public or commercial deployment.

## Roadmap

- Add mocked LLM/Qdrant integration tests and a small redistributable demo fixture.
- Add a pure-vector fallback when the LLM provider is unavailable.
- Replace live poster discovery with an explicitly licensed image catalog.
- Benchmark retrieval quality, latency, and reranking contribution before publishing performance claims.
- Move long asset preparation out of the request-serving process for production deployments.

## License

Source code is released under the [MIT License](LICENSE). Third-party data and model artifacts are governed separately as described in [DATA_POLICY.md](DATA_POLICY.md).