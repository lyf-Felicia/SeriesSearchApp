# SeriesSearch

**Find a show from the character, scene, or feeling you remember.**

[![CI](https://github.com/lyf-Felicia/SeriesSearchApp/actions/workflows/ci.yml/badge.svg)](https://github.com/lyf-Felicia/SeriesSearchApp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0f766e.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776ab.svg)](https://www.python.org/)

SeriesSearch is a Streamlit retrieval product for Chinese TV discovery. It combines precise year/genre/region filtering over SQLite with natural-language search over rich and basic Qdrant indexes, then uses an OpenAI-compatible LLM for intent extraction, bounded candidate reranking, and recommendation explanations.

## At a Glance

| | Product perspective |
|---|---|
| **Problem** | Catalog search works when users know metadata, but not when they only remember a character setup, emotional tone, or isolated scene |
| **Insight** | The same title must be discoverable at multiple levels: structured metadata, overall story, character profile, and episode detail |
| **Product bet** | Combine deterministic filters with semantic recall instead of forcing every request through one search paradigm |
| **Design objective** | Return a small, explainable set of titles while preserving the scene or narrative clue that produced each match |

## Product Thesis

> People often remember a show by a character, a scene, or a feeling—not by its exact title or catalog labels.

SeriesSearch treats TV discovery as an **intent-understanding problem**, not just a keyword-matching problem. A user might ask for “a modern romance where the male lead is a doctor” or “the show with a breakup in the rain.” Those memories live at different levels: metadata, character profiles, overall plots, and individual episodes. The product makes the request interpretable, searches across those levels, and returns evidence that users can recognize.

| User behavior | Product decision | Why it matters |
|---|---|---|
| “I know the year and genre.” | Keep a direct SQLite filter path | Deterministic requests should remain fast, predictable, and independent of an LLM |
| “I remember the protagonist was a doctor.” | Represent `PERSONA` intent and make character/occupation signals searchable | Catalog fields rarely capture the identity users actually remember |
| “There was a breakup in the rain.” | Index episodes as child documents and surface matched snippets | A scene-level memory can be invisible in a series-level synopsis |
| “I want something sweet but not childish.” | Retrieve broadly, then let an LLM rerank the bounded candidate set | Semantic similarity creates recall; contextual judgment improves ordering |
| “Why does this match me?” | Generate explanations from retrieved candidates | Recommendation quality includes user trust, not only ranking |

## What Makes the Design Distinctive

### 1. Two search modes reflect two kinds of intent

The interface does not force every request through an AI pipeline. Structured filters serve users who know their constraints; semantic search serves users who only remember narrative clues. This separation also creates a useful fallback boundary: deterministic discovery does not depend on vector retrieval or model behavior.

### 2. Quality and coverage are balanced explicitly

The corpus is split into a **rich index** for titles with generated plot and character representations and a **basic index** for titles that only have source summaries. Searching both avoids the common product tradeoff of either discarding sparsely described titles or lowering the quality of every document to the weakest schema.

### 3. Retrieval preserves the clue that produced the match

Series and episodes form a parent-child document structure. Results are deduplicated at series level, while matched episode snippets remain attached as evidence. The UI can therefore answer both “which show?” and “which remembered scene led to this result?”

### 4. LLMs are used where judgment adds value

The LLM does not replace the database or search index. It handles three bounded tasks: intent extraction, reranking a retrieved candidate set, and explaining the final recommendations. Exact filtering and first-stage recall remain inspectable, reducing cost and keeping the system easier to debug.

In the current production path, intent classification provides interpretable query analysis; it does **not** dynamically route retrieval. Both rich and basic indexes are searched for every semantic query. Dynamic routing remains an experiment until it can be evaluated against this simpler baseline.

![SeriesSearch architecture](docs/architecture.svg)

## Example User Journey

```text
“I want a sweet modern drama where the male lead is a doctor”
	↓
Intent: PERSONA · keywords: doctor, sweet romance
	↓
Rich + basic vector recall across series and episode documents
	↓
Merge by series_id while retaining episode evidence
	↓
LLM reranks a bounded candidate set
	↓
Ranked titles + matched scenes + conversational recommendation
```

This flow also supports refinement within the same browser session: a follow-up such as “make it more suspenseful” carries recent turns into the next retrieval and explanation request.

## Product and Engineering Tradeoffs

| Decision | Benefit | Current limitation / next validation |
|---|---|---|
| Local BGE embeddings and Qdrant | Chinese semantic retrieval without sending the corpus to an embedding API | Model download and local index increase cold-start and memory cost |
| LLM reranking after retrieval | Applies nuanced judgment to a small candidate set | Needs an offline relevance benchmark and latency/cost measurement |
| Session-scoped conversational context | Enables lightweight iterative discovery without accounts | Context is string-based, short-lived, and not a durable preference model |
| Live poster lookup | Makes results recognizable without bundling image assets | Third-party availability and licensing require a production replacement |
| Versioned Release assets | Keeps large artifacts outside Git while preserving reproducibility | First boot is multi-gigabyte; production should pre-stage assets |
| Deliberately narrow production path | Keeps the shipped experience understandable and testable | HyDE, local reranking, and rule-based routing remain isolated experiments until evaluated |

The repository intentionally avoids publishing unsupported accuracy, recall, latency, or memory claims. Establishing a labeled query set and measuring retrieval/reranking contributions is part of the roadmap.

## End-to-End Product Scope

SeriesSearch is designed as more than an LLM wrapper. The project connects five product layers:

| Layer | Product work represented in the repository |
|---|---|
| Discovery | Framing how users search when they remember constraints versus narrative fragments |
| Information architecture | Modeling titles and episodes, and separating enriched from basic content |
| AI orchestration | Combining embedding recall, candidate fusion, LLM scoring, and grounded explanation |
| Interaction design | Supporting two search modes, visible episode evidence, and lightweight follow-up context |
| Product reliability | Pinning large artifacts, protecting secrets, testing trust boundaries, and documenting data rights |

## Evaluation Plan

The next product question is not “can an LLM generate recommendations?” but “does this design help users identify the right show with less effort?” A credible evaluation would combine:

| Dimension | Proposed measure | Decision it informs |
|---|---|---|
| Retrieval coverage | `Recall@15` on persona, scene, and theme query sets | Whether both indexes retrieve a relevant candidate before reranking |
| Ranking quality | `NDCG@5` / human relevance preference | Whether LLM reranking improves over vector-score ordering |
| Evidence quality | Precision of surfaced episode snippets | Whether users can recognize why a result matched |
| User success | Result-detail engagement, reformulation rate, and successful-session rate | Whether the interaction model reduces search effort |
| Operational quality | p50/p95 latency, LLM parse-failure rate, fallback rate, and cost per search | Whether the experience is viable beyond a demo |

The first ablation should compare basic-only retrieval, dual-index retrieval, and dual-index plus LLM reranking. This isolates whether each layer earns its additional complexity.

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