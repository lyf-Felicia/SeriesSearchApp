# Deployment Guide

## Current Readiness

The application has an executable offline test suite, secret scanning, fixed Release checksums, safe ZIP extraction, and Streamlit secret-based configuration. A full hosted startup has **not** yet been verified: it requires a live LLM credential, approximately 2.55 GB of Release downloads, extracted Qdrant storage, and the BGE embedding model.

Use a host with persistent writable storage, enough cold-start time, and at least 6 GB of free disk. Confirm the provider's memory limit before deploying the BGE model and local Qdrant together. Streamlit Community Cloud may be unsuitable if its current storage or startup limits are lower than these requirements.

## Configuration

Use Python 3.11 and install the pinned top-level dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure these values in the platform's secret manager:

```toml
LLM_API_KEY = "your-provider-key"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL_NAME = "qwen-max"
QDRANT_PATH = "data/qdrant_data"
EMBEDDING_MODEL_PATH = "BAAI/bge-large-zh-v1.5"
DB_PATH = "data/database/final.db"
```

Do not set `GITHUB_REPO` or `RELEASE_TAG`: the trusted repository, version, sizes, and SHA-256 digests are pinned in `src/release_assets.py`.

## Preflight

```bash
python -W error -m compileall -q src scripts
python -m pytest -q
```

Then start the service:

```bash
streamlit run src/app.py --server.address 0.0.0.0 --server.port 8501
```

On first boot, monitor disk use and allow time for verified Release downloads and model initialization. A production platform should persist both `data/` and the Hugging Face model cache between restarts.

## Release Checklist

- CI compile, pytest, and Gitleaks jobs pass.
- `LLM_API_KEY` is stored only in the deployment secret manager.
- Release assets match the manifest in `src/release_assets.py`.
- The host has sufficient disk, memory, egress, and cold-start allowance.
- LLM provider retention and regional-processing terms are acceptable.
- Third-party data and image rights have been reviewed under `DATA_POLICY.md`.
- A real semantic query, structured filter, restart, and corrupted-download recovery have been tested in the target environment.

## Updating Data Assets

Publish new assets under a new immutable Release tag. Update every corresponding `size` and `sha256` value in `src/release_assets.py`, run the offline test suite, and deploy the code change together with the Release. Never replace files under an existing tag without updating the manifest.