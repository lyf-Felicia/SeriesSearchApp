# Data and Model Policy

## Scope

The MIT license covers the source code in this repository. It does **not** grant rights to third-party television metadata, plot text, images, model weights, or Release assets.

## Runtime assets

The application currently expects a versioned SQLite database, generated summaries, and Qdrant index files from GitHub Release `1.0`. These artifacts are distributed separately from the source tree and are integrity-pinned in `src/release_assets.py`.

Before redistributing or using those assets commercially, independently verify the source terms, copyright status, database rights, and model-output obligations for your jurisdiction. The repository does not assert that third-party content is licensed for commercial reuse.

## Images

Poster discovery uses third-party image URLs at runtime. Availability and reuse rights are controlled by the original hosts. Do not treat returned images as repository-owned assets; replace this integration with a licensed catalog for production use.

## Privacy

Search queries are sent to the configured OpenAI-compatible LLM provider for intent extraction, reranking, and recommendation generation. Review that provider's retention and privacy terms before deployment. Do not submit sensitive personal data.

## Removal requests

Open a repository issue that identifies the affected record or asset and the basis for the request. Do not include private personal data in the issue.