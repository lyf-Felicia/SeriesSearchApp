# Security Policy

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting for this repository. Do not open a public issue containing credentials, exploit details, or personal data.

Include the affected revision, reproduction steps, impact, and any suggested mitigation. You should receive an acknowledgement within seven days.

## Secrets

The application reads `LLM_API_KEY` from Streamlit Secrets. Never commit `.streamlit/secrets.toml`, API keys, access tokens, or downloaded user data. If a secret is committed, revoke it at the provider first, then remove it from Git history.

## Release assets

Runtime assets are accepted only from the repository and tag pinned in `src/release_assets.py`. Downloads are checked against committed byte sizes and SHA-256 digests before use. ZIP members are constrained by path, type, count, expanded size, and compression ratio.

## Supported version

Security fixes are applied to the current `main` branch only.