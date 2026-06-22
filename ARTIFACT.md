# Artifact Manifest

This public repository is a sanitized source-code artifact for HarnessFix.

Included:

- HarnessFix pipeline code.
- HTIR and failure-analysis utilities.
- Scoped repair and validation helper code.
- Initial harness snapshots for SWE-Bench, GAIA, AppWorld, and Terminal-Bench.
- Final repaired harness snapshots under `task_agent/final/`.
- Dataset sampling and lightweight evaluation scripts.

Excluded:

- API keys, tokens, credentials, and private endpoint URLs.
- Raw task traces and model conversations.
- Generated evaluation reports and logs.
- Downloaded benchmark datasets.
- Local virtual environments, caches, temporary downloads, and paper drafting assets.

Users should obtain official benchmark data from the original benchmark maintainers and configure credentials locally through `.env`.
