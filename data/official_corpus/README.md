Place full official benchmark exports here before running `benchmarks/run_official_corpus.py`.

Expected default layout:

- `data/official_corpus/gaia/records.jsonl`
- `data/official_corpus/gaia/attachments/`
- `data/official_corpus/swebench/records.jsonl`
- `data/official_corpus/swebench/workspaces/`
- `data/official_corpus/swebench/repo_cache/`

The records file may be JSON or JSONL, but the default commands assume JSONL.

You can populate the staging area with:

- `python benchmarks/download_official_corpus.py --corpus swebench`
- `python benchmarks/download_official_corpus.py --corpus gaia --gaia-token <hf_token>`
- `python benchmarks/run_official_corpus.py --corpus swebench --max-cases 5 --deterministic --safe-runtime`

Notes:

- SWE-bench Verified downloads publicly.
- GAIA is gated on Hugging Face; without access, the downloader writes `data/official_corpus/gaia/download_status.json`.
- SWE-bench supports lazy repo materialization from official metadata, so `workspaces/` can start empty.
- The SWE-bench runner now normalizes malformed upstream `test_patch` hunk counts before applying them to a live workspace.
