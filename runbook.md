## Math Sentinel V7 Runbook

This runbook documents the training workflow that is actually supported by `train_v7.py`.

## Scope

- Working directory: repository root.
- Default training config: `config/default.yaml`.
- Default curriculum config: `config/curriculum.yaml`.
- Default search override config: `config/search.yaml`.
- Default checkpoint output: `checkpoints/last.pt`.
- Default log output: `logs/train_v7.jsonl`.
- Default persistent memory files:
  - `memory/replay.jsonl`
  - `memory/hard_cases.json`
  - `memory/lemma_store.json`
  - `memory/tactic_stats.json`

## Environment Setup

Use PowerShell from the repository root.

```powershell
C:/Python312/python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the active interpreter before starting training:

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__)"
```

Expected result on this machine: `python` should resolve to the activated `.venv` interpreter, not `C:\Users\scsar\AppData\Local\Programs\Python\Python314\python.exe`.

## Supported Training Flags

The training entrypoint supports these operational flags:

- `--config`
- `--curriculum-config`
- `--steps`
- `--batch-size`
- `--micro-batch-size`
- `--lr`
- `--device`
- `--compile`
- `--resume`
- `--checker-plugin`
- `--eval-every`
- `--save-every`
- `--memory-refresh-samples`

## Standard Training Runs

Default run:

```powershell
python train_v7.py
```

Short smoke run:

```powershell
python train_v7.py --steps 20 --batch-size 4 --micro-batch-size 4 --eval-every 10 --save-every 10
```

Longer GPU run:

```powershell
python train_v7.py --steps 2000 --batch-size 16 --micro-batch-size 8 --device cuda --compile
```

RTX 4060 laptop recommended open-weight run:

```powershell
python train_v7.py --config config/product_rtx4060_laptop.yaml --search-config "" --backend math
```

RTX 4060 laptop code-agent benchmark smoke:

```powershell
python benchmark_v7.py --profile rtx4060_coder_local --suite swebench_verified_smoke --deterministic --safe-runtime
```

Official-style manifest benchmark on the RTX 4060:

```powershell
python benchmark_v7.py --profile rtx4060_general_local --suite manifest:benchmarks/manifests/gaia_medium_official_style.json --deterministic --safe-runtime
python benchmark_v7.py --profile rtx4060_coder_local --suite manifest:benchmarks/manifests/swebench_verified_medium_official_style.json --deterministic --safe-runtime
```

Import a local public benchmark export into a runnable manifest:

```powershell
python benchmarks/import_public_manifest.py --format swebench --input data\swebench_export.jsonl --output benchmarks\manifests\swebench_public_import.json --fixtures-root benchmarks\fixtures\swebench_medium_smoke
python benchmarks/import_public_manifest.py --format gaia --input data\gaia_export.jsonl --output benchmarks\manifests\gaia_public_import.json --fixtures-root benchmarks\fixtures\gaia_medium_smoke
```

Full official-corpus workflow:

```powershell
python benchmarks\run_official_corpus.py --corpus all --prepare-only --strict-materialization
python benchmarks\run_official_corpus.py --corpus gaia --deterministic --safe-runtime --results-dir results\official
python benchmarks\run_official_corpus.py --corpus swebench --deterministic --safe-runtime --results-dir results\official
```

The runner expects these staged inputs by default:

- `data\official_corpus\gaia\records.jsonl`
- `data\official_corpus\gaia\attachments\`
- `data\official_corpus\swebench\records.jsonl`
- `data\official_corpus\swebench\workspaces\`

It will build strict manifests in `benchmarks\manifests\gaia_full_official.json` and `benchmarks\manifests\swebench_full_official.json`.

When using a dedicated product config like `config/product_rtx4060_laptop.yaml`, pass `--search-config ""` if you want the config's own search block to stay intact instead of being overridden by `config/search.yaml`.

The RTX 4060 branch assumes the downloaded Qwen 1.5B weights live under the local `models/` folder, so the config can stay offline by default.

Custom learning rate:

```powershell
python train_v7.py --steps 5000 --batch-size 16 --lr 0.0001
```

## Resume Training

Resume from the last checkpoint on disk:

```powershell
python train_v7.py --resume checkpoints/last.pt
```

Resume with explicit overrides:

```powershell
python train_v7.py --resume checkpoints/last.pt --steps 2000 --device cuda
```

`--resume` restores model weights, verifier weights, optimizer state, scaler state, and last step from the checkpoint file. It does not by itself change memory file paths.

## Memory Behavior

Memory is always loaded from the file paths configured under the `memory` section of `config/default.yaml` or the config file passed with `--config`.

Search behavior is loaded from the `search` section in `config/default.yaml`, then overridden by values in `config/search.yaml` when that file exists. This includes beam parameters and scoring weights used by beam ranking.

`--memory-refresh-samples` controls how many sampled evaluation cases are written back into the memory stores during each evaluation interval.

Example with memory refresh enabled explicitly:

```powershell
python train_v7.py --memory-refresh-samples 32
```

Example with memory refresh disabled:

```powershell
python train_v7.py --memory-refresh-samples 0
```

## Starting From Previous Memory Stores

The training code supports loading previous memory stores from files on disk. It does not implement a remote memory endpoint protocol.

There are two supported ways to start from previous stores.

Method 1: replace the default store files before training.

```powershell
Copy-Item C:\backups\replay.jsonl memory\replay.jsonl -Force
Copy-Item C:\backups\hard_cases.json memory\hard_cases.json -Force
Copy-Item C:\backups\lemma_store.json memory\lemma_store.json -Force
Copy-Item C:\backups\tactic_stats.json memory\tactic_stats.json -Force
python train_v7.py
```

Method 2: use a separate config file that points to alternate store paths.

Example memory block for an override config:

```yaml
memory:
  replay_capacity: 5000
  hard_case_capacity: 2000
  lemma_store_path: D:/math-sentinel-store/lemma_store.json
  hard_cases_path: D:/math-sentinel-store/hard_cases.json
  tactic_stats_path: D:/math-sentinel-store/tactic_stats.json
  replay_path: D:/math-sentinel-store/replay.jsonl
```

Run with the override config:

```powershell
python train_v7.py --config config/custom_memory.yaml
```

If your previous stores live behind an HTTP or object-store endpoint, download them first and then use one of the two local-file methods above.

## Evaluation And Sampling

Evaluate a checkpoint:

```powershell
python eval_v7.py --checkpoint checkpoints/last.pt --count 64
```

Sample from a checkpoint:

```powershell
python sample_v7.py --checkpoint checkpoints/last.pt
```

Run a manual problem:

```powershell
python sample_v7.py --checkpoint checkpoints/last.pt --domain linear_equation --problem "Solve: 2x + 3 = 11"
```

## Cleanup

The cleanup script removes checkpoints, logs, persistent memory artifacts, benchmark temp directories, pytest cache, recursive `__pycache__` folders, and generated benchmark result JSON files. It then recreates empty `checkpoints` and `logs` directories.

Interactive cleanup:

```powershell
.\scripts\clean_training.ps1
```

Non-interactive cleanup:

```powershell
.\scripts\clean_training.ps1 -Yes
```

Keep generated benchmark results while still cleaning training/runtime residue:

```powershell
.\scripts\clean_training.ps1 -Yes -KeepResults
```

## Operational Notes

- If `--device auto` is used, training selects CUDA when available and otherwise falls back to CPU.
- `--compile` only enables `torch.compile` when the installed PyTorch build supports it. If compilation fails at runtime, training falls back to eager mode and logs a warning.
- Checkpoints are written on the `save_every` interval.
- Memory stores are persisted when checkpoints are saved.
- The code imports memory stores from local files at process startup.

## Troubleshooting

- If PowerShell blocks venv activation, run `Set-ExecutionPolicy -Scope Process Bypass` in the current shell and retry activation.
- If `python -c "import torch"` fails, you are using the wrong interpreter. On this machine, bare `python` currently resolves to `Python314`, while `torch` is installed under `C:/Python312/python.exe`.
- If needed, bypass the shell alias entirely and run training with the interpreter that has `torch` installed:

```powershell
C:/Python312/python.exe train_v7.py --steps 20000 --batch-size 16 --micro-batch-size 8 --device cuda --save-every 100 --compile
```

- If CUDA is not available, omit `--device cuda` and let the code use CPU.
- If `--compile` reports an Inductor or Triton failure, training now falls back to eager automatically. Remove the flag if you want to avoid the warning and startup retry.
- If resume fails, verify that `checkpoints/last.pt` exists and matches the current model code.
- The correct flag is `--save-every 100`. `--save every 100` is not a valid `train_v7.py` argument.

