# Math Sentinel Runbook

This runbook documents the supported local training, evaluation, sampling, checkpoint, memory, and cleanup workflows for Math Sentinel.

## Scope

Run commands from the repository root.

Default locations:

- training config: `config/default.yaml`
- curriculum config: `config/curriculum.yaml`
- search override config: `config/search.yaml`
- checkpoint: `checkpoints/last.pt`
- training log: `logs/train.jsonl`
- replay store: `memory/replay.jsonl`
- hard-case store: `memory/hard_cases.json`
- lemma store: `memory/lemma_store.json`
- tactic statistics: `memory/tactic_stats.json`

## Environment setup

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the active interpreter and PyTorch installation before a long run:

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__); print('cuda=', torch.cuda.is_available())"
```

For repository development and tests:

```bash
python -m pip install -r requirements-dev.txt
```

## Training flags

`train.py` supports:

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

## Standard training runs

Default run:

```bash
python train.py
```

Short smoke run:

```bash
python train.py --steps 20 --batch-size 4 --micro-batch-size 4 --eval-every 10 --save-every 10
```

Longer CUDA run:

```bash
python train.py --steps 2000 --batch-size 16 --micro-batch-size 8 --device cuda --compile
```

Custom learning rate:

```bash
python train.py --steps 5000 --batch-size 16 --lr 0.0001
```

## Resume training

Resume from the default checkpoint:

```bash
python train.py --resume checkpoints/last.pt
```

Resume with explicit overrides:

```bash
python train.py --resume checkpoints/last.pt --steps 2000 --device cuda
```

`--resume` restores prover weights, verifier weights, optimizer state, scaler state, and the last step stored in the checkpoint. Memory paths continue to come from the active configuration.

## Memory behavior

Memory stores are loaded from the paths under the `memory` section of the selected configuration.

Search behavior is loaded from the `search` section of that configuration and then overridden by `config/search.yaml` when the override file exists.

`--memory-refresh-samples` controls how many sampled evaluation cases are written back into memory at each evaluation interval.

Disable refresh for a short diagnostic run:

```bash
python train.py --memory-refresh-samples 0
```

## Starting from previous memory stores

Math Sentinel reads memory stores from local files. To continue from saved stores, either place those files at the configured default paths or point a separate configuration file at alternate paths.

Example override:

```yaml
memory:
  replay_capacity: 5000
  hard_case_capacity: 2000
  lemma_store_path: /path/to/math-sentinel-memory/lemma_store.json
  hard_cases_path: /path/to/math-sentinel-memory/hard_cases.json
  tactic_stats_path: /path/to/math-sentinel-memory/tactic_stats.json
  replay_path: /path/to/math-sentinel-memory/replay.jsonl
```

Then run:

```bash
python train.py --config config/custom_memory.yaml
```

Remote object stores are not a native memory protocol; materialize the files locally before training.

## Evaluation

Evaluate a checkpoint:

```bash
python eval.py --checkpoint checkpoints/last.pt --count 64
```

The evaluation command reports research diagnostics over sampled curriculum tasks. A successful process exit is a runtime check, not a claim of mathematical correctness.

## Sampling and manual solving

Sample a generated task:

```bash
python sample.py --checkpoint checkpoints/last.pt
```

Run a manual problem:

```bash
python sample.py --checkpoint checkpoints/last.pt --domain linear_equation --problem "Solve: 2x + 3 = 11"
```

## Checker plugins

Load a checker/tool plugin with:

```bash
python train.py --checker-plugin plugins/example_checker_plugin.py
python sample.py --checker-plugin plugins/example_checker_plugin.py
```

Plugins are loaded through `ToolRegistry`; they extend available deterministic tools but do not convert Math Sentinel into a formal proof kernel.

## Test and smoke verification

Install development dependencies and run all tests:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Run the lightweight end-to-end smoke configuration:

```bash
python train.py --config config/smoke.yaml --steps 1 --batch-size 1 --micro-batch-size 1 --eval-every 100 --save-every 100 --memory-refresh-samples 0 --device cpu
python eval.py --config config/smoke.yaml --count 1 --step 1
python sample.py --config config/smoke.yaml --domain arithmetic --problem "What is 2 + 2?"
```

## Cleanup

The PowerShell cleanup script removes generated checkpoints, logs, and persistent memory artifacts and recreates the checkpoint/log directories.

Interactive cleanup:

```powershell
.\scripts\clean_training.ps1
```

Non-interactive cleanup:

```powershell
.\scripts\clean_training.ps1 -Yes
```

## Operational notes

- `--device auto` chooses CUDA when available and CPU otherwise.
- `--compile` attempts `torch.compile` when supported; training falls back to eager execution if compilation setup fails.
- Checkpoints are written at the configured `save_every` interval.
- Memory stores are persisted when checkpoints are saved.
- Persistent memory is loaded from local files at process startup.
- The valid save-frequency flag is `--save-every N`.

## Troubleshooting

- If virtual-environment activation fails, invoke the environment's Python interpreter directly and verify it with `python -c "import sys; print(sys.executable)"`.
- If `import torch` fails, reinstall dependencies in the active environment.
- If CUDA is unavailable or incompatible with the installed PyTorch build, use `--device cpu` or `--device auto`.
- If `--compile` causes backend-specific failures, retry without the flag; compilation is an optimization, not a correctness requirement.
- If resume fails, verify that the checkpoint exists and was produced by code/configuration compatible with the current model definitions.
