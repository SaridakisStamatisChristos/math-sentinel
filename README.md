# Math Sentinel V7

Math Sentinel V7 is a **stateful symbolic proof-agent scaffold** for mathematics built on top of a small domain-neutral reasoning engine. It is designed as the first version in the series where the authoritative object is a **reasoning state**, not just raw generated text, with math as the first backend.

## What V7 is

V7 combines:

- a small causal **prover** transformer
- a small recurrent **verifier**
- an explicit state object
- a typed action system
- deterministic math tools
- verifier-guided beam search
- MCTS-style planning search for planning-heavy domains
- replay, hard-case tracking, and lemma memory
- procedural math curriculum generation

This repo is meant to be a real runnable starting point, not a stub architecture dump.

It now supports two prover runtimes:

- `legacy_tiny`: the original in-repo tiny transformer for fast smoke tests
- `hf_causal_lm`: a local open-weight Hugging Face runtime, with LoRA as the default fine-tuning path

## Backends

The repo now has six wired backends:

- `math`: the original symbolic math backend
- `string_ops`: a non-math backend for deterministic text and sequence operations
- `code_ops`: a code-oriented backend for deterministic Python snippet analysis
- `planning_ops`: a planning-oriented backend for dependency, budget, and time-constrained plan construction
- `swebench_ops`: a repository-patching backend shaped like local SWE-bench-style bug-fix tasks
- `gaia_ops`: a file-and-tool reasoning backend shaped like local GAIA-style tasks

The shared engine lives in `engine/`, while domain adapters live in `domains/`.

## What V7 can do now

It can train on and search over synthetic tasks in these families:

- arithmetic
- fractions
- divmod
- gcd/lcm
- modular arithmetic
- primality
- factorization
- linear equations
- polynomial simplification
- derivatives
- antiderivatives
- short proof-template parity tasks

The `string_ops` backend currently supports:

- reverse text
- uppercase conversion
- vowel counting
- alphabetical word sorting
- duplicate removal with order preservation

The `code_ops` backend currently supports:

- extracting a function name
- counting function parameters
- loop detection
- conditional detection
- assignment counting
- first called-function extraction
- distinct called-function counting
- literal return-value extraction
- repository workspace inspection
- repo-aware patch drafting, application, verification, and rollback
- local repo-patch tasks through the shared code-agent toolchain

The `planning_ops` backend currently supports:

- dependency-respecting project plans
- budget-constrained shopping plans
- time-limited day plans with dependency handling

The `swebench_ops` backend currently supports:

- repository workspace inspection
- test inspection and failure localization
- file reading and code search
- test-guided patch drafting from source plus extracted test cases
- deterministic patch application and rollback
- unit-test execution
- local SWE-bench-style smoke tasks with fixture repos

The `gaia_ops` backend currently supports:

- question planning with file/intent inference
- workspace file inspection
- generic evidence-file inspection
- CSV aggregation from prompt/entity matching
- JSON scalar lookup from path scoring
- meeting-slot overlap resolution from generic JSON evidence
- evidence-aware answer synthesis
- local GAIA-style smoke tasks with fixture data

It can also execute typed proof actions such as:

- `THINK`
- `APPLY`
- `CHECK`
- `ANSWER`
- `REWRITE`
- `LEMMA`
- `SUBGOAL`
- `RESOLVE_SUBGOAL`
- `ASSUME`
- `BACKTRACK`
- `CALL_PLUGIN`
- `SIMPLIFY`

## Architecture in words

1. A domain backend emits or loads a task.
2. The task is wrapped into a structured reasoning state.
3. The prover is conditioned on the serialized state and proposes action traces.
4. The domain parser extracts typed actions.
5. The executor applies actions to child states using exact tools.
6. The verifier scores those child states.
7. Search can route between beam search and MCTS-style tree search, while pruning duplicate states and biasing toward historically useful tactics.
8. Replay, hard-case tracking, and lemma memory retain what mattered.
9. In this repo, math is the first fully wired backend, but the engine now has multiple non-math adapters too.

## Repo layout

```text
math_sentinel_v7/
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── train_v7.py
├── eval_v7.py
├── sample_v7.py
├── engine/
├── domains/
├── config/
├── sentinel/
├── proof/
├── tools/
├── search/
├── memory/
├── curriculum/
├── checkpoints/
├── logs/
└── plugins/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For pinned benchmark and CI reproduction, use:

```bash
pip install -r requirements-lock.txt
```

The `hf_causal_lm` runtime uses:

- `transformers`
- `peft`
- `accelerate`
- `safetensors`

`bitsandbytes` is optional and not required for Windows-safe inference.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For pinned benchmark and CI reproduction on Windows:

```powershell
pip install -r requirements-lock.txt
```

If `python` is not available in your shell after activation, use the interpreter path documented in [runbook.md](/C:/Users/scsar/Desktop/math_sentinel_v7/runbook.md).

## Training quickstart

Light CPU/GPU smoke run:

```bash
python train_v7.py --steps 20 --batch-size 4 --micro-batch-size 4 --eval-every 10 --save-every 10
```

More serious CUDA run:

```bash
python train_v7.py --steps 2000 --batch-size 16 --micro-batch-size 8 --compile
```

Train the non-math backends:

```bash
python train_v7.py --backend string_ops --steps 200
python train_v7.py --backend code_ops --steps 200
python train_v7.py --backend planning_ops --steps 200
```

Open-weight runtime example:

```bash
python train_v7.py --config config/product_openweights.yaml --model-provider hf_causal_lm --backbone Qwen/Qwen2.5-1.5B-Instruct
```

Once a backbone is cached locally, add `--local-files-only` to force fully local model loads.

Recommended local config for an RTX 4060 laptop GPU:

```bash
python train_v7.py --config config/product_rtx4060_laptop.yaml --search-config "" --backend math
python sample_v7.py --config config/product_rtx4060_laptop.yaml --search-config "" --backend math --domain arithmetic --problem "Compute: 12 + 30"
python eval_v7.py --config config/product_rtx4060_laptop.yaml --search-config "" --backend math --count 32
```

The empty `--search-config ""` keeps the generic `config/search.yaml` override from shrinking the dedicated 4060 search settings.

On this branch, the 4060 config assumes the public 1.5B Qwen weights have already been downloaded into local project folders under `models/`, so the run stays fully local.

For code-agent tasks on the same GPU, prefer the coder-tuned 1.5B benchmark profile:

```bash
python benchmark_v7.py --profile rtx4060_coder_local --suite swebench_verified_smoke --deterministic --safe-runtime
```

Official-style manifest benchmarks are also supported. A manifest run uses `--suite manifest:<path>` and drives the same benchmark engine through a locally imported public-benchmark export:

```bash
python benchmark_v7.py --profile rtx4060_general_local --suite manifest:benchmarks/manifests/gaia_medium_official_style.json --deterministic --safe-runtime
python benchmark_v7.py --profile rtx4060_coder_local --suite manifest:benchmarks/manifests/swebench_verified_medium_official_style.json --deterministic --safe-runtime
```

To convert a local benchmark export into a runnable manifest:

```bash
python benchmarks/import_public_manifest.py --format swebench --input data/swebench_export.jsonl --output benchmarks/manifests/swebench_public_import.json --fixtures-root benchmarks/fixtures/swebench_medium_smoke
python benchmarks/import_public_manifest.py --format gaia --input data/gaia_export.jsonl --output benchmarks/manifests/gaia_public_import.json --fixtures-root benchmarks/fixtures/gaia_medium_smoke
```

You can also scale the model in `config/default.yaml` once CUDA is confirmed working.

Resume:

```bash
python train_v7.py --resume checkpoints/last.pt
```

## Evaluation

```bash
python eval_v7.py --checkpoint checkpoints/last.pt --count 64
```

Evaluate the non-math backends:

```bash
python eval_v7.py --backend string_ops --count 64
python eval_v7.py --backend code_ops --count 64
python eval_v7.py --backend planning_ops --count 64
```

Deterministic product-mode evaluation:

```bash
python eval_v7.py --config config/product_openweights.yaml --safe-runtime --deterministic
```

## Sampling / solving

Sample a generated task:

```bash
python sample_v7.py --checkpoint checkpoints/last.pt
```

Manual problem:

```bash
python sample_v7.py --checkpoint checkpoints/last.pt --domain linear_equation --problem "Solve: 2x + 3 = 11"
```

Non-math sample:

```bash
python sample_v7.py --backend string_ops --domain sort_words --problem "Sort words alphabetically: kiwi apple mango"
python sample_v7.py --backend code_ops --domain function_name --problem "Read the Python function and return the function name:\ndef helper(x):\n    return x + 1"
python sample_v7.py --backend planning_ops --domain project_plan --problem "Create a valid project plan.\nTasks:\n- design (duration=1, priority=3, deps=none)\n- build (duration=2, priority=4, deps=design)\n- test (duration=1, priority=2, deps=build)\nReturn the ordered task plan."
python sample_v7.py --backend swebench_ops --domain swebench_patch --problem "Patch the repository so the failing tests pass. Fix the arithmetic bug in app.py and verify with the test suite."
python sample_v7.py --backend gaia_ops --domain gaia_csv_reasoning --problem "Use the files in the workspace to answer this question: what is the total sales amount for the east region in sales.csv? Return only the number."
```

Local serving wrapper:

```bash
python serve_v7.py --backend math --domain arithmetic --problem "Compute: 12 + 30"
```

Or JSONL stdin/stdout mode:

```bash
python serve_v7.py --backend planning_ops --stdin-jsonl
```

## Plugin usage

Reference plugin:

```bash
python train_v7.py --checker-plugin plugins/example_checker_plugin_v7.py
python sample_v7.py --checker-plugin plugins/example_checker_plugin_v7.py
```

The plugin can register new tools with the registry.

## Search modes

The repo currently exposes:

- verifier-guided beam search
- MCTS-style planning search
- fallback repair actions when parsing fails
- structured proposal scoring over canonical `ACTION {...}` candidates

Runtime search parameters are loaded from `config/default.yaml` and then overridden by `config/search.yaml` when that file is present.

The search stack now also includes:

- semantic transposition pruning with bounded capacity
- value-aware scoring
- deterministic strict-decoder product mode
- explicit fallback-action scoring inside search
- deterministic fallback-chain mode for development and non-claim benchmark profiles
- runtime event logs for retrieval hits, schema failures, tool failures, search budget exhaustion, and verifier/value disagreement

For the benchmark configs, the headline public-claim baseline is now strict and unassisted:

- `benchmark.assistance_mode: unassisted`
- `benchmark.oracle_hints_enabled: false`
- `search.guided_fallback_rollout: false`
- `search.deterministic_fallback_chain: false`
- `search.enable_fallback_repairs: false`
- `memory.retrieval_mode: none`
- oracle-style metadata remains available for gold traces and optional analysis, but the default runtime path does not rely on it
- benchmark outputs now record whether guided rollout fired, whether fallback repairs fired, whether a deterministic fallback chain fired, and whether any oracle fields were touched
- the strict claim path is required to pass with `benchmark_integrity_passed=true`, `fallback_chain_used=false`, `fallback_repair_used=false`, and `guided_rollout_used=false`

Curriculum phases are backend-specific:

- `math` uses `config/curriculum.yaml`
- `string_ops` uses `config/string_ops_curriculum.yaml`
- `code_ops` uses `config/code_ops_curriculum.yaml`
- `planning_ops` uses `config/planning_ops_curriculum.yaml`
- `swebench_ops` uses `config/swebench_ops_curriculum.yaml`
- `gaia_ops` uses `config/gaia_ops_curriculum.yaml`

## Benchmarks

Run the fixed benchmark suites across all registered backends:

```bash
python benchmark_v7.py --backends all
```

Run the public-style smoke suites:

```bash
python benchmark_v7.py --suite public_smoke --config config/benchmarks/public_smoke.yaml --profile public_unassisted_strict --deterministic --safe-runtime
python benchmark_v7.py --suite swebench_verified_smoke --config config/benchmarks/public_smoke.yaml --deterministic --safe-runtime
python benchmark_v7.py --suite gaia_smoke --config config/benchmarks/public_smoke.yaml --deterministic --safe-runtime
python benchmark_v7.py --suite math_public_smoke --config config/benchmarks/public_smoke.yaml --deterministic --safe-runtime
```

Run the harder public-style medium suites:

```bash
python benchmark_v7.py --suite public_medium --config config/benchmarks/public_smoke.yaml --profile public_unassisted_strict --deterministic --safe-runtime
python benchmark_v7.py --suite swebench_verified_medium --config config/benchmarks/public_smoke.yaml --deterministic --safe-runtime
python benchmark_v7.py --suite gaia_medium --config config/benchmarks/public_smoke.yaml --deterministic --safe-runtime
```

List the named benchmark profiles and ablations:

```bash
python benchmark_v7.py --list-profiles
python benchmark_v7.py --list-ablations
```

Run a profile-driven ablation campaign with repeatable artifacts:

```bash
python benchmark_v7.py \
  --config config/benchmarks/public_smoke.yaml \
  --suite public_smoke \
  --profile smoke_tiny \
  --ablations baseline,no_retrieval \
  --repeat 2 \
  --campaign-name roadmap_phase_smoke \
  --deterministic \
  --safe-runtime
```

Important benchmark profiles now include:

- `public_claim_no_repairs`: the default local 1.5B public claim profile
- `public_claim_coder_local_1p5b`: the default local 1.5B coder claim profile for SWE-bench-style runs
- `public_unassisted_strict`: the legacy tiny strict profile kept for compatibility
- `public_search_assisted`: a development baseline with guided rollout enabled
- `smoke_tiny`: a fast regression profile
- `rtx4060_general_local`: RTX 4060 laptop tuned general local profile
- `rtx4060_coder_local`: RTX 4060 laptop tuned code-agent local profile

For official-style local claim runs on a machine with the downloaded 1.5B models:

```bash
python benchmark_v7.py --suite manifest:benchmarks/manifests/gaia_medium_official_style.json --deterministic --safe-runtime
python benchmark_v7.py --suite manifest:benchmarks/manifests/swebench_verified_medium_official_style.json --deterministic --safe-runtime
```

On this branch, those manifest commands auto-select the local 1.5B claim profile:

- GAIA / math style suites -> `public_claim_no_repairs`
- SWE-bench style suites -> `public_claim_coder_local_1p5b`

There are also focused runner entrypoints:

```bash
python benchmarks/swebench_runner.py --deterministic --safe-runtime
python benchmarks/gaia_runner.py --deterministic --safe-runtime
python benchmarks/math_runner.py --deterministic --safe-runtime
```

Benchmark JSON summaries are written into `results/`.
Campaign runs additionally write:

- per-run suite results
- per-run manifests
- config snapshots
- campaign summary JSON
- campaign markdown report
- append-only entries in `results/benchmark_ledger.jsonl`

The strict benchmark smoke campaign now has a pinned CI workflow in `.github/workflows/benchmark-strict.yml`.

## Memory modes

Memory currently includes:

- replay buffer for solved/failed examples
- hard-case store
- persistent lemma store
- tactic success statistics
- retrieval-mode selection via config (`hybrid`, `lexical`, `embedding`)
- benchmark-failure harvesting into replay and hard-case memory during training when the benchmark harvest configs are enabled

This is the first durable memory layer, not the final one.

## Limitations

This is **not** a formal theorem prover and not a Lean/Coq backend.

It still has important limits:

- search depth is shallow
- the prover is small
- proof states are explicit but still lightweight
- verifier supervision is synthetic
- theorem discovery is not the target yet
- decoding now uses a structured tokenizer plus scored canonical action candidates, but it is not yet a full parser-level constrained decoder
- MCTS is real now, but still lightweight and value-guided rather than AlphaZero-scale
- public benchmark adapters are real, but the fixture suites are still local smoke proxies rather than full official benchmark runs
- the strict claim profile is now autonomous and benchmark-audited, but the flagship public path is still limited by local proxy fixtures and small-model regime performance

## Best next upgrades

The strongest next steps after this repo are:

- richer proof-state transitions
- fully constrained typed argument decoding rather than partially structured action content
- stronger verifier ranking losses
- proper proof-state beam pruning
- SymPy-extended algebra/calculus checks
- a formal backend bridge later
- learned retrieval over lemma memory

## Practical note

This codebase is intended to be a **working V7 skeleton with real logic**. It is not the final mathematical sentinel, but it is the first repo in the line that genuinely reasons over mathematical state rather than only text, while also beginning to separate a reusable reasoning engine from the math-specific backend.

For the open-weight profile, see [config/product_openweights.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/product_openweights.yaml). That profile turns on deterministic safe runtime defaults and strict structured decoding.

For benchmark profile snapshots and the ablation matrix, see:

- [config/benchmarks/profiles.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/benchmarks/profiles.yaml)
- [config/benchmarks/ablation_matrix.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/benchmarks/ablation_matrix.yaml)
- [config/benchmarks/profile_public_unassisted_strict.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/benchmarks/profile_public_unassisted_strict.yaml)
- [config/benchmarks/profile_public_search_assisted.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/benchmarks/profile_public_search_assisted.yaml)
- [config/benchmarks/train_public_benchmark_harvest.yaml](/C:/Users/scsar/Desktop/math_sentinel_v7/config/benchmarks/train_public_benchmark_harvest.yaml)

## Quick Commands

Use these commands to get the repo running and to reproduce common workflows.

- Setup a Python virtualenv (Linux/macOS):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Setup a Python virtualenv (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

- Run a short training smoke test (CPU):

```bash
python train_v7.py --steps 20 --batch-size 4 --micro-batch-size 4 --eval-every 10 --save-every 10
```

- Resume training from a checkpoint:

```bash
python train_v7.py --resume checkpoints/last.pt --steps 34000 --eval-every 200 --save-every 100
```

- Run evaluation:

```bash
python eval_v7.py --checkpoint checkpoints/last.pt --count 64
```

- Sample or solve a single problem:

```bash
python sample_v7.py --checkpoint checkpoints/last.pt
python sample_v7.py --checkpoint checkpoints/last.pt --domain linear_equation --problem "Solve: 2x + 3 = 11"
```

- Run the unit tests (requires test deps):

```bash
python -m unittest discover -s tests -q
```

- Git: initialize, add remote, and push (if needed):

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-remote-url>
git push -u origin main
```

- Run on CUDA / GPU (example):

```bash
# ensure CUDA drivers + torch with CUDA are installed
python train_v7.py --steps 2000 --batch-size 16 --micro-batch-size 8 --compile
```
