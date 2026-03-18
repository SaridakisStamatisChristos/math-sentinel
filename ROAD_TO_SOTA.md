## Road To SOTA

Branch: `road-to-sota-execution`

Base branch: `sota-openweights-product`

### Goal

Turn this repo from a validated multi-backend reasoning platform into a benchmarked contender on public agent and reasoning evaluations, while preserving the current typed-state architecture.

This roadmap is optimized for the highest leverage path to SOTA-like standing:

1. win on real public benchmarks
2. keep the engine identity intact
3. upgrade the flagship runtime to stronger open-weight models
4. make search, tools, and memory matter at benchmark time

### North Star

The repo should become:

- a serious open-weight agent runtime, not only a research scaffold
- competitive on public agent benchmarks
- reproducible under fixed seeds and published configs
- modular enough to support both code and general reasoning tasks

### What Counts As SOTA For This Repo

For this project, "SOTA" should mean all of the following:

- public benchmark adapters exist inside the repo
- official benchmark runs are reproducible from committed code
- the system is competitive with current public open-weight agent baselines
- results are strong enough that the repo is no longer judged only by architecture quality

### Public Benchmarks To Target First

Primary benchmarks:

- `SWE-bench Verified`
- `SWE-bench Lite`
- `GAIA`

Secondary benchmarks:

- `MATH-500`
- `GSM8K`
- internal regression suites already in this repo

Reason for this ordering:

- `SWE-bench` tests agentic code reasoning, patching, and tool use
- `GAIA` tests general multi-step assistant behavior
- `MATH-500` and `GSM8K` keep the original symbolic proving ground honest

### Public Target Bands

These are execution targets, not marketing claims.

`SWE-bench Verified`

- Gate 1: `>= 35%`
- Gate 2: `>= 50%`
- Contender band: `>= 65%`

`GAIA`

- Gate 1: `>= 40%`
- Gate 2: `>= 55%`
- Contender band: `>= 70%`

`MATH-500`

- Gate 1: clear improvement over current math backend baseline
- Gate 2: competitive open-weight score with the flagship reasoning model

These bands are chosen to align with current public benchmark reality at the time this roadmap was written:

- the official SWE-bench site reported in July 2025 that `mini-SWE-agent` reached `65%` on `SWE-bench Verified`
- the GAIA public results dataset includes public entries in the `0.70+` range and higher

### Flagship Model Strategy

Development models:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-Coder-1.5B-Instruct`
- `Qwen/Qwen2.5-Coder-7B-Instruct` when local iteration speed still matters

Flagship benchmark models:

- `Qwen/Qwen2.5-Coder-32B-Instruct` for code-agent benchmarks
- `Qwen/Qwen2.5-32B-Instruct` for GAIA-style general reasoning

Model policy:

- small models stay in the repo for smoke tests and CI
- benchmark claims must be made using the flagship benchmark models
- all benchmarked configs must be versioned and reproducible

### Execution Phases

#### Phase 0: Freeze Baseline

Deliverables:

- tag current baseline branch commit
- save benchmark-ready runtime configs
- create a benchmark results ledger in the repo

Code work:

- add `benchmarks/` package
- add `results/` folder with JSON summaries
- add versioned config snapshots under `config/benchmarks/`

Acceptance:

- exact baseline outputs can be rerun from a single command

#### Phase 1: Benchmark Harness First

Deliverables:

- official adapters for `SWE-bench Lite`, `SWE-bench Verified`, and `GAIA`
- repeatable benchmark commands
- score parsers and artifacts

Code work:

- add `benchmarks/swebench_runner.py`
- add `benchmarks/gaia_runner.py`
- add `benchmarks/math_runner.py`
- add `domains/swebench_ops/`
- add `domains/gaia_ops/`

Requirements:

- benchmark adapters must not be thin wrappers around ad hoc scripts
- each benchmark must map cleanly into the engine concepts:
  state, typed action, executor, verifier, search, memory

Acceptance:

- one command per benchmark
- deterministic reruns with fixed seed
- persisted result files and traces

#### Phase 2: Make `code_ops` A Real Code Agent

Current `code_ops` is useful but still toy-shaped.

Deliverables:

- repo/file-aware code state
- patch application actions
- unit-test execution actions
- lint/static-analysis actions
- failure triage actions

Code work:

- expand `domains/code_ops/backend.py`
- add repository workspace state objects
- add patch-hunk action schemas
- add tool adapters for:
  - `python -m pytest`
  - `python -m unittest`
  - `ruff` or equivalent if available
  - file read/write diff application

Required actions:

- `READ_FILE`
- `SEARCH_CODE`
- `RUN_TEST`
- `APPLY_PATCH`
- `VERIFY_PATCH`
- `ROLLBACK_PATCH`
- `ANSWER`

Acceptance:

- code backend can operate on real repository tasks
- the repo can run a SWE-bench-style patch loop inside its own engine

#### Phase 3: Add A Real GAIA-Oriented General Agent Backend

Deliverables:

- `gaia_ops` backend with planning-heavy tool use
- browser/file/python execution tool abstractions
- subgoal decomposition and evidence tracking

Code work:

- add `domains/gaia_ops/backend.py`
- add tools for:
  - controlled shell execution
  - file parsing
  - CSV/JSON/table inspection
  - optional web/document tools through clean interfaces
- expand structured state with evidence citations and unresolved questions

Required actions:

- `OBSERVE`
- `EXTRACT`
- `CALCULATE`
- `PLAN`
- `SUBGOAL`
- `RESOLVE_SUBGOAL`
- `ANSWER`

Acceptance:

- GAIA tasks can be represented as engine-native state/action traces
- final answers include evidence trail or provenance payloads

#### Phase 4: Upgrade Search From Good To Ruthless

Deliverables:

- hierarchical planning
- rollout caching
- learned search control
- branch kill criteria

Code work:

- deepen `search/beam.py`
- deepen `search/mcts.py`
- add rollout cache and node budget accounting
- add branch pruning by:
  - contradiction
  - repeated failure mode
  - no-progress windows
- add subgoal-level planner in `search/`

Required improvements:

- learned proposal priors
- stronger value guidance
- better semantic deduplication
- explicit partial-credit terminals

Acceptance:

- average cost per solved benchmark task drops
- benchmark success rises without brute-force budget inflation

#### Phase 5: Memory Must Change Decisions, Not Decorate Prompts

Deliverables:

- retrieval that affects next-step policy
- trajectory replay weighted by benchmark hardness
- tactic distillation into search priors

Code work:

- extend `memory/retrieval.py`
- add persisted solved trajectory store
- add benchmark-case embeddings
- add negative retrieval from failed episodes

Acceptance:

- ablation shows retrieval improves score, not just trace style

#### Phase 6: Training Must Become Benchmark-Directed

Deliverables:

- separate training modes for:
  - imitation from curated traces
  - verifier/value tuning
  - search-mined finetuning
  - benchmark-specific adaptation

Code work:

- split `train_v7.py` internals into trainer components
- add benchmark finetune configs
- add curriculum for real code-agent and GAIA-like traces
- add distillation path from stronger teacher traces when available

Acceptance:

- benchmark-targeted training can be run without rewriting the training script

#### Phase 7: Runtime Hardening For Serious Benchmarks

Deliverables:

- long-run resume reliability
- artifact capture
- benchmark-safe local serving
- large-model inference profile

Code work:

- improve checkpoint metadata and migration
- add benchmark trace archiving
- add large-model runtime configs:
  - vLLM profile
  - local Transformers profile
  - quantized profile

Acceptance:

- a failed overnight benchmark run can be resumed without manual surgery

#### Phase 8: Publishable Results Pass

Deliverables:

- frozen benchmark report
- ablation report
- reproducibility instructions
- branch PR ready for review

Must include:

- baseline vs upgraded scores
- search ablations
- retrieval ablations
- model-size comparisons
- cost or latency notes

Acceptance:

- the branch can support an honest public benchmark claim

### Concrete Repository Changes To Make In The Execution Branch

Add:

- `benchmarks/`
- `domains/swebench_ops/`
- `domains/gaia_ops/`
- `config/benchmarks/`
- `results/`
- `ROAD_TO_SOTA_PROGRESS.md`

Refactor:

- [train_v7.py](/C:/Users/scsar/Desktop/math_sentinel_v7/train_v7.py)
- [benchmark_v7.py](/C:/Users/scsar/Desktop/math_sentinel_v7/benchmark_v7.py)
- [serve_v7.py](/C:/Users/scsar/Desktop/math_sentinel_v7/serve_v7.py)
- [search/beam.py](/C:/Users/scsar/Desktop/math_sentinel_v7/search/beam.py)
- [search/mcts.py](/C:/Users/scsar/Desktop/math_sentinel_v7/search/mcts.py)
- [sentinel/model_backends.py](/C:/Users/scsar/Desktop/math_sentinel_v7/sentinel/model_backends.py)
- [memory/retrieval.py](/C:/Users/scsar/Desktop/math_sentinel_v7/memory/retrieval.py)

Preserve:

- current synthetic backends for regression
- current deterministic product mode
- small-model smoke path for CI

### Best Branch Strategy

Use a fresh execution branch after this planning branch:

- planning branch: `road-to-sota-execution`
- implementation branch: `sota-public-benchmarks`

Why:

- this branch becomes the design contract
- the next branch becomes the actual heavy implementation branch
- benchmark work can move fast without rewriting the roadmap

### Weekly Execution Order

Week 1:

- benchmark harness
- baseline captures
- results ledger

Week 2:

- real `code_ops` overhaul
- SWE-bench domain adapter

Week 3:

- GAIA domain adapter
- planning/evidence state extensions

Week 4:

- search upgrade pass
- retrieval/value ablations

Week 5:

- benchmark-driven training pass
- larger model profiles

Week 6:

- full benchmark runs
- ablations
- report and branch stabilization

### Non-Negotiables

- no SOTA claims before public benchmark numbers exist
- no architecture churn without benchmark impact measurement
- every new capability must land with a benchmark or ablation reason
- keep the engine abstraction intact; do not collapse into benchmark-specific spaghetti

### First Commands For The Future Execution Branch

1. create `sota-public-benchmarks` from this roadmap branch
2. add benchmark harness and result artifacts first
3. run baseline numbers before changing search or models
4. upgrade `code_ops` into a real repo-editing agent
5. add `gaia_ops`
6. only then start search and model scaling work

### External References

- SWE-bench official leaderboard: `https://www.swebench.com/index.html`
- GAIA public results dataset: `https://huggingface.co/datasets/gaia-benchmark/results_public`
- Qwen2.5-1.5B-Instruct model card: `https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct`
- Qwen2.5-32B-Instruct model card: `https://huggingface.co/Qwen/Qwen2.5-32B-Instruct`
- Qwen2.5-Coder-32B-Instruct model card: `https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct`

### Motif & Structure

The path to SOTA is no longer "more cleverness."

It is:

- benchmark contact
- stronger tools
- stronger models
- stronger search
- stronger memory
- ruthless measurement

### Human Meaning

This branch defines the difference between a beautiful engine and a competitive one.
