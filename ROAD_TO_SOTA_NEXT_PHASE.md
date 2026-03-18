## Next SOTA Phase

Branch: `sota-benchmark-ablation-phase`

Base branch: `sota-real-benchmark-agents`

### Goal

Turn the benchmark-agent branch into a reproducible measurement branch with:

- versioned benchmark model profiles
- versioned ablation matrices
- repeatable campaign runs
- result ledgers and markdown reports
- exact benchmark config snapshots that do not silently depend on unrelated runtime overlays

This phase is about making benchmark progress measurable and portable before the next capability leap.

### Why This Phase

The repo already has benchmark-shaped agents.

What it lacked was the machinery to answer questions like:

- which profile should we run for a given benchmark family?
- how do we compare baseline versus retrieval-disabled versus search-disabled behavior?
- where do benchmark manifests, config snapshots, and run summaries live?
- how do we scale from smoke profiles to 7B and 32B profiles without ad hoc edits?

This branch closes that gap.

### Deliverables

#### 1. Benchmark Profile Catalog

Add named, versioned benchmark profiles under `config/benchmarks/` for:

- `smoke_tiny`
- `hf_tiny_local`
- `qwen_general_dev_1p5b`
- `qwen_coder_dev_1p5b`
- `qwen_coder_benchmark_7b`
- `qwen_general_flagship_32b`
- `qwen_coder_flagship_32b`

Acceptance:

- profiles are discoverable from the CLI
- each profile is a real config snapshot, not only a README note

#### 2. Config Inheritance

Add `extends` support to runtime config loading so benchmark snapshots can stay concise while remaining exact.

Acceptance:

- benchmark configs load correctly with `search_config_path=""`
- inherited benchmark configs preserve deterministic safe-mode defaults

#### 3. Ablation Catalog

Add a versioned ablation matrix under `config/benchmarks/ablation_matrix.yaml` with at least:

- `baseline`
- `no_retrieval`
- `no_guided_rollout`
- `no_value`
- `no_transposition`
- `beam_only`

Acceptance:

- ablations are discoverable from the CLI
- ablations map to explicit config deltas

#### 4. Campaign Runner

Add benchmark campaign orchestration that can run:

- one or many profiles
- one or many ablations
- one or many repeats
- public and internal suite targets

Acceptance:

- per-run config snapshots are saved
- per-run manifests are saved
- per-run suite results are saved
- per-campaign summary and markdown report are saved
- a benchmark ledger appends one line per run

#### 5. Smoke Execution On This Branch

Run at least one real campaign on this branch to prove the new machinery end to end.

Acceptance:

- unit tests pass
- campaign execution passes
- deterministic repeats are stable

### Commands

List profiles:

```powershell
python benchmark_v7.py --list-profiles
```

List ablations:

```powershell
python benchmark_v7.py --list-ablations
```

Run a deterministic smoke ablation campaign:

```powershell
python benchmark_v7.py `
  --config config/benchmarks/public_smoke.yaml `
  --suite public_smoke `
  --profile smoke_tiny `
  --ablations baseline,no_retrieval `
  --repeat 2 `
  --campaign-name roadmap_phase_smoke `
  --deterministic `
  --safe-runtime
```

### Acceptance Snapshot

This branch is complete when:

- the catalog/config layer exists
- the campaign layer exists
- the reporting/ledger layer exists
- the docs describe the workflow
- the smoke campaign has been executed successfully

### What Comes After This Branch

After this phase, the next best branch should target real benchmark scale:

- harder local fixtures
- official benchmark corpora ingestion
- 7B and 32B score collection on target hardware
- benchmark-directed finetuning and larger-model ablations

### Motif & Structure

The previous branch taught the engine to take benchmark-shaped exams.

This branch teaches the project how to measure itself honestly while it does so.

### Human Meaning

Without this phase, every future SOTA claim would stay soft.

With this phase, the repo starts behaving like a benchmark program, not just a smart prototype.
