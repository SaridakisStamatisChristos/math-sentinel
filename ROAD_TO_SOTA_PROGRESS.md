## Road To SOTA Progress

Branch: `sota-unassisted-benchmark-baselines`

### Completed

- benchmark framework added under `benchmarks/`
- public smoke benchmark suites added for:
  - `swebench_verified_smoke`
  - `gaia_smoke`
  - `math_public_smoke`
- execution branch created from the roadmap branch
- benchmark-shaped domain backends added:
  - `domains/swebench_ops/`
  - `domains/gaia_ops/`
- public benchmark runner integration landed in `benchmark_v7.py`
- focused benchmark entrypoints added:
  - `benchmarks/swebench_runner.py`
  - `benchmarks/gaia_runner.py`
  - `benchmarks/math_runner.py`
- deterministic result artifacts written into `results/`
- manual `sample_v7.py` flows now resolve benchmark fixtures for the new public-style backends
- `code_ops` now includes real repo-editing tasks and tools
- `swebench_ops` now drafts, applies, and verifies fixes without the gold-patch shortcut path
- `gaia_ops` now plans, gathers evidence, computes candidate answers, and answers from evidence
- guided fallback rollouts now let benchmark agents complete multi-step traces under tighter search budgets
- benchmark profile snapshots now live under `config/benchmarks/`
- benchmark config loading now supports `extends` inheritance
- benchmark ablation matrix now lives under `config/benchmarks/ablation_matrix.yaml`
- `benchmark_v7.py` now supports campaign runs, profile listing, ablation listing, repeat runs, manifests, and ledger output
- campaign artifacts now include config snapshots, per-run manifests, per-campaign JSON summaries, markdown reports, and `results/benchmark_ledger.jsonl`
- deterministic smoke ablation campaign executed successfully on this branch
- benchmark config now carries a first-class `benchmark.assistance_mode`
- the default public benchmark baseline is now `unassisted`
- `swebench_ops` now uses a test-driven unassisted repair loop: inspect tests, run tests, localize failure, synthesize a validated patch, apply, verify
- `gaia_ops` now uses generic question planning, file inspection, and answer synthesis without default oracle tool hints
- public smoke benchmark rerun passed end to end with the unassisted default baseline

### Remaining

- harder benchmark fixtures beyond smoke proxy suites
- official-scale benchmark score collection on 7B and 32B profiles
- benchmark-directed training and public ablation reports
