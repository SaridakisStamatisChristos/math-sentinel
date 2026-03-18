## Road To SOTA Progress

Branch: `sota-real-benchmark-agents`

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

### In Progress

- harder benchmark fixtures beyond smoke proxy suites
- larger-model benchmark profiles and ablations

### Remaining

- larger-model benchmark profiles
- public benchmark score collection and ablations
