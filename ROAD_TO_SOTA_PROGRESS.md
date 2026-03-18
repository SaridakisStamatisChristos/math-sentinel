## Road To SOTA Progress

Branch: `sota-public-benchmarks`

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

### In Progress

- richer repository-editing behavior beyond smoke fixture patching
- deeper GAIA-style evidence tracking and planning state

### Remaining

- real repository-oriented code-agent behavior
- GAIA-style richer evidence and planning tools
- larger-model benchmark profiles
- public benchmark score collection and ablations
