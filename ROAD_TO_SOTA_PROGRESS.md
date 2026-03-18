## Road To SOTA Progress

Branch: `sota-strict-benchmark-hardening`

### Completed

- strict public-claim benchmark profile added:
  - `config/benchmarks/profile_public_unassisted_strict.yaml`
- separate search-assisted development benchmark profile added:
  - `config/benchmarks/profile_public_search_assisted.yaml`
- the strict public benchmark path now disables guided fallback rollout by default
- benchmark integrity defaults now include `benchmark.fail_on_integrity_violation: true`
- unassisted benchmark runtime state now strips oracle fixture metadata before search
- benchmark runners now emit per-case audit fields for:
  - `guided_rollout_used`
  - `fallback_repair_used`
  - `fallback_chain_used`
  - `oracle_fields_touched`
  - `benchmark_integrity_passed`
- benchmark runs now fail loudly on integrity violations
- public benchmark suite catalog now includes harder medium suites:
  - `swebench_verified_medium`
  - `gaia_medium`
- harder local benchmark fixtures landed under:
  - `benchmarks/fixtures/swebench_medium_smoke/`
  - `benchmarks/fixtures/gaia_medium_smoke/`
- repo patch drafting now uses ranked candidate search with:
  - multiple patch candidates
  - extracted-test validation
  - provenance
  - failed-candidate retention
- failed tool payloads are now preserved in executor state for later analysis
- benchmark-directed failure harvesting now feeds replay and hard-case memory during training
- benchmark harvest configs landed under:
  - `config/benchmarks/train_public_benchmark_harvest.yaml`
  - `config/benchmarks/train_swebench_benchmark_harvest.yaml`
  - `config/benchmarks/train_gaia_benchmark_harvest.yaml`
- pinned environment lockfile added:
  - `requirements-lock.txt`
- strict benchmark CI workflow added:
  - `.github/workflows/benchmark-strict.yml`
- deterministic benchmark workspaces are now stable across fixed-seed repeat runs
- strict public smoke campaign passed at:
  - `solved_rate=1.0`
  - `equivalence_rate=1.0`
- strict public medium campaign passed at:
  - `solved_rate=1.0`
  - `equivalence_rate=1.0`
- deterministic repeat campaign for the strict public smoke profile is now stable across two repeats
- 1-step benchmark-harvest training smokes passed for:
  - `swebench_ops`
  - `gaia_ops`

### Remaining

- official-scale benchmark corpus ingestion beyond local proxy fixtures
- 7B and 32B public benchmark score collection on target hardware
- public ablation reports on official-scale runs
- stronger learned policies so the strict claim path relies less on deterministic fallback chains in the tiny-model regime

### Motif & Structure

The benchmark stack is now split into a clean honesty boundary:

- strict public-claim path
- search-assisted development path
- oracle metadata reserved for gold traces and analysis

That means benchmark success is now attached to an explicit contract, not just to intent.

### Human Meaning

This is the phase where the repo stopped merely saying “trust us, it’s unassisted” and started producing auditable evidence for that claim.
