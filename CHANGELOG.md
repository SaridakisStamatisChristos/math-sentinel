# Changelog

This file records user-visible repository and runtime-interface changes. Git history remains the authoritative development record.

## Unreleased

### Changed

- Productized the active repository namespace as **Math Sentinel**.
- Renamed the active training, evaluation, and sampling entry points to `train.py`, `eval.py`, and `sample.py`.
- Renamed the example checker plugin to `plugins/example_checker_plugin.py`.
- Renamed the default training log to `logs/train.jsonl`.
- Reworked README and runbook documentation around stable capabilities rather than implementation-lineage language.
- Added CI covering the complete test suite plus training, evaluation, and solving smoke checks.

### Repository history

Earlier development used the **V7** label in filenames and documentation. That label is retained here only as historical lineage; it is no longer part of the active product namespace.
