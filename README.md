# Math Sentinel V7

Math Sentinel V7 is a **stateful symbolic proof-agent scaffold** for mathematics. It is designed as the first version in the series where the authoritative object is a **proof state**, not just raw generated text.

## What V7 is

V7 combines:

- a small causal **prover** transformer
- a small recurrent **verifier**
- an explicit `ProofState`
- a typed action system
- deterministic math tools
- verifier-guided beam search
- replay, hard-case tracking, and lemma memory
- procedural math curriculum generation

This repo is meant to be a real runnable starting point, not a stub architecture dump.

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

1. A curriculum generator emits an infinite synthetic math task.
2. The task is wrapped into a structured `ProofState`.
3. The prover is conditioned on the serialized proof state and proposes action traces.
4. The parser extracts typed actions.
5. The executor applies actions to child proof states using exact tools.
6. The verifier scores those child states.
7. Beam search keeps the strongest branches.
8. Replay, hard-case tracking, and lemma memory retain what mattered.

## Repo layout

```text
math_sentinel_v7/
├── README.md
├── requirements.txt
├── train_v7.py
├── eval_v7.py
├── sample_v7.py
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

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training quickstart

Light CPU/GPU smoke run:

```bash
python train_v7.py --steps 20 --batch-size 4 --micro-batch-size 4 --eval-every 10 --save-every 10
```

More serious CUDA run:

```bash
python train_v7.py --steps 2000 --batch-size 16 --micro-batch-size 8 --compile

You can also scale the model in `config/default.yaml` once CUDA is confirmed working.
```

Resume:

```bash
python train_v7.py --resume checkpoints/last.pt
```

## Evaluation

```bash
python eval_v7.py --checkpoint checkpoints/last.pt --count 64
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
- fallback repair actions when parsing fails
- an experimental `mcts.py` placeholder that currently reuses beam search

## Memory modes

Memory currently includes:

- replay buffer for solved/failed examples
- hard-case store
- persistent lemma store
- tactic success statistics

This is the first durable memory layer, not the final one.

## Limitations

This is **not** a formal theorem prover and not a Lean/Coq backend.

It still has important limits:

- search depth is shallow
- the prover is small
- proof states are explicit but still lightweight
- verifier supervision is synthetic
- theorem discovery is not the target yet
- grammar-aware decoding is not implemented yet
- MCTS is only scaffolded

## Best next upgrades

The strongest next steps after this repo are:

- richer proof-state transitions
- typed argument decoding rather than free-form action content
- stronger verifier ranking losses
- proper proof-state beam pruning
- SymPy-extended algebra/calculus checks
- a formal backend bridge later
- learned retrieval over lemma memory

## Practical note

This codebase is intended to be a **working V7 skeleton with real logic**. It is not the final mathematical sentinel, but it is the first repo in the line that genuinely reasons over mathematical state rather than only text.
