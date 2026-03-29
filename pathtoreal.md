# Path To Real

This file records the reference constraints for the strict blind generalized GAIA lane.

## Core Principle

The next gain is unlikely to come from making the 1.5B model "smarter".
It is more likely to come from restoring generalized source-family routing under strict-blind rules,
then strengthening the generic retrieval and extraction lanes those routes feed into.

## What Strict Blind Should Mean

Strict blind should not disable entire public families.

Strict blind should allow family-level routing for:

- `public_reference`
- `public_record`
- `github_public_artifact`
- `scholarly_reference`
- `video_transcript`
- `history_archive`
- `public_dataset`
- `map_or_location_public_geography`
- `generic_calculation`
- `unit_normalization`
- `date_reasoning`

Strict blind should disable:

- benchmark-known regexes
- exact GAIA question templates
- hand-coded phrases tied to known cases
- narrow one-off extractors that only work because the benchmark contained that case
- pipelines keyed to one known dataset, site, or entity with no broader utility

## Routing Philosophy

Route using coarse source cues, not benchmark-shaped strings.

Valid generalized cues include:

- asks for a date tied to a public event -> `public_record` or `public_reference`
- asks about code, repo, commit, release, issue, contributor -> `github_public_artifact`
- asks about a paper, citation, DOI, abstract, arXiv, author chronology -> `scholarly_reference`
- asks about a video, speech, transcript, timestamp, quote in a clip -> `video_transcript`
- asks about a historical snapshot, archived webpage, revision state, "as of" version -> `history_archive`
- asks about a schedule, standings, public timetable, public agency table -> `public_record`

A routing feature is acceptable only if it would still make sense in a general-purpose public QA agent outside GAIA.

## Desired Architecture

1. Family-level routing stays on.
2. Case-level specializations stay off.
3. No-file prompts should receive:
   - top-1 family
   - top-2 backup families
   - routing confidence
   - abstention when confidence is too low
4. Solvers should be split conceptually into:
   - `generic_*`
   - `case_specific_*`
5. Strict blind should use only the generic subset.

## Concrete Patch Order

1. Build a pure source-family classifier for no-file prompts.
2. Audit every family for generic vs case-shaped behavior.
3. Add route-only evaluation:
   - no-file family coverage
   - blank-plan rate
   - correct-family rate
   - top-2-family recall
4. Separate infrastructure failures from reasoning failures:
   - planner miss
   - retrieval unavailable
   - retrieval empty
   - extraction failed
   - reasoning failed
   - formatting failed

## Diagnostic Summary

The current no-file weakness is mostly:

- the system often cannot decide where to look
- the few places it does look are fragile

The strong file-backed performance suggests the evidence-grounded solver is real.
The weak no-file performance suggests the underdeveloped source-family router and generic retrieval lanes are now the bottleneck.
