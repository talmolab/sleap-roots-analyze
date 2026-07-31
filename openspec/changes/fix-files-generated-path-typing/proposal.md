## Why

First ratchet step of #161 (CI: ratchet mypy strictness and pay the baseline down toward
zero). #161 assumed the ~19-22 `files_generated ... list[Path] vs list[str | Path] ...
invariant` entries frozen in `.mypy-baseline.txt` were "resolved by #157/#159" (which
centralized path serialization and typed `StepResult.files_generated` as `List[Path]`,
per the `Provenance Path Serialization Is Centralized` requirement in
`openspec/specs/cli-pipeline/spec.md`) and just need `mypy-baseline sync` to drop out.

Verifying against the current baseline shows that's only half true: running
`mypy-baseline sync` today is a no-op (0 lines change) because **one producer was missed**.
`ReduceTraitRedundancyStep` (`reduce_trait_redundancy.py`) still has two live baseline
entries:
- `files_generated = []` (two call sites, `execute()` and `_cluster_experiment()`) has no
  type annotation (`var-annotated`).
- `_cluster_experiment()`'s return signature declares its files-list element as
  `Tuple[pd.DataFrame, List[str], List[str], int]` — the third `List[str]` should be
  `List[Path]` (`_create_dendrogram`/`_create_cluster_heatmap` both return `Path`, and
  `run_dir / f"..."` is a `Path`, so the annotation has always disagreed with the runtime
  type). Passing that list into `StepResult(files_generated=...)` (typed `List[Path]`)
  produces the `arg-type` error.

Neither is a runtime bug — every value stored has always been a real `Path` object, so
JSON serialization is unaffected — but the annotation drift is exactly the class of
"invariant not fully applied to every producer" the `cli-pipeline` spec's `Provenance Path
Serialization Is Centralized` requirement exists to prevent, and it's silently masked by
sitting in the frozen mypy baseline instead of failing CI.

## What Changes

- `reduce_trait_redundancy.py`: annotate both `files_generated` locals as `List[Path]`,
  and correct `_cluster_experiment`'s return-type annotation from `List[str]` to
  `List[Path]` for the files-list element.
- Regenerate `.mypy-baseline.txt` via `mypy-baseline sync` once the fix lands, shrinking it
  from 375 to 373 lines (the two now-resolved entries) — the first concrete count in
  #161's tracked paydown.
- `cli-pipeline` spec: extend `Provenance Path Serialization Is Centralized` with a new
  scenario making explicit that the `List[Path]` contract applies to every intermediate
  local variable and return-type annotation a step uses to build `files_generated`, not
  only to the final `StepResult` field — and that the mypy baseline is the enforcement
  mechanism for it (the existing "producers do not pre-stringify" scenario only covers
  runtime `str(path)` misuse, not static annotation drift).

## Impact

- Affected specs: `cli-pipeline` (MODIFIED: `Provenance Path Serialization Is Centralized`
  — new scenario, no change to the four existing scenarios).
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/reduce_trait_redundancy.py` (type annotations
    only — `execute()` ~L83, `_cluster_experiment()` ~L191, ~L205)
  - `.mypy-baseline.txt` (2 fewer lines)
  - `docs/CHANGELOG.md` `[Unreleased]` — a `### Fixed` entry
- No behavior change: annotations only, no logic touched; every existing test in
  `tests/test_trait_redundancy.py` continues to exercise the same runtime values.
- Not in scope (remaining #161 checklist items, each its own future small PR): per-library
  `pandas-stubs`/`types-*` stub adoption; `disallow_incomplete_defs` /
  `check_untyped_defs` ratchet steps; `disallow_any_*`; the mypy crash-guard (tracked
  separately as companion issue #160).
