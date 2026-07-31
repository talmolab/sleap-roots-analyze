## Why

First ratchet step of #161 (CI: ratchet mypy strictness and pay the baseline down toward
zero). #161 assumed the ~19-22 `files_generated ... list[Path] vs list[str | Path] ...
invariant` entries frozen in `.mypy-baseline.txt` were "resolved by #157/#159" (which
centralized path serialization and typed `StepResult.files_generated` as `List[Path]`,
per the `Provenance Path Serialization Is Centralized` requirement in
`openspec/specs/cli-pipeline/spec.md`) and just need `mypy-baseline sync` to drop out.

Verifying against the current baseline (374 lines) shows that's only half true: running
`mypy-baseline sync` today is a no-op because **`ReduceTraitRedundancyStep`
(`reduce_trait_redundancy.py`) was left with the same annotation drift** #159's fix
addressed for other producers. #159 removed the runtime `str(cluster_file)` /
`str(dendrogram_file)` / `str(heatmap_file)` pre-stringification calls in this file (the
part `cli-pipeline`'s spec directly requires — storing real `Path` objects), but never
updated the surrounding *type annotations*, so three baseline entries remain for this one
file:
- `files_generated = []` in `execute()` (~L83) has no type annotation (`var-annotated`).
- `_cluster_experiment()`'s return signature declares its files-list element as
  `Tuple[pd.DataFrame, List[str], List[str], int]` — the third `List[str]` should be
  `List[Path]` (`_create_dendrogram`/`_create_cluster_heatmap` both return `Path`, and
  `run_dir / f"..."` is a `Path`, so the annotation has always disagreed with the runtime
  type). Passing that list into `StepResult(files_generated=...)` (typed `List[Path]`)
  produces an `arg-type` error at the call site (~L180) **and** a `return-value` error at
  `_cluster_experiment`'s own `return` statement (~L284), since the function's declared
  return type no longer matches what it actually returns once the third element is
  correctly understood as `List[Path]`. Both trace to the same one-line annotation fix.

(`_cluster_experiment()`'s own `files_generated = []` local, ~L205, does **not** produce a
separate baseline entry — mypy infers its type from the immediately-following
`.append(cluster_file)` call. Annotating it explicitly anyway is still worth doing, for
consistency with the fixed return type and the other local, not because mypy requires it.)

None of this is a runtime bug — every value stored has always been a real `Path` object,
so JSON serialization is unaffected — but the annotation drift is exactly the class of
"invariant not fully applied to every producer" the `cli-pipeline` spec's `Provenance Path
Serialization Is Centralized` requirement exists to prevent, and it's silently masked by
sitting in the frozen mypy baseline instead of failing CI.

**Known, deliberately out-of-scope sibling case:** `generate_static_figures.py` has the
same annotation-drift shape — two bare `files = []` locals (in
`_create_phenotype_variation_plots` and a second static-figure helper) inside functions
declared `-> list[Path]`, both still live in the baseline. Fixing `reduce_trait_redundancy.py`
does **not** bring `files_generated`/`List[Path]` mismatches in the codebase to zero; it
closes out this one producer. `generate_static_figures.py` is left for its own small
follow-up PR (same shape, same fix, but this proposal keeps a one-file diff per #161's
own "each its own small PR" preference rather than bundling two unrelated producers).

## What Changes

- `reduce_trait_redundancy.py`: annotate the `execute()` and `_cluster_experiment()`
  `files_generated` locals as `List[Path]`, and correct `_cluster_experiment`'s
  return-type annotation from `List[str]` to `List[Path]` for the files-list element.
- Add a runtime regression assertion to `tests/test_trait_redundancy.py` (e.g.
  `assert all(isinstance(f, Path) for f in result.files_generated)` in the existing
  clustering-path tests) so the `List[Path]` contract for this step is verifiable by
  `pytest` alone, not only by mypy/baseline bookkeeping — closing a gap where every
  existing assertion on `files_generated` items goes through `str(f)` and would pass
  identically whether the runtime values were `Path` or `str`.
- Regenerate `.mypy-baseline.txt` via `mypy-baseline sync` once the fix lands, shrinking it
  from 374 to 371 lines (the three now-resolved entries above) — the first concrete count
  in #161's tracked paydown.
- `cli-pipeline` spec: extend `Provenance Path Serialization Is Centralized` with a new
  scenario making explicit that the `List[Path]` contract applies to every intermediate
  local variable and return-type annotation a step uses to build `files_generated`, not
  only to the final `StepResult` field — and that the mypy baseline is the enforcement
  mechanism for it (the existing "producers do not pre-stringify" scenario only covers
  runtime `str(path)` misuse, not static annotation drift).

## Impact

- Affected specs: `cli-pipeline` (MODIFIED: `Provenance Path Serialization Is Centralized`
  — new scenario, no change to the six existing scenarios).
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/reduce_trait_redundancy.py` (type annotations
    only — `execute()` ~L83, `_cluster_experiment()` ~L191, ~L205)
  - `tests/test_trait_redundancy.py` — new `isinstance(Path)` assertions on existing
    clustering-path tests (see `tasks.md`)
  - `.mypy-baseline.txt` (374 → 371 lines)
  - `docs/CHANGELOG.md` `[Unreleased]` — a `### Fixed` entry
- No behavior change: annotations only, no pipeline logic touched (trait clustering,
  representative selection, and file naming are all untouched); every existing test in
  `tests/test_trait_redundancy.py` continues to exercise the same runtime values, plus one
  new type-level assertion per the point above.
- Not in scope: `generate_static_figures.py`'s matching annotation gap (see "Why" —
  explicit follow-up, not silently left unmentioned); remaining #161 checklist items, each
  its own future small PR: per-library `pandas-stubs`/`types-*` stub adoption;
  `disallow_incomplete_defs` / `check_untyped_defs` ratchet steps; `disallow_any_*`; the
  mypy crash-guard (tracked separately as companion issue #160).
