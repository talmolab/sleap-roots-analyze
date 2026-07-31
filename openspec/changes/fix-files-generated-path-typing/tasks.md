# Tasks: fix-files-generated-path-typing

**Suggested commit grouping**: land the code fix (Task 2), the regression test (Task 3),
and the baseline sync (Task 4.2) in the **same** commit — never commit the annotation fix
without the baseline sync in the same commit. Per the `type-check` CI job's own comment,
`mypy-baseline filter` fails on a baseline entry that "no longer reproduces," so a
commit that fixes the annotations but leaves the stale `.mypy-baseline.txt` entries
in place is a genuinely CI-red intermediate state, exactly the pattern the
`fix-post-pca-trait-names-propagation` precedent (`openspec/changes/archive/...
/tasks.md`) warns against.

**mypy cache note**: `uv run mypy` reuses `.mypy_cache/` across runs; a warm cache from a
prior invocation can reproduce stale errors even after the source is fixed on disk. Clear
it (`rm -rf .mypy_cache`) before any verification run below whose result matters (Tasks 1,
3, 4).

## Task 1: Confirm baseline state before touching code
- [x] 1.1 `rm -rf .mypy_cache`, then run
      `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path /tmp/check-baseline.txt`
      and diff against the committed `.mypy-baseline.txt` (374 lines) to confirm it is
      still a no-op — i.e. the `reduce_trait_redundancy.py` entries are still live and
      this fix is still needed. Do not commit to the scratch path.
- [x] 1.2 Confirm the three baseline lines this fix will resolve are, in
      `src/sleap_roots_analyze/pipeline/steps/reduce_trait_redundancy.py`: a
      `var-annotated` error on the `execute()` local (~L83), an `arg-type` error at the
      `StepResult(...)` call in `execute()` (~L180), and a `return-value` error at
      `_cluster_experiment`'s own `return` statement (~L284) — all three trace to the
      same `List[str]` → `List[Path]` return-type fix in Task 2.2, not three independent
      bugs.

## Task 2: Fix the type annotations
- [x] 2.1 In `execute()` (~L83), annotate `files_generated: List[Path] = []`.
- [x] 2.2 In `_cluster_experiment()` (~L191), change the return-type annotation's third
      element from `List[str]` to `List[Path]`:
      `Tuple[pd.DataFrame, List[str], List[Path], int]`. Update the docstring `Returns:`
      line if it names the type.
- [x] 2.3 In `_cluster_experiment()` (~L205), annotate `files_generated: List[Path] = []`
      for consistency with the corrected return type — note this does not itself resolve
      a distinct baseline entry (mypy already infers the type here from the immediately
      following `.append(cluster_file)`), it's a clarity/consistency change only.
- [x] 2.4 `List` and `Path` are already imported in this file — no import changes needed.

## Task 3: Add regression coverage independent of mypy
- [x] 3.1 In `tests/test_trait_redundancy.py`, add
      `assert all(isinstance(f, Path) for f in result.files_generated)` to the existing
      clustering-path tests that already exercise `_cluster_experiment` end-to-end
      (`test_clustering_both_experiments`, `test_produces_cluster_membership_file`) —
      every current assertion on `files_generated` items goes through `str(f)`, which
      would pass identically for `str` or `Path`, so this closes a real type-level
      coverage gap.
- [x] 3.2 Run `uv run pytest tests/test_trait_redundancy.py -v` — confirm these
      assertions pass on current code (they should, since runtime values are already
      `Path`; this is coverage, not a red/green TDD cycle — the annotation fix changes no
      runtime behavior for this assertion to catch).

## Task 4: Verify the fix resolves the three baseline entries with no new errors
- [x] 4.1 `rm -rf .mypy_cache`, run
      `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt`
      — confirm 0 new errors (the three entries are now fixed code, not newly baselined
      debt).
- [x] 4.2 `rm -rf .mypy_cache`, run
      `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path .mypy-baseline.txt`
      and confirm the committed `.mypy-baseline.txt` shrinks by exactly 3 lines
      (374 → 371), with no unrelated lines reordered/changed.

## Task 5: Full regression check
- [x] 5.1 Run `uv run pytest tests/test_trait_redundancy.py -v` (full file, not just the
      two touched tests) — confirm no regressions.
- [x] 5.2 Run `uv run ruff check src/sleap_roots_analyze` and
      `uv run black --check src/sleap_roots_analyze tests` — confirm clean.

## Task 6: Spec, docs, changelog
- [x] 6.1 `cli-pipeline` spec delta (this change's `specs/cli-pipeline/spec.md`) — done
      as part of this proposal.
- [x] 6.2 Add a `### Fixed` entry to `docs/CHANGELOG.md` `[Unreleased]` noting the
      `List[Path]` annotation fix in `ReduceTraitRedundancyStep`, the new
      `isinstance(Path)` regression assertions, and the baseline shrink
      (374 → 371), referencing #161.
- [x] 6.3 Run
      `npx -y -p @fission-ai/openspec openspec validate fix-files-generated-path-typing --strict`
      and resolve any issues.

## Task 7: Issue bookkeeping
- [x] 7.1 In the PR description (not this proposal), note this closes the first checklist
      item of #161 ("Pay down concrete error classes already captured in the baseline")
      for `reduce_trait_redundancy.py` specifically, but leaves #161 open — both the
      matching `generate_static_figures.py` gap and the rest of #161's checklist are
      separate future PRs.
