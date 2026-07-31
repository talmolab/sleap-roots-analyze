# Tasks: fix-files-generated-path-typing

## Task 1: Confirm baseline state before touching code
- [ ] 1.1 Run `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path /tmp/check-baseline.txt` and diff against the committed `.mypy-baseline.txt` to confirm it is still a no-op (0 lines changed) — i.e. the two `reduce_trait_redundancy.py` `files_generated` entries are still live and this fix is still needed. Do not commit to the scratch path.
- [ ] 1.2 Confirm the two baseline lines are exactly `files_generated ... var-annotated` and `files_generated ... arg-type` in `src/sleap_roots_analyze/pipeline/steps/reduce_trait_redundancy.py`.

## Task 2: Fix the type annotations
- [ ] 2.1 In `execute()` (~L83), annotate `files_generated: List[Path] = []`.
- [ ] 2.2 In `_cluster_experiment()` (~L191), change the return-type annotation's third element from `List[str]` to `List[Path]`: `Tuple[pd.DataFrame, List[str], List[Path], int]`. Update the docstring `Returns:` line if it names the type.
- [ ] 2.3 In `_cluster_experiment()` (~L205), annotate `files_generated: List[Path] = []`.
- [ ] 2.4 `List` and `Path` are already imported in this file — no import changes needed.

## Task 3: Verify the fix resolves the two baseline entries with no new errors
- [ ] 3.1 Run `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt` — confirm 0 new errors (the two entries are now fixed code, not newly baselined debt).
- [ ] 3.2 Run `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline sync --baseline-path .mypy-baseline.txt` and confirm the committed `.mypy-baseline.txt` shrinks by exactly 2 lines (375 → 373), with no unrelated lines reordered/changed.

## Task 4: Regression check — no behavior change
- [ ] 4.1 Run `uv run pytest tests/test_trait_redundancy.py -v` — confirm all existing tests still pass unchanged (annotations only, no logic touched).
- [ ] 4.2 Run `uv run ruff check src/sleap_roots_analyze` and `uv run black --check src/sleap_roots_analyze tests` — confirm clean.

## Task 5: Spec, docs, changelog
- [ ] 5.1 Add the `cli-pipeline` spec delta (this change's `specs/cli-pipeline/spec.md`) — done as part of this proposal.
- [ ] 5.2 Add a `### Fixed` entry to `docs/CHANGELOG.md` `[Unreleased]` noting the `List[Path]` annotation fix in `ReduceTraitRedundancyStep` and the baseline shrink (375 → 373), referencing #161.
- [ ] 5.3 Run `npx -y -p @fission-ai/openspec openspec validate fix-files-generated-path-typing --strict` and resolve any issues.

## Task 6: Issue bookkeeping
- [ ] 6.1 In a PR description (not this proposal), note this closes the first checklist item of #161 ("Pay down concrete error classes already captured in the baseline") but leaves #161 open — the remaining ratchet steps are separate future PRs.
