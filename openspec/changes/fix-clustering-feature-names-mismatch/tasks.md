# Tasks: fix-clustering-feature-names-mismatch

**Suggested commit grouping** (per the `fix-pca-zero-variance-crash` precedent,
which landed as one test commit + one fix commit, not one commit per task
group): Tasks 1-4 → one `test(#183):` commit (red); Task 5 → one `fix(#183):`
commit (green, turns the previous commit's tests passing); Tasks 6-7 → one
`test(#183):` commit; Task 8 → folded into whichever commit is last, plus a
final `docs(#183):` commit for the checklist. Don't push mid-red-phase while
a PR is open — batch commits 1-3 into a single push so CI never sees a
failing intermediate state.

## Task 1: Fixtures
- [x] 1.1 Reuse `pca_constant_feature_data` (`tests/fixtures.py:1046`) for the clustering producer tests instead of hand-rolling a new one
- [x] 1.2 Add a new fixture mixing a constant numeric column and a non-numeric (string plant/genotype ID) column in the same frame — no existing fixture covers the `select_dtypes` half of the bug

## Task 2: Write failing tests — structural invariant (TDD Red Phase)
- [x] 2.1 Add a parametrized test across `perform_kmeans_clustering` / `perform_gmm_clustering` / `perform_hierarchical_clustering`, each with `standardize=True` and `standardize=False`, asserting `len(feature_names) == <array>.shape[1]` (`cluster_centers`, `means`, `data_processed` respectively) on `pca_constant_feature_data`
- [x] 2.2 Assert the dropped constant column's name is absent from `feature_names` and the surviving real trait names are present in original relative order
- [x] 2.3 Run tests, confirm they FAIL on current code (length mismatch)

## Task 3: Write failing tests — silent mislabeling regression (TDD Red Phase)
- [x] 3.1 For KMeans/GMM, build `dict(zip(result["feature_names"], result[<array_key>][k]))` for a known cluster/component and assert each entry matches a hand-computed expected value for that named trait. For hierarchical, build the same mapping per-sample against `data_processed[i]` (there is no per-cluster centroid array for hierarchical) — a pure count assertion would not reliably catch the mislabeling in any of the three
- [x] 3.2 Run tests, confirm they FAIL on current code (values map to wrong names)

## Task 4: Write failing tests — non-numeric columns and all-filtered error (TDD Red Phase)
- [x] 4.1 Add tests using the new mixed constant+non-numeric fixture (Task 1.2): for `standardize=False`, `feature_names` excludes the non-numeric and constant columns and the producer succeeds (does not raise) on the surviving numeric columns. Verified empirically today: `standardize=False` on this fixture currently raises `RuntimeError: ... could not convert string to float: 'G0'` for all three producers — confirm that changes to a successful result post-fix
- [x] 4.2 Add tests confirming a `RuntimeError` whose message contains "No numeric columns with non-zero variance found" still raises when every column is filtered out, for both `standardize` values, across all three producers — build the all-filtered-out `DataFrame` inline (matching the `test_standardize_empty_after_cleaning` precedent in `test_pca.py`), no new fixture needed. Verified empirically: `standardize=True` already raises exactly this today (via `standardize_data`'s `ValueError` wrapped by the producer's own `except Exception`); `standardize=False` currently raises `RuntimeError: ... could not convert string to float` instead — confirm 4.2 fails today for `standardize=False` only
- [x] 4.3 Run tests, confirm 4.1 fails on `standardize=False` (currently raises instead of succeeding) and 4.2 fails on `standardize=False` (currently raises the wrong message)

## Task 5: Implement the fix (TDD Green Phase)
- [x] 5.1 `perform_kmeans_clustering` (~L104-111): on `standardize=True`, capture `df_clean` from `standardize_data`'s return (not `_`) and re-derive `feature_names` from it; on `standardize=False`, apply the numeric-only + non-zero-variance filter before deriving `feature_names`/`X_processed`, raising `ValueError` if nothing survives
- [x] 5.2 Apply the same fix to `perform_gmm_clustering` (~L355-362)
- [x] 5.3 Apply the same fix to `perform_hierarchical_clustering` (~L572-579)
- [x] 5.4 Update the `feature_names` docstring line in all three producers (currently just `feature_names: Feature names (if DataFrame input)`) to note it reflects columns remaining after numeric/non-zero-variance filtering, not necessarily the caller's original columns
- [x] 5.5 Run all Task 2-4 tests, confirm PASS (green); confirm all existing `test_clustering.py` (55 tests) and `test_cluster_result.py` (16 tests) still pass unchanged (no behavior change for already-clean input)

## Task 6: Typed adapter and outlier-detection regression coverage
- [x] 6.1 Extend `tests/test_cluster_result.py`'s `TestClusterAdapters` so `KMeansResult`/`GMMResult` `feature_names` is checked against `cluster_centers.shape[1]` / `means.shape[1]` on the constant-feature fixture (post-fix, corrected values)
- [x] 6.2 `HierarchicalResult` does not exist yet on `main` (PR #182 for #179 is still open) — do NOT add a test for it here. Once #182 merges (whichever order the two PRs land in), add one follow-up test asserting `HierarchicalResult.feature_names` matches `perform_hierarchical_clustering(df)["data_processed"].shape[1]`; no code change should be needed since the adapter passes `feature_names` through
- [x] 6.3 Add tests in `tests/test_outlier_detection.py` for `detect_outliers_kmeans`/`detect_outliers_gmm`/`detect_outliers_hierarchical` on the constant-feature and mixed fixtures (both `standardize` values): assert `feature_names` in the returned dict excludes filtered columns and matches the corresponding array's column count — these three functions currently have zero `feature_names` coverage and dict-spread the producer output verbatim, so they'd otherwise silently carry the bug forward undetected

## Task 7: Pipeline-path guardrail (no regression)
- [x] 7.1 In `tests/test_step_detect_outliers.py`, assert that when `DetectOutliersStep` (`pipeline/steps/detect_outliers.py`, which calls `detect_outliers_kmeans`/`_gmm`/`_hierarchical` directly) runs downstream of the #177 cleanup step, its `feature_names` output matches the cleanup step's surviving trait list exactly — confirms the pipeline path stays protected before *and* after this fix (the cleanup step already excludes constants before this code runs, so this guards against a regression, not the bug itself)

## Task 8: Verify no regressions
- [x] 8.1 All existing `test_clustering.py`, `test_cluster_result.py`, and `test_outlier_detection.py` tests pass unchanged
- [x] 8.2 Full test suite passes (`uv run pytest`)
- [x] 8.3 Linting, formatting, and the frozen mypy baseline pass (`uv run ruff check`, `uv run black --check`, `uv run mypy src/sleap_roots_analyze | uv run mypy-baseline filter --baseline-path .mypy-baseline.txt`)
- [x] 8.4 Update `docs/CHANGELOG.md` `[Unreleased]` with a `### Fixed` entry that explicitly distinguishes this fix from the existing #177 `### Changed` entry — e.g. "This is separate from the #177 cleanup change above: #177 keeps constant traits from ever reaching these functions through the standard cleanup step; #183 fixes the functions' own label bookkeeping for any caller that doesn't go through it (direct callers, or `standardize=False`)."
