# Tasks: fix-pca-zero-variance-crash

## Task 1: Write failing tests — shape mismatch crash (TDD Red Phase)
- [x] 1.1 Add `test_pca_with_zero_variance_traits` — data with some constant traits; verify step completes without error and loadings shape matches actual features used
- [x] 1.2 Add `test_pca_with_all_zero_variance_traits` — all traits constant; verify step raises `ValueError` with clear message
- [x] 1.3 Run tests, confirm both FAIL (red) on current code — confirmed 6 FAILED (shape mismatch ValueError), 1 PASSED (all-zero-variance already errors correctly)

## Task 2: Write failing tests — metadata and logging (TDD Red Phase)
- [x] 2.1 Add `test_pca_zero_variance_metadata_tracking` — verify `excluded_zero_variance_traits` and `n_traits_after_filtering` present in metadata when zero-variance traits exist
- [x] 2.2 Add `test_pca_zero_variance_warning_threshold` — verify `UserWarning` emitted when >50% of traits are zero-variance
- [x] 2.3 Add `test_pca_no_warning_below_threshold` — verify no warning when <=50% of traits are zero-variance
- [x] 2.4 Run tests, confirm all 3 FAIL (red) on current code — confirmed all crash with shape mismatch before reaching metadata/warning assertions

## Task 3: Write failing tests — feature selection correctness (TDD Red Phase)
- [x] 3.1 Add `test_pca_top_features_use_filtered_names` — verify `top_features` metadata contains actual feature names (not indices into original trait list)
- [x] 3.2 Add `test_pca_loadings_csv_index_matches_features` — verify saved loadings.csv has correct feature names as index
- [x] 3.3 Run tests, confirm both FAIL (red) on current code — confirmed both crash with shape mismatch

## Task 4: Implement the fix (TDD Green Phase)
- [x] 4.1 After PCA, compute `feature_names = pca_results["feature_names"]` and derive `excluded_traits = set(trait_cols) - set(feature_names)`
- [x] 4.2 Log excluded traits: count and names
- [x] 4.3 Emit `warnings.warn()` if `len(excluded_traits) / len(trait_cols) > 0.5`
- [x] 4.4 Replace `trait_cols` with `feature_names` in loadings DataFrame index (line 117)
- [x] 4.5 Replace `len(trait_cols)` with `len(feature_names)` in `select_top_features_from_pca()` call (line 91)
- [x] 4.6 Replace `trait_cols[i]` with `feature_names[i]` in top features lookup (line 96)
- [x] 4.7 Add `excluded_zero_variance_traits` and `n_traits_after_filtering` to output metadata
- [x] 4.8 Run all new tests, confirm all PASS (green) — 7 passed, 0 failed

## Task 5: Write integration tests — full viz pipeline with zero-variance traits (TDD Red→Green)
- [x] 5.1 Add `test_viz_pipeline_completes_with_zero_variance_traits` — full 12-step VizPipeline with PCA + static figures, UMAP/clustering disabled; asserts `summary.status == "success"`
- [x] 5.2 Add `test_viz_pipeline_pca_metadata_propagates_to_figures` — verifies `excluded_zero_variance_traits` and `n_traits_after_filtering` in PCA metadata, and `GenerateStaticFiguresStep` completes without errors
- [x] 5.3 Add `test_viz_pipeline_loadings_csv_dimensions_match` — verifies loadings.csv has 4 rows (variable traits) with correct index names
- [x] 5.4 Run integration tests — 3 passed, 0 failed (26s)

## Task 6: Verify no regressions
- [x] 6.1 All existing `TestPCAAnalysisStep` tests pass unchanged — 13 existing + 7 new = 20 passed
- [x] 6.2 All existing `TestPCADataOrganization` tests pass unchanged — 2 passed
- [x] 6.3 Full test suite passes (`uv run pytest`) — 1893 passed, 6 flaky (pre-existing, pass in isolation)
- [x] 6.4 Linting and formatting pass (`uv run ruff check`, `uv run black --check`) — all clean
