# Tasks: Fix Pipeline Runner Config Preservation

## 1. Fix Config File Bug

- [x] 1.1 Update `configs/active/cross_platform/cross_platform_rootcore_vs_cylinder.yaml` line 15 to use `07_data_outliers_removed.csv` instead of `10_final_data.csv`
- [x] 1.2 Update the comment on line 14 to say "QC'd but NOT heritability filtered" for consistency
- [x] 1.3 Fix missing "(QC'd)" suffix on exp2_name in same config

## 2. Refactor `_update_viz_config()` Method

- [x] 2.1 Add helper function `_extract_filename(path: str) -> str` to extract filename from path
- [x] 2.2 Add helper function `_update_yaml_path_preserving_structure(content: str, key: str, new_dir: Path, original_filename: str) -> str` for regex-based path substitution
- [x] 2.3 Refactor `_update_viz_config()` to:
  - Read original config as text
  - Extract filename from `data.csv_path` value
  - Use regex substitution to update only the directory portion
  - Write back preserving structure
- [x] 2.4 Verify updated config file exists with new directory but same filename

## 3. Refactor `_update_cross_platform_config()` Method

- [x] 3.1 Refactor `_update_cross_platform_config()` to:
  - Read original config as text
  - Extract filenames from `exp1_data_path` and `exp2_data_path`
  - Use regex substitution for each path
  - Write back preserving structure
- [x] 3.2 Verify both paths updated correctly with original filenames preserved

## 4. Add Tests

- [x] 4.1 Add test for `_extract_filename()` helper with various path formats (4 tests)
- [x] 4.2 Add test for `_update_yaml_path_preserving_structure()` preserving comments and formatting (4 tests)
- [x] 4.3 Add test verifying `07_data_outliers_removed.csv` is preserved when specified
- [x] 4.4 Add test verifying `10_final_data.csv` is preserved when specified
- [x] 4.5 Add test verifying YAML comments are preserved in updated config

## 5. Validation

- [x] 5.1 Run existing tests: `uv run pytest tests/test_pipeline_runner_summary.py -v` (29 passed)
- [x] 5.2 Run full test suite: `uv run pytest` (1391 passed)
- [ ] 5.3 Manual validation: Run `/run-pipelines --dry-run` and verify paths shown
- [ ] 5.4 Manual validation: Run full pipeline and verify `_updated_*.yaml` files preserve:
  - Original filename choice
  - All comments
  - Key ordering
  - String quoting style
