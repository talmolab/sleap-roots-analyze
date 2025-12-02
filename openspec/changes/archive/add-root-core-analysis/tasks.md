# Implementation Tasks

## 1. Test Fixtures and Infrastructure
- [x] 1.1 Add `create_test_root_core_data()` fixture to `tests/fixtures.py`
- [ ] 1.2 Copy sample root core CSV to `tests/data/root_counting_sample.csv` (simplified version)

## 2. Root Core Analysis Module
- [x] 2.1 Create `src/sleap_roots_analyze/root_core_analysis.py` module file
- [x] 2.2 Write tests for `create_sample_identifier()` in `tests/test_root_core_analysis.py`
- [x] 2.3 Implement `create_sample_identifier()` function
- [x] 2.4 Write tests for `validate_unique_identifiers()`
- [x] 2.5 Implement `validate_unique_identifiers()` function
- [x] 2.6 Write tests for `melt_depth_data()` with depth parsing
- [x] 2.7 Implement `melt_depth_data()` with regex depth extraction
- [x] 2.8 Write tests for `aggregate_by_replicate()`
- [x] 2.9 Implement `aggregate_by_replicate()` function
- [x] 2.10 Add module-level docstring with usage examples
- [x] 2.11 Export public functions in `__init__.py`

## 3. Depth Profile Visualization Module
- [x] 3.1 Create `src/sleap_roots_analyze/depth_profile_plots.py` module file
- [x] 3.2 Write tests for `plot_depth_profile_faceted()` in `tests/test_depth_profile_plots.py`
- [x] 3.3 Implement `plot_depth_profile_faceted()` function
- [x] 3.4 Write tests for `plot_depth_profile_replicates()`
- [x] 3.5 Implement `plot_depth_profile_replicates()` function
- [x] 3.6 Add module-level docstring with usage examples
- [x] 3.7 Export public functions in `__init__.py`

## 4. Data Utilities Extension
- [ ] 4.1 Write tests for `filter_rows_by_values()` in `tests/test_data_utils.py`
- [ ] 4.2 Implement `filter_rows_by_values()` in `data_utils.py`
- [ ] 4.3 Update `data_utils.py` docstring

## 5. Integration Testing
- [ ] 5.1 Create integration test for complete pipeline (raw data → plots)
- [ ] 5.2 Test with actual notebook data formats
- [ ] 5.3 Verify handling of NaN values
- [ ] 5.4 Test edge cases (single core, missing depths)

## 6. Quality Assurance
- [x] 6.1 Run `uv run pytest` - all tests pass (34/34 passing)
- [ ] 6.2 Run `uv run pytest --cov --cov-branch` - verify >95% coverage
- [x] 6.3 Run `uv run black src/sleap_roots_analyze tests` - code formatted
- [x] 6.4 Run `uv run ruff check src/sleap_roots_analyze tests` - no linting errors
- [x] 6.5 Verify all docstrings follow Google format

## 7. Documentation
- [ ] 7.1 Add root core analysis section to README.md
- [ ] 7.2 Include code examples in module docstrings
- [ ] 7.3 Document expected data format (column naming convention)

## 8. Final Validation
- [ ] 8.1 Run `openspec validate add-root-core-analysis --strict`
- [ ] 8.2 Verify all scenarios in spec are tested
- [ ] 8.3 Manual testing with notebook data files
