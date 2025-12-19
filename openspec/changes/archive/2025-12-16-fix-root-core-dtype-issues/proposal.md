# Proposal: Fix Root Core Data Type Issues

## Why

When loading root core data from CSV files, numeric columns (`Plot`, `Rep`, `Core_Replicate`, `core_n`) are read as float64 (e.g., `1.0` instead of `1`). This causes several issues:
1. Sample identifiers contain decimals: `plot1.0_rep1.0_Control_core1.0` instead of `plot1_rep1_Control_core1`
2. Barcode column shows decimals: `1.0-1.0` instead of `1-1`
3. Rep column remains as float throughout the pipeline, making it harder to use as an index
4. Redundant code: LoadRootCoreDataStep duplicates sample_id creation logic from root_core_analysis.py

These are data quality issues that make the output harder to work with and indicate poor data hygiene.

## What Changes

- **Fix 1**: Refactor LoadRootCoreDataStep to use shared `create_sample_identifier()` function from root_core_analysis.py
- **Fix 2**: Convert numeric metadata columns (`Plot`, `Rep`, `Core_Replicate`, `core_n`, `Ent`, `Sub`) to int immediately after loading CSV
- **Fix 3**: Update ReshapeForTraitQCStep to convert Rep/Plot to int before creating Barcode column
- **Fix 4**: Add tests to catch float dtype issues in sample_id and Barcode generation
- **Fix 5**: Document outlier_flag column behavior in pipeline documentation

## Impact

- Affected specs: `root-core-qc`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/load_root_core_data.py` - Remove duplicate method, use shared function
  - `src/sleap_roots_analyze/pipeline/steps/reshape_for_trait_qc.py` - Fix Barcode creation
  - `src/sleap_roots_analyze/root_core_analysis.py` - Already fixed
  - `tests/test_step_load_root_core_data.py` - Add regression test (already added)
  - `tests/test_step_reshape_for_trait_qc.py` - Add Barcode dtype test
- Breaking changes: None (only affects output formatting, not behavior)
- Data quality: Improved - cleaner identifiers without decimal points
