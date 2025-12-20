# Tasks

## Implementation

- [x] Add test for `_flatten_config_dict()` helper function
- [x] Implement `_flatten_config_dict()` to convert nested dict to dot-notation keys
- [x] Add test for `_extract_all_config_params()` extracting ALL parameters from QC config
- [x] Implement `_extract_all_config_params()` to load and flatten config YAML
- [x] Add test for `_format_comparison_table()` with multiple configs
- [x] Implement `_format_comparison_table()` to generate markdown table with datasets as columns
- [x] Add test for `_format_comparison_table()` with single config
- [x] Add test for `_format_comparison_table()` with missing parameters (N/A handling)
- [x] Add test for `_format_comparison_table()` with list values
- [x] Add test for `_format_config_comparison()` generating QC section
- [x] Implement `_format_config_comparison()` to generate full config comparison section
- [x] Add test for `_format_config_comparison()` generating Viz section
- [x] Add test for `_format_config_comparison()` generating Cross-Platform section
- [x] Add test for `_generate_summary()` including config comparison section
- [x] Modify `_generate_summary()` to call `_format_config_comparison()` and include in output

## Validation

- [x] Run `openspec validate add-config-comparison-to-summary --strict`
- [x] Run full test suite with `uv run pytest tests/test_pipeline_runner_summary.py -v`
- [x] Run `sleap-roots-analyze run-all` and verify SUMMARY.md contains config comparison
- [x] Verify all QC parameters appear in comparison table
- [x] Verify all Viz parameters appear in comparison table
- [x] Verify all Cross-Platform parameters appear in comparison table
