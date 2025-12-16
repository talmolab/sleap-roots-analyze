## 1. TDD Tests

- [x] 1.1 Add test: `test_qc_uses_config_log_file_when_cli_not_specified` - verifies log file created at `output_dir/config.log_file`
- [x] 1.2 Add test: `test_qc_cli_log_file_overrides_config` - verifies `--log-file` takes precedence
- [x] 1.3 Add test: `test_qc_no_log_file_when_log_to_file_false` - verifies `log_to_file: false` prevents file logging
- [x] 1.4 Add test: `test_qc_config_log_file_with_subdirectory` - verifies nested paths work correctly
- [ ] 1.5 Add test: `test_viz_uses_config_log_file` - same behavior for viz command (deferred - viz uses same pattern)

## 2. Implementation

- [x] 2.1 Modify `qc` command to resolve log file from config when `--log-file` not provided
- [x] 2.2 Modify `viz` command with same logic
- [ ] 2.3 Modify `cross-platform` command with same logic (N/A - CrossPlatformConfig doesn't have logging field)
- [x] 2.4 Ensure log file path is resolved relative to output directory

## 3. Verification

- [x] 3.1 Run all tests: `uv run pytest tests/test_cli.py -v` (34 passed)
- [x] 3.2 Manual test: run pipeline with config log file, verify file created in output dir
- [x] 3.3 Manual test: run with `--log-file`, verify it overrides config