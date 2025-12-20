## 1. Test First (TDD)

- [x] 1.1 Add test: config log file is created in run_dir, not output_dir
- [x] 1.2 Add test: CLI `--log-file` flag still works for explicit paths
- [x] 1.3 Add test: no stray log files in output_dir when using config logging
- [x] 1.4 Verify existing per-run logging tests still pass

## 2. Implementation

- [x] 2.1 Refactor CLI to defer config-based log file setup until after run_dir creation
- [x] 2.2 Pass config's `logging.log_file` to pipeline for file handler creation
- [x] 2.3 Update pipeline's `_setup_logger` to optionally use config's log filename
- [x] 2.4 Remove CLI's early log file creation for config-based logging

## 3. Verification

- [x] 3.1 Run all tests (`uv run pytest`)
- [x] 3.2 Run black and ruff linting
- [x] 3.3 Manual test: run root core pipeline, verify no stray log files
- [x] 3.4 Manual test: verify `--log-file` CLI flag still works