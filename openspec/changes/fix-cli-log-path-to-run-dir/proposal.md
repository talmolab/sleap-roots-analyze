## Why

The CLI's config-based log file (`logging.log_file` setting) is created in `output_dir/` (e.g., `qc_runs/`) instead of the run-specific directory (`qc_runs/pipeline_timestamp/`). This creates stray log files outside of run directories, breaking the principle that all run artifacts should be contained within the timestamped run folder.

## What Changes

- **MODIFIED**: Config-based log file resolution now targets `run_dir` instead of `output_dir`
- CLI defers log file creation until after the pipeline creates its run directory
- Config's `logging.log_file` setting writes to `{run_dir}/{log_file}` instead of `{output_dir}/{log_file}`
- Removes duplicate logging: CLI's config-based log and pipeline's per-run log are now unified

## Impact

- Affected specs: `cli-pipeline` (Config-Based Log File Resolution requirement)
- Affected code: `src/sleap_roots_analyze/cli.py` (qc, viz, cross-platform commands)
- **No breaking changes**: The `--log-file` CLI flag still works for explicit paths
- Existing behavior preserved: Pipeline's per-run `pipeline.log` continues to work unchanged
