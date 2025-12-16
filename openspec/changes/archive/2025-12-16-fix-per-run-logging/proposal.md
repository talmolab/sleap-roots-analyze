## Why

Pipeline logs are not being saved to individual run directories. Currently:

1. **Logs saved at parent directory level**: Log files appear at `qc_runs/pipeline.log` or `pipeline_runs/2025-12-15_222551/qc/pipeline.log` instead of inside individual run directories like `qc_runs/EDPIE Root Core Full QC_20251215_213809/pipeline.log`.

2. **Log files overwritten**: When multiple pipelines run in parallel, they all write to the same `pipeline.log` file at the parent level, causing logs to be interleaved or overwritten.

3. **Empty log files**: Some parent-level log files are 0 bytes, indicating logging conflicts or race conditions.

4. **No per-run traceability**: Users cannot review the log for a specific run because logs aren't co-located with the run's output files.

### Root Cause

The `BasePipeline._setup_logger()` method (base_pipeline.py:95-113) only creates a `StreamHandler` for console output. It never creates a `FileHandler` to write logs to the run directory. The CLI's `setup_logging()` function creates a file handler, but it writes to the parent output directory, not the per-run directory.

### Expected Behavior

Each pipeline run should have its own `pipeline.log` file inside its timestamped run directory (e.g., `qc_runs/EDPIE Root Core Full QC_20251215_213809/pipeline.log`).

## What Changes

- **FIX** `BasePipeline._setup_logger()` to add a `FileHandler` that writes to `{run_dir}/pipeline.log`
- **FIX** Logger setup to occur after `run_dir` is created (reorder initialization)
- **ADD** Config option `logging.run_log_file` for customizing the per-run log filename (default: `pipeline.log`)
- **MODIFY** Existing CLI `setup_logging()` to work alongside per-run logging (CLI log captures all runs, per-run log captures individual run)

### Design Decision

Two-tier logging approach:
1. **CLI-level log** (existing): Captures all pipeline runs in a session, written to `output_dir/pipeline.log`
2. **Per-run log** (new): Captures single run, written to `run_dir/pipeline.log`

Both logs contain the same information for their scope, enabling:
- Review of individual runs (per-run log)
- Review of batch runs (CLI-level log)

## Impact

### Affected Specs

- `cli-pipeline` - Add per-run logging requirement (ADDED)

### Affected Code

- `src/sleap_roots_analyze/pipeline/pipelines/base_pipeline.py`:
  - `__init__()` - Reorder to create run_dir before logger
  - `_setup_logger()` - Add FileHandler for per-run log
  - `_add_file_handler()` - New method to add file handler after run_dir exists

### New Tests (TDD)

- `tests/test_base_pipeline.py`:
  - `test_per_run_log_file_created()` - Log file exists in run_dir
  - `test_per_run_log_contains_pipeline_messages()` - Log content matches execution
  - `test_multiple_runs_have_separate_logs()` - Parallel runs don't interfere
  - `test_log_file_not_empty()` - Log file has content after run

### Risk Assessment

- **Low Risk**: Only affects logging infrastructure, not pipeline logic
- **Backwards Compatible**: Existing CLI-level logging unchanged
- **Easy Verification**: Check for `pipeline.log` in run directories after execution
