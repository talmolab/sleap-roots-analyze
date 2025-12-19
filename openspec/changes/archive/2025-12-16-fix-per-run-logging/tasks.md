## 1. TDD - Write Failing Tests First

- [x] 1.1 Create `tests/test_per_run_logging.py` with test cases:
  - `test_per_run_log_file_created()` - Verify `pipeline.log` exists in run_dir
  - `test_per_run_log_contains_start_message()` - Log contains "Starting pipeline"
  - `test_per_run_log_contains_task_messages()` - Log contains task execution logs
  - `test_per_run_log_contains_completion_message()` - Log contains success/failure
  - `test_per_run_log_not_empty()` - Log file has content
  - `test_multiple_runs_have_separate_logs()` - Parallel runs don't interfere

- [x] 1.2 Run tests to confirm they fail (red phase)
  - All 10 tests failed as expected

## 2. Implementation

- [x] 2.1 Refactor `BasePipeline.__init__()` to create run_dir before logger setup
  - Moved `self.run_dir = self._create_run_directory()` before `self.logger = self._setup_logger()`

- [x] 2.2 Modify `BasePipeline._setup_logger()` to add FileHandler
  - Created `FileHandler` writing to `self.run_dir / "pipeline.log"`
  - Used same formatter as StreamHandler
  - Added `_file_handler` attribute to track the handler

- [x] 2.3 Add `_close_file_handler()` method to release file lock after run
  - Called at end of `run()` for both success and failure paths
  - Prevents Windows file locking issues

- [x] 2.4 Run tests to confirm they pass (green phase)
  - All 10 tests pass

## 3. Verification

- [x] 3.1 Run full test suite: `uv run pytest tests/ -v`
  - 1072 tests pass, no regressions

- [x] 3.2 Manual verification:
  - Verified `pipeline.log` is created in run_dir
  - Verified log content matches console output
  - Log file contains all expected messages

- [x] 3.3 Verify existing CLI-level logging still works:
  - Per-run logs work independently of CLI logging

## 4. Cleanup

- [x] 4.1 Update docstrings for modified methods
- [x] 4.2 Mark tasks complete in this file