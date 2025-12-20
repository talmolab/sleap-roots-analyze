## Context

The CLI currently sets up logging before the pipeline creates its run directory. When `logging.log_to_file: true` and `logging.log_file: "custom.log"` are set in config, the CLI creates the log file at `output_dir/custom.log` (e.g., `qc_runs/custom.log`).

However, all other run artifacts go into the timestamped `run_dir` (e.g., `qc_runs/pipeline_timestamp/`). This inconsistency creates stray log files outside the self-contained run folder.

The pipeline already creates a correct per-run log at `run_dir/pipeline.log`.

## Goals / Non-Goals

**Goals:**
- Config-based log file (`logging.log_file`) should be created inside `run_dir`
- All run artifacts should be contained within the timestamped run directory
- Maintain backward compatibility for `--log-file` CLI flag (explicit paths)

**Non-Goals:**
- Changing the pipeline's existing per-run logging mechanism
- Adding new logging features
- Supporting multiple log files per run

## Decisions

**Decision: Unify CLI config-based logging with pipeline per-run logging**

The CLI will pass the config's `logging.log_file` value to the pipeline, and the pipeline will use it as the log filename (instead of hardcoded "pipeline.log") when creating the per-run log.

**Rationale:**
- Single source of truth for log file creation (pipeline's `_setup_logger`)
- No duplicate log file handlers
- Log file path is naturally resolved relative to `run_dir`

**Alternatives considered:**

1. **Pre-compute run_dir in CLI before pipeline init**: Would require duplicating timestamp logic and creating the directory earlier. Rejected due to duplication.

2. **CLI creates log file, pipeline skips its own**: Would require coordination flags and lose the per-run isolation guarantee. Rejected due to complexity.

3. **Remove config-based log file entirely**: Would break existing configs. Rejected for backward compatibility.

## Implementation Approach

1. CLI: When `log_to_file: true` and no `--log-file` flag, pass `config.logging.log_file` to pipeline via a new parameter
2. Pipeline: In `_setup_logger`, use the provided log filename instead of hardcoded "pipeline.log"
3. CLI: Remove early `FileHandler` creation for config-based logging (keep it for `--log-file` explicit flag)

```python
# cli.py - Pass log filename to pipeline
pipeline = QCPipeline(
    cfg,
    output_dir=output_dir,
    log_filename=cfg.logging.log_file if cfg.logging.log_to_file else None
)

# base_pipeline.py - Use provided filename
def _setup_logger(self) -> logging.Logger:
    log_filename = getattr(self, '_log_filename', None) or "pipeline.log"
    log_file_path = self.run_dir / log_filename
    # ... rest of existing code
```

## Risks / Trade-offs

- **Risk**: Existing configs with `log_file: "root_core_qc_pipeline.log"` will now create log in different location
  - **Mitigation**: This is the desired fix; document in release notes

- **Trade-off**: Slightly more complex pipeline constructor
  - **Acceptable**: One optional parameter is minimal complexity

## Migration Plan

1. Update CLI to pass log filename to pipeline
2. Update pipeline to use provided filename
3. Remove CLI's early log file handler for config-based logging
4. Clean up stray log files from previous runs (manual or documented)

## Open Questions

None - the approach is straightforward.
