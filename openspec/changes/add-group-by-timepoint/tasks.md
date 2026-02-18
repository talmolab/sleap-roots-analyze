# Implementation Tasks

## Status Note (2026-02-17)
PR #63 is open (not yet merged). Core QC grouping (tasks 1–4) was implemented
on the feature branch but tasks were never marked complete in this file.
Run-All integration (task 5) and per-group viz fan-out (GitHub #69) are now
implemented in pipeline_runner.py. Tasks 5.1–5.3 and 6.2–6.3 written and
passing as of 2026-02-17. Documentation cleanup (7.2–7.4) deferred.

## 1. Configuration Support
- [x] 1.1 Add `group_by: str | None = None` field to DataConfig in components.py
- [x] 1.2 Write unit test for DataConfig validation with group_by field
- [ ] 1.3 Update config validation to check group_by column exists in data

## 2. CLI Flag Support
- [x] 2.1 Add `--group-by` argument to `qc` command in cli.py
- [x] 2.2 Add `--group-by` argument to `viz` command in cli.py
- [x] 2.3 Write CLI integration test for group-by flag parsing
- [x] 2.4 Implement CLI override logic (CLI takes precedence over config)

## 3. Data Grouping Logic
- [x] 3.1 Write test for data splitting by group column (TDD - write test first)
- [x] 3.2 Implement `run_grouped_pipelines()` in pipeline/utils.py
- [x] 3.3 Write test for group validation (min samples per group)
- [x] 3.4 Implement group validation using existing `min_samples_per_trait` threshold
- [x] 3.5 Write test for logging warnings when groups skipped

## 4. Pipeline Execution
- [x] 4.1 Write test for running pipeline once per group (TDD)
- [x] 4.2 Implement grouped execution via `run_grouped_pipelines()` in pipeline/utils.py
- [x] 4.3 Write test for output directory naming: `<column>_<value>_<timestamp>`
- [x] 4.4 Implement isolated output directories per group
- [x] 4.5 Write test for preserving all group metadata in outputs

## 5. Run-All Integration (GitHub #69)
The critical missing piece: when run-all executes a QC config with group_by,
it must fan out the per-group QC outputs to per-group viz runs, not just
auto-update the viz config to the last group's output.

Required behavior:
1. After grouped QC completes, discover all group output directories
2. For each group output, spawn a viz run pointing to that group's 10_final_data.csv
3. Per-group viz output written to run_dir/viz/{group_label}/
4. Per-group updated config written to run_dir/viz/{group_label}/_updated_{config_name}
   (consistent with existing _updated_* convention used for non-grouped runs)
5. Track each group's result in run_results["viz"]["{config_rel}:{group_label}"]

New helper methods in pipeline_runner.py:
- `_get_qc_config_group_by(config_path)` — reads data.group_by from YAML; None if absent
- `_find_grouped_qc_outputs(base_dir, group_by_column)` — sorted list of matching group dirs
- `_extract_group_label(dir_name)` — strips YYYYMMDD_HHMMSS suffix from dir name
- `_run_viz_for_group(config_path, config_rel, group_qc_dir, viz_output_dir)` — runs viz for one group

### Unit Tests (tests/test_run_all_grouped_viz.py — written BEFORE implementation per TDD)
- [x] 5.1a Test: `_find_grouped_qc_outputs` discovers all matching group dirs
- [x] 5.1b Test: `_find_grouped_qc_outputs` ignores non-matching dirs
- [x] 5.1c Test: `_find_grouped_qc_outputs` returns empty for nonexistent base
- [x] 5.1d Test: `_find_grouped_qc_outputs` returns deterministically sorted list
- [x] 5.1e Test: `_get_qc_config_group_by` returns column name when set
- [x] 5.1f Test: `_get_qc_config_group_by` returns None when absent, null, or file missing
- [x] 5.3a Test: `_extract_group_label` strips timestamp from integer group dir name
- [x] 5.3b Test: `_extract_group_label` strips timestamp from string group dir name
- [x] 5.3c Test: `_extract_group_label` returns name unchanged when no timestamp pattern
- [x] 5.3d Test: `_run_viz_pipelines` calls pipeline N times for N groups (fan-out)
- [x] 5.3e Test: `_run_viz_pipelines` result keys include group label per group
- [x] 5.3f Test: each group gets distinct output subdir under run_dir/viz/
- [x] 5.3g Test: updated viz config written under group subdir with _updated_ prefix
- [x] 5.3h Test: updated viz config content references group's 10_final_data.csv path
- [x] 5.3i Test: non-grouped viz runs exactly once (regression guard)
- [x] 5.3j Test: group with missing CSV recorded as failure, not raised; others continue

### Implementation
- [x] 5.2a Add `self.qc_grouped_outputs: dict[str, list[Path]] = {}` to `__init__`
- [x] 5.2b Add `_get_qc_config_group_by(config_path)` static method
- [x] 5.2c Add `_find_grouped_qc_outputs(base_dir, group_by_column)` static method
- [x] 5.2d Add `_extract_group_label(dir_name)` static method
- [x] 5.2e Add `_run_viz_for_group(config_path, config_rel, group_qc_dir, viz_output_dir)` method
- [x] 5.2f Modify `_run_qc_pipelines()` to populate `qc_grouped_outputs` when group_by detected
- [x] 5.2g Modify `_run_viz_pipelines()` to fan out when `qc_grouped_outputs` is populated
- [ ] 5.4 Update /run-pipelines slash command: remove interim workaround, document native fan-out

## 6. Integration Tests

### Root cause of the gap
Tasks 5.1 and 5.3 (unit tests) and 6.2–6.3 (integration tests) were defined here but
never implemented. Without these tests, the run-all QC→viz path with group_by was only
exercised manually, allowing the fan-out bug to ship undetected.

- [x] 6.1 End-to-end test: QC pipeline with group_by (test_grouped_pipeline_integration.py)
- [x] 6.2 Integration test: PipelineRunner with grouped QC config produces N viz run_results entries
      (uses mocked _run_pipeline_command — fast and deterministic, in test_run_all_grouped_viz.py)
- [x] 6.3 Integration test: PipelineRunner grouped viz fan-out produces N group output dirs == N QC groups
- [x] 6.4 Test edge cases: empty groups, single group (test_ungrouped_qc_pipeline_with_real_data)

## 7. Documentation
- [x] 7.1 Update config templates README with group_by examples
- [x] 7.2 Remove known limitation note from templates README (issue #69 now fixed)
- [x] 7.3 Update run manifest header: remove workaround, reference native fan-out
- [ ] 7.4 Update /run-pipelines slash command: remove workaround steps, document native behavior
