# Implementation Tasks

## Status Note (2026-02-17)
PR #63 is open (not yet merged). Core QC grouping (tasks 1–4) was implemented
on the feature branch but tasks were never marked complete in this file.
Run-All integration (task 5) and per-group viz fan-out are the known gap
tracked in GitHub issue #69. The /run-pipelines workaround (task 7) is needed
until issue #69 is resolved.

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

## 5. Run-All Integration (NOT DONE — GitHub #69)
The critical missing piece: when run-all executes a QC config with group_by,
it must fan out the per-group QC outputs to per-group viz runs, not just
auto-update the viz config to the last group's output.

Required behavior:
1. After grouped QC completes, discover all group output directories
2. For each group output, spawn a viz run pointing to that group's 10_final_data.csv
3. Name viz output dirs to match QC group dirs: viz_<pipeline>_<column>_<value>_<ts>/

- [ ] 5.1 Write test for run-all detecting multiple QC group outputs from a grouped QC config
- [ ] 5.2 Implement grouped viz fan-out in run_all: after QC, iterate group outputs and run viz per group
- [ ] 5.3 Write test for per-group viz output directory naming
- [ ] 5.4 Update /run-pipelines slash command to handle this behavior

## 6. Integration Tests
- [x] 6.1 End-to-end test: QC pipeline with group_by (test_grouped_pipeline_integration.py)
- [ ] 6.2 End-to-end test: Viz pipeline fan-out via run-all with group_by
- [ ] 6.3 End-to-end test: run-all with group_by, verify N viz outputs == N QC groups
- [x] 6.4 Test edge cases: empty groups, single group (test_ungrouped_qc_pipeline_with_real_data)

## 7. Documentation
- [x] 7.1 Update config templates README with group_by examples
- [x] 7.2 Add known limitation note to templates README (run-all + group_by, GitHub #69)
- [x] 7.3 Document workaround in run manifest header comments
- [ ] 7.4 Update /run-pipelines slash command to automate per-group viz workaround

## /run-pipelines Interim Workaround (until issue #69 is resolved)
When the active QC config has group_by set, /run-pipelines SHOULD:
1. Run `sleap-roots-analyze run-all --qc-only` first
2. Discover group output directories under output/qc/
3. For each group dir: update viz config csv_path, run `sleap-roots-analyze viz`
4. Report all viz output paths to the user
