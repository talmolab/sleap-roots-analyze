# Implementation Tasks

## 1. Configuration Support
- [ ] 1.1 Add `group_by: str | None = None` field to DataConfig in components.py
- [ ] 1.2 Write unit test for DataConfig validation with group_by field
- [ ] 1.3 Update config validation to check group_by column exists in data

## 2. CLI Flag Support
- [ ] 2.1 Add `--group-by` argument to `qc` command in cli.py
- [ ] 2.2 Add `--group-by` argument to `viz` command in cli.py
- [ ] 2.3 Write CLI integration test for group-by flag parsing
- [ ] 2.4 Implement CLI override logic (CLI takes precedence over config)

## 3. Data Grouping Logic
- [ ] 3.1 Write test for data splitting by group column (TDD - write test first)
- [ ] 3.2 Implement `split_data_by_group()` function in load_data.py
- [ ] 3.3 Write test for group validation (min samples per group)
- [ ] 3.4 Implement group validation using existing `min_samples_per_trait` threshold
- [ ] 3.5 Write test for logging warnings when groups skipped

## 4. Pipeline Execution
- [ ] 4.1 Write test for running pipeline once per group (TDD)
- [ ] 4.2 Modify base_pipeline.py to detect group_by and loop over groups
- [ ] 4.3 Write test for output directory naming: `<column>_<value>_<timestamp>`
- [ ] 4.4 Implement isolated output directories per group
- [ ] 4.5 Write test for preserving all group metadata in outputs

## 5. Run-All Integration
- [ ] 5.1 Write test for manifest with group_by field (TDD)
- [ ] 5.2 Add `group_by` field to run-all manifest schema
- [ ] 5.3 Implement manifest-level group_by support in run_all command
- [ ] 5.4 Write test for CLI --group-by override of manifest value

## 6. Integration Tests
- [ ] 6.1 End-to-end test: QC pipeline with --group-by plant_age_days
- [ ] 6.2 End-to-end test: Viz pipeline with group_by in config
- [ ] 6.3 End-to-end test: run-all with manifest group_by
- [ ] 6.4 Test edge cases: missing group column, empty groups, single group

## 7. Documentation
- [ ] 7.1 Update config templates with group_by examples
- [ ] 7.2 Update CLI help text for --group-by flag
- [ ] 7.3 Add usage examples to README or docs
- [ ] 7.4 Document output directory structure convention

## Test-First Checklist
- Every function has a test written BEFORE implementation
- Tests cover success cases, edge cases, and error conditions
- Integration tests verify end-to-end workflows
- All tests pass before marking task complete
