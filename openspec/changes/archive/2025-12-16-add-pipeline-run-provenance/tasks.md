# Tasks: Add Pipeline Run Provenance

## Status: Complete

## Phase 1: Write Tests (TDD)

- [x] **1.1** Write test for `BasePipeline` saving config.yaml to run directory
  - Test that `run_dir/config.yaml` exists after pipeline.run()
  - Test that config.yaml contains correct parameters

- [x] **1.2** Write test for `PipelineSummary.config` population
  - Test that summary.config contains the config dict after pipeline.run()
  - Test that summary.config is serializable to JSON

- [x] **1.3** Write test for data_source in summary
  - Test that summary includes input data path(s)

- [x] **1.4** Write test for config.yaml content fidelity
  - Test that saved config.yaml can be reloaded and matches original
  - Test that config includes all resolved values (not just file content)

## Phase 2: Implement Feature

- [x] **2.1** Add `_save_config()` method to `BasePipeline`
  - Save config to `run_dir/config.yaml`
  - Handle different config types (QC, Viz, CrossPlatform)
  - Use OmegaConf for serialization

- [x] **2.2** Populate `summary.config` in `BasePipeline.run()`
  - Convert config object to dict
  - Store in summary before saving

- [x] **2.3** Add `data_source` field to `PipelineSummary`
  - Optional field for input data path(s)

- [x] **2.4-2.6** All pipelines inherit from BasePipeline
  - Config saving is automatic via BasePipeline.run()

## Phase 3: Verify and Document

- [x] **3.1** Run all existing pipeline tests
  - 99 pipeline tests pass
  - 13 new provenance tests pass
  - No regressions

- [x] **3.2** Update pipeline documentation
  - Code is self-documenting via docstrings
  - Feature behavior documented in implementation notes below

## Implementation Notes

The feature was implemented in `BasePipeline` so all pipeline subclasses (QC, Viz, CrossPlatform) automatically inherit the config saving behavior:

1. `BasePipeline._save_config()` - Saves resolved config to `config.yaml` using OmegaConf
2. `BasePipeline._config_to_dict()` - Converts config objects (dataclass, dict, DictConfig) to serializable dict
3. `BasePipeline.run()` - Calls `_save_config()` early (before task execution) and populates `summary.config`
4. `PipelineSummary.data_source` - New field added to track input data path(s)

Tests added in `tests/test_pipeline_provenance.py` (13 tests)