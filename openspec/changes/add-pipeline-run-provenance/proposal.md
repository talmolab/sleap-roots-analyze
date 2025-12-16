# Proposal: Add Pipeline Run Provenance

## Problem Statement

When a pipeline run completes, users cannot easily identify what config and data were used to generate the output. Currently:

1. The `pipeline_summary.json` has a `config` field but it's never populated
2. No copy of the config file is saved to the run folder
3. Data paths are embedded in the config but not explicitly recorded in the summary

This makes it difficult to:
- Reproduce a run by re-using the same config
- Audit which parameters generated specific results
- Track data lineage for scientific reproducibility

## Proposed Solution

Add provenance information to pipeline run output by:

1. **Save resolved config as YAML** - Copy the full config to `config.yaml` in the run folder
2. **Populate summary.config** - Include the config dict in `pipeline_summary.json`
3. **Add data_source to summary** - Explicitly record the input data path(s) used

## Scope

- Affects: `BasePipeline.run()`, `QCPipeline`, `VizPipeline`, `CrossPlatformPipeline`
- Minimal changes: Add config saving after run directory creation
- Backward compatible: No API changes, only additional output files

## Benefits

- Easy to see what config was used for any run
- Reproducible runs by copying the saved config
- Better audit trail for scientific publications
- Aligns with existing `config-management` spec requirements

## Implementation Approach

Use TDD (Test-Driven Development):
1. Write tests for config saving behavior first
2. Write tests for summary.config population
3. Implement the feature to pass tests
4. Update existing pipeline tests to verify provenance

## Related Specs

- `config-management` - Requirement for reproducibility guarantees
- `cli-pipeline` - Existing pipeline behavior

## Change ID

`add-pipeline-run-provenance`
