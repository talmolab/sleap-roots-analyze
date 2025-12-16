# Implementation Tasks

**Status**: ✅ COMPLETED (2025-12-15)

## 1. Create Config Directory Structure

- [x] 1.1 Create `configs/active/qc/` directory
- [x] 1.2 Create `configs/active/viz/` directory
- [x] 1.3 Create `configs/active/cross_platform/` directory
- [x] 1.4 Create `configs/examples/` directory for example configs
- [x] 1.5 Move active QC configs to `configs/active/qc/`
- [x] 1.6 Move active Viz configs to `configs/active/viz/`
- [x] 1.7 Move active Cross-Platform configs to `configs/active/cross_platform/`

## 2. Create Run Manifest System

- [x] 2.1 Create `configs/active/run_manifest.yaml` listing all active configs
- [x] 2.2 Define manifest schema (run_name, description, qc/viz/cross_platform configs)
- [x] 2.3 Use relative paths from `configs/active/` directory
- [x] 2.4 Document manifest format in design.md

## 3. Update Gitignore

- [x] 3.1 Add `pipeline_runs/` to `.gitignore`
- [x] 3.2 Keep existing `qc_runs/`, `viz_runs/`, `cross_platform_runs/` patterns

## 4. Implement Pipeline Runner Module

- [x] 4.1 Create `src/sleap_roots_analyze/pipeline_runner.py`
- [x] 4.2 Implement `PipelineRunner` class with:
  - `__init__()` - Load manifest, setup output directory
  - `run_all()` - Orchestrate pipeline execution
  - `_run_qc_pipelines()` - Execute QC configs
  - `_run_viz_pipelines()` - Execute Viz configs
  - `_run_cross_platform_pipelines()` - Execute Cross-Platform configs
  - `_update_dependent_configs()` - Update paths between pipelines
  - `generate_summary()` - Create markdown summary
- [x] 4.3 Implement manifest loading and validation
- [x] 4.4 Implement timestamped run directory creation
- [x] 4.5 Implement symlink creation for `latest` run

## 5. Add CLI Command

- [x] 5.1 Add `run-all` command to `cli.py`
- [x] 5.2 Implement CLI options:
  - `--manifest` to specify manifest file (default: configs/active/run_manifest.yaml)
  - `--output` to specify output directory (default: pipeline_runs/)
  - `--dry-run` to preview without running
  - `--qc-only`, `--viz-only`, `--cross-only` for selective runs
  - `--no-summary` to skip summary generation
  - `-v/--verbose` for detailed output
- [x] 5.3 Wire CLI to PipelineRunner module
- [x] 5.4 Add progress output during execution

## 6. Create Slash Command

- [x] 6.1 Create `.claude/commands/run-pipelines.md`
- [x] 6.2 Slash command should call CLI command internally
- [x] 6.3 Add TodoWrite integration for progress tracking
- [x] 6.4 Support same arguments as CLI
- [x] 6.5 Document usage examples

## 7. Implement Summary Generation

- [x] 7.1 Read `10_pipeline_summary.json` from each QC run
- [x] 7.2 Read `pipeline_summary.json` from each Viz run
- [x] 7.3 Read correlation summaries from Cross-Platform runs
- [x] 7.4 Generate summary with sections:
  - Header with timestamp, git commit, manifest info
  - QC Pipeline Runs table
  - Visualization Pipeline Runs table
  - Cross-Platform Analysis Runs table
- [x] 7.5 Write to `pipeline_runs/.../SUMMARY.md`
- [x] 7.6 Also update root `PIPELINE_RUNS_SUMMARY.md` for quick access

## 8. Add Progress Tracking

- [x] 8.1 Use TodoWrite to create task list for each pipeline (slash command)
- [x] 8.2 Update task status as pipelines complete
- [x] 8.3 Report final completion status
- [x] 8.4 CLI progress output via verbose mode

## 9. Testing

- [x] 9.1 Test manifest parsing (via dry-run)
- [x] 9.2 Test CLI command with single QC config
- [x] 9.3 Test full pipeline set
- [x] 9.4 Test dry-run mode
- [x] 9.5 Test selective modes (--qc-only, --viz-only, --cross-only)
- [x] 9.6 Test error handling for missing configs
- [x] 9.7 Verify summary accuracy against actual run outputs
- [x] 9.8 Test slash command invocation

## 10. Documentation

- [x] 10.1 Update CLAUDE.md with reference to `run-all` CLI command and `/run-pipelines` slash command
- [x] 10.2 Document config directory structure
- [x] 10.3 Document manifest format
- [x] 10.4 Add migration guide for existing configs (N/A - new structure)
- [x] 10.5 Add CLI help text and examples
