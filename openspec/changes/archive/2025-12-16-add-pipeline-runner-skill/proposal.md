## Why

Running all QC, Viz, and Cross-Platform pipelines and generating a comprehensive summary document is a common workflow that currently requires multiple manual steps:
1. Identify all config files
2. Run each pipeline individually
3. Update config paths to point to new QC outputs
4. Collect run metadata and statistics
5. Generate a publication-ready summary markdown

This is error-prone, time-consuming, and the exact workflow users need before publication or sharing results.

## What Changes

- **ADDED**: New CLI command `sleap-roots-analyze run-all` that:
  - Discovers all QC, Viz, and Cross-Platform config files in `configs/`
  - Runs all pipelines in the correct order (QC first, then Viz/Cross-Platform)
  - Automatically updates Viz/Cross-Platform config paths to point to new QC outputs
  - Collects statistics from each run (samples, traits, genotypes, heritability)
  - Generates/updates `PIPELINE_RUNS_SUMMARY.md` with comprehensive documentation
  - Supports selective runs (e.g., only QC, only a specific platform)

- **ADDED**: Summary document generation logic that:
  - Reads `10_pipeline_summary.json` from each QC run
  - Reads `pipeline_summary.json` from each Viz run
  - Formats results in publication-ready markdown
  - Includes methods section template
  - Documents all config files and run paths

- **ADDED**: Claude Code slash command `/run-pipelines` that:
  - Provides the same functionality as the CLI command
  - Integrates with Claude Code's task tracking via TodoWrite
  - Ideal for interactive development sessions

## Impact

- **Affected specs**: New capability `pipeline-runner-skill`
- **Affected code**:
  - `src/sleap_roots_analyze/cli.py` - New `run-all` command
  - `src/sleap_roots_analyze/pipeline_runner.py` - Core execution logic (new module)
  - `.claude/commands/run-pipelines.md` - Slash command wrapper
- **User benefit**: One-command execution of complete analysis workflow via CLI or Claude Code
