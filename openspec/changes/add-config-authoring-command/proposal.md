## Why

Creating pipeline configs for a new analysis currently requires manually writing three YAML files (QC, Viz, run manifest), understanding dozens of parameters, navigating reproducibility risks (overwriting active configs, no git anchor), and doing all of this without guardrails. The process is error-prone and inconsistent across analyses. A guided slash command codifies best practices—statistical accuracy, reproducibility, and metadata preservation—into an interactive workflow that catches mistakes before they propagate into published results.

## What Changes

- **New slash command `/configure-run-all`**: Interactively guides the user through creating QC config, Viz config, and run manifest for a new analysis, one question at a time.
- **Dataset-aware defaults**: The command inspects the input CSV to infer column names, sample counts, group structures, and trait counts, then tailors default recommendations to the dataset.
- **Critical parameter review**: Before writing any file, the command presents a summary of the most statistically consequential parameters (heritability threshold, outlier method, min_samples_per_trait) with recommendations and flagged risks.
- **Backup-before-overwrite**: When a config file already exists in `configs/active/`, the command MUST offer to save a timestamped backup before overwriting.
- **Git-anchored configs**: After writing configs, the command commits them to git so they are permanently associated with a specific git SHA. This is the reproducibility anchor that allows any future user to exactly reproduce the analysis.
- **User validation gate**: The command shows the final configs and waits for user approval before running. It points out the most important values and flags any parameters that deviate significantly from recommended defaults.
- **New config-management requirements**: Adds requirements for backup behavior and git-anchored config versioning.

## Impact

- Affected specs: `developer-tooling` (new slash command), `config-management` (backup + git-anchored configs)
- Affected code: New file `.claude/commands/configure-run-all.md` (slash command definition)
- No breaking changes to existing CLI or config format
- Complements existing `/run-pipelines` command (configure first, then run)
