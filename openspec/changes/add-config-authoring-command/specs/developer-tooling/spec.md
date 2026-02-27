## ADDED Requirements

### Requirement: Interactive Analysis Configuration Command

The system SHALL provide a `/configure-run-all` Claude Code slash command that interactively guides the user through creating a complete, scientifically sound set of pipeline configuration files (QC config, Viz config, run manifest) for a new analysis.

The command SHALL embody three core scientific values:
1. **Reproducibility**: All config files SHALL be committed to git so they are permanently tied to a specific codebase state (git SHA).
2. **Metadata preservation**: Config file headers SHALL document dataset identity, analysis date, author intent, and parameter rationale.
3. **Statistical accuracy**: The command SHALL warn the user when parameter choices are likely to produce statistically unreliable results.

#### Scenario: Dataset inspection

- **WHEN** the user provides a CSV path
- **THEN** the command SHALL read the CSV and report: total sample count, column names (with candidates for barcode/genotype/replicate roles), numeric trait count, and candidate group_by columns (columns with ≤20 unique values)
- **AND** the command SHALL flag any candidate group with fewer than 30 samples with a WARNING (Mahalanobis chi-squared reliability requires n≥30)
- **AND** the command SHALL flag any experiment where fewer than 3 replicates per genotype exist in any group with a WARNING (heritability estimation requires ≥3 replicates per genotype)

#### Scenario: Interactive Q&A sequence

- **WHEN** the user invokes `/configure-run-all`
- **THEN** the command SHALL collect the following information interactively, one topic at a time:
  1. Dataset CSV path and output directory
  2. Column name assignments: barcode, genotype, replicate
  3. Whether to use `group_by` (and which column if yes)
  4. Cleanup thresholds: max_nan_fraction, max_zeros_per_trait, max_nans_per_trait
  5. Whether to enable outlier detection and which method(s)
  6. Whether to enable heritability filtering and the H² threshold
  7. PCA settings: n_components, feature_selection_strategy, n_top_features, pca_biplot_top_features
  8. Whether to enable UMAP (and n_neighbors recommendation based on sample size)
  9. Whether images are available (and image directory path if yes)
- **AND** for each question, the command SHALL show the recommended default and explain why
- **AND** for PCA specifically, the command SHALL clarify the following parameter relationships that are commonly confused:
  - `pca.feature_selection_strategy` controls WHICH traits are selected for UMAP coloring and PCA metadata. Available strategies include `"top_variance"`, `"extreme"`, `"top_absolute"`, `"top_contribution"`, and `"vector_length"`. The two most commonly used are `"top_variance"` (selects traits with highest total variance contribution — good for general exploration) and `"extreme"` (selects the traits with the most extreme positive and negative PC loadings — better for mechanistic interpretation of PC axes). This setting does NOT affect the feature contribution bar chart, which always ranks traits by total variance contribution regardless of strategy.
  - `pca.n_top_features` controls HOW MANY traits are selected by the above strategy for UMAP coloring and metadata. With UMAP disabled, this has limited visual impact.
  - `static_viz.pca_biplot_top_features` independently controls how many arrows appear on the PCA biplot. This is separate from `n_top_features` and should be kept small (1–5) on high-dimensional datasets (>100 traits) to prevent arrow crowding. For `"extreme"` strategy, a value of 1 produces 2 arrows per PC (one positive-loading trait, one negative-loading trait).

#### Scenario: Critical parameter review before writing

- **WHEN** all parameters have been collected
- **THEN** the command SHALL present a critical parameter review table showing:
  - Heritability threshold and whether it is appropriate for the sample size
  - Mahalanobis chi2_percentile and whether the sample size supports it
  - min_samples_per_trait relative to group sizes
  - UMAP n_neighbors relative to smallest group size
- **AND** the command SHALL flag parameters that deviate significantly from recommended defaults
- **AND** the command SHALL ask the user to confirm or modify each flagged parameter before proceeding

#### Scenario: Backup before overwriting active configs

- **WHEN** a config file already exists at the target path in `configs/active/`
- **THEN** the command SHALL inform the user that an existing config will be overwritten
- **AND** the command SHALL offer to save a timestamped backup to `configs/archive/<original-name>_backup_<YYYYMMDD_HHMMSS>.yaml`
- **AND** the command SHALL NOT overwrite any existing config without explicit user confirmation

#### Scenario: Config file writing with self-documenting headers

- **WHEN** the user approves the configuration
- **THEN** the command SHALL write QC config, Viz config, and run manifest to `configs/active/`
- **AND** each config header SHALL include: dataset name, input CSV path, analysis date, and key parameter choices with brief rationale
- **AND** the run manifest header SHALL include the CLI command to reproduce the run

#### Scenario: User validation gate

- **WHEN** configs have been written to disk
- **THEN** the command SHALL display the full content of each config file for the user to review
- **AND** the command SHALL highlight (in text) the most consequential parameters: heritability threshold, outlier method, group_by column, min_samples_per_trait
- **AND** the command SHALL wait for explicit user approval ("looks good" / "yes" / "run it") before offering to proceed
- **AND** the command SHALL NOT invoke `/run-pipelines` automatically — it SHALL remind the user of the exact command to run

#### Scenario: Git commit after user approval

- **WHEN** the user approves the configs
- **THEN** the command SHALL stage the new/modified config files in `configs/active/`
- **AND** the command SHALL create a git commit with a message that includes: analysis run_name, dataset path, and ISO date
- **AND** the command SHALL report the resulting git SHA to the user as the reproducibility anchor
- **AND** if git commit fails (e.g., no changes, detached HEAD), the command SHALL warn the user clearly and continue without crashing

#### Scenario: Statistical accuracy guardrails for heritability

- **GIVEN** the user has chosen a heritability threshold
- **AND** the dataset has fewer than 3 replicates per genotype in one or more groups
- **THEN** the command SHALL warn: "Heritability estimation requires ≥3 replicates per genotype. Group {X} has only {N} replicates. H² estimates will be unreliable."
- **AND** the command SHALL recommend disabling heritability for that group OR increasing the replicate threshold

#### Scenario: Statistical accuracy guardrails for UMAP

- **GIVEN** the user has enabled UMAP
- **AND** a group has fewer than 15 samples
- **THEN** the command SHALL recommend n_neighbors = max(2, n_samples // 4)
- **AND** the command SHALL warn that UMAP with very small N may not produce meaningful structure

#### Scenario: Statistical accuracy guardrails for Mahalanobis

- **GIVEN** the user has chosen Mahalanobis outlier detection with chi-squared
- **AND** a group has fewer than 30 samples
- **THEN** the command SHALL warn: "Mahalanobis chi-squared outlier detection is unreliable for n<30. Group {X} has only {N} samples."
- **AND** the command SHALL suggest either disabling outlier detection for small groups or using a more permissive chi2_percentile (e.g., 95.0 instead of 99.0)
