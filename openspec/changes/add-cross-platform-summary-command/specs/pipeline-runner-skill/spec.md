## ADDED Requirements

### Requirement: Detailed Cross-Platform Summary in Run Output

The system SHALL generate a detailed cross-platform analysis summary section in `SUMMARY.md` when cross-platform pipelines complete successfully.

The detailed summary SHALL include for each cross-platform comparison:

**Comparison Overview Table**:
| Column | Description |
|--------|-------------|
| Comparison | "{exp1_name} vs {exp2_name}" |
| Genotypes | Number of common genotypes (n) |
| Trait Reduction | "X → Y (Z% reduction)" or "N/A" if disabled |
| Correlations | Total number of trait pair correlations |
| Nominal Sig | Count where p < 0.05 |
| FDR Sig | Count where q < 0.05 (FDR-adjusted) |
| Top |r| | Maximum absolute correlation coefficient |
| Power ≥80% | Count/percentage of correlations with achieved_power ≥ 0.80 |

**Top Correlations Table** (per comparison, top 5 by |r|):
| Column | Description |
|--------|-------------|
| Rank | 1-5 |
| Exp1 Trait | Trait name from experiment 1 |
| Exp2 Trait | Trait name from experiment 2 |
| r | Spearman correlation coefficient (signed) |
| p | Raw p-value |
| q | FDR-adjusted p-value |
| Power | Achieved statistical power |
| n | Number of genotypes used |

**Metadata Table** (per comparison):
- FDR correction method
- Trait reduction method and parameters
- Minimum genotypes threshold
- Significance level (α)

**Embedded Visualizations** (per comparison):
- Correlation summary plot (4-panel): `cross_platform_correlation_summary.png`
- Top 3 joint plots: Scatter + regression for highest |r| correlations
- Exp1 clustering dendrogram (if exp1 clustered): `exp1_trait_clustering_dendrogram.png`
- Exp1 cluster heatmap (if exp1 clustered): `exp1_trait_cluster_heatmap.png`
- Exp2 clustering dendrogram (if exp2 clustered): `exp2_trait_clustering_dendrogram.png`
- Exp2 cluster heatmap (if exp2 clustered): `exp2_trait_cluster_heatmap.png`
- Cross-platform representative heatmap (if any clustering): `cross_platform_representative_heatmap.png`

#### Scenario: Cross-platform summary generated after run-all

- **WHEN** `sleap-roots-analyze run-all` completes with cross-platform pipelines
- **THEN** `SUMMARY.md` SHALL contain a "## Detailed Cross-Platform Analysis" section
- **AND** the section SHALL appear after the existing Cross-Platform Results table
- **AND** each comparison SHALL have its own subsection

#### Scenario: Trait reduction statistics displayed

- **WHEN** a cross-platform run used `trait_reduction_method: clustering`
- **THEN** the Trait Reduction column SHALL show "819 → 121 (85% reduction)" format
- **AND** the values SHALL match `trait_clusters.csv` exactly

#### Scenario: Trait reduction disabled

- **WHEN** a cross-platform run used `trait_reduction_method: none`
- **THEN** the Trait Reduction column SHALL show "Disabled"
- **AND** no `trait_clusters.csv` file is expected

#### Scenario: Correlation counts verified against source

- **WHEN** summary reports correlation counts
- **THEN** total correlations SHALL equal row count in `cross_platform_correlations.csv`
- **AND** nominal significant SHALL equal count where `spearman_p < 0.05`
- **AND** FDR significant SHALL equal count where `significant_fdr == True`

#### Scenario: Top correlations match CSV ordering

- **WHEN** summary displays top 5 correlations
- **THEN** they SHALL be the 5 rows with highest `abs(spearman_r)` from CSV
- **AND** all displayed values (r, p, q, power, n) SHALL match CSV values exactly

#### Scenario: Power statistics computed correctly

- **WHEN** summary displays power statistics
- **THEN** Power ≥80% count SHALL equal count where `achieved_power >= 0.80` in CSV
- **AND** percentage SHALL be calculated as (count / total) * 100

#### Scenario: Multiple comparisons summarized

- **WHEN** run includes 4 cross-platform configs
- **THEN** summary SHALL include subsections for all 4 comparisons
- **AND** each subsection SHALL have complete statistics

#### Scenario: Visualizations embedded inline

- **WHEN** summary is generated
- **THEN** images SHALL be embedded using markdown syntax: `![Description](relative/path/to/image.png)`
- **AND** paths SHALL be relative to the run directory
- **AND** images SHALL render correctly in GitHub, VS Code, and other markdown viewers

#### Scenario: Correlation summary plot included

- **WHEN** a cross-platform comparison has `cross_platform_correlation_summary.png`
- **THEN** the summary SHALL embed this 4-panel visualization
- **AND** it SHALL appear at the start of the comparison subsection

#### Scenario: Top joint plots included

- **WHEN** a cross-platform comparison has joint plots
- **THEN** the summary SHALL embed the top 3 joint plots by |r|
- **AND** plots SHALL be ordered from strongest to weakest correlation

#### Scenario: Exp1 clustering visualizations included when exp1 clustered

- **WHEN** a cross-platform run used `trait_reduction_target: exp1` or `trait_reduction_target: both`
- **THEN** the summary SHALL embed `exp1_trait_clustering_dendrogram.png`
- **AND** the summary SHALL embed `exp1_trait_cluster_heatmap.png`

#### Scenario: Exp2 clustering visualizations included when exp2 clustered

- **WHEN** a cross-platform run used `trait_reduction_target: exp2` or `trait_reduction_target: both`
- **THEN** the summary SHALL embed `exp2_trait_clustering_dendrogram.png`
- **AND** the summary SHALL embed `exp2_trait_cluster_heatmap.png`

#### Scenario: Representative heatmap included when any clustering enabled

- **WHEN** a cross-platform run used `trait_reduction_method: clustering`
- **THEN** the summary SHALL embed `cross_platform_representative_heatmap.png`

#### Scenario: Clustering visualizations omitted when disabled

- **WHEN** a cross-platform run used `trait_reduction_method: none`
- **THEN** the summary SHALL NOT include any clustering visualizations
- **AND** no error SHALL occur for missing visualization files

### Requirement: Cross-Platform Summary Validation Guardrails

The system SHALL validate that all reported statistics match their source data files before generating the summary.

#### Scenario: Validation passes for accurate summary

- **WHEN** summary generator reads `cross_platform_correlations.csv` and `trait_clusters.csv`
- **AND** all computed statistics match source data
- **THEN** validation SHALL pass
- **AND** summary SHALL be generated normally

#### Scenario: Validation fails on correlation count mismatch

- **WHEN** reported total correlations does not match CSV row count
- **THEN** validation SHALL fail with error message
- **AND** summary SHALL include warning about discrepancy
- **AND** the mismatch SHALL be logged for debugging

#### Scenario: Validation fails on trait reduction mismatch

- **WHEN** reported trait reduction percentage does not match computed from `trait_clusters.csv`
- **THEN** validation SHALL fail with error message
- **AND** summary SHALL include warning about discrepancy

#### Scenario: Missing source files handled gracefully

- **WHEN** `cross_platform_correlations.csv` is missing
- **THEN** validation SHALL skip that comparison
- **AND** summary SHALL show "Data unavailable" for that comparison
- **AND** no crash SHALL occur

### Requirement: Cross-Platform Summary Slash Command

The system SHALL provide a `/cross-platform-summary` Claude command for generating detailed cross-platform analysis reports on demand.

#### Scenario: Basic command invocation

- **WHEN** user invokes `/cross-platform-summary pipeline_runs/2026-02-02_133904`
- **THEN** the system SHALL read all cross-platform results from that directory
- **AND** the system SHALL generate a detailed summary report
- **AND** the system SHALL display the report to the user

#### Scenario: Command with latest run

- **WHEN** user invokes `/cross-platform-summary` without arguments
- **THEN** the system SHALL find the most recent pipeline run directory
- **AND** the system SHALL generate summary for that run

#### Scenario: Command validates results

- **WHEN** summary is generated via command
- **THEN** validation guardrails SHALL be applied
- **AND** any discrepancies SHALL be reported to the user
- **AND** user SHALL be warned if data integrity issues are detected

#### Scenario: No cross-platform results found

- **WHEN** specified directory contains no cross-platform results
- **THEN** command SHALL report "No cross-platform results found in {directory}"
- **AND** command SHALL suggest checking the directory path

## MODIFIED Requirements

### Requirement: Run Summary Generation

The system SHALL generate a comprehensive markdown summary document after each run.

#### Scenario: Summary document created

- **WHEN** pipeline run completes
- **THEN** a `SUMMARY.md` file SHALL be created in the run directory
- **AND** the summary SHALL include generation timestamp
- **AND** the summary SHALL include git commit hash
- **AND** the summary SHALL include manifest file reference
- **AND** the file SHALL be written with UTF-8 encoding to properly display Unicode characters (H², mm², etc.)

#### Scenario: QC results in summary

- **WHEN** QC pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Dataset, Samples, Traits, Genotypes, H² Threshold, Mean H², Status, Run Path
- **AND** for each QC run, the summary SHALL read `10_pipeline_summary.json` to extract scientific metrics
- **AND** the summary SHALL include a "Removed Traits" subsection listing traits filtered by heritability threshold for each dataset
- **AND** each removed trait SHALL display its heritability value in parentheses (e.g., "Depth (mm) (H²=0.32)")

#### Scenario: QC results with failed pipeline

- **WHEN** a QC pipeline fails during execution
- **THEN** the summary table SHALL show "Failed" status for that config
- **AND** numeric columns (Samples, Traits, etc.) SHALL display "N/A"
- **AND** other successful QC runs SHALL still display full metrics

#### Scenario: Viz results in summary

- **WHEN** Viz pipelines complete
- **THEN** the summary SHALL include a table with columns: Dataset, Figures Generated, Interactive Plots, Status, Time, Run Path
- **AND** the summary SHALL count static figures from the `static_figures/` directory
- **AND** the summary SHALL count interactive plots from the `pca/` and `umap/` directories

#### Scenario: Cross-Platform results in summary

- **WHEN** Cross-Platform pipelines complete successfully
- **THEN** the summary SHALL include a basic table with columns: Comparison, Common Genotypes, Exp1 Samples, Exp2 Samples, Top Correlation, Status, Run Path
- **AND** the summary SHALL include a detailed "## Detailed Cross-Platform Analysis" section
- **AND** the detailed section SHALL include trait reduction statistics, correlation counts, significance counts, power analysis, and top correlations for each comparison

#### Scenario: Cross-Platform results with missing data

- **WHEN** Cross-Platform pipeline completes but correlation CSV is missing
- **THEN** the summary SHALL display "N/A" for detailed statistics
- **AND** the summary SHALL still display status and run path
- **AND** a warning SHALL be included noting missing data
