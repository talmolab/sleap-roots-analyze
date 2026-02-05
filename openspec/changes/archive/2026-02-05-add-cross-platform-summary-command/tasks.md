# Implementation Tasks

## 1. TDD: Write Tests First (Red Phase)

- [x] 1.1 Create `tests/test_cross_platform_summary.py` with test structure
- [x] 1.2 Test: `test_generate_summary_from_single_run` - verify summary generation from one cross-platform run
- [x] 1.3 Test: `test_generate_summary_from_multiple_runs` - verify aggregation across multiple comparisons
- [x] 1.4 Test: `test_trait_reduction_statistics_accuracy` - verify reported reduction matches `trait_clusters.csv`
- [x] 1.5 Test: `test_correlation_counts_match_csv` - verify total/nominal/FDR counts match source CSV
- [x] 1.6 Test: `test_top_correlations_match_csv` - verify top N correlations match sorted CSV values
- [x] 1.7 Test: `test_power_statistics_accuracy` - verify power statistics match CSV `achieved_power` column
- [x] 1.8 Test: `test_metadata_extraction` - verify config parameters extracted correctly
- [x] 1.9 Test: `test_validation_guardrails_pass` - verify guardrails pass for valid data
- [x] 1.10 Test: `test_validation_guardrails_fail_on_mismatch` - verify guardrails catch discrepancies
- [x] 1.11 Test: `test_missing_files_handled_gracefully` - verify graceful handling of missing CSVs
- [x] 1.12 Test: `test_empty_correlations_handled` - verify empty/zero correlation case handled
- [x] 1.13 Test: `test_config_requires_target_when_clustering` - verify validation fails without trait_reduction_target
- [x] 1.14 Test: `test_exp1_dendrogram_generated_when_exp1_clustered` - verify exp1 dendrogram PNG created
- [x] 1.15 Test: `test_exp2_dendrogram_generated_when_exp2_clustered` - verify exp2 dendrogram PNG created
- [x] 1.16 Test: `test_exp1_heatmap_generated_when_exp1_clustered` - verify exp1 cluster heatmap created
- [x] 1.17 Test: `test_exp2_heatmap_generated_when_exp2_clustered` - verify exp2 cluster heatmap created
- [x] 1.18 Test: `test_representative_heatmap_generated` - verify cross-platform representative heatmap created
- [x] 1.19 Test: `test_visualizations_skipped_when_clustering_disabled` - verify no viz files when method=none
- [x] 1.20 Test: `test_summary_embeds_visualizations` - verify markdown includes image references
- [x] 1.21 Test: `test_top_3_joint_plots_included` - verify top 3 joint plots embedded in summary
- [x] 1.22 Run tests, confirm all fail (Red phase complete)

## 2. Implementation (Green Phase)

- [x] 2.1 Create `src/sleap_roots_analyze/summary/__init__.py`
- [x] 2.2 Create `src/sleap_roots_analyze/summary/cross_platform_summary.py`
- [x] 2.3 Implement `CrossPlatformSummaryGenerator` class:
  - [x] 2.3.1 `__init__(self, run_dir: Path)` - initialize with pipeline run directory
  - [x] 2.3.2 `_find_cross_platform_runs(self) -> List[Path]` - discover cross-platform output dirs
  - [x] 2.3.3 `_read_trait_clusters(self, run_dir: Path) -> TraitReductionStats` - parse trait_clusters.csv
  - [x] 2.3.4 `_read_correlations(self, run_dir: Path) -> CorrelationStats` - parse correlations CSV
  - [x] 2.3.5 `_read_metadata(self, run_dir: Path) -> Dict` - parse pipeline_summary.json
  - [x] 2.3.6 `_validate_statistics(self, reported: Stats, source: Path) -> ValidationResult` - guardrails
  - [x] 2.3.7 `generate(self) -> CrossPlatformSummary` - main entry point
  - [x] 2.3.8 `to_markdown(self) -> str` - render summary as markdown
- [x] 2.4 Implement data classes:
  - [x] 2.4.1 `TraitReductionStats` - original, clusters, representatives, reduction_pct
  - [x] 2.4.2 `CorrelationStats` - total, nominal_sig, fdr_sig, top_correlations
  - [x] 2.4.3 `TopCorrelation` - exp1_trait, exp2_trait, r, p, q, power, n
  - [x] 2.4.4 `PowerStats` - min, median, max, pct_above_80
  - [x] 2.4.5 `CrossPlatformRunSummary` - all stats for one comparison
  - [x] 2.4.6 `CrossPlatformSummary` - aggregated results from all runs
  - [x] 2.4.7 `ValidationResult` - passed, errors, warnings
- [x] 2.5 Run summary tests, confirm pass

## 2b. Configuration Updates

- [x] 2b.1 Update `CrossPlatformConfig` in `src/sleap_roots_analyze/pipeline/config/components.py`:
  - [x] 2b.1.1 Add `trait_reduction_target: Literal["exp1", "exp2", "both"]` field
  - [x] 2b.1.2 Add validation: require `trait_reduction_target` when `trait_reduction_method == "clustering"`
  - [x] 2b.1.3 Add validation error message: "trait_reduction_target must be specified when trait_reduction_method is 'clustering'"
- [x] 2b.2 Update all active cross-platform config YAML files:
  - [x] 2b.2.1 Add `trait_reduction_target: both` to each config
- [x] 2b.3 Write tests for config validation:
  - [x] 2b.3.1 Test: `test_config_requires_target_when_clustering_enabled`
  - [x] 2b.3.2 Test: `test_config_accepts_valid_reduction_targets`
  - [x] 2b.3.3 Test: `test_config_allows_no_target_when_clustering_disabled`

## 2c. Trait Clustering Pipeline Updates

- [x] 2c.1 Update `ReduceTraitRedundancyStep` to support configurable clustering target:
  - [x] 2c.1.1 Refactor to cluster exp1, exp2, or both based on `trait_reduction_target`
  - [x] 2c.1.2 Output `exp1_trait_clusters.csv` when exp1 clustered
  - [x] 2c.1.3 Output `exp2_trait_clusters.csv` when exp2 clustered
  - [x] 2c.1.4 Update metadata with explicit clustering info per experiment
- [x] 2c.2 Add visualization methods to `ReduceTraitRedundancyStep`:
  - [x] 2c.2.1 Add `_create_dendrogram(self, linkage_matrix, traits, clusters, threshold, exp_name) -> Figure`
  - [x] 2c.2.2 Add `_create_cluster_heatmap(self, trait_data, clusters, representatives, exp_name) -> Figure`
  - [x] 2c.2.3 Save `exp1_trait_clustering_dendrogram.png` when exp1 clustered
  - [x] 2c.2.4 Save `exp1_trait_cluster_heatmap.png` when exp1 clustered
  - [x] 2c.2.5 Save `exp2_trait_clustering_dendrogram.png` when exp2 clustered
  - [x] 2c.2.6 Save `exp2_trait_cluster_heatmap.png` when exp2 clustered
- [x] 2c.3 Update `VisualizeCrossPlatformStep`:
  - [x] 2c.3.1 Add `_create_representative_heatmap(self, corr_df, exp1_traits, exp2_traits, metadata) -> Figure`
  - [x] 2c.3.2 Save `cross_platform_representative_heatmap.png` with significance annotations
  - [x] 2c.3.3 Handle cases: exp1 only clustered, exp2 only clustered, both clustered
- [x] 2c.4 Write tests for pipeline steps:
  - [x] 2c.4.1 Test: `test_clustering_exp1_only`
  - [x] 2c.4.2 Test: `test_clustering_exp2_only`
  - [x] 2c.4.3 Test: `test_clustering_both_experiments`
  - [x] 2c.4.4 Test: `test_dendrogram_output_matches_exp_name`
  - [x] 2c.4.5 Test: `test_heatmap_output_matches_exp_name`
  - [x] 2c.4.6 Test: `test_metadata_includes_clustering_info_per_experiment`

## 2d. Summary Visualization Embedding

- [x] 2d.1 Update `to_markdown()` to embed visualizations:
  - [x] 2d.1.1 Embed correlation summary plot
  - [x] 2d.1.2 Embed top 3 joint plots (by |r|)
  - [x] 2d.1.3 Embed `exp1_trait_clustering_dendrogram.png` (if exp1 clustered)
  - [x] 2d.1.4 Embed `exp1_trait_cluster_heatmap.png` (if exp1 clustered)
  - [x] 2d.1.5 Embed `exp2_trait_clustering_dendrogram.png` (if exp2 clustered)
  - [x] 2d.1.6 Embed `exp2_trait_cluster_heatmap.png` (if exp2 clustered)
  - [x] 2d.1.7 Embed `cross_platform_representative_heatmap.png` (if any clustering)
- [x] 2d.2 Run all visualization tests, confirm pass (Green phase complete)

## 3. Integration with run-pipelines

- [x] 3.1 Update `src/sleap_roots_analyze/pipeline_runner.py`:
  - [x] 3.1.1 Import `CrossPlatformSummaryGenerator`
  - [x] 3.1.2 After cross-platform pipelines complete, call summary generator
  - [x] 3.1.3 Append cross-platform detailed summary to output
- [x] 3.2 Test integration with `uv run sleap-roots-analyze run-all --cross-only`
- [x] 3.3 Verify summary contains new detailed cross-platform section

## 4. Create Claude Command

- [x] 4.1 Create `.claude/commands/cross-platform-summary.md`:
  - [x] 4.1.1 Define command purpose and arguments
  - [x] 4.1.2 Document usage with `$ARGUMENTS` placeholder
  - [x] 4.1.3 Include instructions for reading and validating results
  - [x] 4.1.4 Specify output format requirements
- [x] 4.2 Test command manually with `/cross-platform-summary pipeline_runs/<run_dir>`

## 5. Refactor (if needed)

- [x] 5.1 Review code for DRY violations
- [x] 5.2 Extract common parsing utilities if duplicated
- [x] 5.3 Ensure all public functions have docstrings
- [x] 5.4 Run `uv run ruff check` and `uv run black`

## 6. Documentation

- [x] 6.1 Update `docs/CROSS_PLATFORM_ANALYSIS.md` with summary command usage
- [x] 6.2 Add example output to documentation
- [x] 6.3 Document validation guardrails and their purpose

## 7. Final Validation

- [x] 7.1 Run full test suite: `uv run pytest tests/test_cross_platform_summary.py -v`
- [x] 7.2 Run integration test: `uv run sleap-roots-analyze run-all --cross-only`
- [x] 7.3 Verify summary accuracy against source CSVs manually
- [x] 7.4 Run `/pre-merge-check` to ensure CI passes
