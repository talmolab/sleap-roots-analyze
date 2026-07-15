# QC Pipeline User Guide

This guide explains how to use the Quality Control (QC) pipeline for root trait data analysis.

## Quick Start

```bash
# Using a template (recommended for new users)
cp configs/templates/qc_full_pipeline_template.yaml configs/my_qc.yaml
# Edit my_qc.yaml with your dataset-specific values
sleap-roots-analyze qc configs/my_qc.yaml

# Using an example config
sleap-roots-analyze qc configs/examples/qc_turface_150genotypes.yaml
```

Outputs will be saved to: `./qc_runs/{pipeline_name}_YYYYMMDD_HHMMSS/`

### CLI Options

```bash
# Specify custom output directory
sleap-roots-analyze qc my_config.yaml -o ./my_results

# Enable verbose logging
sleap-roots-analyze qc my_config.yaml --verbose

# Dry run (validate config without running)
sleap-roots-analyze qc my_config.yaml --dry-run

# Get help
sleap-roots-analyze qc --help
```

## Configuration

### Starting Points

**For new users:**
- Start with `configs/templates/qc_full_pipeline_template.yaml` for full QC with outlier detection
- Or use `configs/templates/qc_cleanup_only_template.yaml` for data cleanup only
- See `configs/templates/README.md` for detailed guidance

**Example configs:**
Browse `configs/examples/` for real-world configurations:
- `qc_turface_150genotypes.yaml` - Full pipeline with Mahalanobis outlier detection
- `qc_root_core_edpie.yaml` - Root core processing with trait merging
- `qc_consensus_6method.yaml` - Robust outlier detection with 6 methods
- `qc_mahalanobis.yaml` - Simple single-method template
- `qc_clustering_strict.yaml` - Clustering-based outlier detection
- `qc_permissive.yaml` - Lenient thresholds for exploratory analysis

### Required Parameters

You MUST explicitly set these in your config:

**Column Mappings** (dataset-specific):
```yaml
columns:
  genotype: "geno"      # Your genotype column name (required)
  replicate: "rep"      # Optional — omit or set null if no replicate column
                        # (e.g. cylinder data). Never used in any computation.
  barcode: "Barcode"    # Sample ID column
```

`replicate` is **optional**: its values are never used in any computation and it
is not a term in the heritability model (`value ~ 1 + (1|genotype)`). Omit it (or
set `replicate: null`) for datasets with no replicate factor.

**Data Source**:
```yaml
data:
  csv_path: "path/to/your/data.csv"  # REQUIRED
  image_dir: null                     # Optional: for image linking
```

**Cleanup Thresholds**:
```yaml
cleanup:
  max_nan_fraction: 0.25      # Max NaN per sample (0.0-1.0)
  max_zeros_per_trait: 0.5    # Max zeros per trait (0.0-1.0)
  max_nans_per_trait: 0.2     # Max NaN per trait (0.0-1.0)
  min_samples_per_trait: 10   # Min samples required
```

**PCA Configuration**:
```yaml
pca:
  n_components: 0.95          # Variance explained (0.90-0.99)
  standardize: true
```

### Optional Features

**Outlier Detection** (can be empty for cleanup-only):
```yaml
outlier_detection:
  traditional_methods:
    - mahalanobis          # Fast, reliable
    - isolation_forest     # Robust to complex patterns
  clustering_methods:
    - kmeans               # Density-based detection

  mahalanobis:
    variance_threshold: 0.95
    use_chi_squared: true
    chi2_percentile: 99.0
```

**Heritability Filtering**:
```yaml
heritability:
  enabled: true           # Set to false to disable
  threshold: 0.40         # Min H² for trait retention (0.3-0.6)
  generate_diagnostics: true
```

## Pipeline Steps

The QC pipeline executes 10 steps in sequence:

1. **LoadData** - Load and validate CSV data
2. **CleanupTraits** - Remove problematic traits (high zeros/NaNs)
3. **ValidateClean** - Validate cleaned data
4. **ExploratoryAnalysis** - Generate EDA visualizations
5. **DetectOutliers** - Run configured outlier detection methods
6. **VisualizeOutliers** - Create outlier visualization plots
7. **RemoveOutliers** - Remove detected outliers
8. **StatisticalAnalysis** - Calculate ANOVA and heritability
9. **FilterHeritability** - Remove low heritability traits (if enabled)
10. **GenerateSummary** - Create comprehensive pipeline summary

## Output Files

### Data Files
- `00_data_loaded.csv` - Raw data after loading
- `01_data_traits_cleaned.csv` - After trait cleanup
- `02_data_samples_cleaned.csv` - After sample cleanup
- `07_data_outliers_removed.csv` - After outlier removal
- `09_data_high_heritability.csv` - After heritability filtering (if enabled)
- `10_final_data.csv` - Final cleaned dataset

### Analysis Files
- `08_anova_results.csv` - ANOVA F-statistics and p-values
- `08_heritability_results.csv` - Heritability estimates (H²) per trait
- `08_blup_adjusted_means.csv` - BLUP-adjusted genotype means per trait (when
  `generate_blup_table` and `calculate_heritability` are both enabled)
- `08_trait_statistics.json` - Descriptive statistics for all traits
- `10_pipeline_summary.json` - Complete pipeline execution summary

### Figures
- Trait distributions (histograms, boxplots by genotype)
- Correlation heatmaps
- Missing data patterns
- PCA biplots and scree plots
- Outlier detection visualizations
- Heritability threshold analysis (variance decomposition)

### Logs
- `01_trait_cleanup_log.json` - Trait removal details and reasons
- `02_cleanup_log.json` - Sample removal details
- `07_outlier_removal_log.json` - Outlier removal details
- `09_heritability_filter_summary.json` - Heritability filtering summary

## Validation

### Validate Configuration

```bash
sleap-roots-analyze config validate my_config.yaml
```

This checks:
- Required parameters are set
- Values are valid
- File paths exist
- Method names are correct

### View Configuration

```bash
sleap-roots-analyze config show my_config.yaml
```

## Customization Examples

### Adjust Outlier Detection Stringency

**More outliers detected** (lower threshold):
```yaml
outlier_detection:
  mahalanobis:
    chi2_percentile: 95.0  # Was 99.0
```

**Fewer outliers** (higher threshold):
```yaml
outlier_detection:
  mahalanobis:
    chi2_percentile: 99.5  # Was 99.0
```

### Change Heritability Threshold

**More permissive** (keep more traits):
```yaml
heritability:
  threshold: 0.3  # Was 0.4
```

**More stringent** (keep fewer, higher H² traits):
```yaml
heritability:
  threshold: 0.5  # Was 0.4
```

### Adjust Data Cleanup Strictness

**More permissive** (keep more samples with NaN):
```yaml
cleanup:
  max_nan_fraction: 0.3  # Allow up to 30% NaN per sample (was 0.25)
```

**More strict** (remove any sample with NaN):
```yaml
cleanup:
  max_nan_fraction: 0.0  # Remove any sample with any NaN
```

### Use Multiple Outlier Detection Methods

**Consensus approach** (require multiple methods to agree):
```yaml
outlier_detection:
  traditional_methods:
    - mahalanobis
    - isolation_forest
  clustering_methods:
    - kmeans

outlier_removal:
  strategy: "subset"
  min_methods: 2  # Require 2/3 methods to agree
```

## Technical Details

### Pipeline Infrastructure
- NetworkX-based DAG execution for automatic step ordering
- Comprehensive test coverage (1220+ tests passing)
- OmegaConf configuration validation
- Automatic code snapshot and git tracking
- Structured JSON outputs for reproducibility

### Statistical Methods
- **ANOVA**: One-way ANOVA per trait to test genotype effects
- **Heritability**: Mixed-effects model (similar to R lme4)
- **Outlier Detection**:
  - Mahalanobis distance on PCA-transformed data (chi-squared distribution)
  - Isolation Forest (tree-based anomaly detection)
  - K-Means clustering (density-based outlier scoring)

### Configuration Philosophy

The pipeline uses **explicit configuration** to prevent silent failures:
- Critical parameters must be set (validation error if missing)
- Important-but-optional features trigger warnings (e.g., empty outlier detection)
- Sensible defaults provided but validation encourages awareness
- See `docs/configuration_review.md` for details

## Advantages Over Notebooks

1. **Reproducibility**: Configuration-driven, no manual parameter changes
2. **Testability**: All pipeline steps have comprehensive unit tests
3. **Traceability**: Automatic git tracking and code snapshots
4. **Maintainability**: Modular step-based architecture
5. **Extensibility**: Easy to add new steps or modify existing ones
6. **Documentation**: Self-documenting with JSON outputs and detailed logs
7. **Scalability**: Process multiple datasets with same pipeline

## Troubleshooting

### Common Issues

**"Configuration Validation Failed"**:
- Check that all required parameters are set in your config
- See error message for specific missing parameters
- Refer to templates for correct structure

**"No outlier detection methods configured" warning**:
- This is expected for cleanup-only configs
- Add outlier detection methods if you want robust QC
- Or acknowledge the warning and proceed with cleanup only

**Pipeline runs but produces unexpected results**:
- Use `--dry-run` to validate config without running
- Check `pipeline_summary.json` for step-by-step details
- Review cleanup logs to see what was removed and why

### Getting Help

```bash
# View all available commands
sleap-roots-analyze --help

# Get help for specific command
sleap-roots-analyze qc --help
sleap-roots-analyze config --help
```

## Examples

### Example 1: Basic QC with Outlier Detection

See `configs/examples/qc_turface_150genotypes.yaml` for a complete working example that:
- Uses Mahalanobis outlier detection
- Filters traits by heritability (H² >= 0.4)
- Generates comprehensive visualizations
- Matches results from original Jupyter notebook

### Example 2: Root Core Processing

See `configs/examples/qc_root_core_edpie.yaml` for processing root core data:
- Aggregates 3 cores per plot using median
- Merges with above-ground traits before QC
- Demonstrates best practices for core-level data

### Example 3: Robust Multi-Method QC

See `configs/examples/qc_consensus_6method.yaml` for maximum robustness:
- Uses 6 different outlier detection methods
- Requires 3/6 methods to agree before removing outliers
- Best for production analyses requiring high confidence

## References

- Configuration templates: `configs/templates/README.md`
- Example configs: `configs/examples/`
- Configuration philosophy: `CLAUDE.md` (Configuration Philosophy section)
- Pipeline implementation: `src/sleap_roots_analyze/pipeline/pipelines/qc_pipeline.py`
- Test coverage: Issue #19 (comprehensive test coverage)
