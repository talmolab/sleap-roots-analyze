# Turface 150 Genotypes QC Pipeline

This document describes how to run the QC pipeline on the Turface 150 genotypes dataset, replicating the analysis from `trait_qc_150_genotypes_turface_20251024.ipynb`.

## Quick Start

```bash
# Run the QC pipeline
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
```

Outputs will be saved to: `./qc_runs/turface_150genotypes_qc_YYYYMMDD_HHMMSS/`

### Options

```bash
# Specify custom output directory
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml -o ./my_results

# Enable verbose logging
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml --verbose

# Dry run (validate config without running)
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml --dry-run

# Get help
sleap-roots-analyze qc --help
```

### Legacy Script (Deprecated)

The `python run_turface_qc.py` script still works but is deprecated and will be removed in v0.1.0.

## Configuration

The pipeline configuration is in `configs/qc_turface_150genotypes.yaml`.

### Key Settings

**Data Configuration:**
- Input CSV: Turface 150 genotypes trait data
- Column names: barcode, geno (genotype), rep (replicate)
- Additional metadata columns excluded from analysis

**Data Cleanup:**
- Max NaN fraction per sample: 0.0 (strict - removes any sample with NaN)
- Max zeros per trait: 0.5 (removes traits with >50% zeros)
- Max NaNs per trait: 0.2 (removes traits with >20% NaNs)
- Min samples per trait: 10

**Outlier Detection:**
- Method: Mahalanobis distance only
- Variance threshold: 0.95 (keep PCA components explaining 95% variance)
- Chi-squared threshold: 99th percentile
- Strategy: Single method removal

**Heritability Filtering:**
- Enabled: Yes
- Threshold: 0.40 (keep traits with H² >= 0.4)

## Results Summary

### Notebook vs Pipeline Comparison

| Metric | Notebook | Pipeline | Match |
|--------|----------|----------|-------|
| Original samples | 926 | 926 | ✓ |
| Final samples | 890 | 890 | ✓ |
| Samples removed | 36 | 36 | ✓ |
| Original traits | 20 | 20 | ✓ |
| Final traits | 13 | 13 | ✓ |
| Traits removed | 7 | 7 | ✓ |
| Outliers removed (Mahalanobis) | 35 | 35 | ✓ |
| H² threshold | 0.40 | 0.40 | ✓ |

### Pipeline Steps

The QC pipeline executes 10 steps in sequence:

1. **LoadData** - Load and validate CSV data
2. **CleanupTraits** - Remove problematic traits (high zeros/NaNs)
3. **ValidateClean** - Validate cleaned data
4. **ExploratoryAnalysis** - Generate EDA visualizations
5. **DetectOutliers** - Run Mahalanobis outlier detection
6. **VisualizeOutliers** - Create outlier visualization plots
7. **RemoveOutliers** - Remove detected outliers
8. **StatisticalAnalysis** - Calculate ANOVA and heritability
9. **FilterHeritability** - Remove low heritability traits
10. **GenerateSummary** - Create comprehensive pipeline summary

### Output Files

**Data Files:**
- `00_data_loaded.csv` - Raw data after loading
- `01_data_traits_cleaned.csv` - After trait cleanup
- `02_data_samples_cleaned.csv` - After sample cleanup
- `07_data_outliers_removed.csv` - After outlier removal
- `09_data_high_heritability.csv` - After heritability filtering
- `10_final_data.csv` - Final cleaned dataset

**Analysis Files:**
- `08_anova_results.csv` - ANOVA F-statistics and p-values
- `08_heritability_results.csv` - Heritability estimates for each trait
- `08_trait_statistics.json` - Basic statistics for all traits
- `10_pipeline_summary.json` - Complete pipeline summary

**Figures:**
- Trait distributions (histograms, boxplots)
- Correlation heatmaps
- Missing data patterns
- Outlier detection plots
- Heritability threshold analysis

**Logs:**
- `01_trait_cleanup_log.json` - Trait removal details
- `02_cleanup_log.json` - Sample removal details
- `07_outlier_removal_log.json` - Outlier removal details
- `09_heritability_filter_summary.json` - Heritability filtering summary

## CLI Commands

### Run QC Pipeline

```bash
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml [OPTIONS]
```

**Options:**
- `-o, --output-dir PATH` - Output directory (default: ./qc_runs)
- `-v, --verbose` - Enable DEBUG logging
- `-q, --quiet` - Only show warnings and errors
- `--log-file FILE` - Save logs to file
- `--dry-run` - Validate config without running

### Validate Configuration

```bash
sleap-roots-analyze config validate configs/qc_turface_150genotypes.yaml
```

### View Configuration

```bash
sleap-roots-analyze config show configs/qc_turface_150genotypes.yaml
```

## Customization

To modify the pipeline behavior, edit `configs/qc_turface_150genotypes.yaml`:

**Change outlier detection stringency:**
```yaml
outlier_detection:
  mahalanobis:
    chi2_percentile: 95.0  # More outliers (was 99.0)
```

**Change heritability threshold:**
```yaml
heritability:
  threshold: 0.5  # More stringent (was 0.4)
```

**Change NaN tolerance:**
```yaml
cleanup:
  max_nan_fraction: 0.1  # Allow up to 10% NaN per sample
```

## Technical Details

**Pipeline Infrastructure:**
- Uses NetworkX-based DAG execution
- All 10 steps have comprehensive unit tests (1006 tests passing)
- Configuration validation with OmegaConf
- Automatic code snapshot and git tracking
- Structured JSON outputs for reproducibility

**Statistical Methods:**
- ANOVA: One-way ANOVA per trait
- Heritability: Mixed-effects model (similar to R lme4)
- Outlier detection: Mahalanobis distance on PCA-transformed data (chi-squared distribution)

## Advantages Over Notebook

1. **Reproducibility**: Configuration-driven, no manual parameter changes
2. **Testability**: All pipeline steps have unit tests
3. **Traceability**: Automatic git tracking and code snapshots
4. **Maintainability**: Modular step-based architecture
5. **Extensibility**: Easy to add new steps or modify existing ones
6. **Documentation**: Self-documenting with JSON outputs and logs

## References

- Original notebook: `trait_qc_150_genotypes_turface_20251024.ipynb`
- Configuration: `configs/qc_turface_150genotypes.yaml`
- Pipeline implementation: `src/sleap_roots_analyze/pipeline/pipelines/qc_pipeline.py`
- Test coverage: Issue #19 (PR #37)
