# QC Pipeline Configuration Templates

Template configuration files to help you get started with the QC pipeline.

## Quick Start

1. **Choose a template** based on your needs:
   - `qc_cleanup_only_template.yaml` - Data cleanup without outlier detection
   - `qc_full_pipeline_template.yaml` - Full QC with outlier detection

2. **Copy and customize**:
   ```bash
   cp configs/templates/qc_full_pipeline_template.yaml configs/my_analysis.yaml
   # Edit my_analysis.yaml with your dataset-specific values
   ```

3. **Run the pipeline**:
   ```bash
   sleap-roots-analyze qc configs/my_analysis.yaml
   ```

## Required Parameters

You MUST set these parameters in your config:

- **`columns.genotype`** - Your genotype column name (e.g., "geno", "accession")
- **`columns.replicate`** - Your replicate column name (e.g., "rep", "block")
- **`data.csv_path`** - Path to your trait CSV file
- **`cleanup.max_nan_fraction`** - Max NaN per sample (typical: 0.25)
- **`cleanup.max_zeros_per_trait`** - Max zeros per trait (typical: 0.5)
- **`cleanup.low_variance_threshold`** - Min trait variance (typical: 1e-10)
- **`pca.variance_threshold`** - PCA variance threshold (typical: 0.95)

## Conditionally Required

These parameters are required only if certain features are enabled:

- **`heritability.threshold`** - Required if `heritability.enabled: true` (typical: 0.3-0.6)
- **`outlier_removal.strategy`** - Required if outlier detection methods configured
- **`root_core.sources[].aggregation_method`** - Required if processing root cores (use "median")

## Optional But Important

- **`outlier_detection.traditional_methods`** - Can be empty for cleanup-only pipeline
  - **WARNING**: You will be warned if this is empty (to ensure conscious choice)
  - This ensures you make a deliberate decision about whether to include outlier detection

- **`data.group_by`** - Column name to group data by for separate per-group analyses
  - Common use: "plant_age_days" for multi-timepoint experiments
  - Creates independent output directories and statistics for each group
  - See "Grouped Analysis by Timepoint" section below for details

## Templates Explained

### Cleanup-Only Template

Use `qc_cleanup_only_template.yaml` when you only need data cleanup (removing NaNs, zeros, low-variance traits) without outlier detection.

**Best for**:
- Quick exploratory analysis
- Well-characterized datasets without outliers
- When you want to handle outliers manually

**Outlier detection**: Disabled (empty `traditional_methods` and `clustering_methods`)

### Full Pipeline Template

Use `qc_full_pipeline_template.yaml` for complete QC including outlier detection using multiple methods.

**Best for**:
- Production analyses requiring robust QC
- Datasets with potential outliers
- Maximizing data quality before downstream analysis

**Outlier detection**: Enabled with multiple methods (Mahalanobis, Isolation Forest, DBSCAN)

## Real-World Configs

The parent `configs/` directory contains several real-world configurations you can learn from and reuse:

```bash
# Browse available configs
ls ../configs/*.yaml

# View a specific config
cat ../configs/qc_turface_150genotypes.yaml
```

These configs show real parameter choices from actual analyses. Feel free to use them directly or as starting points for your own experiments.

## Common Workflows

### Workflow 1: Cleanup Only

```bash
# 1. Copy template
cp configs/templates/qc_cleanup_only_template.yaml configs/my_cleanup.yaml

# 2. Edit required fields:
#    - columns.genotype
#    - columns.replicate
#    - data.csv_path
#    - cleanup thresholds

# 3. Run
sleap-roots-analyze qc configs/my_cleanup.yaml
```

You will see a warning that outlier detection is disabled. This is normal and expected for cleanup-only pipelines.

### Workflow 2: Full QC with Outlier Detection

```bash
# 1. Copy template
cp configs/templates/qc_full_pipeline_template.yaml configs/my_qc.yaml

# 2. Edit required fields (same as cleanup-only, plus):
#    - outlier_removal.strategy
#    - heritability.threshold (if enabled)

# 3. Run
sleap-roots-analyze qc configs/my_qc.yaml
```

## Parameter Recommendations

### Cleanup Thresholds

- **`max_nan_fraction: 0.25`** - Removes samples with >25% missing data
  - Lower (0.10-0.20) = stricter, keeps only very complete samples
  - Higher (0.30-0.40) = permissive, allows more missing data

- **`max_zeros_per_trait: 0.5`** - Removes traits with >50% zeros
  - Lower (0.30-0.40) = stricter, removes traits with many zeros
  - Higher (0.60-0.70) = permissive, allows traits with frequent zeros

- **`low_variance_threshold: 1e-10`** - Removes near-constant traits
  - Standard value works for most cases
  - Increase (1e-8, 1e-6) if you want to be more aggressive

### Heritability Threshold

- **0.30** - Permissive (retains more traits)
- **0.40** - Moderate (balanced)
- **0.50-0.60** - Stringent (only highly heritable traits)

Choose based on your downstream analysis requirements. Higher thresholds give you fewer but more reliable traits.

### PCA Variance Threshold

- **0.90** - Fewer components (faster, less complete)
- **0.95** - Standard (good balance)
- **0.99** - More components (slower, more complete)

## Validation Warnings

When you run the pipeline, you may see validation warnings:

### Warning: No Outlier Detection

```
UserWarning: No outlier detection methods configured...
  This is valid if you only want data cleanup (NaN/zero removal).
  Consider adding outlier detection for robust QC:
    - traditional_methods: ['mahalanobis_pca', 'isolation_forest']
```

**This is normal** for cleanup-only pipelines. The warning ensures you're making a conscious choice not to include outlier detection.

### Error: Missing Required Parameters

```
ValueError: Configuration Validation Failed
Critical parameters must be explicitly set to avoid silent failures.

cleanup.max_nan_fraction must be explicitly set
  Recommended: 0.25 (removes samples with >25% missing data)
  Range: 0.0-1.0 (lower = stricter)
```

This error means you forgot to set a required parameter. Edit your config file to set the missing parameters.

## Getting Help

If you encounter issues:

1. Check that all REQUIRED parameters are set
2. Review the examples in `configs/` directory
3. Read the error messages carefully - they include recommended values
4. Consult the full documentation at `docs/configuration_guide.md`

## Grouped Analysis by Timepoint

For multi-timepoint experiments (e.g., plants measured at 7, 14, 21 days), you can use `data.group_by` to analyze each timepoint independently.

### Why Use Grouped Analysis?

Combining data across timepoints can **confound temporal and genetic effects**, making heritability estimates invalid. Grouping ensures:
- Independent statistics per timepoint (ANOVA, heritability)
- Separate PCA analyses (PC loadings differ by developmental stage)
- Clean comparison of genetic effects within homogeneous groups

### Configuration

```yaml
data:
  csv_path: "multi_timepoint_data.csv"
  group_by: "plant_age_days"  # Column containing timepoint values
```

### CLI Usage

```bash
# Group by config value
sleap-roots-analyze qc my_config.yaml

# Override with CLI flag
sleap-roots-analyze qc my_config.yaml --group-by plant_age_days

# Run-all with grouping (applies to all pipelines in manifest)
sleap-roots-analyze run-all manifest.yaml --group-by plant_age_days
```

### run-all with group_by: automatic viz fan-out

When a QC config uses `group_by`, `run-all` automatically fans out the downstream viz pipeline
to run once per QC group output. Each group gets its own viz subdirectory and updated config:

```
run_dir/
├── qc/
│   ├── plant_age_days_7_20260217_143052/    # QC output for day 7
│   ├── plant_age_days_14_20260217_143108/   # QC output for day 14
│   └── plant_age_days_21_20260217_143124/   # QC output for day 21
└── viz/
    ├── plant_age_days_7/
    │   ├── _updated_my_viz_config.yaml      # csv_path → day 7 10_final_data.csv
    │   └── viz_output_20260217_144000/
    ├── plant_age_days_14/
    │   └── ...
    └── plant_age_days_21/
        └── ...
```

No manual workaround is needed. `run-all` handles the fan-out natively.

### Output Structure

Each group gets an independent output directory:

```
qc_runs/
├── plant_age_days_7_20260216_143052/
│   ├── config.yaml
│   ├── pipeline_summary.json
│   ├── 10_final_data.csv          # Only day 7 samples
│   ├── 08_heritability_results.csv  # H² for day 7
│   └── figures/
├── plant_age_days_14_20260216_143108/
│   └── ...
└── plant_age_days_21_20260216_143124/
    └── ...
```

### Group Validation

Groups with fewer than `cleanup.min_samples_per_trait` samples are automatically skipped with a warning:

```
WARNING: Skipping group plant_age_days=28 (3 samples < 10 minimum)
```

### When to Use Grouping

✅ **Use grouping when:**
- Data has multiple timepoints/developmental stages
- Samples were collected at different sites/batches
- You need per-group heritability estimates

❌ **Don't use grouping when:**
- All samples are from the same timepoint/condition
- You intentionally want to analyze temporal trends
- Groups would have insufficient samples

## Tips

1. **Start with a template** - Don't write configs from scratch
2. **Document your choices** - Add comments explaining why you chose specific values
3. **Test with small datasets** - Validate your config before running on full data
4. **Review outputs** - Check the generated plots and summaries to ensure QC worked as expected
5. **Learn from examples** - The configs in `configs/` show real-world usage patterns
6. **Use grouping for multi-timepoint data** - Prevents confounding temporal and genetic effects
