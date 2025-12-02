# Root Core Data Merge Guide

This guide explains how to merge root core trait data with above-ground phenotype data using the QC pipeline's merge functionality (Steps 11-12).

## Overview

The root core QC pipeline can automatically merge root trait data (depth profiles from core sampling) with above-ground trait data (e.g., biomass, height, flowering time). This creates a unified dataset for downstream analysis.

## Quick Start

### 1. Prepare Your Data

You need two datasets:
1. **Root core data**: Processed through Steps 0a-10 (core loading, QC, aggregation)
2. **Above-ground CSV**: A separate CSV file with phenotype data

### 2. Configure Merge Settings

Add `merge_traits` configuration to your YAML config:

```yaml
root_core:
  # ... Steps 0a-0e configuration ...

  merge_traits:
    above_ground_csv: "path/to/above_ground_phenotypes.csv"
    join_keys: ["Rep", "geno"]  # Columns to merge on (MUST exist in both datasets)
    join_type: "inner"          # Options: inner, left, right, outer
    duplicate_strategy: "fail"  # Options: fail, skip, suffix
    output_path: "merged_traits.csv"
```

### 3. Run Pipeline

```bash
uv run sleap-roots-analyze qc configs/your_config.yaml -o ./output
```

The pipeline will:
- Load and validate your above-ground CSV (Step 11)
- Merge root + above-ground data (Step 12)
- Save merged CSV and metadata JSON

## Critical Configuration: Join Keys

### What Are Join Keys?

Join keys are the columns used to match rows between your root and above-ground datasets. **Both datasets must have these columns with matching values.**

### Common Scenarios

#### Scenario 1: Replicate-Level Above-Ground Data (Most Common)

**When to use**: Above-ground measurements are taken once per replicate-genotype combination (no plot-level data).

**Data structure**:
- Root cores: 3 cores per plot → aggregated to 60 plots (20 genotypes × 3 reps)
- Above-ground: 1 measurement per rep-genotype (20 genotypes × 3 reps = 60 rows)

**Configuration**:
```yaml
join_keys: ["Rep", "geno"]
```

**Result**: 60 merged rows. The `Plot` column from root data is preserved.

**Example**:
```
Root data (after Step 0d):
  Plot  Rep  geno      RootDW_15cm  RootDW_45cm
  1     1    GH_7386   2.5          1.2
  2     1    GH_7420   2.1          0.9
  ...   ...  ...       ...          ...
  60    3    Control   2.8          1.5

Above-ground data:
  Rep  geno      Height_cm  Biomass_g
  1    GH_7386   35.2       22.5
  1    GH_7420   32.1       20.1
  ...  ...       ...        ...
  3    Control   38.5       25.2

Merged (60 rows):
  Plot  Rep  geno      RootDW_15cm  RootDW_45cm  Height_cm  Biomass_g
  1     1    GH_7386   2.5          1.2          35.2       22.5
  2     1    GH_7420   2.1          0.9          32.1       20.1
  ...
```

#### Scenario 2: Plot-Level Above-Ground Data

**When to use**: Above-ground measurements are taken per plot (same granularity as root cores).

**Data structure**:
- Root cores: 60 plots after aggregation
- Above-ground: 60 plots (one measurement per plot)

**Configuration**:
```yaml
join_keys: ["Plot", "Rep", "geno"]
```

**Result**: 60 merged rows with exact Plot-Rep-Geno matches.

#### Scenario 3: Genotype-Only Data

**When to use**: Above-ground data is genotype-level averages (no replicates).

**Data structure**:
- Root cores: 60 plots (20 genotypes × 3 reps)
- Above-ground: 20 genotypes (averaged across reps)

**Configuration**:
```yaml
join_keys: ["geno"]
```

**Result**: 60 merged rows (each root plot matched to its genotype average). **WARNING**: This creates redundancy - all 3 reps of a genotype get the same above-ground values.

### Validation Rules

The pipeline validates that:
1. ✅ All `join_keys` columns exist in the above-ground CSV
2. ✅ Both datasets have **exactly one row per join key combination** (no duplicates)
3. ✅ Join key values match between datasets (or are compatible with join type)

**Common error**:
```
ValueError: Missing join keys in above-ground CSV: ['Plot'].
Available columns: ['Rep', 'geno', 'Height_cm', 'Biomass_g']
```

**Solution**: Change `join_keys` to only include columns present in both datasets (e.g., `["Rep", "geno"]`).

## Join Type Options

### `inner` (default, recommended)

**Behavior**: Keep only samples present in **both** datasets.

**Use when**: You want a clean dataset with complete data for all traits.

**Example**: If a genotype is missing from above-ground data, it's excluded from the merged output.

### `left`

**Behavior**: Keep all root samples, even if above-ground data is missing.

**Use when**: Root data is primary and you want to preserve all root samples.

**Result**: Above-ground columns will have `NaN` for unmatched root samples.

### `right`

**Behavior**: Keep all above-ground samples, even if root data is missing.

**Use when**: Above-ground data is primary and you want to preserve all above-ground samples.

**Result**: Root columns will have `NaN` for unmatched above-ground samples.

### `outer`

**Behavior**: Keep all samples from **both** datasets.

**Use when**: You want maximum data retention and will handle missing values downstream.

**Result**: Many `NaN` values in both root and above-ground columns for unmatched samples.

## Duplicate Column Handling

### What Are Duplicate Columns?

If both datasets have a column with the same name (besides join keys), it's a "duplicate." For example, both might have a `Biomass` column.

### Strategy Options

#### `fail` (default, recommended)

**Behavior**: Raise an error if duplicates are found.

**Use when**: You want to catch naming conflicts and fix them manually.

**Example error**:
```
ValueError: Duplicate columns found between root and above-ground traits: {'Biomass'}.
Use duplicate_strategy='skip' or 'suffix' to handle this.
```

#### `skip`

**Behavior**: Keep the root version and drop the above-ground version.

**Use when**: You trust root data more and want to ignore above-ground duplicates.

**Result**: Only `RootDW_15cm` is kept; `RootDW_15cm` from above-ground is dropped.

#### `suffix`

**Behavior**: Rename duplicate columns with `_root` and `_ag` suffixes.

**Use when**: You want to keep both versions for comparison.

**Result**: `Biomass_root` and `Biomass_ag` both appear in the merged dataset.

## Core Aggregation Method

### What Is Aggregation?

Root core data typically has **3 cores per plot**. Before QC and merging, these 3 cores are aggregated to a single plot-level value in Step 0d.

### Why Median is Recommended Over Statistical Outlier Detection

**CRITICAL: Statistical outlier detection does NOT work at the core level.**

Root core experiments have a fundamental limitation: each plot has only ~3 cores, but statistical methods like Mahalanobis distance require 30+ samples for reliable detection. This means:

1. **Statistical methods fail with small samples**:
   - Mahalanobis distance needs stable covariance estimation (requires large n)
   - Chi-squared threshold assumes asymptotic distribution (violated with n=3)
   - PCA compression loses outlier signal with few features (2-12 depths)

2. **Median aggregation solves the problem**:
   - Robust to outliers without requiring detection (breaks down only when >50% are outliers)
   - Handles typos, miscounts, and measurement errors automatically
   - Works with any sample size (including n=3)
   - Simpler and more reliable than statistical methods with insufficient samples

3. **Trait-level QC still catches outliers**:
   - After aggregation, you have 60+ plots (sufficient for Mahalanobis)
   - Step 5 (DetectOutliers) uses statistical methods on plot-level data
   - This catches biological outliers (e.g., genotypes with extreme phenotypes)

**RECOMMENDATION: Use `median` aggregation and disable core-level QC (`core_qc.enabled: false`).**

### Choosing Mean vs Median

#### Use `median` (recommended default):
- ✅ Robust to core-level outliers without requiring statistical detection
- ✅ Any dataset where you haven't manually inspected all cores
- ✅ Data with potential typos, miscounts, or measurement errors
- ✅ Within-plot variance CV > 0.3
- ✅ You want reliable results without manual data curation

#### Use `mean` (use with caution):
- ⚠️ Only if within-plot variance is very low (CV < 0.3)
- ⚠️ Only after manual inspection confirms no outlier cores
- ⚠️ Only if mean-median difference is negligible (< 5% of typical values)
- ⚠️ You need maximum precision for subtle genetic effects

### Example Data Analysis (EDPIE Biomass)

```python
import pandas as pd
import numpy as np

df = pd.read_csv("root_biomass_cores.csv")  # Before aggregation

# Calculate within-plot variance
plot_stats = df.groupby('Plot')['0-30'].agg(['mean', 'median', 'std'])
cv = plot_stats['std'] / plot_stats['mean']
print(f"Mean CV: {cv.mean():.3f}")  # 0.317 (moderate)

# Compare aggregation methods
mean_agg = df.groupby('Plot')['0-30'].mean()
median_agg = df.groupby('Plot')['0-30'].median()
diff = abs(mean_agg - median_agg).mean()
print(f"Mean-Median Diff: {diff:.3f}g")  # 0.034g (< 10% of values)

# Recommendation: MEAN is appropriate
# - CV ~0.3 (moderate, not high)
# - Mean-median diff is small
# - No extreme outliers detected
```

**Results for EDPIE data**:
- Within-plot CV: 0.3-0.4 (moderate variability)
- Mean vs median difference: < 0.2g (< 10% of typical values)
- IQR outliers: 0 plots
- **Recommendation**: MEDIAN is recommended despite low variance, because:
  - Provides robustness to undetected core-level errors (typos, miscounts)
  - Statistical outlier detection at core level is unreliable (n=3, need 30+)
  - Mean-median difference is small enough that median doesn't lose precision
  - Median is the safer default unless you've manually inspected all cores

### Configuration

```yaml
root_core:
  sources:
    - csv_path: "biomass.csv"
      aggregation_method: "median"  # Recommended: robust to outliers
      # ... other settings ...
```

## Troubleshooting

### Error: Missing join keys in above-ground CSV

**Problem**: Above-ground CSV doesn't have a column specified in `join_keys`.

**Solution**:
1. Check your above-ground CSV columns: `pd.read_csv("above_ground.csv").columns`
2. Update `join_keys` to only include shared columns
3. Most common fix: Change from `["Plot", "Rep", "geno"]` to `["Rep", "geno"]`

### Error: Duplicate samples found in above-ground CSV

**Problem**: Above-ground CSV has multiple rows with the same join key combination.

**Example**: Two rows with `Rep=1, geno=GH_7386`

**Solution**:
1. Check for duplicates: `df.duplicated(subset=['Rep', 'geno'], keep=False)`
2. Decide how to handle:
   - Average duplicates: `df.groupby(['Rep', 'geno']).mean()`
   - Remove duplicates: `df.drop_duplicates(subset=['Rep', 'geno'])`
   - Investigate data quality issues

### Merged dataset has wrong number of rows

**Problem**: Expected 60 rows but got 45 (with `inner` join).

**Cause**: Some samples in root data don't have matches in above-ground data.

**Solution**:
1. Check which samples are missing:
   ```python
   root_keys = set(zip(root_df['Rep'], root_df['geno']))
   ag_keys = set(zip(ag_df['Rep'], ag_df['geno']))
   missing = root_keys - ag_keys
   ```
2. Options:
   - Add missing samples to above-ground CSV
   - Use `join_type: "left"` to keep all root samples (NaN for missing above-ground)
   - Accept reduced sample size with `inner` join

### NaN values in merged dataset

**Expected**: With `left`, `right`, or `outer` joins, unmatched samples get NaN.

**Check**:
```python
merged = pd.read_csv("merged_output.csv")
print(merged.isna().sum())  # Count NaN per column
```

**Handle**:
- Remove samples with too many NaN: `merged.dropna(thresh=n_required_cols)`
- Impute values: Use genotype means or other imputation methods
- Use `inner` join to avoid NaN entirely

## Complete Example

### Directory Structure
```
project/
├── data/
│   ├── root_biomass_dw.csv           # Raw core data
│   ├── root_counting.csv              # Raw counting data
│   └── above_ground_phenotypes.csv    # Above-ground traits
├── configs/
│   └── qc_root_core_merge.yaml       # Pipeline config
└── output/
    ├── Field_2024_final.csv          # Merged output
    └── Field_2024_final_metadata.json # Merge metadata
```

### Configuration File

```yaml
# configs/qc_root_core_merge.yaml
pipeline_name: "Root Core + Above-Ground Merge"
version: "1.0"

root_core:
  sources:
    - csv_path: "data/root_biomass_dw.csv"
      data_type: "biomass"
      depth_column_prefix: "RootDW"
      value_column_name: "Root_DW_g"
      aggregation_method: "median"  # Recommended: robust to outliers
      genotype_column: "salk_geno"
      depth_mapping:
        "0-30": 15.0
        "30-60": 45.0

  core_qc:
    enabled: false  # Recommended: use median aggregation instead
    max_missing_proportion: 0.5  # If enabled, only flag missing data
    remove_outliers: true

  merge_traits:
    above_ground_csv: "data/above_ground_phenotypes.csv"
    join_keys: ["Rep", "geno"]  # Rep-level merge
    join_type: "inner"           # Only keep matching samples
    duplicate_strategy: "fail"   # Error if duplicate columns
    output_path: "output/Field_2024_final.csv"

# Standard QC pipeline config (Steps 1-10)
columns:
  barcode: "Barcode"
  genotype: "geno"
  replicate: "Rep"

cleanup:
  max_zeros_per_trait: 0.3
  max_nans_per_trait: 0.3
  max_nan_fraction: 0.1

outlier_detection:
  traditional_methods: [mahalanobis]

outlier_removal:
  strategy: "single"
  method: "mahalanobis"

heritability:
  enabled: true
  threshold: 0.3

visualization:
  create_pca_plots: true
  create_outlier_plots: true
```

### Running the Pipeline

```bash
# Activate environment
uv sync --group dev

# Run pipeline
uv run sleap-roots-analyze qc configs/qc_root_core_merge.yaml -o ./output

# Check outputs
ls output/
# → Field_2024_final.csv (merged data)
# → Field_2024_final_metadata.json (merge statistics)
# → plots/ (PCA, outlier plots, etc.)
```

### Inspecting Outputs

```python
import pandas as pd
import json

# Load merged data
merged = pd.read_csv("output/Field_2024_final.csv")
print(f"Merged dataset: {len(merged)} rows × {len(merged.columns)} columns")

# Load metadata
with open("output/Field_2024_final_metadata.json") as f:
    metadata = json.load(f)

print(f"Root traits: {metadata['num_root_traits']}")
print(f"Above-ground traits: {metadata['num_above_ground_traits']}")
print(f"Total samples: {metadata['num_samples']}")
print(f"Join type: {metadata['merge_type']}")

# Check for any issues
print("\nMissing values per column:")
print(merged.isna().sum()[merged.isna().sum() > 0])
```

## Best Practices

1. **Validate your data first**
   - Check that join key columns exist in both datasets
   - Verify no duplicate samples (one row per join key combo)
   - Inspect data types (numeric columns should be numeric)

2. **Start with `inner` join and `fail` duplicate strategy**
   - Catches data quality issues early
   - Switch to other strategies only if needed

3. **Document your choices**
   - Add comments to config explaining why you chose specific join keys
   - Note any data quality issues in your analysis notebook

4. **Analyze aggregation method before choosing**
   - Calculate within-plot CV to understand variance
   - Compare mean vs median to see if they differ substantially
   - Default to `mean` unless you have specific reasons for `median`

5. **Inspect merge metadata**
   - Check `num_samples` matches expectations
   - Review `duplicate_columns_found` to catch naming conflicts
   - Verify trait counts are correct

## See Also

- [QC Pipeline Documentation](../README.md)
- [Configuration Reference](../openspec/changes/add-root-core-qc-pipeline/proposal.md)
- [Test Examples](../tests/test_step_merge_all_traits.py)
