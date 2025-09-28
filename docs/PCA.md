# PCA Analysis Documentation

This document provides a comprehensive guide to the PCA analysis capabilities in the `sleap-roots-analyze` package, with mathematical background and practical usage examples.

## Table of Contents
- [Overview](#overview)
- [Mathematical Background](#mathematical-background)
- [API Walkthrough](#api-walkthrough)
- [Usage Examples](#usage-examples)
- [Artifact Descriptions](#artifact-descriptions)

## Overview

The PCA module provides a complete pipeline for dimensionality reduction and feature importance analysis of root trait data. The main entry points are:

- `perform_pca_analysis()` - Core PCA computation with automatic component selection
- `run_pca_and_export_artifacts()` - Comprehensive analysis with CSV exports
- `calculate_pca_metrics()` - Detailed metrics calculation
- `build_feature_metrics_df()` - Tidy DataFrame for downstream analysis

## Standard PCA Result Keys

The `perform_pca_analysis()` function returns a dictionary with standardized keys. All PCA-related functions in the package expect and use these exact keys:

### Core PCA Outputs
- `pca`: Fitted sklearn PCA object
- `transformed_data`: PC scores array (n_samples × n_components)
- `loadings`: Eigenvectors/loadings matrix (n_features × n_components)
- `eigenvalues`: Explained variance per PC (array of length n_components)

### Variance Explained Metrics
- `explained_variance_ratio`: Fraction of variance per PC (sums to ≤ 1)
- `cumulative_variance_ratio`: Cumulative fraction explained
- `total_variance_explained`: Total fraction explained by selected components

### Feature-Level Metrics
- `feature_names`: List of feature names
- `feature_contributions`: DataFrame with per-feature importance metrics
  - Index: feature names
  - Columns: 
    - `total_contribution`: Total variance contribution across selected PCs
    - `fractional_contribution`: Normalized contribution [0,1] that sums to 1
- `explained_variance_per_feature`: Variance explained per original feature
- `explained_variance_ratio_per_feature`: Fraction explained per feature [0,1]

### Processing Metadata
- `n_components_selected`: Number of components selected/used
- `scaler`: StandardScaler object if standardize=True, else None
- `data_processed`: Preprocessed data array (standardized or cleaned)
- `feature_variances`: Per-feature variances of fitted data
- `feature_variance_ddof`: Degrees of freedom used for variance

## Mathematical Background

### PCA Basics

Given a trait matrix $X \in \mathbb{R}^{n \times p}$ with $n$ samples and $p$ features:

After optional standardization, PCA diagonalizes the empirical covariance matrix:

$$\hat{\Sigma} = \frac{1}{n-1} X^\top X = V \Lambda V^\top$$

Where:
- $V \in \mathbb{R}^{p \times m}$ are the loadings (unit eigenvectors, orthonormal)
- $\Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_m)$ are eigenvalues (variances of PC scores)
- $T = X V \in \mathbb{R}^{n \times m}$ are the transformed PC scores

In code:
- `pca_results["loadings"]` → $V$
- `pca_results["eigenvalues"]` → $\lambda_k$
- `pca_results["transformed_data"]` → $T$
- `pca_results["explained_variance_ratio"]` → $\lambda_k / \sum_i \lambda_i$

### Per-Trait Variance Contribution

Each original feature $j$ contributes to PC $k$ proportionally to the squared loading, weighted by the eigenvalue:

$$\text{Contribution}_{j,k} = \lambda_k \cdot v_{jk}^2$$

This has units of variance (same as $\lambda_k$).

In code:
```python
trait_pc_variance_contrib = (loadings ** 2) * eigenvalues
```

### Total Contribution Per Trait

Summing across selected PCs:

$$\text{TotalContribution}_j = \sum_{k=1}^m \lambda_k \cdot v_{jk}^2$$

This tells you **how much variance in the selected PCA subspace** is attributable to each original feature.

### Fractional Contribution

To allow direct comparison between traits, we normalize:

$$\text{FractionalContribution}_j = \frac{\text{TotalContribution}_j}{\sum_{j=1}^p \text{TotalContribution}_j} = \frac{\sum_k \lambda_k v_{jk}^2}{\sum_k \lambda_k}$$

Properties:
- Unitless (sums to 1)
- Column: `trait_fractional_contrib`
- Interpretation: The share of explained variance attributable to trait $j$

## API Walkthrough

### `perform_pca_analysis()`

Core PCA computation with automatic component selection:

```python
from sleap_roots_analyze.pca import perform_pca_analysis

result = perform_pca_analysis(
    data=df_traits,                    # DataFrame or array
    standardize=True,                   # Standardize features
    explained_variance_threshold=0.95,  # Auto-select components
    n_components=None,                  # Or specify directly
    random_state=42
)
```

Returns dictionary with:
- `pca`: Fitted sklearn PCA object
- `transformed_data`: PC scores (n_samples × n_components)
- `loadings`: Eigenvectors (n_features × n_components)
- `eigenvalues`: Explained variance per component
- `explained_variance_ratio`: Fraction of variance per PC
- `cumulative_variance_ratio`: Cumulative variance explained
- `feature_names`: List of feature names
- `scaler`: StandardScaler if standardize=True

### `run_pca_and_export_artifacts()`

Comprehensive PCA analysis with CSV exports:

```python
from sleap_roots_analyze.pca import run_pca_and_export_artifacts

results = run_pca_and_export_artifacts(
    df_traits=df,
    trait_cols=trait_columns,
    analysis_dir="pca_results",
    n_components=10,
    save_csv=True,
    save_prefix="experiment1_"
)
```

Returns dictionary with DataFrames:
- `pca_results`: Full PCA results dictionary
- `scores_df`: PC scores with metadata
- `trait_contrib_df`: Per-trait variance contributions
- `variance_df`: Variance explained per PC

### `calculate_pca_metrics()`

Low-level metrics calculation:

```python
from sleap_roots_analyze.pca import calculate_pca_metrics

metrics = calculate_pca_metrics(
    pca=fitted_pca,
    X_transformed=pc_scores,
    X_fitted=standardized_data,
    ddof_for_feature_var=1  # Degrees of freedom
)
```

### `build_feature_metrics_df()`

Create tidy DataFrame for visualization:

```python
from sleap_roots_analyze.pca import build_feature_metrics_df

feature_df = build_feature_metrics_df(
    pca_result,
    include_loadings=True,
    sort_by="fraction_explained"
)
```

### `select_top_features_from_pca()`

Select top features based on PCA loadings using various strategies:

```python
from sleap_roots_analyze.pca import select_top_features_from_pca

# Select features with extreme loadings (most positive and negative)
selected_features = select_top_features_from_pca(
    loadings=result['loadings'],
    eigenvalues=result['eigenvalues'],
    n_features_total=len(result['feature_names']),
    n_features_to_select=10,  # Per direction for "extreme" method
    method='extreme',          # Options: 'extreme', 'top_absolute', 'top_contribution', 'top_variance'
    pc_indices=[0, 1]         # Which PCs to consider (0-based)
)
```

Methods available:
- `extreme`: Select top N most positive and negative loadings for specified PCs
- `top_absolute`: Select top N by absolute loading magnitude on specified PCs
- `top_contribution`: Select top N by variance contribution to specified PCs
- `top_variance`: Select top N by total variance contribution across all PCs

## Usage Examples

### Example 1: Basic PCA Analysis

```python
import pandas as pd
from sleap_roots_analyze.pca import perform_pca_analysis
from sleap_roots_analyze.data_cleanup import get_trait_columns

# Load data
df = pd.read_csv("traits.csv")
trait_cols = get_trait_columns(df)

# Run PCA
result = perform_pca_analysis(
    df[trait_cols],
    standardize=True,
    explained_variance_threshold=0.95
)

print(f"Selected {result['n_components_selected']} components")
print(f"Total variance explained: {result['total_variance_explained']:.2%}")
```

### Example 2: Export Comprehensive Artifacts

```python
from sleap_roots_analyze.pca import run_pca_and_export_artifacts

# Run with automatic exports
results = run_pca_and_export_artifacts(
    df_traits=df,
    trait_cols=trait_cols,
    analysis_dir="./pca_output",
    n_components=5,
    save_csv=True,
    save_prefix="rice_"
)

# Access DataFrames
loadings_df = results["loadings_df"]
scores_df = results["scores_df"]
contributions_df = results["trait_contrib_df"]

# Top contributing traits
top_traits = contributions_df.nlargest(10, "trait_fractional_contrib")
print("Top 10 contributing traits:")
print(top_traits[["trait_fractional_contrib"]])
```

### Example 3: Visualization with Feature Importance

```python
from sleap_roots_analyze.visualization import (
    create_pca_biplot,
    create_feature_contribution_heatmap
)

# Create biplot
fig_biplot = create_pca_biplot(
    result,
    color_by="genotype",
    metadata_df=df[["Barcode", "genotype"]]
)

# Feature contribution heatmap
fig_heatmap = create_feature_contribution_heatmap(
    result["feature_contributions"],
    n_components=5,
    n_features=20
)
```

### Example 4: Interactive Visualization

```python
from sleap_roots_analyze.interactive_visualization import (
    create_interactive_pca_with_images
)

# Create interactive PCA with sample images
fig = create_interactive_pca_with_images(
    pca_result=result,
    df=df,
    image_links=image_paths,  # Dict: sample_id -> image_path
    color_by="genotype",
    components=(0, 1),
    show_loadings=True
)

# Save as HTML
fig.write_html("pca_interactive.html")
```

## Artifact Descriptions

### CSV Outputs from `run_pca_and_export_artifacts()`

1. **`{prefix}pca_loadings.csv`**
   - Raw eigenvectors $v_{jk}$
   - Rows: Features, Columns: PCs
   - Interpretation: Direction and magnitude of feature influence

2. **`{prefix}trait_variance_contrib.csv`**
   - Columns: PC1, PC2, ..., trait_total_variance_contrib, trait_fractional_contrib
   - Per-trait variance contributions in variance units
   - Fractional contributions (sum to 1)

3. **`{prefix}pca_variance_explained.csv`**
   - Per-PC variance statistics
   - Columns: `Explained Variance (%)`, `Cumulative Variance (%)`

4. **`{prefix}pca_transformed_data.csv`**
   - PC scores for each sample
   - Includes metadata columns (Barcode, genotype, replicate)

5. **`{prefix}feature_metrics.csv`** (optional)
   - Tidy format with one row per feature
   - Comprehensive metrics for plotting

### Key Metrics Interpretation

| Metric | Description | Use Case |
|--------|-------------|----------|
| `eigenvalues` | Variance of each PC | Component importance |
| `explained_variance_ratio` | Fraction of total variance | Component selection |
| `loadings` | Feature weights per PC | Feature-PC relationships |
| `trait_total_variance_contrib` | Absolute variance contribution | Feature importance (variance scale) |
| `trait_fractional_contrib` | Normalized contribution [0,1] | Feature importance (relative) |
| `cumulative_variance_ratio` | Running sum of variance | Dimensionality assessment |

## Advanced Topics

### Component Selection Strategies

1. **Variance Threshold** (default):
   ```python
   result = perform_pca_analysis(
       data, 
       explained_variance_threshold=0.95
   )
   ```

2. **Fixed Components**:
   ```python
   result = perform_pca_analysis(
       data,
       n_components=10
   )
   ```

3. **Scree Plot Analysis**:
   ```python
   from sleap_roots_analyze.visualization import create_pca_scree_plot
   
   fig = create_pca_scree_plot(result)
   # Visually identify elbow point
   ```

### Standardization Considerations

- **When to standardize**: Different units or scales across traits
- **When not to standardize**: All traits in same units, preserving scale matters
- **Check standardization effect**:
  ```python
  # Compare with and without
  result_std = perform_pca_analysis(data, standardize=True)
  result_raw = perform_pca_analysis(data, standardize=False)
  ```

### Integration with Outlier Detection

PCA results feed directly into outlier detection:

```python
from sleap_roots_analyze.outlier_detection import detect_outliers_pca

outliers = detect_outliers_pca(
    result["transformed_data"],
    method="reconstruction",
    threshold_percentile=95
)
```

## Troubleshooting

### Common Issues

1. **Too few components selected**:
   - Lower `explained_variance_threshold` (e.g., 0.90)
   - Check data quality and feature correlations

2. **Memory issues with large datasets**:
   - Use `n_components` to limit computation
   - Consider incremental PCA for very large datasets

3. **NaN handling**:
   - `perform_pca_analysis` automatically drops NaN rows
   - Check `result["data_indices"]` for retained samples

4. **Zero variance features**:
   - Automatically removed during standardization
   - Check `result["feature_names"]` for retained features

## References

- Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
- Abdi, H., & Williams, L. J. (2010). Principal component analysis. WIREs Computational Statistics, 2(4), 433-459.
- [scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)