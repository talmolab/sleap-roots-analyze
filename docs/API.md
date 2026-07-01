# API Reference

Complete API documentation for `sleap-roots-analyze`.

## Table of Contents

- [data_cleanup](#data_cleanup-module)
- [statistics](#statistics-module)
- [data_utils](#data_utils-module)
- [pca](#pca-module)
- [outlier_detection](#outlier_detection-module)
- [outlier_visualization](#outlier_visualization-module)
- [visualization](#visualization-module)

---

## `data_cleanup` Module

Data loading, cleaning, and preprocessing utilities.

### Functions

#### `load_trait_data`

```python
load_trait_data(
    file_path: Union[str, Path],
    required_cols: Optional[List[str]] = None
) -> pd.DataFrame
```

Load trait data from CSV or Excel file with validation.

**Parameters:**
- `file_path`: Path to the data file (CSV or Excel)
- `required_cols`: Optional list of required column names to validate

**Returns:**
- `pd.DataFrame`: Loaded data

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `ValueError`: If required columns are missing or file format unsupported

**Example:**
```python
df = load_trait_data("traits.csv", required_cols=["geno", "rep"])
```

---

#### `get_trait_columns`

```python
get_trait_columns(
    df: pd.DataFrame,
    exclude_patterns: Optional[List[str]] = None,
    additional_exclude: Optional[List[str]] = None
) -> List[str]
```

Identify numeric trait columns, excluding metadata columns.

**Parameters:**
- `df`: Input DataFrame
- `exclude_patterns`: Regex patterns to exclude (default: metadata patterns)
- `additional_exclude`: Additional column names to exclude

**Returns:**
- `List[str]`: Names of numeric trait columns

**Default Excluded Patterns:**
- Identifiers: `Barcode`, `geno`, `rep`, `genotype`, `replicate`
- QC: `QC_*`, `outlier*`
- Experimental: `wave_*`, `scan_*`, `plant_id`, `experiment_*`
- Dates: `*_date`, `*_time`, `*_day`
- Paths: `*_path`, `*_file`, `*.png`, `*.jpg`

**Example:**
```python
trait_cols = get_trait_columns(df)
print(f"Found {len(trait_cols)} trait columns")
```

---

#### `apply_data_cleanup_filters`

```python
apply_data_cleanup_filters(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_zeros_per_trait: float = 0.5,
    max_nans_per_trait: float = 0.2,
    max_nans_per_sample: float = 0.0,
    min_samples_per_trait: int = 10,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> Tuple[pd.DataFrame, Dict]
```

Canonical smart cleanup used by QC step 02 (`CleanupTraitsStep`). Drops bad
**traits** first (zero-inflated, too-many-NaN, low-sample) and *then* the
remaining NaN rows, to minimize sample loss.

**Returns:**
- `Tuple[pd.DataFrame, Dict]`: `(cleaned_df, cleanup_log)`; the log records
  removed traits/samples and per-step counts.

**Example:**
```python
clean_df, log = apply_data_cleanup_filters(df, trait_cols)
```

---

#### `build_clean_validation_report`

```python
build_clean_validation_report(
    df: pd.DataFrame,
    trait_cols: List[str],
) -> Dict
```

Build the no-NaN validation report for a cleaned trait table (the report QC
step 03 emits). Pure — no I/O, no raising.

**Returns:**
- `Dict`: report with `validation_passed`, `total_samples`,
  `nan_values_in_traits`, `trait_nan_counts`, etc.

---

#### `validate_clean_traits`

```python
validate_clean_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
) -> Dict
```

Validate that a cleaned table has no NaNs in its trait columns (the importable
form of QC step 03's check, shared by `ValidateCleanStep` and
`clean_traits_for_analysis`).

**Returns:**
- `Dict`: the validation report (on success).

**Raises:**
- `ValueError`: If any NaN values remain in `trait_cols` (message names the
  affected traits).

---

#### `clean_traits_for_analysis`

```python
clean_traits_for_analysis(
    df: pd.DataFrame,
    trait_cols: Optional[List[str]] = None,
    *,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    **cleanup_kwargs,
) -> Tuple[pd.DataFrame, List[str], Dict]
```

Public minimal-QC entry point. Composes `apply_data_cleanup_filters` (step 02)
with `validate_clean_traits` (step 03) and adds analysis-readiness gates so the
result is safe for PCA / UMAP / clustering without silently dropping rows.
Validation runs in order: (1) empty input, (2) no NaN in surviving traits,
(3) ≥2 surviving samples, (4) ≥1 non-constant numeric trait (`var(ddof=0) > 0`).

> **Note:** default thresholds are the **QC pipeline's canonical** values
> (`max_zeros_per_trait=0.5`, `max_nans_per_trait=0.2`, `max_nans_per_sample=0.0`,
> `min_samples_per_trait=10`), so the analysis-ready frame matches what the QC
> pipeline produces rather than a looser clean. (`apply_data_cleanup_filters`' own
> signature defaults now equal these canonical values — `max_nans_per_trait=0.2`,
> `max_nans_per_sample=0.0` — aligned in #167.) Caller kwargs override. With
> `max_nans_per_sample=0.0`, any sample carrying a NaN in a surviving trait is
> dropped. Two samples is the runnability floor only.

**Returns:**
- `Tuple[pd.DataFrame, List[str], Dict]`: `(clean_df, trait_cols, cleanup_log)`,
  where `trait_cols` are the surviving traits and `cleanup_log` is enriched with
  `effective_thresholds` and `validation_summary`.

**Raises:**
- `ValueError`: On empty input, residual NaN, fewer than 2 surviving samples, or
  no non-constant numeric trait remaining.

**Example:**
```python
from sleap_roots_analyze import clean_traits_for_analysis, perform_pca_analysis

clean_df, trait_cols, log = clean_traits_for_analysis(df)
pca = perform_pca_analysis(clean_df[trait_cols])  # no rows silently dropped
```

---

#### `remove_nan_samples`

```python
remove_nan_samples(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_nan_fraction: float = 0.2,
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    save_removed_path: Optional[Union[Path, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]
```

Remove samples with too many NaN values.

**Parameters:**
- `df`: Input DataFrame
- `trait_cols`: List of trait columns to check
- `max_nan_fraction`: Maximum fraction of NaN values allowed (default: 0.2)
- `barcode_col`: Name of barcode/ID column
- `genotype_col`: Name of genotype column
- `replicate_col`: Name of replicate column (optional)
- `save_removed_path`: Optional path to save removed samples

**Returns:**
- `Tuple[DataFrame, DataFrame, Dict]`:
  - Cleaned DataFrame
  - DataFrame of removed samples
  - Statistics dictionary with removal details

**Example:**
```python
df_clean, df_removed, stats = remove_nan_samples(
    df, 
    trait_cols,
    max_nan_fraction=0.3,
    save_removed_path="removed_samples.csv"
)

print(f"Removed {stats['samples_removed']} samples")
print(f"Genotypes affected: {stats['genotypes_affected']}")
```

---

#### `remove_zero_inflated_traits`

```python
remove_zero_inflated_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    max_zero_fraction: float = 0.5
) -> Tuple[pd.DataFrame, List[str]]
```

Remove traits with excessive zero values.

**Parameters:**
- `df`: Input DataFrame
- `trait_cols`: List of trait columns to check
- `max_zero_fraction`: Maximum fraction of zeros allowed (default: 0.5)

**Returns:**
- `Tuple[DataFrame, List[str]]`: Cleaned DataFrame and list of removed traits

**Example:**
```python
df_clean, removed = remove_zero_inflated_traits(df, trait_cols, 0.4)
print(f"Removed zero-inflated traits: {removed}")
```

---

#### `remove_low_variance_traits`

```python
remove_low_variance_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    min_variance: float = 0.01
) -> Tuple[pd.DataFrame, List[str]]
```

Remove traits with insufficient variance.

**Parameters:**
- `df`: Input DataFrame  
- `trait_cols`: List of trait columns to check
- `min_variance`: Minimum variance threshold (default: 0.01)

**Returns:**
- `Tuple[DataFrame, List[str]]`: Cleaned DataFrame and list of removed traits

---

#### `link_rhizovision_images_to_samples`

```python
link_rhizovision_images_to_samples(
    df: pd.DataFrame,
    image_dir: Union[str, Path],
    barcode_col: str = "Barcode",
    image_types: List[str] = ["features.png", "seg.png"]
) -> Dict[str, Dict[str, Optional[Path]]]
```

Link Rhizovision images to their corresponding sample barcodes.

**Parameters:**
- `df`: Input DataFrame
- `image_dir`: Directory containing image files
- `barcode_col`: Name of barcode column
- `image_types`: List of image file suffixes to search for

**Returns:**
- `pd.DataFrame`: DataFrame with added image path columns

---

## `statistics` Module

Statistical analysis functions for trait data.

### Functions

#### `calculate_heritability_estimates`

```python
calculate_heritability_estimates(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    replicate_col: str = "rep",
    force_method: Optional[str] = None,
    remove_low_h2: bool = False,
    h2_threshold: float = 0.3,
    barcode_col: str = "Barcode",
    additional_exclude: Optional[List[str]] = None
) -> Union[Dict, Tuple[Dict, pd.DataFrame, List[str], Dict]]
```

Calculate broad-sense heritability (H²) for traits using mixed models.

**Parameters:**
- `df`: Input DataFrame with trait data
- `trait_cols`: List of trait columns to analyze
- `genotype_col`: Name of genotype column (default: "geno")
- `replicate_col`: Name of replicate column (default: "rep")
- `force_method`: Force specific method ("mixed_model" or "anova_based")
- `remove_low_h2`: If True, filter out low heritability traits
- `h2_threshold`: Heritability threshold for filtering (default: 0.3)
- `barcode_col`: Name of sample ID column for preservation
- `additional_exclude`: Additional columns to exclude from filtering

**Returns:**
- If `remove_low_h2=False`: Dictionary with heritability results
- If `remove_low_h2=True`: Tuple of (results, filtered_df, removed_traits, details)

**Heritability Calculation:**
```
H² = σ²_G / (σ²_G + σ²_E)

where:
- σ²_G = genetic variance (between genotypes)
- σ²_E = environmental variance (within genotypes)
```

**Example:**
```python
# Basic usage
h2_results = calculate_heritability_estimates(df, trait_cols)

# With filtering
results, df_filtered, removed, details = calculate_heritability_estimates(
    df, trait_cols,
    remove_low_h2=True,
    h2_threshold=0.3
)
```

---

#### `calculate_trait_statistics`

```python
calculate_trait_statistics(
    df: pd.DataFrame,
    trait_cols: List[str]
) -> Dict[str, Dict[str, float]]
```

Calculate comprehensive statistics for each trait.

**Parameters:**
- `df`: Input DataFrame
- `trait_cols`: List of trait columns

**Returns:**
- Dictionary with statistics for each trait:
  - Basic: `mean`, `std`, `min`, `max`, `median`
  - Percentiles: `q25`, `q75`
  - Shape: `skewness`, `kurtosis`
  - Data quality: `count`, `cv` (coefficient of variation as a raw ratio,
    `std / mean`; `np.inf` if `mean` is 0)

**Example:**
```python
stats = calculate_trait_statistics(df, ["trait1", "trait2"])
for trait, values in stats.items():
    print(f"{trait}: mean={values['mean']:.2f}, CV={values['cv']:.2f}")
```

---

#### `perform_anova_by_genotype`

```python
perform_anova_by_genotype(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    alpha: float = 0.05
) -> Dict[str, Dict]
```

Perform one-way ANOVA for each trait by genotype.

**Parameters:**
- `df`: Input DataFrame
- `trait_cols`: List of trait columns
- `genotype_col`: Name of genotype column (default: "geno")
- `alpha`: Significance level for hypothesis testing (default: 0.05)

**Returns:**
- Dictionary with ANOVA results for each trait:
  - `f_statistic`: F-statistic value
  - `p_value`: Statistical significance
  - `eta_squared`: Effect size (proportion of variance explained by genotype)
  - `significant`: Whether `p_value < alpha`
  - `n_groups`: Number of genotype groups with data
  - `total_n`: Total number of observations across groups
  - `group_stats`: Per-genotype dictionary of `n`, `mean`, `std`, and `sem`

**Example:**
```python
anova_results = perform_anova_by_genotype(df, trait_cols)
for trait, result in anova_results.items():
    if result["p_value"] < 0.05:
        print(f"{trait}: Significant genotype effect (p={result['p_value']:.4f})")
```

---

#### `identify_high_heritability_traits`

```python
identify_high_heritability_traits(
    heritability_results: Dict,
    threshold: float = 0.5
) -> List[str]
```

Identify traits with heritability above threshold.

**Parameters:**
- `heritability_results`: Dictionary from `calculate_heritability_estimates`
- `threshold`: Minimum heritability threshold (default: 0.5)

**Returns:**
- List of trait names with H² at or above threshold

---

#### `analyze_heritability_thresholds`

```python
analyze_heritability_thresholds(
    heritability_results: Dict,
    thresholds: Optional[np.ndarray] = None
) -> Dict
```

Analyze trait retention at different heritability thresholds.

**Parameters:**
- `heritability_results`: Dictionary from `calculate_heritability_estimates`
- `thresholds`: Array of threshold values (default: 0.0 to 1.0 in 101 steps)

**Returns:**
- Dictionary with threshold analysis:
  - `thresholds`: Array of threshold values
  - `traits_retained`: Number retained at each threshold
  - `traits_removed`: Number removed at each threshold
  - `fraction_retained`: Fraction retained at each threshold

---

#### `analyze_trait_variance`

```python
analyze_trait_variance(
    df: pd.DataFrame,
    trait: str,
    genotype_col: str = "geno",
    replicate_col: str = "rep"
) -> Dict[str, Any]
```

Decompose a single trait's variance into between- and within-genotype components.

**Parameters:**
- `df`: Input DataFrame with trait data
- `trait`: Name of the trait column to analyze
- `genotype_col`: Name of genotype column (default: "geno")
- `replicate_col`: Name of replicate column (default: "rep")

**Returns:**
- Dictionary of variance metrics: `n_observations`, `n_genotypes`,
  `mean_reps_per_geno`, `min_reps_per_geno`, `max_reps_per_geno`, `trait_mean`,
  `trait_std`, `trait_cv`, `overall_variance`, `between_genotype_variance`,
  `within_genotype_variance`, and `pct_variance_between_geno`. If fewer than 3 valid
  observations exist, returns `{"error": ..., "n_observations": ...}` instead.
- Note: `trait_cv` here is a **percentage**, `(std / mean) * 100` — unlike the `cv`
  from `calculate_trait_statistics`, which is a raw ratio (`std / mean`).

---

#### `diagnose_heritability_issues`

```python
diagnose_heritability_issues(
    df: pd.DataFrame,
    trait: str,
    heritability_result: Dict[str, Any],
    genotype_col: str = "geno",
    replicate_col: str = "rep"
) -> Dict[str, Any]
```

Identify likely causes of low or zero heritability with actionable explanations.

**Parameters:**
- `df`: Input DataFrame with trait data
- `trait`: Name of the trait to diagnose
- `heritability_result`: Per-trait dictionary from `calculate_heritability_estimates`
- `genotype_col`: Name of genotype column (default: "geno")
- `replicate_col`: Name of replicate column (default: "rep")

**Returns:**
- Dictionary with `has_issues` (bool), `issues` (list of descriptions), `severity`
  (`"critical"`, `"warning"`, or `"info"`), and `recommendations` (list of suggested
  actions).

---

#### `compare_trait_heritabilities`

```python
compare_trait_heritabilities(
    df: pd.DataFrame,
    traits: List[str],
    heritability_results: Dict[str, Dict[str, Any]],
    genotype_col: str = "geno",
    replicate_col: str = "rep",
    sort_by: Optional[str] = None
) -> pd.DataFrame
```

Build a comparison table of variance components and heritability across traits.

**Parameters:**
- `df`: Input DataFrame with trait data
- `traits`: List of trait names to compare
- `heritability_results`: Dictionary mapping trait names to heritability results
- `genotype_col`: Name of genotype column (default: "geno")
- `replicate_col`: Name of replicate column (default: "rep")
- `sort_by`: Optional column name to sort the result by (default: None)

**Returns:**
- DataFrame with one row per trait and columns `trait`, `heritability`,
  `var_genetic`, `var_residual`, `between_geno_var`, `within_geno_var`,
  `pct_var_between`, `n_observations`, `n_genotypes`, `mean_reps_per_geno`,
  `trait_mean`, `trait_cv`, and `model_type`. As in `analyze_trait_variance`,
  `trait_cv` is a **percentage** (`(std / mean) * 100`).

---

## `pca` Module

Principal Component Analysis for dimensionality reduction of root trait data.

### Functions

#### `perform_pca_analysis`

```python
perform_pca_analysis(
    data: Union[pd.DataFrame, np.ndarray],
    standardize: bool = True,
    explained_variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    random_state: int = 42
) -> Dict
```

Perform complete PCA analysis pipeline with optional standardization.

**Parameters:**
- `data`: Input data as DataFrame or array
- `standardize`: Whether to standardize data (default: True)
- `explained_variance_threshold`: Cumulative variance threshold (default: 0.95)
- `n_components`: Number of components (overrides automatic selection if specified)
- `random_state`: Random state for reproducibility

**Returns:**
Dictionary containing:
- `pca`: Fitted PCA model
- `transformed_data`: Transformed data in PC space
- `loadings`: Component loadings
- `explained_variance_ratio`: Variance explained by each component
- `n_components_selected`: Number of components selected
- `scaler`: StandardScaler if standardize=True, else None
- `feature_names`: List of feature names used

**Example:**
```python
result = perform_pca_analysis(df_traits, standardize=True)
print(f"Selected {result['n_components_selected']} components")
print(f"Explained variance: {result['cumulative_variance_ratio'][-1]:.2%}")
```

---

#### `standardize_data`

```python
standardize_data(
    df: pd.DataFrame
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]
```

Standardize numeric columns and remove zero-variance features.

**Parameters:**
- `df`: Input DataFrame

**Returns:**
- Standardized data array
- Fitted StandardScaler
- Cleaned DataFrame (without zero-variance columns)

---

#### `calculate_mahalanobis_distances`

```python
calculate_mahalanobis_distances(
    X_transformed: np.ndarray,
    robust: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Calculate Mahalanobis distances for outlier detection.

**Parameters:**
- `X_transformed`: PCA-transformed data
- `robust`: Use robust covariance estimation (default: True)

**Returns:**
- Mahalanobis distances for each sample
- Mean of transformed data
- Covariance matrix

**Example:**
```python
distances, mean, cov = calculate_mahalanobis_distances(pca_result['transformed_data'])
outliers = distances > np.percentile(distances, 95)
```

---

#### `calculate_pca_metrics`

```python
calculate_pca_metrics(
    pca: PCA,
    X_transformed: np.ndarray,
    X_fitted: Optional[np.ndarray] = None,
    ddof_for_feature_var: int = 1
) -> Dict
```

Calculate comprehensive PCA metrics including per-feature variance explained.

**Parameters:**
- `pca`: Fitted sklearn PCA object
- `X_transformed`: Transformed data (scores)
- `X_fitted`: Original fitted data for variance calculations
- `ddof_for_feature_var`: Degrees of freedom for variance (default: 1)

**Returns:**
Dictionary containing:
- `loadings`: Component loadings matrix
- `explained_variance`: Eigenvalues
- `explained_variance_ratio`: Per-component variance ratios
- `cumulative_variance_ratio`: Cumulative variance explained
- `feature_variances`: Per-feature variance explained
- `feature_fraction_explained`: Fraction of each feature's variance explained

**Example:**
```python
metrics = calculate_pca_metrics(pca, transformed, X_fitted=X_processed)
print(f"Feature variances: {metrics['feature_variances']}")
```

---

#### `build_feature_metrics_df`

```python
build_feature_metrics_df(
    pca_result: Dict,
    ddof_feature_var: Optional[int] = None,
    include_loadings: bool = True,
    loading_prefix: str = "loading_pc",
    sort_by: str = "fraction_explained"
) -> pd.DataFrame
```

Build DataFrame with per-feature PCA metrics.

**Parameters:**
- `pca_result`: Dictionary from `perform_pca_analysis`
- `ddof_feature_var`: Override ddof for variance calculations
- `include_loadings`: Include loading columns
- `loading_prefix`: Prefix for loading column names
- `sort_by`: Column to sort by

**Returns:**
DataFrame with columns:
- `feature`: Feature name
- `variance_total`: Total feature variance
- `variance_explained`: Variance explained by retained PCs
- `fraction_explained`: Fraction of variance explained
- Optional: `loading_pc1`, `loading_pc2`, etc.

**Example:**
```python
feature_df = build_feature_metrics_df(pca_result, sort_by="fraction_explained")
top_features = feature_df.head(10)["feature"].tolist()
```

---

## `data_utils` Module

Utility functions for data processing.

### Functions

#### `create_run_directory`

```python
create_run_directory(base_dir: Path) -> Path
```

Create timestamped directory for output files.

**Parameters:**
- `base_dir`: Base directory path

**Returns:**
- `Path`: Created directory path with format `run_YYYYMMDD_HHMMSS`

**Example:**
```python
output_dir = create_run_directory(Path("results"))
# Creates: results/run_20250103_143025/
```

---

#### `convert_to_json_serializable`

```python
convert_to_json_serializable(obj) -> Any
```

Convert numpy types to JSON-serializable Python types.

**Parameters:**
- `obj`: Object to convert (supports nested structures)

**Returns:**
- JSON-serializable version of the object

**Conversions:**
- `np.integer` → `int`
- `np.floating` → `float`
- `np.bool_` → `bool`
- `np.ndarray` → `list`
- Nested `dict`, `list`, `tuple` are processed recursively

---

## `visualization` Module

Visualization functions for trait data exploration and publication-ready figures.

### Boxplot Functions

#### `create_trait_boxplots_by_genotype`

```python
create_trait_boxplots_by_genotype(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
    adaptive_config: Optional[Any] = None,
    orientation: str = "auto",
    horizontal_threshold: int = 8,
) -> plt.Figure
```

Create boxplots for traits grouped by genotype with adaptive layout.

**Parameters:**
- `df`: DataFrame with trait and genotype data
- `trait_cols`: List of trait column names to plot
- `genotype_col`: Name of genotype column (default: "geno")
- `n_cols`: Number of columns in subplot grid
- `figsize`: Base figure size. May be overridden by adaptive_config or adjusted for genotype count/orientation
- `adaptive_config`: Optional adaptive sizing configuration
- `orientation`: Boxplot orientation — "vertical", "horizontal", or "auto". "auto" switches to horizontal when genotype count exceeds `horizontal_threshold`
- `horizontal_threshold`: Threshold for auto orientation switch (default: 8)

**Returns:**
- `plt.Figure`: Matplotlib figure. Callers are responsible for calling `tight_layout()` if needed (e.g., after adding suptitle).

**Layout behavior:**
- **Vertical** (≤ threshold genotypes): Uses `df.boxplot()` with rotated x-axis labels. Subplot width scales adaptively (0.5 in/genotype, min 4.0, max 20.0 inches).
- **Horizontal** (> threshold genotypes): Uses `ax.boxplot(orientation="horizontal")` with genotype names as y-axis labels.
- Both orientations use consistent styling: unfilled outline boxes, blue (`#1f77b4`) outlines, green (`#2ca02c`) medians, and gridlines.

**Example:**
```python
from sleap_roots_analyze.visualization import create_trait_boxplots_by_genotype

fig = create_trait_boxplots_by_genotype(df, trait_cols, orientation="auto")
fig.suptitle("Trait Distributions by Genotype")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("boxplots.png")
```

---

#### `create_trait_boxplots_by_genotype_batched`

```python
create_trait_boxplots_by_genotype_batched(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    batch_size: int = 16,
    n_cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    subplot_size: Tuple[float, float] = (4.0, 4.0),
    orientation: str = "auto",
    horizontal_threshold: int = 8,
) -> List[plt.Figure]
```

Create batched boxplots for many traits across multiple figures.

**Parameters:**
- `df`: DataFrame with trait data
- `trait_cols`: List of trait column names
- `genotype_col`: Column name for genotype grouping (default: "geno")
- `batch_size`: Number of traits per figure (default: 16)
- `n_cols`: Number of columns in subplot grid
- `figsize`: Optional explicit figure size. If None, calculated adaptively from `subplot_size`
- `subplot_size`: Size of each subplot in inches when figsize is None (default: (4.0, 4.0))
- `orientation`: Boxplot orientation — "vertical", "horizontal", or "auto"
- `horizontal_threshold`: Threshold for auto orientation switch (default: 8)

**Returns:**
- `List[plt.Figure]`: List of figures, one per batch. Each includes suptitle and `tight_layout(rect=[0, 0, 1, 0.96])`.

**Example:**
```python
from sleap_roots_analyze.visualization import create_trait_boxplots_by_genotype_batched

figures = create_trait_boxplots_by_genotype_batched(
    df, trait_cols, batch_size=12, orientation="auto"
)
for i, fig in enumerate(figures):
    fig.savefig(f"boxplots_batch_{i}.png")
```

---

## Error Handling

### Common Exceptions

#### `ValueError`
Raised when:
- Required columns are missing
- File format is unsupported
- Invalid parameter values provided
- Insufficient data for analysis

#### `FileNotFoundError`
Raised when:
- Input file doesn't exist
- Image directory not found

#### `TypeError`
Raised when:
- Invalid data types provided
- Incompatible DataFrame structure

### Error Examples

```python
try:
    df = load_trait_data("missing_file.csv")
except FileNotFoundError as e:
    print(f"File not found: {e}")

try:
    results = calculate_heritability_estimates(
        df, trait_cols,
        genotype_col="missing_column"
    )
except ValueError as e:
    print(f"Missing column: {e}")
```

## Performance Considerations

### Memory Usage

- **Large datasets**: Process in chunks when possible
- **Trait filtering**: Remove unnecessary traits early
- **Copy operations**: Use `inplace=True` where appropriate

### Computation Time

- **Heritability calculation**: O(n × m) for n samples, m traits
- **Mixed models**: Can be slow for large datasets (>1000 samples)
- **ANOVA**: Fast for reasonable group sizes

### Optimization Tips

```python
# Process traits in batches for large datasets
batch_size = 50
for i in range(0, len(trait_cols), batch_size):
    batch_traits = trait_cols[i:i+batch_size]
    results = calculate_heritability_estimates(df, batch_traits)
    
# Filter data early in pipeline
df_clean = remove_nan_samples(df, trait_cols)[0]
df_clean = remove_zero_inflated_traits(df_clean, trait_cols)[0]
# Then run expensive calculations
```

## `outlier_detection` Module

Outlier detection using Mahalanobis distance on PCA-transformed data.

### Functions

#### `remove_outlier_samples`

```python
remove_outlier_samples(
    clean_df: pd.DataFrame,
    trait_cols: Optional[List[str]] = None,
    *,
    method: str = "mahalanobis",
    barcode_col: str = "Barcode",
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    random_state: int = 42,
    **detect_kwargs,
) -> Tuple[pd.DataFrame, Dict]
```

Public outlier-removal entry point — the **quality**-step follow-up to
[`clean_traits_for_analysis`](#clean_traits_for_analysis) (#164). Takes a clean
(NaN-free) trait table and detects + removes outlier **samples** so PCA / UMAP /
clustering can optionally run on outlier-trimmed data. Composes the existing
public primitives — [`detect_outliers_mahalanobis`](#detect_outliers_mahalanobis)
/ [`detect_outliers_isolation_forest`](#detect_outliers_isolation_forest) for
detection and [`remove_outliers_from_data`](#remove_outliers_from_data) for row
removal — and defines no detection/removal logic of its own, so its semantics
cannot drift from the QC pipeline's outlier steps. Intended chain:
`clean_traits_for_analysis` → (optional) `remove_outlier_samples` →
`perform_pca_analysis` / UMAP / clustering.

Enforces, before any detector runs, a **NaN-free** and **unique-index** input
(both correctness guards for one-to-one index alignment — the detectors run PCA
which silently drops NaN rows, and removal is label-based). After removal it
re-applies the #164 readiness gates (≥2 surviving samples, ≥1 non-constant trait)
so the trimmed frame stays analysis-ready.

> **Note:** the default `method="mahalanobis"` with `chi2_percentile=97.5` trims
> roughly the top 2.5% of samples *by construction* on a well-fit chi-squared tail
> — i.e. it removes ~2.5% even on clean data. Tighten `chi2_percentile` or switch
> to `method="isolation_forest"` with an explicit `contamination` to control this.

**Parameters:**
- `clean_df`: A clean (NaN-free in trait columns), unique-indexed wide trait table
  (e.g. the first element returned by `clean_traits_for_analysis`)
- `trait_cols`: Trait columns to score; inferred via `get_trait_columns` if `None`
- `method`: `"mahalanobis"` (default) or `"isolation_forest"`; unknown raises
- `barcode_col` / `genotype_col` / `replicate_col`: metadata columns excluded from
  inferred traits (`barcode_col` is also read to populate `outlier_barcodes`)
- `random_state`: Seed forwarded to the detector for reproducibility (default: 42)
- `**detect_kwargs`: Per-method parameters forwarded to the chosen detector
  (`contamination` for isolation forest; `chi2_percentile`, `variance_threshold`,
  `use_chi_squared`, `distance_threshold`, `robust_covariance` for Mahalanobis)

**Returns:**
- `Tuple[pd.DataFrame, Dict]`: `(trimmed_df, outlier_report)`. `trimmed_df` is
  `clean_df` with the flagged rows removed (all columns preserved). `outlier_report`
  is an auditable, JSON-serializable dict: `method`, `method_params`, `random_state`,
  `n_input_samples`, `n_outliers`, `n_output_samples`, `removal_fraction`,
  `outlier_indices`, `outlier_barcodes`, `threshold_type`, `threshold_value`,
  `n_components`, `variance_threshold`, `goodness_of_fit` (the last four are
  populated for Mahalanobis and `None` for isolation forest).

**Warns:**
- `UserWarning`: when the removed fraction exceeds 0.5 (likely mis-set
  `contamination`/threshold), and in the `p > n` regime after trimming. On the
  Mahalanobis path it also warns on a small sample (`n < 30` — fragile chi-squared
  tail / covariance) and when the detector's chi-squared goodness-of-fit reports the
  distributional assumption is violated (so the `chi2_percentile` threshold's meaning
  is questionable).

**Raises:**
- `ValueError`: empty input; duplicate columns; explicit `trait_cols`
  missing/non-numeric; unknown `method`; unknown / cross-method `**detect_kwargs`
  for the chosen detector (names the keys + supported set); non-unique index; NaN in
  trait columns (message points to `clean_traits_for_analysis`); or a detector failure.
- `OutlierRemovalError` (a `ValueError`): when trimming leaves <2 samples or no
  non-constant trait; carries `outlier_report` as an attribute.

**Example:**
```python
from sleap_roots_analyze import (
    clean_traits_for_analysis,
    remove_outlier_samples,
    perform_pca_analysis,
)

clean_df, trait_cols, _ = clean_traits_for_analysis(df)
trimmed_df, report = remove_outlier_samples(clean_df, trait_cols)
print(f"Removed {report['n_outliers']} outliers: {report['outlier_barcodes']}")
pca = perform_pca_analysis(trimmed_df[trait_cols])  # outlier-trimmed PCA
```

---

#### `detect_outliers_mahalanobis`

```python
detect_outliers_mahalanobis(
    data: Union[pd.DataFrame, np.ndarray],
    standardize: bool = True,
    variance_threshold: float = 0.95,
    use_chi_squared: bool = True,
    chi2_percentile: float = 97.5,
    distance_threshold: Optional[float] = None,
    robust_covariance: bool = False,
    random_state: int = 42
) -> Dict
```

Detect outliers using Mahalanobis distance on PCA-transformed data.

**Parameters:**
- `data`: Input data as DataFrame or array
- `standardize`: Whether to standardize data before PCA (default: True)
- `variance_threshold`: Cumulative variance threshold for PCA component selection (default: 0.95)
- `use_chi_squared`: Use chi-squared distribution threshold (default: True)
- `chi2_percentile`: Percentile for chi-squared threshold (default: 97.5)
- `distance_threshold`: Direct Mahalanobis distance threshold (if not using chi-squared)
- `robust_covariance`: Use robust covariance estimation (MinCovDet) (default: False)
- `random_state`: Random seed for reproducibility (default: 42)

**Returns:**
Dictionary containing:
- `outlier_indices`: List of outlier sample indices
- `mahalanobis_distances`: Distance for each sample
- `n_outliers`: Number of outliers detected
- `n_components`: Number of PCA components used
- `threshold_type`: Type of threshold used ("chi_squared" or "distance")
- `threshold_value`: Threshold value used
- `feature_names`: List of feature names
- `error`: Error message if detection failed (only if error occurred)

**Example:**
```python
# Basic outlier detection
result = detect_outliers_mahalanobis(df_traits, standardize=True)
print(f"Found {result['n_outliers']} outliers")
print(f"Outlier indices: {result['outlier_indices']}")

# With custom threshold
result = detect_outliers_mahalanobis(
    df_traits,
    chi2_percentile=99.0,  # More conservative threshold
    robust_covariance=True  # Use robust estimation
)

# Using direct distance threshold instead of chi-squared
result = detect_outliers_mahalanobis(
    df_traits,
    use_chi_squared=False,
    distance_threshold=3.5  # 3.5 standard deviations
)
```

---

#### `calculate_outlier_threshold`

```python
calculate_outlier_threshold(
    n_components: int,
    use_chi_squared: bool = True,
    chi2_percentile: float = 97.5,
    distance_threshold: Optional[float] = None
) -> Tuple[float, str]
```

Calculate threshold for outlier detection.

**Parameters:**
- `n_components`: Number of PCA components (degrees of freedom)
- `use_chi_squared`: Whether to use chi-squared distribution (default: True)
- `chi2_percentile`: Percentile for chi-squared threshold (0-100)
- `distance_threshold`: Direct distance threshold

**Returns:**
- Tuple of (threshold value, threshold type string)

**Raises:**
- `ValueError`: If parameters are invalid

**Example:**
```python
# Chi-squared threshold for 5 components
threshold, threshold_type = calculate_outlier_threshold(
    n_components=5,
    use_chi_squared=True,
    chi2_percentile=97.5
)
print(f"{threshold_type} threshold: {threshold:.2f}")

# Direct distance threshold
threshold, threshold_type = calculate_outlier_threshold(
    n_components=5,
    use_chi_squared=False,
    distance_threshold=3.0
)
```

---

#### `identify_outliers_from_distances`

```python
identify_outliers_from_distances(
    distances: np.ndarray,
    threshold: float,
    threshold_type: str = "chi_squared",
    indices: Optional[pd.Index] = None
) -> Dict
```

Identify outliers from Mahalanobis distances.

**Parameters:**
- `distances`: Array of Mahalanobis distances
- `threshold`: Threshold value for outlier detection
- `threshold_type`: Type of threshold ("chi_squared" or "distance")
- `indices`: Optional custom indices for the samples

**Returns:**
Dictionary with:
- `outlier_mask`: Boolean mask of outliers
- `outlier_indices`: List of outlier indices
- `n_outliers`: Number of outliers

**Example:**
```python
# Calculate distances first
distances, _, _ = calculate_mahalanobis_distances(pca_transformed_data)

# Identify outliers using chi-squared threshold
threshold, _ = calculate_outlier_threshold(n_components=5)
outliers = identify_outliers_from_distances(
    distances,
    threshold,
    threshold_type="chi_squared"
)

print(f"Found {outliers['n_outliers']} outliers")
```

---

#### `detect_outliers_isolation_forest`

```python
detect_outliers_isolation_forest(
    data: Union[pd.DataFrame, np.ndarray],
    contamination: float = 0.1,
    random_state: int = 42
) -> Dict
```

Detect outliers using an Isolation Forest. Anomalies require fewer random splits
to isolate, so they receive more negative anomaly scores. `contamination` is the
expected outlier proportion (a quota), so it may flag exactly that fraction even on
clean data. One of the detectors composed by
[`remove_outlier_samples`](#remove_outlier_samples).

**Parameters:**
- `data`: DataFrame with numeric trait data or numpy array
- `contamination`: Expected proportion of outliers, 0–0.5 (default: 0.1)
- `random_state`: Random seed for reproducibility (default: 42)

**Returns:**
Dictionary containing:
- `outlier_indices`: List of row indices identified as outliers
- `anomaly_scores`: Per-sample anomaly scores (more negative = more anomalous)
- `contamination`: Contamination parameter used
- `outlier_labels`: `-1` for outliers, `1` for inliers
- `data_indices`: Original indices of the data
- `error`: Error message if detection failed (only if error occurred)

**Example:**
```python
result = detect_outliers_isolation_forest(df_traits, contamination=0.2)
print(f"Found {result['n_outliers']} outliers")
```

---

#### `remove_outliers_from_data`

```python
remove_outliers_from_data(
    df: pd.DataFrame,
    outlier_indices: Union[List, np.ndarray, pd.Index],
    keep_metadata: bool = True,
    return_outliers: bool = True,
    reset_index: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]
```

Remove outlier samples (by index label) from a dataset and optionally return them.
Handles integer/string/custom indices and preserves them. The row-dropping helper
composed by [`remove_outlier_samples`](#remove_outlier_samples).

**Parameters:**
- `df`: Original DataFrame with data
- `outlier_indices`: Indices of outliers to remove (from a detection function)
- `keep_metadata`: Preserve all columns (`True`) or keep only numeric (`False`)
- `return_outliers`: Also return the removed-rows DataFrame (default: True)
- `reset_index`: Reset the index of the cleaned frame after removal (default: False)

**Returns:**
- If `return_outliers=False`: the cleaned DataFrame
- If `return_outliers=True`: a tuple `(cleaned_df, outliers_df)`

**Example:**
```python
result = detect_outliers_mahalanobis(df_traits)
cleaned_df, outlier_df = remove_outliers_from_data(
    df_traits, result["outlier_indices"]
)
```

---

### Complete Outlier Detection Pipeline

```python
from sleap_roots_analyze import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
    detect_outliers_mahalanobis
)

# Load and prepare data
df = load_trait_data("experiment_traits.csv")
trait_cols = get_trait_columns(df)

# Clean NaN samples first
df_clean, _, _ = remove_nan_samples(df, trait_cols)

# Select only trait columns for outlier detection
df_traits = df_clean[trait_cols]

# Detect outliers using default settings
result = detect_outliers_mahalanobis(
    df_traits,
    standardize=True,
    variance_threshold=0.95,
    chi2_percentile=97.5
)

# Report results
print(f"Analysis used {result['n_components']} PCA components")
print(f"Cumulative variance explained: {result['cumulative_variance_at_selection']:.2%}")
print(f"Found {result['n_outliers']} outliers out of {len(df_traits)} samples")
print(f"Outlier threshold ({result['threshold_type']}): {result['threshold_value']:.2f}")

# Get outlier samples
outlier_samples = df_clean.iloc[result['outlier_indices']]
print(f"Outlier barcodes: {outlier_samples['Barcode'].tolist()}")

# Optional: Remove outliers from dataset
df_no_outliers = df_clean.drop(index=result['outlier_indices'])
```

### Visualizing Outliers

```python
import matplotlib.pyplot as plt
import numpy as np

# Run outlier detection
result = detect_outliers_mahalanobis(df_traits, standardize=True)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of Mahalanobis distances
ax1.hist(result['mahalanobis_distances'], bins=30, edgecolor='black')
ax1.axvline(x=np.sqrt(result['threshold_value']), color='red', 
           linestyle='--', label=f"Threshold ({result['threshold_type']})")
ax1.set_xlabel("Mahalanobis Distance")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Mahalanobis Distances")
ax1.legend()

# PCA biplot with outliers highlighted
pca_components = np.array(result['pca_components'])
outlier_mask = np.zeros(len(pca_components), dtype=bool)
outlier_mask[result['outlier_indices']] = True

ax2.scatter(pca_components[~outlier_mask, 0], 
           pca_components[~outlier_mask, 1],
           alpha=0.5, label='Normal')
ax2.scatter(pca_components[outlier_mask, 0],
           pca_components[outlier_mask, 1],
           color='red', s=100, label='Outliers')
ax2.set_xlabel(f"PC1 ({result['explained_variance_ratio'][0]:.1%} var)")
ax2.set_ylabel(f"PC2 ({result['explained_variance_ratio'][1]:.1%} var)")
ax2.set_title("PCA Biplot with Outliers")
ax2.legend()

plt.tight_layout()
plt.show()
```

---

## `outlier_visualization` Module

Outlier-detection figure composition. The `create_*_outlier` functions each render an
individual figure; `plot_outlier_analysis` composes the method-appropriate set for a
consumer (e.g. the bloom-mcp `remove_outliers` tool's optional plots).

### Functions

#### `plot_outlier_analysis`

```python
plot_outlier_analysis(
    clean_df: pd.DataFrame,
    trait_cols: Optional[List[str]] = None,
    *,
    method: str = "mahalanobis",
    random_state: Optional[int] = 42,
    which: Optional[Union[str, List[str]]] = None,
    **detect_kwargs,
) -> Dict[str, plt.Figure]
```

The plotting sibling of [`remove_outlier_samples`](#remove_outlier_samples) (#173).
**Re-detects** outliers with the same detector, seed, and per-method parameters — so,
under the shared NaN-free + unique-index preconditions, it flags the same samples
`remove_outlier_samples` removes — then composes the existing public `create_*_outlier`
figure functions and returns the figures. **IO-free**: returns open `matplotlib`
`Figure` objects; the caller saves/persists them. Covers the two `remove_outlier_samples`
methods (`"mahalanobis"`, `"isolation_forest"`); the `detect_outliers_pca`/`_kmeans`/
`_gmm`/`_hierarchical` plots stay pipeline-only.

**Parameters:**
- `clean_df`: Clean (NaN-free in trait columns), unique-indexed wide trait table.
- `trait_cols`: Trait columns to score; inferred via `get_trait_columns` if `None`.
- `method`: `"mahalanobis"` (default) or `"isolation_forest"`; unknown raises.
- `random_state`: Seed forwarded to the detector (default 42; accepts `None`).
- `which`: Figure-key string or list of keys to return; `None` returns the full set. An
  unavailable key raises.
- `**detect_kwargs`: Per-method detector parameters; unknown/cross-method keys raise.

**Returns:**
- `Dict[str, plt.Figure]`: stable figure key → figure. For `"mahalanobis"`, the
  [`create_mahalanobis_outlier_plots`](#create_mahalanobis_outlier_plots) figures; for
  `"isolation_forest"`,
  [`create_isolation_forest_plots`](#create_isolation_forest_plots); plus
  [`create_outliers_per_genotype_plot`](#create_outliers_per_genotype_plot) (key
  `outliers_per_genotype`) when a `geno` column is present.

**Raises:**
- `ValueError`: on empty input; duplicate columns; invalid `trait_cols`; unknown
  `method`; a non-unique index; NaN traits (points to `clean_traits_for_analysis`);
  unknown `detect_kwargs`; an unavailable `which` key; or a detector failure.

#### `create_mahalanobis_outlier_plots`

```python
create_mahalanobis_outlier_plots(df: pd.DataFrame, mahal_results: Dict) -> Dict[str, plt.Figure]
```

Render the Mahalanobis outlier figures (`mahalanobis_outlier_detection`,
`mahalanobis_pc_analysis`, `mahalanobis_threshold_analysis`) from a
`detect_outliers_mahalanobis` result. Returns `{}` on an `error` result.

#### `create_isolation_forest_plots`

```python
create_isolation_forest_plots(df: pd.DataFrame, iso_results: Dict) -> Dict[str, plt.Figure]
```

Render the isolation-forest outlier figure (`isolation_forest_analysis`) from a
`detect_outliers_isolation_forest` result. Returns `{}` on an `error` result.

#### `create_outliers_per_genotype_plot`

```python
create_outliers_per_genotype_plot(
    df: pd.DataFrame, all_outlier_results: Dict, genotype_col: str = "geno"
) -> plt.Figure
```

Render a per-genotype outlier bar chart (absolute counts and proportions) across the
methods present in `all_outlier_results`.

---

## Examples

### Complete Pipeline

```python
from sleap_roots_analyze import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
    calculate_heritability_estimates,
    identify_high_heritability_traits
)

# Load data
df = load_trait_data("experiment_traits.csv")

# Identify traits
trait_cols = get_trait_columns(df)
print(f"Found {len(trait_cols)} traits")

# Clean data
df_clean, removed_samples, stats = remove_nan_samples(
    df, trait_cols, 
    max_nan_fraction=0.2
)

# Calculate heritability with filtering
results = calculate_heritability_estimates(
    df_clean, trait_cols,
    remove_low_h2=True,
    h2_threshold=0.3
)

h2_results, df_filtered, removed_traits, details = results

# Report results
print(f"Retained {len(df_filtered.columns)} columns")
print(f"Removed {len(removed_traits)} low heritability traits")
print(f"High heritability traits: {identify_high_heritability_traits(h2_results)}")
```

### Batch Processing

```python
from pathlib import Path
import pandas as pd

# Process multiple experiments
experiment_files = Path("data").glob("*_traits.csv")
all_results = {}

for file_path in experiment_files:
    experiment_name = file_path.stem
    
    # Load and process
    df = load_trait_data(file_path)
    trait_cols = get_trait_columns(df)
    
    # Calculate heritability
    h2_results = calculate_heritability_estimates(df, trait_cols)
    
    # Store results
    all_results[experiment_name] = h2_results
    
# Combine results
summary = pd.DataFrame(all_results).T
summary.to_csv("heritability_summary.csv")
```