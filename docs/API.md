# API Reference

Complete API documentation for `sleap-roots-analyze`.

## Table of Contents

- [data_cleanup](#data_cleanup-module)
- [statistics](#statistics-module)
- [data_utils](#data_utils-module)

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

#### `link_images_to_samples`

```python
link_images_to_samples(
    df: pd.DataFrame,
    image_dir: Union[str, Path],
    barcode_col: str = "Barcode",
    image_types: List[str] = ["features.png", "seg.png"]
) -> pd.DataFrame
```

Add image file paths to DataFrame based on barcodes.

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
  - Data quality: `count`, `cv` (coefficient of variation)

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
    genotype_col: str = "geno"
) -> Dict[str, Dict]
```

Perform one-way ANOVA for each trait by genotype.

**Parameters:**
- `df`: Input DataFrame
- `trait_cols`: List of trait columns
- `genotype_col`: Name of genotype column

**Returns:**
- Dictionary with ANOVA results for each trait:
  - `f_statistic`: F-statistic value
  - `p_value`: Statistical significance
  - `n_groups`: Number of genotype groups
  - `group_means`: Mean value per genotype

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
    threshold: float = 0.3
) -> List[str]
```

Identify traits with heritability above threshold.

**Parameters:**
- `heritability_results`: Dictionary from `calculate_heritability_estimates`
- `threshold`: Minimum heritability threshold (default: 0.3)

**Returns:**
- List of trait names with H² above threshold

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

#### `_convert_to_json_serializable`

```python
_convert_to_json_serializable(obj) -> Any
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