# SLEAP Roots Analyze

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Coverage: 97%](https://img.shields.io/badge/Coverage-97%25-brightgreen)
![Tests: 134+](https://img.shields.io/badge/Tests-134%2B-brightgreen)
![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)

Statistical analysis tools for root trait data from [SLEAP Roots](https://github.com/talmolab/sleap-roots).

## Installation

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-roots-analyze.git
cd sleap-roots-analyze

# Install with uv
uv sync --group dev  # Includes development dependencies
```

## Quick Start

### Load and Clean Data

```python
from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
)

# Load data
df = load_trait_data("path/to/traits.csv")

# Get trait columns (excludes metadata automatically)
trait_cols = get_trait_columns(df)

# Remove samples with >20% missing data
df_clean, df_removed, stats = remove_nan_samples(
    df, trait_cols, max_nan_fraction=0.2
)
```

### Calculate Heritability

```python
from sleap_roots_analyze.statistics import calculate_heritability_estimates

# Calculate heritability for all traits
h2_results = calculate_heritability_estimates(
    df_clean,
    trait_cols,
    genotype_col="geno",
    replicate_col="rep"
)

# Filter low heritability traits
h2_results, df_filtered, removed, details = calculate_heritability_estimates(
    df_clean,
    trait_cols,
    remove_low_h2=True,
    h2_threshold=0.3
)
```

### PCA Analysis

```python
from sleap_roots_analyze.pca import perform_pca_analysis

# Run PCA with automatic component selection
result = perform_pca_analysis(
    df_filtered,
    standardize=True,
    explained_variance_threshold=0.95
)

# Access results
pca_model = result['pca']
transformed_data = result['transformed_data']
loadings = result['loadings']
```

### Outlier Detection

```python
from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_isolation_forest,
    remove_outliers_from_data
)

# Detect outliers using Mahalanobis distance
outliers_maha = detect_outliers_mahalanobis(
    df_filtered[trait_cols],
    use_robust=True
)

# Or use Isolation Forest for complex patterns
outliers_iso = detect_outliers_isolation_forest(
    df_filtered[trait_cols],
    contamination=0.1
)

# Remove outliers from data
df_clean, df_outliers = remove_outliers_from_data(
    df_filtered,
    outliers_maha['outlier_indices'],
    return_outliers=True
)
```

## Features

- **Data Cleaning**: Automatic metadata detection, NaN handling, zero-inflated trait removal
- **Statistical Analysis**: Broad-sense heritability (H²), ANOVA, trait statistics
- **PCA Analysis**: Dimensionality reduction with automatic component selection
- **Outlier Detection**: Mahalanobis, PCA reconstruction, and Isolation Forest methods

## Data Format

Expected CSV structure:
```csv
Barcode,geno,rep,trait1,trait2,trait3,...
BC001,Genotype1,1,100.5,200.3,50.2,...
BC002,Genotype1,2,102.3,195.8,48.9,...
```

Required columns:
- **Genotype**: `geno` (configurable)
- **Replicate**: `rep` (configurable)
- **Sample ID**: `Barcode` (configurable)
- **Traits**: Any numeric columns

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black src tests

# Lint code
uv run ruff check src tests

# Coverage report
uv run pytest --cov --cov-branch
```

## Project Structure

```
sleap-roots-analyze/
├── src/sleap_roots_analyze/
│   ├── data_cleanup.py      # Data loading and cleaning
│   ├── statistics.py         # Statistical analysis
│   ├── pca.py               # PCA analysis
│   └── outlier_detection.py # Outlier detection
├── tests/                   # Test suite
├── docs/                    # Documentation
└── pyproject.toml          # Project configuration
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{sleap_roots_analyze,
  title = {SLEAP Roots Analyze},
  author = {Elizabeth Berrigan},
  year = {2025},
  url = {https://github.com/talmolab/sleap-roots-analyze}
}
```