# SLEAP Roots Analyze

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

## Features

- **Data Cleaning**: Automatic metadata detection, NaN handling, zero-inflated trait removal
- **Statistical Analysis**: Broad-sense heritability (H²), ANOVA, trait statistics
- **PCA Analysis**: Dimensionality reduction with automatic component selection
- **Outlier Detection**: Statistical outlier identification (in development)

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

MIT License - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{sleap_roots_analyze,
  title = {SLEAP Roots Analyze},
  author = {Elizabeth Berrigan},
  year = {2025},
  url = {https://github.com/talmolab/sleap-roots-analyze}
}
```