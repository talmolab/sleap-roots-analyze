# SLEAP Roots Analyze

Statistical analysis and visualization tools for root trait data from [SLEAP Roots](https://github.com/talmolab/sleap-roots).

## Overview

`sleap-roots-analyze` provides comprehensive tools for analyzing root system architecture traits, with a focus on:
- **Data quality control** and cleaning
- **Statistical analysis** including heritability estimation
- **Trait filtering** based on statistical thresholds
- **Batch processing** of large-scale phenotyping experiments

## Features

### ✅ Implemented

- **Data Loading & Cleaning** (`data_cleanup.py`)
  - Automatic metadata detection and exclusion
  - NaN handling with configurable thresholds
  - Zero-inflated trait detection and removal
  - Sample filtering based on data quality

- **Statistical Analysis** (`statistics.py`)
  - Broad-sense heritability (H²) calculation using mixed models
  - ANOVA by genotype
  - Trait statistics (mean, std, skewness, kurtosis, percentiles)
  - Heritability threshold analysis
  - Optional trait filtering based on heritability

- **Utility Functions** (`data_utils.py`)
  - Timestamped output directory creation
  - JSON serialization helpers for numpy types

### 🚧 In Development

- **Outlier Detection** (`outlier_detection.py`)
- **Visualization** tools for root traits
- **GWAS preparation** utilities

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/talmolab/sleap-roots-analyze.git
cd sleap-roots-analyze

# Install with uv (main dependencies only)
uv sync

# Install with development dependencies
uv sync --group dev
```

### Using pip

```bash
pip install sleap-roots-analyze  # When published to PyPI
```

## Quick Start

### Basic Data Cleaning

```python
from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
)

# Load trait data
df = load_trait_data("path/to/traits.csv")

# Identify numeric trait columns (excludes metadata automatically)
trait_cols = get_trait_columns(df)

# Remove samples with >20% NaN values
df_clean, df_removed, stats = remove_nan_samples(
    df, 
    trait_cols,
    max_nan_fraction=0.2,
    save_removed_path="removed_samples.csv"  # Optional: save removed samples
)

print(f"Retained {stats['samples_retained']} of {stats['total_samples']} samples")
```

### Heritability Analysis

```python
from sleap_roots_analyze.statistics import (
    calculate_heritability_estimates,
    identify_high_heritability_traits,
)

# Calculate heritability for all traits
h2_results = calculate_heritability_estimates(
    df_clean,
    trait_cols,
    genotype_col="geno",
    replicate_col="rep"
)

# Print heritability for each trait
for trait, results in h2_results.items():
    if "heritability" in results:
        print(f"{trait}: H² = {results['heritability']:.3f}")

# Identify high heritability traits (H² > 0.3)
high_h2_traits = identify_high_heritability_traits(h2_results, threshold=0.3)
print(f"High heritability traits: {high_h2_traits}")
```

### Integrated Filtering Pipeline

```python
# Calculate heritability and filter low H² traits in one step
results = calculate_heritability_estimates(
    df_clean,
    trait_cols,
    remove_low_h2=True,      # Enable filtering
    h2_threshold=0.3,         # Remove traits with H² < 0.3
    barcode_col="Barcode",    # Preserve sample identifiers
)

# When filtering is enabled, returns tuple
h2_results, df_filtered, removed_traits, removal_details = results

print(f"Removed {len(removed_traits)} low heritability traits")
print(f"Retained {len(df_filtered.columns)} columns in filtered dataset")
```

## Data Format Requirements

### Expected CSV Structure

```csv
Barcode,geno,rep,trait1,trait2,trait3,...
BC001,Genotype1,1,100.5,200.3,50.2,...
BC002,Genotype1,2,102.3,195.8,48.9,...
BC003,Genotype2,1,95.2,210.5,55.3,...
```

### Required Columns

- **Genotype identifier**: Default `"geno"`, configurable
- **Replicate identifier**: Default `"rep"`, configurable  
- **Sample identifier**: Default `"Barcode"`, configurable
- **Trait columns**: Any numeric columns not matching metadata patterns

### Automatically Excluded Metadata Patterns

The package automatically detects and excludes these columns from trait analysis:
- Standard: `Barcode`, `geno`, `rep`, `genotype`, `replicate`
- QC flags: `QC_*`, `outlier*`
- Experimental: `wave_*`, `scan_*`, `plant_id`, `experiment_*`
- Dates: Columns ending in `_date`, `_time`, `_day`
- Image paths: Columns ending in `_path`, `_file`, `.png`, `.jpg`

## Testing

The package includes comprehensive test coverage:

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov --cov-branch

# Run specific test module
uv run pytest tests/test_statistics.py -v

# Current test coverage:
# - data_cleanup.py: 99%
# - statistics.py: 95%+ 
# - 45 tests passing
```

### Test Organization

- `tests/fixtures.py`: Centralized test fixtures with known correct answers
- `tests/test_data_cleanup.py`: Data loading and cleaning tests
- `tests/test_statistics.py`: Statistical analysis and numerical accuracy tests
- `tests/data/`: Sample CSV files for testing

## Development

### Code Formatting and Linting

```bash
# Format code with black
uv run black src/sleap_roots_analyze tests

# Check linting with ruff
uv run ruff check src/sleap_roots_analyze tests

# Fix linting issues
uv run ruff check --fix src/sleap_roots_analyze tests
```

### Project Structure

```
sleap-roots-analyze/
├── src/sleap_roots_analyze/
│   ├── __init__.py
│   ├── data_cleanup.py      # Data loading and cleaning
│   ├── statistics.py         # Statistical analysis
│   ├── data_utils.py         # Utility functions
│   └── outlier_detection.py  # Outlier detection (in development)
├── tests/
│   ├── fixtures.py           # Test fixtures
│   ├── test_data_cleanup.py  # Data cleanup tests
│   ├── test_statistics.py    # Statistics tests
│   └── data/                 # Test data files
├── docs/
│   ├── RELEASE_PROCESS.md   # Release workflow
│   └── testing.md            # Testing guide
├── pyproject.toml            # Project configuration
├── CLAUDE.md                 # AI assistant guidelines
└── README.md                 # This file
```

### Dependencies

Core dependencies:
- `pandas >= 2.0.0`: Data manipulation
- `numpy >= 1.24.0`: Numerical operations
- `scipy >= 1.10.0`: Statistical functions
- `statsmodels >= 0.14.0`: Mixed models for heritability

Development dependencies:
- `pytest >= 8.0.0`: Testing framework
- `pytest-cov >= 6.0.0`: Coverage reporting
- `black >= 24.0.0`: Code formatting
- `ruff >= 0.8.0`: Linting

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `uv run pytest`
5. Format code: `uv run black src tests`
6. Commit with descriptive message
7. Push and create a Pull Request

## Documentation

- [API Reference](docs/API.md): Complete API documentation
- [Testing Guide](docs/TESTING.md): Writing and running tests  
- [Contributing Guide](docs/CONTRIBUTING.md): How to contribute
- [Changelog](docs/CHANGELOG.md): Version history and changes
- [Release Process](docs/RELEASE_PROCESS.md): Creating releases
- [Claude Guidelines](CLAUDE.md): Development guidelines for AI assistants

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{sleap_roots_analyze,
  title = {SLEAP Roots Analyze: Statistical Analysis Tools for Root Traits},
  author = {Elizabeth Berrigan},
  year = {2025},
  url = {https://github.com/talmolab/sleap-roots-analyze}
}
```

## Related Projects

- [SLEAP Roots](https://github.com/talmolab/sleap-roots): Deep learning-based root system architecture analysis
- [SLEAP](https://github.com/talmolab/sleap): General framework for animal pose tracking

## Support

- **Issues**: [GitHub Issues](https://github.com/talmolab/sleap-roots-analyze/issues)
- **Discussions**: [GitHub Discussions](https://github.com/talmolab/sleap-roots-analyze/discussions)
- **Documentation**: [Read the Docs](https://sleap-roots-analyze.readthedocs.io) (coming soon)