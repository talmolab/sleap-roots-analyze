# SLEAP Roots Analyze

Analyze, visualize, and interpret root traits output from [SLEAP Roots](https://github.com/talmolab/sleap-roots).

## Features

- **Data Loading & Cleaning**: Robust data loading with automatic metadata detection and NaN handling
- **Statistical Analysis**: Heritability estimation, outlier detection, and trait correlations
- **Visualization**: Generate publication-ready plots for root system traits
- **Batch Processing**: Process multiple experiments and datasets efficiently
- **Quality Control**: Automatic detection of low-quality samples and traits

## Installation

### From PyPI

```bash
pip install sleap-roots-analyze
```

### From Source

```bash
git clone https://github.com/yourusername/sleap-roots-analyze.git
cd sleap-roots-analyze
uv pip install -e .
```

## Quick Start

```python
import pandas as pd
from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
)

# Load your trait data
df = load_trait_data("path/to/traits.csv")

# Get numeric trait columns (excluding metadata)
trait_cols = get_trait_columns(df)

# Remove samples with too many NaN values
df_clean, df_removed, stats = remove_nan_samples(
    df, 
    trait_cols, 
    max_nan_fraction=0.2
)

print(f"Retained {stats['samples_retained']} samples")
print(f"Removed {stats['samples_removed']} samples with NaN values")
```

## Core Modules

### Data Cleanup (`data_cleanup.py`)

Utilities for loading and cleaning root trait data:

- `load_trait_data()`: Load CSV data with validation
- `get_trait_columns()`: Identify numeric trait columns
- `remove_nan_samples()`: Remove samples with missing data
- `remove_low_heritability_traits()`: Filter traits by heritability
- `link_images_to_samples()`: Connect trait data to image files

### Statistical Analysis (Coming Soon)

- Heritability estimation
- GWAS preparation
- Outlier detection
- Trait correlations

### Visualization (Coming Soon)

- Root system architecture plots
- Trait distributions
- QTL mapping visualizations

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/sleap-roots-analyze.git
cd sleap-roots-analyze

# Install with uv (including dev dependencies for testing, linting, etc.)
uv sync --group dev

# Note: The project uses dependency-groups, not extras
# Main dependencies only: uv sync
# With dev tools: uv sync --group dev
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-branch

# Run specific test file
uv run pytest tests/test_data_cleanup.py
```

### Code Quality

```bash
# Format code with black
uv run black src/sleap_roots_analyze tests

# Lint with ruff
uv run ruff check src/sleap_roots_analyze tests

# Type checking (if configured)
uv run mypy src/sleap_roots_analyze
```

### Test Coverage

Current test coverage:
- `data_cleanup.py`: 99%
- Additional modules: In development

## Data Format

### Input CSV Format

Expected columns:
- **Metadata**: `Barcode`, `geno` (genotype), `rep` (replicate)
- **Traits**: Numeric columns with root measurements
- **Optional**: Date columns, QC flags, experimental metadata

The package automatically detects and excludes metadata columns when processing traits.

### Example Data Structure

```csv
Barcode,geno,rep,Total.Root.Length.mm,Depth.mm,Network.Area.mm2
BC001,Genotype1,1,2500.5,450.2,12000.8
BC002,Genotype1,2,2600.3,460.5,12500.2
BC003,Genotype2,1,2300.1,420.8,11500.5
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Format code (`uv run black src/sleap_roots_analyze tests`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Documentation

- [Release Process](docs/RELEASE_PROCESS.md) - How to create releases
- [Testing Guide](docs/testing.md) - Running and writing tests
- [API Reference](docs/api.md) - Detailed API documentation (coming soon)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{sleap_roots_analyze,
  title = {SLEAP Roots Analyze},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/sleap-roots-analyze}
}
```

## Acknowledgments

- Built to work with [SLEAP Roots](https://github.com/talmolab/sleap-roots)
- Uses [pandas](https://pandas.pydata.org/) for data manipulation
- Tested with [pytest](https://docs.pytest.org/)
- Formatted with [black](https://github.com/psf/black)

## Support

For issues and questions:
- [GitHub Issues](https://github.com/yourusername/sleap-roots-analyze/issues)
- [Discussions](https://github.com/yourusername/sleap-roots-analyze/discussions)