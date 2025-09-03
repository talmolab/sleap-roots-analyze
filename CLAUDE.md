# Claude Development Guidelines

This document provides guidelines for AI assistants (particularly Claude) when working on the `sleap-roots-analyze` project.

## Project Overview

`sleap-roots-analyze` is a Python package for analyzing root trait data output from SLEAP Roots. The package focuses on:
- Data loading and cleaning
- Statistical analysis of root traits
- Visualization of root system architecture
- Quality control and outlier detection

## Development Environment

### Tools and Commands

The project uses `uv` for dependency management. Key commands:

```bash
# Environment setup
uv sync

# Run tests
uv run pytest

# Coverage analysis
uv run pytest --cov --cov-branch

# Code formatting
uv run black src/sleap_roots_analyze tests

# Linting
uv run ruff check src/sleap_roots_analyze tests
```

### Command Documentation

Detailed command documentation is available in `.claude/commands/`:
- `coverage.md` - Running tests with coverage
- `lint.md` - Code linting with ruff
- `black.md` - Code formatting with black

## Code Structure

```
sleap-roots-analyze/
├── src/
│   └── sleap_roots_analyze/
│       ├── __init__.py
│       └── data_cleanup.py      # Data loading and cleaning utilities
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── fixtures.py              # Centralized test fixtures
│   ├── test_data_cleanup.py    # Tests for data_cleanup module
│   └── data/                   # Test data files
│       ├── features.csv
│       ├── traits_summary.csv
│       └── ...
├── docs/
│   ├── RELEASE_PROCESS.md      # Release workflow documentation
│   └── testing.md               # Testing guide
└── pyproject.toml              # Project configuration
```

## Testing Guidelines

### Test Coverage Goals

- Target: 100% coverage for critical modules
- Current status:
  - `data_cleanup.py`: 99% coverage ✅
  - Other modules: To be developed

### Writing Tests

1. **Use centralized fixtures** from `tests/fixtures.py`
2. **Test edge cases**: Include tests for error conditions and boundary values
3. **Mock external dependencies**: Use `unittest.mock` for file I/O when appropriate
4. **Keep tests fast**: Use sample data fixtures for quick testing
5. **Document tests**: Use clear test names and docstrings

### Test Data Fixtures

The project includes several CSV fixtures in `tests/data/`:
- `features.csv` - Root system features
- `traits_summary.csv` - Summarized trait data
- `traits_summary_lateral.csv` - Lateral root data
- `traits_11DAG_cleaned_qc_scanner_independent.csv` - 11 DAG trait data
- `Turface_all_traits_2024.csv` - Turface experiment data
- `Wheat_EDPIE_cylinder_master_data.xlsx` - Wheat EDPIE data

## Code Style

### Formatting Rules

- **Line length**: 88 characters (black default)
- **Imports**: Sorted with `from __future__ import annotations` at top
- **Docstrings**: Google style
- **Type hints**: Use when beneficial for clarity

### Black Configuration

```toml
[tool.black]
line-length = 88
```

### Ruff Configuration

```toml
[tool.ruff.lint]
select = ["D"]  # pydocstyle

[tool.ruff.lint.pydocstyle]
convention = "google"
```

## Module Development

### data_cleanup.py

Key functions to maintain:
- `load_trait_data()` - Load and validate CSV data
- `get_trait_columns()` - Identify numeric traits vs metadata
- `remove_nan_samples()` - Handle missing data
- `remove_low_heritability_traits()` - Filter by heritability
- `link_images_to_samples()` - Connect traits to images

#### Metadata Detection

The module automatically excludes these metadata patterns:
- Standard columns: `Barcode`, `geno`, `rep`
- QC columns: `QC_*`, `outlier*`
- Experimental metadata: `wave_name`, `scan_*`, `plant_id`
- Date/time columns
- Non-numeric columns

### Future Modules

Planned modules to develop:
- `statistical_analysis.py` - Heritability, GWAS prep
- `visualization.py` - Plotting utilities
- `outlier_detection.py` - Statistical outlier detection
- `gwas_prep.py` - GWAS data preparation

## Best Practices

### When Adding New Features

1. **Write tests first** (TDD approach)
2. **Update fixtures** if new test data is needed
3. **Document functions** with clear docstrings
4. **Run coverage** to ensure tests are comprehensive
5. **Format code** with black before committing
6. **Update documentation** in relevant .md files

### Common Patterns

#### Loading Data with Validation
```python
def load_data(path, required_cols):
    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df
```

#### Handling Optional Parameters
```python
def process_data(df, optional_col=None):
    if optional_col and optional_col in df.columns:
        # Process with optional column
    else:
        # Process without it
```

## Release Process

1. **Run tests**: `uv run pytest`
2. **Check coverage**: `uv run pytest --cov --cov-branch`
3. **Format code**: `uv run black src/sleap_roots_analyze tests`
4. **Update version**: `uv version --bump patch/minor/major`
5. **Update CHANGELOG.md**
6. **Create release**: Via GitHub Actions or manually

## Troubleshooting

### Common Issues

1. **Import errors**: Run `uv sync` to install dependencies
2. **Coverage not working**: Use full module paths with `--cov`
3. **Test data missing**: Ensure CSV files exist in `tests/data/`
4. **Black formatting**: Run `uv run black` to auto-format

### Performance Warnings

DataFrame fragmentation warnings in tests are expected and can be ignored:
```
PerformanceWarning: DataFrame is highly fragmented
```

## References

- [UV Documentation](https://docs.astral.sh/uv/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Black Documentation](https://black.readthedocs.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [SLEAP Roots](https://github.com/talmolab/sleap-roots)