<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Claude Development Guidelines

This document provides guidelines for AI assistants (particularly Claude) when working on the `sleap-roots-analyze` project.

## Project Overview

`sleap-roots-analyze` is a Python package for analyzing root trait data output from SLEAP Roots. The package focuses on:
- Data loading and cleaning
- Statistical analysis of root traits
- Visualization of root system architecture
- Quality control and outlier detection

## Configuration Philosophy

The sleap-roots-analyze pipeline uses explicit configuration to ensure reproducibility and prevent silent failures from unintended defaults.

### Explicit Configuration Principles

1. **Critical parameters must be explicitly set** - Parameters that significantly affect results (cleanup thresholds, heritability thresholds, aggregation methods) must be defined in your config file
2. **Validation at pipeline start** - Configuration is validated before execution to catch errors early
3. **Two-tier validation**:
   - **Explicit config validation**: Checks that required parameters are set (errors for required, warnings for optional-but-important)
   - **Structural validation**: Checks that values are valid and internally consistent
4. **Sensible defaults provided** - Default values exist for convenience but validation encourages awareness
5. **Templates for common use cases** - Pre-configured templates in `configs/templates/` demonstrate best practices

### Configuration Templates

Two templates are provided in `configs/templates/`:

1. **qc_cleanup_only_template.yaml** - For data cleanup only (NaN/zero removal)
   - No outlier detection
   - Will generate warning about empty outlier detection (this is expected)
   - Use when you only want basic data cleaning

2. **qc_full_pipeline_template.yaml** - Complete QC pipeline
   - Data cleanup + outlier detection + heritability filtering
   - Multiple detection methods (Mahalanobis, Isolation Forest, K-Means)
   - Subset strategy for robust outlier removal

### Required Parameters

These must be explicitly set in your configuration:

**Cleanup Configuration:**
- `cleanup.max_nan_fraction` - Max fraction of NaN values per sample (recommended: 0.25)
- `cleanup.max_zeros_per_trait` - Max fraction of zero values per trait (recommended: 0.5)
- `cleanup.max_nans_per_trait` - Max fraction of NaN values per trait (recommended: 0.2)

**Column Mappings (dataset-specific):**
- `columns.genotype` - Your genotype column name (e.g., "geno", "accession")
- `columns.replicate` - Your replicate column name (e.g., "rep", "block")

**PCA Configuration:**
- `pca.n_components` - Variance explained by selected components (recommended: 0.95)

**Outlier Removal (if detection enabled):**
- `outlier_removal.strategy` - How to handle outliers ("single", "subset", or "flag")

**Root Core Aggregation (if using root core data):**
- `root_core.sources[*].aggregation_method` - Method for aggregating cores ("median" or "mean")
  - Recommended: "median" (robust to outliers and measurement errors)

**Heritability (if filtering enabled):**
- `heritability.threshold` - Minimum H² for trait retention (typical range: 0.3-0.6)

### Configuration Review

All existing QC configs have been reviewed and are compliant with these requirements. See `docs/configuration_review.md` for detailed status of each config file.

### Validation Warnings

**Expected warnings:**
- Empty outlier detection in cleanup-only configs - This is valid if you only want data cleanup
- Consider adding outlier detection suggestion - Informational, not an error

**Configuration errors will prevent pipeline execution** - Fix any validation errors before running the pipeline.

## Development Environment

### Dependency Management

The project uses `uv` with dependency groups (not extras):
- Main dependencies: Defined in `[project.dependencies]`
- Dev dependencies: Defined in `[dependency-groups.dev]`

To install:
```bash
# Install only main dependencies
uv sync

# Install main + dev dependencies (recommended for development)
uv sync --group dev
```

### Tools and Commands

The project uses `uv` for dependency management. Key commands:

```bash
# Environment setup (installs main + dev dependencies)
uv sync --group dev

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
│       ├── data_cleanup.py      # Data loading and cleaning utilities
│       ├── statistics.py        # Statistical analysis
│       ├── pca.py               # PCA analysis module
│       ├── data_utils.py        # Utility functions
│       └── outlier_detection.py # Outlier detection using Mahalanobis distance
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Pytest configuration
│   ├── fixtures.py              # Centralized test fixtures
│   ├── test_data_cleanup.py    # Tests for data_cleanup module
│   ├── test_statistics.py      # Tests for statistics module
│   ├── test_pca.py             # Tests for PCA module
│   ├── test_outlier_detection.py # Tests for outlier detection module
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

- Target: 95%+ coverage for critical modules
- Current status:
  - `data_cleanup.py`: 98% coverage ✅
  - `statistics.py`: 92% coverage ✅
  - `pca.py`: 94% coverage ✅
  - `data_utils.py`: 100% coverage ✅
  - `outlier_detection.py`: 95% coverage ✅

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

### outlier_detection.py

Key functions to maintain:
- `detect_outliers_mahalanobis()` - Detect outliers using Mahalanobis distance on PCA-transformed data
- `calculate_outlier_threshold()` - Calculate chi-squared or distance thresholds
- `identify_outliers_from_distances()` - Identify outliers from pre-calculated distances

#### Implementation Details

The module integrates with the PCA module to:
- Perform PCA transformation for dimensionality reduction
- Select components based on variance threshold (default 95%)
- Calculate Mahalanobis distances in PCA space
- Use chi-squared distribution for automatic threshold determination
- Support robust covariance estimation via MinCovDet

### pca.py

Key functions to maintain:
- `perform_pca_analysis()` - Complete PCA pipeline with standardization
- `select_top_features_from_pca()` - Select top features based on PCA loadings using various strategies
- `calculate_mahalanobis_distances()` - Calculate distances with optional robust estimation
- `calculate_pca_metrics()` - Comprehensive metrics including per-feature variance
- `build_feature_metrics_df()` - Build DataFrame with per-feature PCA metrics

### Additional Implemented Modules

- `visualization.py` - Comprehensive plotting utilities for PCA, UMAP, and trait analysis
- `cross_experiment_analysis.py` - Cross-experiment correlation and comparison tools
- `outlier_visualization.py` - Specialized plots for outlier detection results
- `umap.py` - UMAP dimensionality reduction
- `data_utils.py` - Utility functions for data manipulation
  - Key function: `sanitize_trait_names()` - Cleans trait names for better visualization (converts `Median.Number.of.Roots` → `Med Num Roots`)

### Future Modules

Planned modules to develop:
- Pipeline modules (currently in development):
  - `pipeline_config.py` - Configuration management
  - `pipeline_utils.py` - Pipeline utilities
  - `visualization_pipeline.py` - Automated visualization pipeline
  - `interactive_visualization.py` - Interactive plot generation

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

#### Git Commit Best Practices
- **NEVER use `git add -A` or `git add .`** - Only add relevant files
- **Review staged files** before committing with `git status`
- **Keep commits focused** - Each commit should have a single purpose
- **Write clear commit messages** that describe the why, not just the what

## Release Process

1. **Run tests**: `uv run pytest`
2. **Check coverage**: `uv run pytest --cov --cov-branch`
3. **Format code**: `uv run black src/sleap_roots_analyze tests`
4. **Update version**: `uv version --bump patch/minor/major`
5. **Update docs/CHANGELOG.md**
6. **Create release**: Via GitHub Actions or manually

## Troubleshooting

### Common Issues

1. **Import errors**: Run `uv sync --group dev` to install all dependencies
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