# Project Context

## Purpose

**sleap-roots-analyze** is a Python package for analyzing root trait data output from [SLEAP Roots](https://github.com/talmolab/sleap-roots). The project focuses on:

- **Data Quality Control**: Load, clean, and validate root trait datasets with automatic metadata detection and NaN handling
- **Statistical Analysis**: Calculate broad-sense heritability (H²), ANOVA, and trait statistics for root phenotyping
- **Dimensionality Reduction**: PCA and UMAP analysis for identifying patterns in high-dimensional root trait data
- **Outlier Detection**: Multiple methods (Mahalanobis distance, Isolation Forest, PCA reconstruction) for quality control
- **Visualization**: Publication-ready static plots and interactive Plotly visualizations
- **Pipeline Infrastructure**: NetworkX-based DAG pipeline for automated, reproducible QC workflows

**Key Goal**: Provide plant biologists with robust, reproducible tools for analyzing root system architecture traits from high-throughput phenotyping experiments.

## Tech Stack

### Core Dependencies
- **Python**: >=3.11 (primary language)
- **pandas**: >=2.3.2 (data manipulation)
- **numpy**: >=2.3.2 (numerical computing)
- **scikit-learn**: >=1.7.1 (machine learning, PCA, outlier detection)
- **scipy**: >=1.16.1 (statistical tests)
- **statsmodels**: >=0.14.5 (ANOVA, heritability calculations)

### Visualization
- **matplotlib**: >=3.10.6 (static plots)
- **seaborn**: >=0.13.2 (statistical visualizations)
- **plotly**: >=6.3.0 (interactive visualizations)
- **Pillow**: >=11.0.0 (image handling)

### Advanced Analysis
- **umap-learn**: >=0.5.9.post2 (non-linear dimensionality reduction)
- **networkx**: >=3.5 (pipeline DAG infrastructure)

### Configuration & Build
- **omegaconf**: >=2.3.0 (configuration management)
- **uv**: Dependency management and build tool (replaces pip/poetry)

### Development Tools
- **pytest**: >=8.4.1 (testing framework)
- **pytest-cov**: >=6.2.1 (coverage reporting)
- **black**: >=25.1.0 (code formatting)
- **ruff**: >=0.12.11 (linting)
- **pydocstyle**: >=6.3.0 (docstring validation)
- **ipykernel**: >=6.29.5 (Jupyter notebook support)

## Project Conventions

### Code Style

**Formatting:**
- **Line length**: 88 characters (Black default)
- **Formatter**: Black (opinionated, PEP 8 compliant)
- **Linter**: Ruff with pydocstyle rules enabled
- **Imports**: `from __future__ import annotations` at top of all files

**Naming Conventions:**
- **Functions**: `snake_case` (e.g., `calculate_heritability_estimates`)
- **Classes**: `PascalCase` (e.g., `LoadDataStep`, `PCAResult`)
- **Variables**: `snake_case` (e.g., `trait_cols`, `df_clean`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_H2_THRESHOLD`)
- **Private functions**: `_leading_underscore` (e.g., `_validate_inputs`)

**Docstrings:**
- **Style**: Google format (enforced by ruff)
- **Required for**: All public functions, classes, and modules
- **Include**: Args, Returns, Raises, Examples where appropriate

**Type Hints:**
- Use when beneficial for clarity
- Especially important for public API functions
- Use `from __future__ import annotations` for modern syntax

### Architecture Patterns

**Module Organization:**
```
src/sleap_roots_analyze/
├── Core Analysis Modules:
│   ├── data_cleanup.py          # Data loading, validation, NaN handling
│   ├── statistics.py            # Heritability, ANOVA, trait statistics
│   ├── pca.py                   # PCA analysis and metrics
│   ├── outlier_detection.py     # Multiple outlier detection methods
│   ├── umap.py                  # UMAP dimensionality reduction
│   └── clustering.py            # Clustering analysis
├── Visualization Modules:
│   ├── visualization.py         # Static matplotlib/seaborn plots
│   ├── interactive_visualization.py  # Plotly interactive plots
│   ├── outlier_visualization.py # Outlier-specific visualizations
│   ├── cluster_visualization.py # Clustering visualizations
│   └── viz_utils.py            # Shared visualization utilities
├── Analysis Tools:
│   ├── cross_experiment_analysis.py  # Cross-experiment comparisons
│   └── data_utils.py           # General utility functions
└── Pipeline Infrastructure:
    └── pipeline/
        ├── config/              # Configuration dataclasses
        ├── steps/               # Pipeline step implementations
        └── utils/               # Pipeline utilities
```

**Design Patterns:**
1. **Separation of Concerns**: Analysis logic separate from visualization
2. **Pipeline Pattern**: NetworkX DAG for reproducible workflows
3. **Configuration-Driven**: OmegaConf dataclasses for type-safe config
4. **Fixture-Based Testing**: Centralized test fixtures for consistency
5. **Metadata Preservation**: Pass through metadata columns throughout pipeline

**Key Architectural Decisions:**
- Use `uv` dependency groups (not extras) for dev dependencies
- Automatic metadata detection (columns like `Barcode`, `geno`, `rep`, `QC_*`)
- Always return both cleaned data AND removed samples for transparency
- Export artifacts as CSV for reproducibility and inspection
- Support both programmatic API and future CLI usage

### Testing Strategy

**Coverage Goals:**
- **Target**: 95%+ coverage for critical modules
- **Current**: 88-98% across all core modules
- **Total Tests**: 150+ tests (growing with each feature)

**Test Organization:**
```
tests/
├── Core Test Files:
│   ├── test_data_cleanup.py      # Data loading/cleaning (98% coverage)
│   ├── test_statistics.py        # Statistical functions (92% coverage)
│   ├── test_pca.py              # PCA analysis (94% coverage)
│   ├── test_outlier_detection.py # Outlier methods (95% coverage)
│   └── test_data_utils.py       # Utilities (100% coverage)
├── Pipeline Tests:
│   ├── test_step_*.py           # Individual pipeline steps
│   └── test_qc_pipeline.py      # Integration tests
├── Fixture Files:
│   ├── conftest.py              # Pytest configuration
│   ├── fixtures.py              # Core test fixtures
│   └── fixtures_visualization.py # Visualization test fixtures
└── data/                        # Test CSV/Excel files
    ├── features.csv
    ├── traits_summary.csv
    ├── Turface_all_traits_2024.csv
    └── ...
```

**Testing Best Practices:**
1. **Use centralized fixtures** from `fixtures.py` and `fixtures_visualization.py`
2. **Test edge cases**: Empty data, missing columns, invalid inputs
3. **Mock file I/O** when appropriate to keep tests fast
4. **Use real data fixtures** for integration tests
5. **Document test intent** with clear names and docstrings
6. **Run coverage** after every feature addition
7. **Test both success and failure paths**

**Test Execution:**
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov --cov-branch

# Run specific module
uv run pytest tests/test_pca.py

# Run with verbose output
uv run pytest -v
```

### Git Workflow

**Branch Strategy:**
- **main**: Stable production code, all tests must pass
- **Feature branches**: `<username>/issue-<number>-<description>` (e.g., `elizabeth/issue-20-outlier-validation-warning`)
- **PR requirement**: All features merged via pull requests
- **Code review**: GitHub Copilot + manual review before merge

**Commit Conventions:**
- **NEVER use** `git add -A` or `git add .` - Only add relevant files
- **Review staged files** before committing with `git status`
- **Keep commits focused**: Each commit should have a single purpose
- **Commit message format**:
  ```
  <type>: <description>

  [optional body]
  ```
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `style`, `chore`
  - Example: `feat: add validation warnings for no outlier detection methods`

**PR Workflow:**
1. Create feature branch from main
2. Implement feature with tests
3. Run `uv run pytest --cov --cov-branch`
4. Run `uv run black src/sleap_roots_analyze tests`
5. Run `uv run ruff check src/sleap_roots_analyze tests`
6. Push branch and create PR
7. Address review feedback
8. Merge to main (squash commits)

## Domain Context

### Plant Phenotyping & Root Trait Analysis

**SLEAP Roots Integration:**
- This package analyzes output from [SLEAP Roots](https://github.com/talmolab/sleap-roots), a computer vision tool for root trait extraction
- Input data: CSV files with root system architecture measurements (e.g., primary root length, lateral root count, network convex area)
- Typical datasets: 100-300 samples, 20-50 traits, multiple genotypes with biological replicates

**Experimental Design:**
- **Genotypes**: Different plant varieties/lines being compared
- **Replicates**: Biological replicates (typically 3-6 per genotype)
- **Traits**: Quantitative measurements of root architecture
- **Metadata**: Experimental conditions, scanner info, QC flags

**Heritability (H²):**
- **Broad-sense heritability**: Proportion of trait variation due to genetic factors
- **Range**: 0.0 (no genetic component) to 1.0 (fully heritable)
- **Threshold**: Traits with H² < 0.3 typically filtered out as too noisy
- **Formula**: H² = σ²_G / (σ²_G + σ²_E/n) where σ²_G is genetic variance, σ²_E is environmental variance

**Common Traits:**
- **Primary root**: length, diameter, angle
- **Lateral roots**: count, density, average length
- **Network**: convex area, solidity, depth-to-width ratio
- **Temporal**: growth rates, emergence timing

**Quality Control Considerations:**
- **NaN values**: Missing measurements due to tracking failures or image quality
- **Outliers**: Measurement errors, tracking failures, biological anomalies
- **Scanner effects**: Batch effects from different imaging sessions
- **Zero inflation**: Some traits naturally have many zero values (e.g., lateral root count at early timepoints)

### Statistical Methods

**PCA (Principal Component Analysis):**
- Reduce 30-50 correlated traits to 5-10 uncorrelated principal components
- Identify traits that contribute most to phenotypic variation
- Variance threshold: Typically keep components explaining 95% of variance

**Outlier Detection Methods:**
1. **Mahalanobis Distance**: Distance in multivariate space accounting for correlations
2. **PCA Reconstruction Error**: Samples that don't fit the PCA model well
3. **Isolation Forest**: Machine learning method for complex outlier patterns

**UMAP (Uniform Manifold Approximation and Projection):**
- Non-linear dimensionality reduction for visualization
- Better preserves local structure than PCA
- Useful for identifying clusters and patterns

### Configuration Philosophy

The sleap-roots-analyze pipeline uses explicit configuration to ensure reproducibility and prevent silent failures from unintended defaults.

**Explicit Configuration Principles:**
1. **Critical parameters must be explicitly set** - Parameters that significantly affect results (cleanup thresholds, heritability thresholds, aggregation methods) must be defined in your config file
2. **Validation at pipeline start** - Configuration is validated before execution to catch errors early
3. **Two-tier validation**:
   - **Explicit config validation**: Checks that required parameters are set (errors for required, warnings for optional-but-important)
   - **Structural validation**: Checks that values are valid and internally consistent
4. **Sensible defaults provided** - Default values exist for convenience but validation encourages awareness
5. **Templates for common use cases** - Pre-configured templates in `configs/templates/` demonstrate best practices

**Configuration Templates:**
- `qc_cleanup_only_template.yaml` - For data cleanup only (NaN/zero removal)
- `qc_full_pipeline_template.yaml` - Complete QC pipeline with outlier detection and heritability filtering

**Required Parameters:**
- `cleanup.max_nan_fraction` - Max fraction of NaN values per sample (recommended: 0.25)
- `cleanup.max_zeros_per_trait` - Max fraction of zero values per trait (recommended: 0.5)
- `cleanup.max_nans_per_trait` - Max fraction of NaN values per trait (recommended: 0.2)
- `columns.genotype` - Your genotype column name (e.g., "geno", "accession")
- `columns.replicate` - Your replicate column name (e.g., "rep", "block")
- `pca.n_components` - Variance explained by selected components (recommended: 0.95)
- `outlier_removal.strategy` - How to handle outliers ("single", "subset", or "flag")
- `root_core.sources[*].aggregation_method` - Method for aggregating cores ("median" recommended)
- `heritability.threshold` - Minimum H² for trait retention (typical range: 0.3-0.6)

### Release Process

**Release Workflow:**
1. Run tests: `uv run pytest`
2. Check coverage: `uv run pytest --cov --cov-branch`
3. Format code: `uv run black src/sleap_roots_analyze tests`
4. Lint code: `uv run ruff check src/sleap_roots_analyze tests`
5. Update version: `uv version --bump patch/minor/major`
6. Update `docs/CHANGELOG.md`
7. Create release via GitHub Actions or manually

### Troubleshooting

**Common Issues:**
- **Import errors**: Run `uv sync --group dev` to install all dependencies
- **Coverage not working**: Use full module paths with `--cov`
- **Test data missing**: Ensure CSV files exist in `tests/data/`
- **Black formatting**: Run `uv run black` to auto-format
- **DataFrame fragmentation warnings**: Expected in tests, can be ignored

## Important Constraints

### Technical Constraints
- **Python Version**: Requires Python >=3.11 for modern type hints and performance
- **Memory**: Large datasets (1000+ samples, 100+ traits) may require 4-8GB RAM
- **File Format**: Input must be CSV or Excel with specific column structure
- **Required Columns**: Must have genotype, replicate, and sample ID columns
- **Numeric Traits**: Trait columns must be numeric (int/float)

### Data Constraints
- **Minimum Sample Size**: Heritability requires >=3 replicates per genotype, >=3 genotypes
- **Missing Data**: Samples with >50% NaN values are typically removed
- **Outlier Thresholds**: Chi-squared distribution requires sufficient sample size (n >= 30 recommended)
- **Zero-Inflated Traits**: Traits with >90% zeros are flagged but not automatically removed

### Performance Constraints
- **Test Speed**: All tests should complete in <2 minutes
- **Coverage Measurement**: Coverage can fail due to scipy/numpy compatibility issues (expected)
- **DataFrame Fragmentation**: Warnings expected in tests due to iterative DataFrame construction

### Code Quality Constraints
- **Line Length**: Max 88 characters (enforced by Black)
- **Docstring Coverage**: All public functions must have Google-style docstrings
- **Type Hints**: Encouraged but not strictly required
- **Test Coverage**: Minimum 90% for new features

### License & Attribution
- **License**: GNU General Public License v3.0
- **Citation Required**: Academic use should cite the package
- **Dependencies**: All dependencies must be GPL-compatible

## External Dependencies

### Required External Systems
- **SLEAP Roots**: Upstream tool that generates the trait CSV files this package analyzes
  - GitHub: https://github.com/talmolab/sleap-roots
  - Provides root trait extraction from images

### Optional External Systems
- **Image Files**: Optional sample images for interactive visualization features
  - Expected format: PNG/JPG
  - Linked via `link_images_to_samples()` function

### Development Services
- **GitHub**: Version control and CI/CD
  - Repository: https://github.com/talmolab/sleap-roots-analyze
  - Pull requests required for all changes
  - GitHub Copilot used for code review

- **PyPI** (future): Package distribution (not yet published)

### Key Python Dependencies
- **scikit-learn**: Machine learning and statistical modeling
- **statsmodels**: Advanced statistical tests (ANOVA, variance components)
- **scipy**: Scientific computing and statistical distributions
- **networkx**: DAG pipeline infrastructure
- **plotly**: Interactive web-based visualizations
- **umap-learn**: UMAP dimensionality reduction algorithm

### Documentation References
- **UV Documentation**: https://docs.astral.sh/uv/
- **Pytest**: https://docs.pytest.org/
- **Black**: https://black.readthedocs.io/
- **Pandas**: https://pandas.pydata.org/
- **scikit-learn**: https://scikit-learn.org/
