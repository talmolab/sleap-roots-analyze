# Changelog

All notable changes to `sleap-roots-analyze` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **FDR Correction for Cross-Platform Correlations** (PR #45)
  - Configurable False Discovery Rate (FDR) correction via `fdr_correction_method` config parameter
  - Three correction methods: `fdr_bh` (Benjamini-Hochberg), `fdr_by` (Benjamini-Yekutieli, default), `none`
  - New CSV output columns: `spearman_p_adjusted`, `pearson_p_adjusted`, `significant_fdr`
  - Updated visualization to show FDR-corrected significance counts in summary plots
  - Comprehensive documentation in `docs/CROSS_PLATFORM_ANALYSIS.md` with mathematical formulations
  - Pipeline summary JSON now includes FDR metadata (`fdr_correction_method`, `significant_correlations`)
- **Visualization Pipeline** with DAG-based architecture for automated visualization workflows
  - 10 new pipeline steps (PCAAnalysisStep, LoadDataAndImagesStep, UMAPAnalysisStep, ClusterAnalysisStep, etc.)
  - 4 configuration presets: minimal, standard, comprehensive, publication
  - Example scripts and comprehensive documentation
- **Unified pipeline architecture** with modular configuration system
  - Reorganized config into reusable components (25 components in `config/components.py`)
  - Composition-based config for QC and Viz pipelines
  - All steps unified in single `pipeline/steps/` directory
  - Pipeline orchestrators in `pipeline/pipelines/` subdirectory
- **Adaptive sizing utilities** (`viz_utils.py`) for automatic plot dimension calculations
  - `calculate_figure_size()` with layout-aware sizing (single, horizontal, vertical, grid)
  - `calculate_grid_dimensions()` for optimal subplot layouts
  - `calculate_subplot_grid_size()` for multi-trait plots
  - `calculate_correlation_matrix_size()` and `calculate_barplot_size()` for specific plot types
  - 347 comprehensive tests for adaptive sizing functions
- **Comprehensive test coverage improvements**
  - PCAAnalysisStep: 31% → 100% coverage (11 new tests)
  - RemoveOutliersStep: 55% → 100% coverage (17 new tests)
  - Overall pipeline coverage improved from 66% to 68%
  - Tests cover all removal strategies, feature selection methods, edge cases, and file outputs
- **Effect size-based goodness-of-fit evaluation** for large samples (n > 500) in Mahalanobis outlier detection
  - K-S test becomes hypersensitive with large n, now uses K-S statistic magnitude instead of p-values
  - New thresholds: excellent (<0.05), good (<0.10), acceptable (<0.15), poor (<0.20), very poor (≥0.20)
  - `print_goodness_of_fit_summary()` function for formatted console output with interpretation
  - References to Massey (1951) and Sullivan & Feinn (2012) for statistical methodology
- **Configurable ID column** in interactive visualization functions
  - Added `id_col` parameter to `create_interactive_scatter_with_images()`, `create_interactive_pca_with_images()`, and `create_interactive_umap_with_images()`
  - Fixes hardcoded "Barcode" column assumption, now supports lowercase or custom column names
- **Pipeline validation warnings** for outlier detection configuration (Issue #20)
  - Early detection when no outlier detection methods are configured
  - Clear, actionable warning messages in pipeline output
  - Graceful handling with pipeline continuing successfully

### Changed
- **Adaptive boxplot layout** for trait visualizations (Issue #73):
  - Auto-switch from vertical to horizontal orientation when genotype count exceeds threshold (default: 8)
  - Configurable via `orientation` ("vertical", "horizontal", "auto") and `horizontal_threshold` parameters
  - Consistent unfilled boxplot styling across both orientations: blue (`#1f77b4`) outlines, green (`#2ca02c`) medians, gridlines
  - Adaptive figure sizing: subplot width scales with genotype count (0.5 in/genotype, min 4.0, max 20.0 inches)
  - Font scaling for x-axis labels when genotype count exceeds 10 (min 6pt)
  - `tight_layout()` called by batched wrapper after suptitle (not in base function) to prevent overlap
  - Replaced seaborn horizontal boxplot with matplotlib for consistent styling
- **PC boxplot layout** now stacks vertically (1 column) instead of grid layout for better display with many genotypes
  - Default figsize updated from (16, 10) to (20, 6) for wider genotype labels
- **Goodness-of-fit display** removed from outlier detection plots (too crowded)
  - Results still available in JSON output and via `print_goodness_of_fit_summary()`
  - Cleaner, more focused visualization

### Fixed
- **JSON serialization in pipeline summaries** (PR #45)
  - Added Path object handling with `as_posix()` in `convert_to_json_serializable()`
  - Excluded non-serializable sklearn PCA object from StepResult metadata
  - Fixed numpy type serialization (int64, float64) in pipeline summary JSON
- **TypeError in interactive image gallery** when image paths are None
  - Added null check before Path conversion in `create_interactive_image_gallery()`
- **ipykernel hanging bug** in VS Code Jupyter notebooks
  - Pinned ipykernel to <7.0.0 (ipykernel 7.x has known kernel hanging issues)

### Added
- Comprehensive test suite with 150+ tests achieving 97%+ coverage across all modules
- Complete PCA module with mathematical validation (88 tests)
  - Per-feature variance explained calculations with configurable ddof
  - Mathematical validation test suite (11 properties verified)
  - `calculate_pca_metrics()` for comprehensive PCA metrics
  - `build_feature_metrics_df()` for per-feature analysis
  - Edge case handling for single samples and constant features
- **Outlier Detection Module** (`sleap_roots_analyze.outlier_detection`) with three complementary methods:
  - `detect_outliers_mahalanobis()`: Statistical detection using Mahalanobis distance with chi-squared and custom thresholds
  - `detect_outliers_pca()`: Outlier detection based on PCA reconstruction error
  - `detect_outliers_isolation_forest()`: Tree-based anomaly detection for complex, non-linear patterns
  - `remove_outliers_from_data()`: Utility to remove outliers while preserving DataFrame structure and metadata
  - `calculate_outlier_threshold()`: Calculate chi-squared or direct distance thresholds
  - `identify_outliers_from_distances()`: Identify outliers from pre-calculated distances
  - Support for robust covariance estimation using MinCovDet
  - Automatic index preservation through NaN removal
  - Comprehensive test suite with 94% coverage (74 tests)
- Improved PCA documentation with scikit-learn references and mathematical proofs
- Numerical accuracy tests with known correct answers
- Edge case fixtures for boundary condition testing
- `.gitattributes` file for consistent line endings across platforms
- Integrated heritability filtering in `calculate_heritability_estimates()`
- Optional saving of removed samples in `remove_nan_samples()`
- Detailed removal statistics and metadata tracking
- Modular data cleanup functions: `remove_zero_inflated_traits()`, `remove_traits_with_many_nans()`, `remove_low_sample_traits()`
- Claude commands for PR review (`.claude/commands/review-pr.md`) and changelog updates (`.claude/commands/update-changelog.md`)

- **Visualization Module** (`sleap_roots_analyze.visualization`):
  - `create_feature_contribution_heatmap()`: Heatmap showing feature contributions to principal components
  - `save_publication_figure()`: Save figures in publication-ready formats (PDF, EPS, PNG, SVG)
  - `identify_extreme_phenotypes()`: Identify genotypes with extreme phenotypes for each trait
  - `create_phenotype_variation_plot()`: Box plots with jittered points showing phenotypic variation
  - `create_feature_contribution_plot()`: Now uses pre-calculated contributions from `run_pca_and_export_artifacts` for efficiency
  - All visualization functions now use Google-style docstrings for consistency
- **PCA Module Enhancements**:
  - `run_pca_and_export_artifacts()`: Comprehensive PCA analysis with CSV export functionality
    - Exports loadings, trait variance contributions, PC scores, and variance explained
    - Calculates trait fractional contributions that sum to 1.0
    - Integration with existing visualization functions
  - Added tests verifying fractional contributions sum to 1 in all scenarios
  - Added metadata hygiene tests for `trait_cols=None` behavior
- **Outlier Visualization Module** (`sleap_roots_analyze.outlier_visualization`):
  - Support for all three outlier detection methods
  - `create_comprehensive_outlier_comparison()`: Compare results from multiple detection methods
  - Integration with new PCA artifact export functionality

### Changed
- **Outlier Detection Refactoring**:
  - Removed redundant validation checks from outlier detection functions (now handled by `perform_pca_analysis`)
  - Isolation Forest now uses shared `standardize_data` function for consistency
  - Standardized feature naming convention across all methods (using "Feature_" prefix)
- Made `statsmodels` a required dependency (removed `mixed_model_available` checks)
- Integrated `save_nan_removed_rows` functionality into `remove_nan_samples()`
- Moved utility functions to `data_utils.py` module
- Improved test fixtures organization with categories (heritability, ANOVA, edge cases)
- Updated documentation to reflect actual implementation
- Renamed `link_images_to_samples()` to `link_rhizovision_images_to_samples()` for clarity
- Made `_convert_to_json_serializable()` public API by removing underscore prefix
- Added configurable `alpha` parameter to `perform_anova_by_genotype()` (default: 0.05)
- Changed metadata key from `_metadata` to `__calculation_metadata__` to avoid trait name conflicts
- Refactored `apply_data_cleanup_filters()` to use new modular functions

### Fixed
- Line ending consistency issues across different platforms
- Test accuracy for heritability calculations with mixed models
- Handling of infinity values in statistical calculations
- Edge case handling for insufficient data conditions
- Duplicate imports in `test_statistics.py` (PR #2 review)
- Misplaced docstring between test classes (PR #2 review)
- Brittle test dependency in heritability tests (PR #2 review)

### Development
- Added `black` code formatter configuration
- Added `ruff` linter with Google docstring convention
- Improved test organization and fixture management
- Enhanced numerical stability tests

## [0.1.0] - 2025-01-XX (Upcoming)

### Added
- **Core Modules**:
  - `data_cleanup.py`: Data loading and cleaning utilities
  - `statistics.py`: Statistical analysis including heritability estimation
  - `data_utils.py`: Utility functions for data processing
  - `outlier_detection.py`: Placeholder for outlier detection (in development)

- **Data Cleaning Features**:
  - `load_trait_data()`: Load CSV/Excel files with validation
  - `get_trait_columns()`: Automatic metadata detection and exclusion
  - `remove_nan_samples()`: Sample filtering based on missing data
  - `remove_zero_inflated_traits()`: Detection and removal of zero-inflated traits
  - `remove_low_variance_traits()`: Filter traits with insufficient variation
  - `link_images_to_samples()`: Connect trait data to image files

- **Statistical Analysis**:
  - `calculate_heritability_estimates()`: Broad-sense heritability using mixed models
  - `perform_anova_by_genotype()`: ANOVA analysis for genotype effects
  - `calculate_trait_statistics()`: Comprehensive trait statistics
  - `identify_high_heritability_traits()`: Threshold-based trait identification
  - `analyze_heritability_thresholds()`: Threshold sensitivity analysis

- **Testing Infrastructure**:
  - Centralized fixtures in `tests/fixtures.py`
  - Test data files for various experimental designs
  - Coverage reporting configuration
  - Edge case and numerical accuracy testing

- **Documentation**:
  - Comprehensive README with examples
  - Testing guide with best practices
  - Release process documentation
  - Claude AI development guidelines

- **Development Tools**:
  - `uv` package manager support with dependency groups
  - `black` code formatting configuration
  - `ruff` linting with Google docstring convention
  - `pytest` with coverage reporting

### Dependencies
- Core: `pandas>=2.0.0`, `numpy>=1.24.0`, `scipy>=1.10.0`, `statsmodels>=0.14.0`
- Development: `pytest>=8.0.0`, `pytest-cov>=6.0.0`, `black>=24.0.0`, `ruff>=0.8.0`

## Version History

### Versioning Scheme

We use [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for backwards-compatible functionality additions  
- **PATCH** version for backwards-compatible bug fixes

### Pre-release Versions

- `0.0.1-alpha` - Initial development
- `0.0.2-alpha` - Core data loading functionality
- `0.0.3-alpha` - Statistical analysis implementation
- `0.0.4-alpha` - Test suite development

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Authors

* **Elizabeth Berrigan** - *Initial work* - [GitHub Profile](https://github.com/eberrigan)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.