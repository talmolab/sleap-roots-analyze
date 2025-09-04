# Changelog

All notable changes to `sleap-roots-analyze` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive test suite with 45+ tests achieving 95%+ coverage
- Numerical accuracy tests with known correct answers
- Edge case fixtures for boundary condition testing
- `.gitattributes` file for consistent line endings across platforms
- Integrated heritability filtering in `calculate_heritability_estimates()`
- Optional saving of removed samples in `remove_nan_samples()`
- Detailed removal statistics and metadata tracking

### Changed
- Made `statsmodels` a required dependency (removed `mixed_model_available` checks)
- Integrated `save_nan_removed_rows` functionality into `remove_nan_samples()`
- Moved utility functions to `data_utils.py` module
- Improved test fixtures organization with categories (heritability, ANOVA, edge cases)
- Updated documentation to reflect actual implementation

### Fixed
- Line ending consistency issues across different platforms
- Test accuracy for heritability calculations with mixed models
- Handling of infinity values in statistical calculations
- Edge case handling for insufficient data conditions

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