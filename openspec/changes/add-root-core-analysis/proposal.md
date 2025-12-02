# Proposal: Add Root Core Analysis Capability

## Why

Root core experiments are a common phenotyping methodology where multiple soil cores are extracted from each plot at different depths, with each depth segment counted for root intersections. Currently, `sleap-roots-analyze` lacks specialized functionality to process this experimental design, requiring researchers to write custom preprocessing code before using existing analysis tools.

This change adds first-class support for root core data, enabling researchers to go from raw core count data to publication-ready depth profile visualizations with proper statistical aggregation.

## What Changes

- **NEW**: Root core data processing module (`root_core_analysis.py`)
  - Sample identifier creation for multi-core experiments
  - Wide-to-long data melting with automatic depth calculation from column names
  - Aggregation across technical replicates (cores) to biological replicates (plots)
  - Validation of unique sample identifiers

- **NEW**: Depth profile visualization module (`depth_profile_plots.py`)
  - Faceted line plots showing mean ± error bars by genotype
  - Spaghetti plots showing individual biological replicates
  - Support for custom error bars (SE, SD, CI)

- **EXTENDED**: Data utilities (`data_utils.py`)
  - Add `filter_rows_by_values()` helper function

- **EXTENDED**: Test fixtures (`tests/fixtures.py`)
  - Add `create_test_root_core_data()` fixture for root core testing

## Impact

### Affected Specs
- **NEW**: `specs/root-core-analysis/spec.md` - Core processing and visualization requirements

### Affected Code
- **NEW**: `src/sleap_roots_analyze/root_core_analysis.py` - Core data processing functions
- **NEW**: `src/sleap_roots_analyze/depth_profile_plots.py` - Visualization functions
- **EXTENDED**: `src/sleap_roots_analyze/data_utils.py` - Add utility function
- **EXTENDED**: `tests/fixtures.py` - Add test fixture
- **NEW**: `tests/test_root_core_analysis.py` - Unit tests for processing
- **NEW**: `tests/test_depth_profile_plots.py` - Unit tests for visualization

### Dependencies
- No new external dependencies required
- Uses existing: pandas, numpy, matplotlib, seaborn, PIL

### Breaking Changes
- None - This is purely additive functionality

### Migration
- Not applicable (new feature)

### Documentation
- Update README.md with root core analysis examples
- Add docstrings following Google format
- Include usage examples in module docstrings
