# Cross-Platform Analysis Pipeline

## Why

Researchers need to compare root trait measurements across different experimental platforms (e.g., cylinder vs turface, field vs growth chamber) to validate trait consistency and understand platform-specific effects. Currently, this analysis exists only as ad-hoc Jupyter notebooks with hardcoded paths and manual parameter tuning. This creates reproducibility issues and prevents systematic cross-platform validation across multiple experiments.

The existing `cross_experiment_spearman_turface_cylinder_20250919.ipynb` notebook demonstrates the need: it manually loads two datasets, aligns genotypes, calculates Spearman correlations for all trait pairs, and generates publication-quality visualizations. This workflow should be automated through the pipeline infrastructure to ensure reproducibility and enable batch processing.

**Important Note on Platform Descriptions**: Both Turface and Cylinder experiments use single-timepoint imaging, not time-series. Turface uses RhizoVision imaging (not 3D), and Cylinder uses SLEAP Roots imaging. Any references to "3D", "2D Time-Series", or multi-timepoint analysis are incorrect and should be removed from configurations and documentation.

## What Changes

- Add **CrossPlatformConfig** dataclass to `pipeline/config/components.py` with:
  - Experiment 1 and 2 data paths, names, and genotype column mappings
  - Configurable correlation method (spearman, pearson, kendall)
  - Minimum samples per genotype threshold
  - Significance level and visualization parameters (top N correlations, plot sizes)
- Add **LoadCrossPlatformDataStep** pipeline step that:
  - Loads two experiment datasets
  - Aligns by common genotypes
  - Extracts trait columns using existing `get_trait_columns()`
  - Validates minimum sample requirements
- Add **CalculateCrossPlatformCorrelationsStep** pipeline step that:
  - Calculates genotype means for each experiment
  - Computes pairwise trait correlations using selected method
  - Performs significance testing
  - Exports correlation results CSV
- Add **VisualizeCrossPlatformStep** pipeline step that:
  - Generates 4-panel summary visualization (distribution, volcano plot, top positive/negative)
  - Creates joint plots for top N correlations
  - Creates genotype boxplots for top N correlations
  - Saves all figures to output directory
- Add helper functions to `cross_experiment_analysis.py`:
  - `calculate_correlations()` - Unified interface for Spearman/Pearson/Kendall
  - `create_correlation_summary_plot()` - 4-panel summary visualization
- Add example configuration `configs/cross_platform_template.yaml`
- Add comprehensive test suite following TDD principles

**Breaking changes**: None (new capability, existing functionality unchanged)

## Impact

- **Affected specs**: New `cross-platform-analysis` capability (no existing specs affected)
- **Affected code**:
  - `src/sleap_roots_analyze/pipeline/config/components.py` - Add `CrossPlatformConfig`
  - `src/sleap_roots_analyze/pipeline/steps/load_cross_platform_data.py` - New step
  - `src/sleap_roots_analyze/pipeline/steps/calculate_cross_platform_correlations.py` - New step
  - `src/sleap_roots_analyze/pipeline/steps/visualize_cross_platform.py` - New step
  - `src/sleap_roots_analyze/pipeline/steps/__init__.py` - Export new steps
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` - Add helper functions
  - `configs/cross_platform_template.yaml` - New template config
  - `tests/test_cross_platform_pipeline.py` - New comprehensive test suite
  - `tests/fixtures.py` - Add cross-platform test fixtures
- **Dependencies**: Leverages existing `cross_experiment_analysis.py` functions (`load_and_align_experiments`, `calculate_genotype_means`, `create_joint_plot`, `create_genotype_boxplots`)
- **Documentation**: Add cross-platform pipeline section to project docs
