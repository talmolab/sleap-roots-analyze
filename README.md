# SLEAP Roots Analyze

[![PyPI version](https://img.shields.io/pypi/v/sleap-roots-analyze)](https://pypi.org/project/sleap-roots-analyze/)
[![Python: 3.11+](https://img.shields.io/pypi/pyversions/sleap-roots-analyze)](https://pypi.org/project/sleap-roots-analyze/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![Tests: 1900+](https://img.shields.io/badge/Tests-1900%2B-brightgreen)

Statistical analysis tools for root trait data from [SLEAP Roots](https://github.com/talmolab/sleap-roots).

## Installation

```bash
pip install sleap-roots-analyze
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sleap-roots-analyze
```

For development:

```bash
git clone https://github.com/talmolab/sleap-roots-analyze.git
cd sleap-roots-analyze
uv sync --group dev
```

## Quick Start

### New Analysis? Start here.

Use the interactive `/configure-run-all` slash command (in Claude Code) to create a complete,
scientifically validated set of configs for a new analysis:

```
/configure-run-all
```

It inspects your CSV, walks you through every parameter with statistical guardrails, writes
QC + Viz + run manifest configs, and commits them to git as a reproducibility anchor.

Optionally validate a specific config before running:
```
/validate-config configs/active/qc/<your_analysis>.yaml
```

Then run:
```
/run-pipelines --manifest configs/active/run_manifest_<your_analysis>.yaml
```

See [configs/templates/README.md](configs/templates/README.md) for manual config authoring.

### Command-Line Interface

The package provides a CLI for running QC and visualization pipelines:

```bash
# Run QC pipeline
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml

# Run with custom output directory
sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml -o ./my_results

# Validate configuration
sleap-roots-analyze config validate configs/qc_turface_150genotypes.yaml

# List example configs
sleap-roots-analyze config list

# Run all pipelines from a manifest
sleap-roots-analyze run-all configs/active/run_manifest.yaml

# Get help
sleap-roots-analyze --help
sleap-roots-analyze qc --help
```

See [docs/QC_PIPELINE_GUIDE.md](docs/QC_PIPELINE_GUIDE.md) for a complete guide to using the QC pipeline.

### Python API

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

### Outlier Detection

```python
from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_isolation_forest,
    remove_outliers_from_data
)

# Detect outliers using Mahalanobis distance
outliers_maha = detect_outliers_mahalanobis(
    df_filtered[trait_cols],
    use_robust=True
)

# Or use Isolation Forest for complex patterns
outliers_iso = detect_outliers_isolation_forest(
    df_filtered[trait_cols],
    contamination=0.1
)

# Remove outliers from data
df_clean, df_outliers = remove_outliers_from_data(
    df_filtered,
    outliers_maha['outlier_indices'],
    return_outliers=True
)
```

### Visualization

```python
from sleap_roots_analyze.visualization import (
    create_heritability_plot,
    create_pca_biplot,
    create_feature_contribution_heatmap,
    create_phenotype_variation_plot,
    save_publication_figure
)

# Create heritability plot
fig = create_heritability_plot(h2_results, threshold=0.3)

# Create PCA biplot
fig_biplot = create_pca_biplot(
    pca_result,
    color_by="geno",
    metadata_df=df_filtered[["Barcode", "geno"]]
)

# Create feature contribution heatmap
fig_heatmap = create_feature_contribution_heatmap(
    pca_result['feature_contributions'],
    n_components=5
)

# Save in publication format
save_publication_figure(fig, "heritability", formats=["pdf", "png"])
```

### Comprehensive PCA Analysis with Export

```python
from sleap_roots_analyze.pca import run_pca_and_export_artifacts

# Run comprehensive PCA analysis with CSV exports
results = run_pca_and_export_artifacts(
    df_filtered,
    trait_cols=trait_cols,
    analysis_dir="pca_results",
    n_components=10,
    save_csv=True,
    save_prefix="experiment1_"
)

# Access results DataFrames
loadings_df = results['loadings_df']
pc_scores_df = results['pc_scores_df']
variance_df = results['variance_explained_df']
contributions_df = results['trait_variance_contributions_df']
```

### Interactive Visualization

```python
from sleap_roots_analyze.interactive_visualization import (
    create_interactive_pca_with_images,
    create_interactive_umap_with_hover_highlight,
    create_trait_explorer_dashboard,
    create_interactive_image_gallery
)

# Create interactive PCA with sample images
fig = create_interactive_pca_with_images(
    pca_result,
    image_paths,  # Dict mapping sample IDs to image paths
    show_images=True,
    metadata_df=df_filtered[["Barcode", "geno"]]
)

# Interactive UMAP with hover highlights
fig_umap = create_interactive_umap_with_hover_highlight(
    umap_result,
    highlight_on_hover=True,
    size=8
)

# Create comprehensive trait explorer dashboard
dashboard = create_trait_explorer_dashboard(
    df_filtered,
    trait_cols,
    groupby_col="geno"
)

# Generate interactive HTML gallery with images
html = create_interactive_image_gallery(
    image_paths,
    metadata_df=df_filtered[["Barcode", "geno", "trait1"]],
    images_per_row=4,
    image_width=200
)
```

## Features

- **Data Cleaning**: Automatic metadata detection, NaN handling, zero-inflated trait removal
- **Statistical Analysis**: Broad-sense heritability (H²), ANOVA, trait statistics
- **PCA Analysis**: Dimensionality reduction with automatic component selection, comprehensive export artifacts
- **Outlier Detection**: Mahalanobis, PCA reconstruction, and Isolation Forest methods
- **Visualization**: Publication-ready plots for heritability, PCA, outliers, and phenotype variation
- **Interactive Visualization**: Plotly-based interactive plots with image integration and hover effects
- **UMAP Analysis**: Non-linear dimensionality reduction for complex trait relationships
- **Cross-Experiment Analysis**: Compare and correlate data across multiple experiments

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
│   ├── cli.py                        # Command-line interface
│   ├── data_cleanup.py               # Data loading and cleaning
│   ├── statistics.py                 # Statistical analysis
│   ├── pca.py                        # PCA analysis
│   ├── outlier_detection.py          # Outlier detection
│   ├── visualization.py              # Plotting and visualization
│   ├── outlier_visualization.py      # Outlier-specific plots
│   ├── interactive_visualization.py  # Interactive Plotly visualizations
│   ├── cross_experiment_analysis.py  # Cross-experiment comparisons
│   ├── depth_profile_plots.py        # Depth profile visualizations
│   ├── pipeline_runner.py            # Pipeline orchestration (run-all)
│   ├── umap.py                       # UMAP dimensionality reduction
│   ├── data_utils.py                 # Utility functions
│   └── pipeline/                     # QC/Viz pipeline steps
├── configs/                     # Pipeline configurations
│   ├── active/                  # Active run manifests
│   └── examples/                # Example configs for different use cases
├── tests/                       # Test suite (1900+ tests)
├── docs/                        # Documentation
└── pyproject.toml              # Project configuration
```

## License

GNU General Public License v3.0 - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{sleap_roots_analyze,
  title = {SLEAP Roots Analyze},
  author = {Elizabeth Berrigan},
  year = {2026},
  url = {https://github.com/talmolab/sleap-roots-analyze}
}
```