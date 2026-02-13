## 1. Remove Duplicate Heritability Plot Generation

- [x] 1.1 Write test: `statistical_analysis` step does NOT generate heritability plots into `figures/`
- [x] 1.2 Remove heritability plot generation from `statistical_analysis.py`
- [x] 1.3 Write test: `generate_static_figures` step generates heritability plots into `figures/heritability/`
- [x] 1.4 Update `generate_static_figures.py` to save heritability plots to `figures/heritability/` subdirectory

## 2. Reorganize Static Figures into Subdirectories

- [x] 2.1 Write test: PCA plots saved to `figures/pca/` subdirectory
- [x] 2.2 Update `generate_static_figures.py` to save PCA plots to `figures/pca/`
- [x] 2.3 Write test: phenotype variation plots saved to `figures/phenotype_variation/`
- [x] 2.4 Update phenotype variation plot saving path
- [x] 2.5 Write test: trait histograms saved to `figures/trait_histograms/`
- [x] 2.6 Update trait histogram saving path
- [x] 2.7 Write test: trait boxplots saved to `figures/trait_boxplots/`
- [x] 2.8 Update trait boxplot saving path
- [x] 2.9 Write test: overview plots (correlation heatmap, eda, variance decomposition) saved to `figures/overview/`
- [x] 2.10 Update overview plot saving paths

## 3. Consolidate Interactive Figures

- [x] 3.1 Write test: interactive figures saved to `figures/interactive/` (not `interactive_figures/`)
- [x] 3.2 Update `generate_interactive.py` to use `figures/interactive/` path
- [x] 3.3 Remove creation of `interactive_figures/` directory

## 4. Reorganize Data Outputs

- [x] 4.1 Write test: PCA CSVs saved to `data/pca/` subdirectory
- [x] 4.2 Update PCA analysis step to save to `data/pca/`
- [x] 4.3 Write test: heritability/anova CSVs saved to `data/` (not `figures/`)
- [x] 4.4 Update statistical analysis outputs to save to `data/`

## 5. Remove Legacy Folder Creation

- [x] 5.1 Remove creation of `static_figures/` directory
- [x] 5.2 Remove creation of `interactive_figures/` directory (replaced by `figures/interactive/`)
- [x] 5.3 Update figure manifest to reflect new paths (automatic via relative_to(run_dir))

## 6. Update Tests and Documentation

- [x] 6.1 Update existing tests that check for old paths
- [x] 6.2 Run full test suite to verify no regressions
- [x] 6.3 Update viz pipeline documentation if any (no user-facing docs reference old paths)
