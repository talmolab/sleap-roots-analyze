## Why

The **viz pipeline** output organization is confusing:

1. **Duplicated heritability plots**: Same 12 heritability plots exist in BOTH:
   - `figures/08_heritability_analysis_page01.png` (12 files)
   - `static_figures/heritability_estimates_page01.png` (24 files - png + pdf)

2. **Inconsistent subfolders**: 4 different output folders with no clear logic:
   - `figures/` - heritability from stats step (12 files)
   - `static_figures/` - heritability AGAIN + everything else (386 files flat)
   - `interactive_figures/` - Plotly HTML files
   - `pca/` - PCA analysis CSV outputs

3. **386+ files dumped flat in `static_figures/`**: All plot types mixed together:
   - `heritability_estimates_page01.png` through `page12.png`
   - `pca_biplot.png`, `pca_scree_plot.png`, etc.
   - `phenotype_variation_*.png` (many files)
   - `trait_histograms_batch01.png` through `batch66.png`
   - `trait_boxplots_batch01.png` through `batch98.png`

4. **Hard to find specific plots**: Users must scroll through 386 files to find a correlation heatmap

## What Changes

### Consolidate to Two Output Folders

Replace inconsistent subfolders with clear structure:

```
viz/{dataset}_{timestamp}/
├── figures/                              # ALL figures here
│   ├── overview/                         # Key summary figures
│   │   ├── heritability_page01.png       # Single source, no duplication
│   │   ├── heritability_page02.png
│   │   ├── correlation_heatmap.png
│   │   ├── eda_overview.png
│   │   └── variance_decomposition.png
│   │
│   ├── pca/                              # PCA-related figures
│   │   ├── biplot.png
│   │   ├── scree_plot.png
│   │   ├── feature_loadings.png
│   │   ├── feature_contributions.png
│   │   └── pc_boxplots.png
│   │
│   ├── phenotype_variation/              # Per-trait variation plots
│   │   ├── network_length_med.png
│   │   └── ...
│   │
│   ├── trait_histograms/                 # Batched histograms
│   │   ├── batch01.png
│   │   └── ...
│   │
│   ├── trait_boxplots/                   # Batched boxplots
│   │   ├── batch01.png
│   │   └── ...
│   │
│   └── interactive/                      # Plotly HTML files
│       ├── pca_3d.html
│       └── trait_explorer.html
│
└── data/                                 # Analysis outputs (CSVs, JSONs)
    ├── pca/
    │   ├── components.csv
    │   ├── loadings.csv
    │   └── variance_explained.csv
    ├── heritability_results.csv
    ├── anova_results.csv
    └── trait_statistics.json
```

### Key Changes

1. **Remove duplication**: Heritability plots generated ONCE, not twice
2. **Consistent structure**: All figures in `figures/`, all data in `data/`
3. **Logical subfolders**: Group by plot type (pca/, heritability/, trait_histograms/)
4. **Easy navigation**: Find PCA plots in `figures/pca/`, histograms in `figures/trait_histograms/`

## Impact

- Affected code:
  - `src/sleap_roots_analyze/pipeline/steps/generate_static_figures.py` (file save paths)
  - `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` (remove duplicate heritability plot)
  - `src/sleap_roots_analyze/pipeline/steps/pca_analysis.py` (output paths)
  - `src/sleap_roots_analyze/pipeline/steps/generate_interactive.py` (move to figures/interactive/)
- Affected specs: `visualization-pipeline`
- **No changes to QC or cross-platform pipelines**
