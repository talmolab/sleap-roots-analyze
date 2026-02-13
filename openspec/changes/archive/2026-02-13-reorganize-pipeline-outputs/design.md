## Context

The viz pipeline currently outputs files to 4 different subdirectories with no consistent logic:
- `figures/` - 12 heritability plots from statistical_analysis step
- `static_figures/` - 386 files including DUPLICATE heritability plots + everything else
- `interactive_figures/` - Plotly HTML files
- `pca/` - PCA analysis CSV outputs

This creates confusion and duplication.

### Current State

```
viz/{dataset}_{timestamp}/
├── figures/                 # Heritability from stats step
│   └── 08_heritability_analysis_page*.png (12 files)
├── static_figures/          # FLAT dump of 386 files
│   ├── heritability_estimates_page*.png (12 files - DUPLICATES!)
│   ├── heritability_estimates_page*.pdf (12 files)
│   ├── pca_*.png/pdf (10+ files)
│   ├── phenotype_variation_*.png/pdf (20+ files)
│   ├── trait_histograms_batch*.png/pdf (130+ files)
│   └── trait_boxplots_batch*.png/pdf (200+ files)
├── interactive_figures/     # Plotly HTML
└── pca/                     # PCA CSVs
```

## Goals / Non-Goals

**Goals:**
- Eliminate duplicate heritability plot generation
- Organize figures into logical subdirectories by plot type
- Consistent folder naming (no mix of `figures/`, `static_figures/`, `interactive_figures/`)
- Easy to find specific plots

**Non-Goals:**
- Changing the QC pipeline output structure (it's fine)
- Changing the cross-platform pipeline output structure (it's fine)
- Changing file naming conventions within folders
- Changing timestamp format in folder names

## Decisions

### Decision 1: Single `figures/` directory with subdirectories

**What**: Merge `figures/`, `static_figures/`, and `interactive_figures/` into a single `figures/` directory with subdirectories by plot type.

**Why**: Clear, consistent structure. All visualizations in one place.

### Decision 2: Remove duplicate heritability generation

**What**: The `statistical_analysis` step currently generates heritability plots into `figures/`. The `generate_static_figures` step ALSO generates heritability plots into `static_figures/`. Remove one.

**Why**: Wasteful duplication. The `generate_static_figures` version is more polished (paginated, PDF output), so keep that one.

**Implementation**: Remove heritability plot generation from `statistical_analysis.py`, keep it in `generate_static_figures.py`.

### Decision 3: Separate `data/` directory for analysis outputs

**What**: Put CSVs and JSONs (PCA outputs, heritability results, trait statistics) in a `data/` subdirectory.

**Why**: Clear separation between figures (for viewing) and data (for further analysis).

### Decision 4: Figure subdirectory structure

**What**: Organize `figures/` with these subdirectories:
- `overview/` - Summary plots (correlation heatmap, eda_overview, variance_decomposition)
- `pca/` - PCA-related plots (biplot, scree, loadings, contributions, pc_boxplots)
- `heritability/` - Paginated heritability plots
- `phenotype_variation/` - Per-trait variation plots
- `trait_histograms/` - Batched histogram plots
- `trait_boxplots/` - Batched boxplot plots
- `interactive/` - Plotly HTML files

**Why**: Logical grouping by analysis type. Easy to find specific plots.

## Migration

Existing pipeline runs will NOT be migrated. New runs will use the new structure.

## Open Questions

1. Should we add a `manifest.json` listing all generated files with paths and descriptions?
   - **Recommendation**: Yes, but as a follow-up task
