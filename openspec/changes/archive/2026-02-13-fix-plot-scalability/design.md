## Context

The pipeline generates ~20 distinct plot types across QC, Visualization, and Cross-Platform pipelines. Systematic review of the Feb 2026 pipeline outputs revealed:

1. Config templates had misaligned PCA biplot settings (analysis=1, display=10)
2. Every label-based plot (bar charts, heatmaps, EDA panels) becomes unreadable at 100+ traits
3. UMAP is a stub but config silently enables it
4. Cross-platform plots use ad-hoc formatting instead of reusing `sanitize_trait_names()`
5. Cylinder experiment generates 656+ batch files

The pipeline must handle datasets from 19 traits (turface 19 genotypes) to 500+ traits (cylinder EDPIE) without manual intervention.

### Stakeholders
- Plant biologists who need interpretable, publication-ready figures
- Pipeline operators who need reliable, non-crashing runs

## Goals / Non-Goals

**Goals:**
- Fix config templates so `pca_biplot_top_features` matches intended extreme selection count
- All text on all plots is legible at the figure's native resolution
- Pipeline handles 500+ traits without freezing or excessive memory use
- Reduce batch file count for high-trait experiments
- Reuse existing `sanitize_trait_names()` from `data_utils.py` for all label formatting
- Preserve all data (full data always available in CSV exports even when plots show subsets)

**Non-Goals:**
- Reducing figure resolution (scientific figures need high DPI)
- Implementing UMAP analysis (that's Phase 2C, separate scope)
- Changing the statistical computations behind any plots
- Creating new plotting functions (all functions already exist in the codebase)
- Changing existing interactive Plotly visualization behavior

## Decisions

### Decision 1: Clarify PCA biplot config in templates
**What**: Update config templates to set `static_viz.pca_biplot_top_features` to match the intended behavior (e.g., 1 for extreme selection). Add comments clarifying that `pca.n_top_features` (analysis) and `static_viz.pca_biplot_top_features` (visualization) are separate and intentional.
**Why**: These are legitimately separate concerns (analysis vs display), but templates had them misaligned (analysis=1, display=10), causing confusion.

### Decision 2: Adaptive pagination for bar-style plots
**What**: Heritability, variance decomposition, and EDA plots will paginate into batches when trait count exceeds a threshold (similar to existing histogram/boxplot batching).
**Why**: Consistent with existing batching pattern. Preserves all traits across pages.
**Alternatives considered**:
- Top-N only display: Loses information for middle-ranked traits
- Adaptive font sizing: Font becomes unreadable below ~6pt

### Decision 3: Adaptive figure sizing with font floor for heatmaps
**What**: Correlation heatmaps and similar matrix plots will scale figure dimensions with trait count, enforcing minimum 6pt font.
**Why**: Ensures labels remain readable. For very large matrices (200+), labels may be omitted with tick marks only.

### Decision 4: adjustText for PCA biplots
**What**: Use the `adjustText` library to prevent label overlap on PCA biplots.
**Why**: PCA biplots have arbitrary label positions that can't be solved by rotation alone.

### Decision 5: Reuse existing sanitize_trait_names()
**What**: Use `sanitize_trait_names()` from `data_utils.py` for all axis label formatting instead of ad-hoc string replacement.
**Why**: This function already exists, handles abbreviations, units, and depth ranges. The cross-platform code uses `.replace('_', ' ').title()` which is redundant and inconsistent.

### Decision 6: Memory management via explicit figure closure
**What**: Every plotting function will call `plt.close(fig)` after saving, and batch generation loops will call `gc.collect()` periodically.
**Why**: Matplotlib figures accumulate in memory. Generating 100+ figures without closing them can consume multiple GB.

### Decision 7: Configurable batch size and PDF toggle
**What**: Add adaptive `traits_per_page` and `save_pdf` (default: True) to visualization config.
**Why**: Reduces 650+ files to a manageable count.

### Decision 8: Wire existing notebook plots into pipeline steps
**What**: Add calls to existing visualization functions that are used in all Jupyter notebooks but missing from pipeline output. No new plotting code needs to be written -- only pipeline wiring.
**Why**: Systematic comparison of 38 notebooks vs pipeline revealed 8 non-UMAP plot types used in notebooks but absent from the pipeline. Scientists using the pipeline get fewer outputs than they get from running notebooks manually.
**Functions to wire (static, in `generate_static_figures.py`):**
- `create_feature_contribution_plot()` — PCA stacked bar chart (all viz notebooks)
- `create_phenotype_variation_plot()` — genotype distribution box+strip plots (all viz notebooks, looped over configurable trait list)
- `create_regression_plot()` — trait-trait regression scatter (configurable trait pairs)
- `create_genotype_image_grid()` — root image grids for extreme genotypes (conditional on image paths being available)

**Functions to wire (interactive, in `generate_interactive.py`):**
- `create_interactive_scatter_with_images()` — general Plotly scatter with image hover
- `create_html_with_image_viewer()` — HTML page with click-to-view image panel
- `create_interactive_image_gallery()` — browsable HTML image gallery

**Functions to wire (QC, in `visualize_outliers.py`):**
- Outlier method comparison bar chart — extract from inline notebook code into a reusable function

### Decision 9: Config-driven plot enablement for new pipeline plots
**What**: Each newly wired plot type will be controlled by a config flag (default enabled) so users can disable any plot they don't need. Image-dependent plots will only run when image paths are available in the pipeline context.
**Why**: Not all experiments have images. Some users may not want all plots. Config flags maintain pipeline flexibility.

## Risks / Trade-offs

- **Risk**: Config template changes may change biplot appearance for existing users
  - **Mitigation**: Only change templates; existing user configs are untouched
- **Risk**: Pagination changes the visual experience for users accustomed to single-page plots
  - **Mitigation**: Only paginate when needed (> threshold traits); small datasets unchanged
- **Risk**: adjustText may not be in current dependencies
  - **Mitigation**: Lightweight pure-Python package; implement simple manual repulsion as fallback
- **Risk**: Image-dependent plots fail when image paths aren't configured or available
  - **Mitigation**: Guard all image-dependent plots with path existence checks; skip gracefully with log message
- **Risk**: Phenotype variation plots in a loop could generate many figures for high-trait datasets
  - **Mitigation**: Use configurable trait list (e.g., top N by heritability) rather than plotting every trait; apply same memory management (plt.close + gc.collect) as other batch plots

## Open Questions

1. Should `static_viz.pca_biplot_top_features` be removed entirely or kept as an override? (Recommendation: remove, keep only `pca.n_top_features`)
2. For correlation heatmaps with 500+ traits, should we show clustered/grouped view by default? (Recommendation: yes, group by hierarchical clustering)
3. What should the pagination threshold be for heritability plots? (Recommendation: ~50 traits per page)
