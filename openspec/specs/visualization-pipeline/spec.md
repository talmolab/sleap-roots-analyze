# Visualization Pipeline

## Purpose
Provides configurable static and interactive visualization generation for trait analysis data, including PCA plots, UMAP visualizations, trait distributions, and genotype comparisons.
## Requirements
### Requirement: Genotype Highlighting Configuration
The `StaticVisualizationConfig` dataclass SHALL provide optional `genotypes_to_color` and `highlight_genotypes` parameters to enable selective genotype highlighting in PCA plots.

When the `color_by` column in `create_pca_biplot` has an **integer dtype** (e.g., int64
accession IDs like `12305183`), the values SHALL be cast to string before coloring so that
distinct tab10 colors and a categorical legend are produced rather than a continuous colorbar.
Float-typed columns retain their current continuous-colormap behavior.

#### Scenario: Configure genotypes to color
- **WHEN** user provides a list of genotype names in `static_viz.genotypes_to_color`
- **THEN** only those genotypes SHALL be colored with distinct colors in PCA biplot
- **AND** all other genotypes SHALL be rendered in gray labeled as "Other"
- **AND** the config SHALL validate successfully

#### Scenario: Configure genotypes to highlight
- **WHEN** user provides a list of genotype names in `static_viz.highlight_genotypes`
- **THEN** those genotypes SHALL be highlighted with:
  - Larger point sizes in PCA biplot
  - Edge colors in PCA biplot
  - Gold fill color in PC boxplots
  - Bold labels in PC boxplots
- **AND** highlighting SHALL work independently of `genotypes_to_color`

#### Scenario: Default behavior with no highlighting
- **WHEN** user omits `genotypes_to_color` (None value)
- **THEN** all genotypes SHALL be colored with distinct colors automatically
- **AND** backward compatibility SHALL be maintained

#### Scenario: Empty list behavior
- **WHEN** user provides an empty list for `genotypes_to_color`
- **THEN** all genotypes SHALL appear in gray
- **AND** the pipeline SHALL execute successfully

#### Scenario: String genotype IDs use discrete colors
- **GIVEN** a dataset where the Genotype column contains string IDs (e.g., `"GEN_A"`)
- **WHEN** `create_pca_biplot` is called with `color_by="Genotype"`
- **THEN** each unique genotype SHALL be assigned a distinct tab10 color
- **AND** a categorical legend SHALL be shown (no colorbar)

#### Scenario: Numeric genotype IDs use discrete colors
- **GIVEN** a dataset where the Genotype column contains integer IDs (e.g., `12305183`)
- **WHEN** `create_pca_biplot` is called with `color_by="Genotype"`
- **THEN** each unique genotype SHALL be assigned a distinct tab10 color
- **AND** a categorical legend SHALL be shown (no continuous colorbar)
- **AND** the integer values SHALL be displayed as their string representation
  (e.g., `"12305183"`) in the legend

#### Scenario: Float color_by column retains continuous colormap
- **GIVEN** a dataset where `color_by` column contains float trait values (e.g., `0.4967`)
- **WHEN** `create_pca_biplot` is called with that column as `color_by`
- **THEN** the continuous viridis colormap SHALL still be used
- **AND** a colorbar SHALL be shown (not a categorical legend)

#### Scenario: Continuous UMAP coloring is unaffected
- **GIVEN** a UMAP plot colored by a continuous trait value
- **WHEN** the plot is generated via `create_umap_colored_by_top_traits`
- **THEN** the continuous viridis colormap SHALL still be used
- **AND** this code path does NOT use the `color_by` parameter

### Requirement: Pipeline Step Parameter Passing

The `GenerateStaticFiguresStep` SHALL pass genotype highlighting parameters from configuration to underlying plotting functions.

The `PCAAnalysisStep` SHALL use the post-filtering feature names returned by
`perform_pca_analysis()` (via `pca_results["feature_names"]`) instead of the original
`trait_cols` list when constructing the loadings DataFrame index, computing
`n_features_total` for feature selection, and mapping feature indices to names.

The `PCAAnalysisStep` SHALL log the names and count of any traits excluded due to zero
variance, and SHALL emit a Python `UserWarning` when more than 50% of input traits are
excluded.

The `PCAAnalysisStep` SHALL store `excluded_zero_variance_traits` (list of excluded trait
names) and `n_traits_after_filtering` (int) in its output metadata for downstream
inspection and reproducibility.

`create_pca_biplot` SHALL honor every `feature_selection` value it documents
— `"vector_length"`, `"extreme"`, `"top_absolute"`, `"top_contribution"`,
and `"top_variance"` — by selecting features using the matching method in
`select_top_features_from_pca()`, and SHALL raise `ValueError` for any other
value instead of silently substituting `"vector_length"`. When
`feature_selection == "top_variance"`, `create_pca_biplot` SHALL call
`select_top_features_from_pca(..., pc_indices=None)` rather than passing the
biplot's two displayed PC indices, since `select_top_features_from_pca`'s
`"top_variance"` method ranks across all retained PCs regardless of
`pc_indices` and passing the 2-index list would misleadingly imply the
biplot's PC scope is honored.

`create_umap_colored_by_top_traits` SHALL scope `pc_indices` to all retained PCs (derived
from `pca_results["n_components_selected"]`, `variance_threshold`, or the 95% cumulative
variance default, in that priority order) for **every** `feature_selection` method, not
only `"top_variance"`.

For `feature_selection == "extreme"`, `create_umap_colored_by_top_traits` SHALL build the
plotted trait set by round-robin selection across (PC, direction) pairs spanning all
scoped `pc_indices`, instead of taking a single block-ordered `select_top_features_from_pca(...,
n_features_to_select=n_traits)` call and truncating it to `top_indices[:n_traits]`. The
round-robin SHALL:
- maintain one sorted-loading iterator per (PC, direction) pair, each advancing its own
  position monotonically across passes (never re-scanning from the start);
- check candidates against a single **global** `seen` set at the moment a trait is popped,
  so a trait already claimed by one pair is skipped (not re-selected) by every other pair;
- order passes **direction-major, PC-minor**: pass 1 takes each retained PC's single
  most-negative unseen trait (PC1, PC2, PC3, ... in order), pass 2 takes each PC's single
  most-positive unseen trait, pass 3 takes each PC's second-most-negative unseen trait, and
  so on, continuing until `n_traits` traits are collected or every pair is exhausted;
- record the (PC, direction) pair that actually claimed each selected trait, on a
  first-come-first-claimed basis per the pass order above, so the source is deterministic
  even when a trait is extreme on more than one PC.

The direction-major/PC-minor pass order is required specifically so that when `n_traits`
is smaller than `2 * len(pc_indices)`, every retained PC still receives at least one
representative before any PC receives a second — a PC-major pass order (all of PC1's
extremes before moving to PC2) would reproduce the same class of bug one level up,
crowding out later PCs whenever per-PC-pair supply exceeds the `n_traits` budget.

This round-robin construction lives in `create_umap_colored_by_top_traits` only;
`select_top_features_from_pca`'s own `"extreme"` method in `pca.py` (and its
per-direction-per-PC `n_features_to_select` count semantics, relied on unsliced by
`PCAAnalysisStep`) SHALL remain unchanged.

For `feature_selection == "extreme"`, each subplot's subtitle SHALL report the actual (PC,
direction) pair that selected that trait, rather than always re-deriving direction from
`loadings[trait_idx, 0]` (PC1's loading).

#### Scenario: Pass parameters to PCA biplot
- **WHEN** generating PCA biplot via `create_pca_biplot`
- **THEN** the step SHALL pass `config.static_viz.genotypes_to_color` to the function
- **AND** the step SHALL pass `config.static_viz.highlight_genotypes` to the function

#### Scenario: Pass parameters to PC boxplots
- **WHEN** generating PC boxplots via `create_pc_genotype_boxplots`
- **THEN** the step SHALL pass `config.static_viz.highlight_genotypes` to the function
- **AND** the highlighted genotypes SHALL appear in gold with bold labels

#### Scenario: PCA step handles zero-variance traits gracefully
- **WHEN** the input DataFrame contains traits with zero variance (constant values)
- **THEN** the PCA step SHALL complete successfully using only non-zero-variance traits
- **AND** the loadings CSV index SHALL match the actual features used in PCA
- **AND** `excluded_zero_variance_traits` SHALL list the excluded trait names in metadata
- **AND** `n_traits_after_filtering` SHALL reflect the count of traits actually used

#### Scenario: PCA step warns on high zero-variance fraction
- **WHEN** more than 50% of input traits have zero variance
- **THEN** the PCA step SHALL emit a `UserWarning` indicating potential data quality issues
- **AND** the step SHALL still complete successfully with the remaining traits

#### Scenario: PCA step with no zero-variance traits
- **WHEN** all input traits have non-zero variance
- **THEN** the PCA step SHALL behave identically to current behavior
- **AND** `excluded_zero_variance_traits` SHALL be an empty list
- **AND** no warning SHALL be emitted

#### Scenario: create_pca_biplot honors top_variance feature selection
- **WHEN** `create_pca_biplot` is called with `feature_selection="top_variance"`
- **THEN** the features selected for display SHALL be the same set returned
  by `select_top_features_from_pca(method="top_variance", pc_indices=None,
  ...)` called directly with the same loadings, eigenvalues, feature count,
  and `top_n_features`
- **AND** the selection SHALL NOT silently fall back to the
  `"vector_length"` method

#### Scenario: create_pca_biplot rejects an unrecognized feature_selection value
- **WHEN** `create_pca_biplot` is called with a `feature_selection` value
  that is not one of `"vector_length"`, `"extreme"`, `"top_absolute"`,
  `"top_contribution"`, or `"top_variance"`
- **THEN** the function SHALL raise `ValueError`
- **AND** it SHALL NOT silently substitute `"vector_length"`

#### Scenario: create_pca_biplot continues to honor pre-existing feature_selection methods
- **WHEN** `create_pca_biplot` is called with `feature_selection` set to
  `"vector_length"`, `"extreme"`, `"top_absolute"`, or `"top_contribution"`
- **THEN** the features selected for display SHALL match a direct
  `select_top_features_from_pca(method=<same value>, pc_indices=[pc_x_idx,
  pc_y_idx])` call with the same loadings, eigenvalues, feature count, and
  `top_n_features`

#### Scenario: create_umap_colored_by_top_traits scopes pc_indices beyond PC1/PC2 for non-top_variance methods
- **GIVEN** `pca_results` resolves to 3 or more retained PCs (via
  `n_components_selected` or the cumulative variance threshold)
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection` set to `"extreme"`, `"top_absolute"`, or
  `"top_contribution"`
- **THEN** the PC indices considered for feature selection SHALL include every
  retained PC, not just PC1 and PC2

#### Scenario: create_umap_colored_by_top_traits shows traits from multiple PCs and both directions for extreme selection
- **GIVEN** `pca_results` resolves to 3 or more retained PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and `n_traits` large enough to span more than
  one (PC, direction) pair
- **THEN** the plotted trait set SHALL NOT be a subset of PC1's `n_traits`
  most-negative-loading indices
- **AND** the plotted trait set SHALL include at least one trait whose
  extreme loading comes from a PC other than PC1
- **AND** the plotted trait set SHALL include at least one trait selected for
  its positive loading

#### Scenario: create_umap_colored_by_top_traits extreme selection round-robins fairly across PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`, `n_traits=6`, and 3 retained PCs each with
  distinct most-extreme traits in both directions
- **THEN** the round-robin construction SHALL select traits from PC1, PC2,
  and PC3 before exhausting `n_traits`, rather than filling the entire
  `n_traits` budget from PC1 alone

#### Scenario: create_umap_colored_by_top_traits gives every retained PC at least one representative before any PC gets a second
- **GIVEN** `pca_results` resolves to 5 retained PCs, each with distinct
  most-extreme traits in both directions
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and `n_traits=6` (fewer than `2 * 5` PC×direction
  pairs)
- **THEN** the plotted trait set SHALL include at least one trait from each of
  the 5 retained PCs
- **AND** no PC SHALL contribute a second trait until every retained PC has
  contributed at least one

#### Scenario: create_umap_colored_by_top_traits extreme selection handles fewer distinct extreme traits than n_traits
- **GIVEN** the total number of distinct traits reachable across all (PC,
  direction) pairs (after deduplication) is less than `n_traits`
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`
- **THEN** the function SHALL return a plotted trait set shorter than
  `n_traits` without raising an exception
- **AND** the figure's unused subplot axes SHALL be removed as they are today

#### Scenario: create_umap_colored_by_top_traits deduplicates a trait that is extreme on more than one PC
- **GIVEN** a single trait is the most-extreme (same or opposite direction)
  loading on two different retained PCs
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"`
- **THEN** that trait SHALL appear exactly once in the plotted trait set
- **AND** the (PC, direction) pair whose round-robin turn claims it first
  (per pass order) SHALL be recorded as its source for subtitle purposes
- **AND** the freed round-robin slot for the other PC SHALL be filled by that
  PC's next-ranked unseen candidate rather than left empty

#### Scenario: create_umap_colored_by_top_traits subtitle reports the true source PC and direction for extreme selection
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="extreme"` and a plotted trait was selected for its
  extreme loading on PC2 (not PC1)
- **THEN** that trait's subplot subtitle SHALL reference PC2 (e.g. `"PC2+"` or
  `"PC2-"`), not PC1

#### Scenario: create_umap_colored_by_top_traits leaves select_top_features_from_pca's extreme method unchanged
- **WHEN** `select_top_features_from_pca(method="extreme", ...)` is called
  directly (e.g. from `PCAAnalysisStep`)
- **THEN** it SHALL continue to return the block-ordered list
  (`PC1_neg, PC1_pos, PC2_neg, PC2_pos, ...`) with `n_features_to_select`
  traits per direction per PC, unchanged from current behavior

#### Scenario: create_umap_colored_by_top_traits top_variance behavior is unchanged
- **WHEN** `create_umap_colored_by_top_traits` is called with
  `feature_selection="top_variance"`
- **THEN** the selected traits and PC scoping SHALL be identical to current
  behavior (unaffected by this change)

### Requirement: Configuration Validation
The visualization pipeline configuration loading SHALL validate genotype highlighting parameters.

#### Scenario: Valid genotype list
- **WHEN** user provides a list of strings for `genotypes_to_color`
- **THEN** the configuration SHALL validate successfully
- **AND** the values SHALL be accessible as `List[str]`

#### Scenario: None value accepted
- **WHEN** user omits genotype highlighting parameters
- **THEN** the configuration SHALL default to None
- **AND** no validation errors SHALL occur

#### Scenario: Invalid type rejected
- **WHEN** user provides non-list value for genotype highlighting
- **THEN** configuration validation SHALL fail with clear error message
- **AND** the error SHALL indicate expected type

### Requirement: Backward Compatibility
Existing visualization configs without genotype highlighting SHALL continue to function identically.

#### Scenario: Legacy config without highlighting
- **WHEN** running a viz config without `genotypes_to_color` or `highlight_genotypes`
- **THEN** all plots SHALL generate successfully
- **AND** all genotypes SHALL be colored automatically as before
- **AND** no changes to existing behavior SHALL occur

#### Scenario: Partial highlighting configuration
- **WHEN** user provides `highlight_genotypes` but not `genotypes_to_color`
- **THEN** all genotypes SHALL be colored distinctly
- **AND** only specified genotypes SHALL have highlight styling
- **AND** the pipeline SHALL execute successfully

### Requirement: Test Coverage
The genotype highlighting feature SHALL have comprehensive test coverage following TDD principles.

#### Scenario: Config tests pass
- **WHEN** tests for `StaticVisualizationConfig` are run
- **THEN** tests SHALL verify default None values
- **AND** tests SHALL verify list of strings accepted
- **AND** tests SHALL verify configuration can be loaded from YAML

#### Scenario: Pipeline step tests pass
- **WHEN** tests for `GenerateStaticFiguresStep` are run
- **THEN** tests SHALL verify parameters passed to `create_pca_biplot`
- **AND** tests SHALL verify parameters passed to `create_pc_genotype_boxplots`
- **AND** tests SHALL use mocking to verify function calls

#### Scenario: Integration tests pass
- **WHEN** full viz pipeline is run with highlighting enabled
- **THEN** PCA biplot files SHALL be generated
- **AND** PC boxplot files SHALL be generated
- **AND** plots SHALL contain highlighted genotypes visually distinct

### Requirement: Image Linking Method Configuration
The `DataConfig` dataclass SHALL provide `image_linking_method` and `scan_path_col` parameters to support different image organization patterns.

#### Scenario: Configure RhizoVision image linking (default)
- **WHEN** user omits `image_linking_method` or sets it to "rhizovision"
- **THEN** the pipeline SHALL use `link_rhizovision_images_to_samples()` function
- **AND** the function SHALL look for files like `{barcode}_c1_p1_{image_type}` in flat directory
- **AND** backward compatibility SHALL be maintained

#### Scenario: Configure cylinder image linking
- **WHEN** user sets `image_linking_method` to "cylinder" in config
- **THEN** the pipeline SHALL use `link_cylinder_images_from_scan_path()` function
- **AND** the function SHALL read `scan_path` column from data
- **AND** the function SHALL build paths like `{base_dir}/{scan_path}/{image_type}`

#### Scenario: Cylinder linking without scan_path column
- **WHEN** user configures cylinder linking but data lacks `scan_path` column
- **THEN** the pipeline SHALL log a warning
- **AND** no images SHALL be linked
- **AND** the pipeline SHALL continue execution without image grids

#### Scenario: Custom scan_path column name
- **WHEN** user sets `scan_path_col` to a custom column name
- **THEN** the pipeline SHALL use that column for cylinder image linking
- **AND** the default SHALL remain "scan_path" for compatibility

### Requirement: Image Paths Metadata Flow
The visualization pipeline SHALL preserve `image_paths` metadata through all analysis steps.

#### Scenario: Metadata preserved through StatisticalAnalysisStep
- **WHEN** `LoadDataAndImagesStep` produces `image_paths` in metadata
- **AND** `StatisticalAnalysisStep` executes
- **THEN** `image_paths` SHALL be present in the step's output metadata
- **AND** downstream steps SHALL have access to the image paths

#### Scenario: Metadata preserved through UMAPAnalysisStep
- **WHEN** `image_paths` metadata exists from previous steps
- **AND** `UMAPAnalysisStep` executes
- **THEN** `image_paths` SHALL be present in the step's output metadata
- **AND** the UMAP step SHALL NOT modify or remove image_paths

#### Scenario: GenerateStaticFiguresStep receives image_paths
- **WHEN** `GenerateStaticFiguresStep` executes
- **THEN** it SHALL receive `image_paths` from accumulated metadata
- **AND** it SHALL pass image_paths to `_create_genotype_image_grids()` function
- **AND** genotype image grids SHALL be generated with correct images

### Requirement: Image Paths Format Handling
The `_create_genotype_image_grids` function SHALL handle both nested dict format and legacy Series format for `image_paths`.

#### Scenario: Handle nested dict format from link functions
- **WHEN** `image_paths` is in format `Dict[barcode, Dict[image_type, Path]]`
- **THEN** the function SHALL detect this as nested dict format
- **AND** the function SHALL use the paths directly (with Path conversion)
- **AND** genotype image grids SHALL be created successfully

#### Scenario: Handle legacy Series format
- **WHEN** `image_paths` is a pd.Series indexed by DataFrame row numbers
- **THEN** the function SHALL detect this as legacy format
- **AND** the function SHALL convert to `Dict[barcode, Dict[image_type, Path]]` format
- **AND** backward compatibility SHALL be maintained

### Requirement: Genotype Image Grid Configuration
The `StaticVizConfig` dataclass SHALL provide configurable parameters for genotype image grids to support different imaging platforms (RhizoVision, cylinder scanners).

#### Scenario: Configure image type for RhizoVision
- **WHEN** user omits `static_viz.genotype_image_grid_image_type` from config
- **THEN** the default value SHALL be "features.png"
- **AND** the genotype image grids SHALL display RhizoVision feature images

#### Scenario: Configure image type for cylinder scanner
- **WHEN** user sets `static_viz.genotype_image_grid_image_type` to "1.jpg"
- **THEN** the genotype image grids SHALL display the front rotation image
- **AND** the image_links dictionary SHALL use "1.jpg" as the key

#### Scenario: Configure trait columns for statistics display
- **WHEN** user provides a list of trait names in `static_viz.genotype_image_grid_trait_cols`
- **THEN** the genotype image grid SHALL display statistics for those traits
- **AND** the traits SHALL be passed to `create_genotype_image_grid()` as `trait_cols`

#### Scenario: Default trait columns behavior
- **WHEN** user omits `static_viz.genotype_image_grid_trait_cols` from config
- **THEN** the default value SHALL be None
- **AND** no trait statistics SHALL be shown in the image grid

### Requirement: Pipeline Step Image Type Handling
The `GenerateStaticFiguresStep._create_genotype_image_grids()` method SHALL use the configured image type when building image_links and calling the visualization function.

#### Scenario: Use configured image type in image_links
- **WHEN** building the image_links dictionary for genotype image grids
- **THEN** the method SHALL use `config.static_viz.genotype_image_grid_image_type` as the dictionary key
- **AND** the method SHALL NOT hardcode "features.png"

#### Scenario: Pass trait columns to visualization function
- **WHEN** calling `create_genotype_image_grid()`
- **THEN** the method SHALL pass `config.static_viz.genotype_image_grid_trait_cols` as the `trait_cols` parameter
- **AND** the method SHALL pass `config.static_viz.genotype_image_grid_image_type` as the `image_type` parameter

### Requirement: Config Template Clarity for PCA Biplot
Config templates SHALL clearly document that `static_viz.pca_biplot_top_features` controls biplot arrow count (separate from `pca.n_top_features` which controls PCA analysis), and templates using `extreme` feature selection SHALL set `pca_biplot_top_features` to match the intended per-extreme count.

#### Scenario: Config template with extreme selection
- **WHEN** a config template uses `pca.feature_selection_strategy: extreme`
- **THEN** the template SHALL set `static_viz.pca_biplot_top_features` to the intended per-extreme count (e.g., 1)
- **AND** the template SHALL include a comment explaining the relationship between the two parameters

### Requirement: Adaptive Plot Readability for High Trait Counts
All static plots generated by the visualization pipeline SHALL produce readable, interpretable figures regardless of trait count (from 19 to 500+ traits). Text labels SHALL never overlap to the point of illegibility.

#### Scenario: Heritability plot with 200+ traits
- **WHEN** generating heritability bar plot for a dataset with 200+ traits
- **THEN** the plot SHALL paginate into multiple figures with readable x-axis labels
- **AND** each page SHALL contain a manageable number of traits (e.g., 40-60 per page)
- **AND** all pages SHALL be saved with sequential batch numbering
- **AND** the full heritability data SHALL remain available in CSV export

#### Scenario: Correlation heatmap with 100+ traits
- **WHEN** generating correlation heatmap for a dataset with 100+ traits
- **THEN** the figure size SHALL scale adaptively with trait count
- **AND** label font size SHALL be at minimum 6pt
- **AND** labels SHALL be readable at native figure resolution

#### Scenario: EDA overview with 100+ traits
- **WHEN** generating EDA overview panels for a dataset with 100+ traits
- **THEN** x-axis labels SHALL NOT overlap into illegibility
- **AND** figure width SHALL scale with trait count or panels SHALL paginate

#### Scenario: Variance decomposition with 100+ traits
- **WHEN** generating variance decomposition plot for a dataset with 100+ traits
- **THEN** x-axis labels SHALL be readable
- **AND** the plot SHALL paginate or display top-N traits per panel when count exceeds threshold

#### Scenario: Small dataset backward compatibility
- **WHEN** generating any plot for a dataset with fewer than 50 traits
- **THEN** plot appearance SHALL be identical to current behavior
- **AND** no pagination SHALL occur

### Requirement: PCA Biplot Label Clarity
PCA biplot feature loading labels SHALL NOT overlap, using automatic text placement to avoid collisions.

#### Scenario: Biplot with many overlapping feature vectors
- **WHEN** generating PCA biplot with multiple feature loading arrows in close proximity
- **THEN** labels SHALL be repositioned automatically to avoid overlap
- **AND** leader lines or offsets SHALL connect displaced labels to their arrows

#### Scenario: Biplot with few features
- **WHEN** generating PCA biplot with fewer than 10 feature loading arrows
- **THEN** label placement SHALL be unchanged from current behavior

### Requirement: Memory-Safe Figure Generation
The visualization pipeline SHALL manage matplotlib figure memory to prevent freezing or crashes
during batch generation.

#### Scenario: Generating 50+ figures sequentially
- **WHEN** the pipeline generates 50+ batch figures (histograms, boxplots) in sequence
- **THEN** each figure SHALL be closed via `plt.close()` after saving
- **AND** memory SHALL be reclaimed periodically via garbage collection
- **AND** the pipeline SHALL NOT accumulate figures in memory

#### Scenario: Batched figures are saved and closed incrementally in ExploratoryAnalysisStep
- **WHEN** `ExploratoryAnalysisStep.execute()` generates batched histogram or boxplot figures for a
  dataset with many traits (enough to trigger `enable_batched_plots`)
- **THEN** each batch figure SHALL be saved and closed before the next batch figure is generated
- **AND** the peak number of simultaneously-open matplotlib figures during the step SHALL NOT scale
  with the total number of batches generated

#### Scenario: Non-batched figures are saved and closed incrementally in ExploratoryAnalysisStep
- **WHEN** `ExploratoryAnalysisStep.execute()` generates summary plots, EDA plots, or the full
  correlation heatmap
- **THEN** each figure SHALL be saved and closed before the next figure is generated
- **AND** no `all_figures`-style accumulation of not-yet-saved figures SHALL occur

#### Scenario: Batched figures are saved and closed incrementally in GenerateStaticFiguresStep
- **WHEN** `GenerateStaticFiguresStep` generates batched histogram or boxplot figures for a dataset
  with many traits
- **THEN** each batch figure SHALL be saved and closed before the next batch figure is generated
  (via the same underlying generator functions `ExploratoryAnalysisStep` uses)
- **AND** the peak number of simultaneously-open matplotlib figures during figure generation SHALL
  NOT scale with the total number of batches generated

### Requirement: Adaptive Batch Sizing
The batch generation for trait histograms and boxplots SHALL adapt batch size based on total trait count to reduce excessive file generation.

#### Scenario: High trait count experiment (100+ traits)
- **WHEN** generating batched histograms or boxplots for 100+ traits
- **THEN** the batch size (traits per page) SHALL increase to reduce total file count
- **AND** total batch count SHALL be reasonable (e.g., fewer than 30 pages per plot type)
- **AND** each subplot SHALL remain readable

#### Scenario: Low trait count experiment (< 50 traits)
- **WHEN** generating batched plots for fewer than 50 traits
- **THEN** batch size SHALL remain at the current default

### Requirement: UMAP Stub Transparency
When UMAP analysis is enabled in configuration but not yet implemented, the pipeline SHALL clearly communicate this to the user.

#### Scenario: UMAP enabled but not implemented
- **WHEN** config specifies `umap.enabled: true` and the UMAP step is a stub
- **THEN** config validation SHALL emit a warning indicating UMAP is not yet available
- **AND** the pipeline summary SHALL clearly state UMAP was skipped with the reason

### Requirement: PCA Feature Contribution Bar Chart
The visualization pipeline SHALL generate a stacked horizontal bar chart showing per-PC variance contributions for top features, matching notebook output.

`create_feature_contribution_plot` SHALL always select the displayed
features by total variance contribution (equivalent to
`select_top_features_from_pca(method="top_variance", pc_indices=None,
...)`) and SHALL NOT accept a `feature_selection` parameter, since the
chart's bars always plot true per-PC variance contribution regardless of
which traits are shown — a non-contribution selection criterion would make
the chart's title (which asserts the displayed traits are the top
contributors) misdescribe its own content. `GenerateStaticFiguresStep`
SHALL NOT pass a `feature_selection` argument to this function.

#### Scenario: Standard PCA analysis complete
- **WHEN** PCA results are available and `static_viz.create_pca_plots` is enabled
- **THEN** the pipeline SHALL generate a feature contribution bar chart via `create_feature_contribution_plot()`
- **AND** the chart SHALL be saved alongside other PCA figures (scree plot, biplot, heatmaps)

#### Scenario: create_feature_contribution_plot has no feature_selection parameter
- **WHEN** `create_feature_contribution_plot`'s signature is inspected
- **THEN** it SHALL NOT include a `feature_selection` parameter
- **AND** calling it with a `feature_selection` keyword argument SHALL raise `TypeError`

#### Scenario: On-the-fly contribution ranking matches select_top_features_from_pca
- **WHEN** `create_feature_contribution_plot` computes contributions on the
  fly (no pre-calculated `trait_contrib_df`/`feature_contributions` in
  `pca_results`)
- **THEN** the top features selected SHALL be identical, in the same order,
  to calling `select_top_features_from_pca(method="top_variance",
  pc_indices=None, ...)` directly with the same loadings and eigenvalues

### Requirement: Phenotype Variation Plots
The visualization pipeline SHALL generate phenotype variation plots (box+strip with genotype distributions) for configurable traits.

#### Scenario: Heritability results available
- **WHEN** heritability results are available and `static_viz.create_phenotype_variation_plots` is enabled
- **THEN** the pipeline SHALL generate phenotype variation plots for the top N traits by heritability (configurable)
- **AND** each plot SHALL show genotype distributions with extreme genotype highlighting
- **AND** figures SHALL be closed after saving to manage memory

#### Scenario: No heritability results
- **WHEN** heritability results are NOT available
- **THEN** the pipeline SHALL skip phenotype variation plots with a log message

### Requirement: Regression Plots for Configurable Trait Pairs
The visualization pipeline SHALL generate regression scatter plots for user-specified trait pairs when configured.

#### Scenario: Regression trait pairs configured
- **WHEN** `static_viz.regression_trait_pairs` is specified in config (list of [x, y] pairs)
- **THEN** the pipeline SHALL generate a regression plot for each pair via `create_regression_plot()`
- **AND** each plot SHALL include R-squared, p-value, and Pearson correlation annotation

#### Scenario: No regression pairs configured
- **WHEN** `static_viz.regression_trait_pairs` is empty or not specified
- **THEN** no regression plots SHALL be generated

### Requirement: Genotype Image Grids for Extreme Genotypes
The visualization pipeline SHALL generate image grids showing root images for genotypes identified as extreme by PCA, when image paths are available.

#### Scenario: Image paths available and PCA results exist
- **WHEN** image paths are configured and PCA results are available and `static_viz.create_genotype_image_grids` is enabled
- **THEN** the pipeline SHALL identify extreme genotypes via `identify_extreme_genotypes_by_pc()`
- **AND** generate image grids for each extreme genotype via `create_genotype_image_grid()`
- **AND** figures SHALL be closed after saving to manage memory

#### Scenario: No image paths available
- **WHEN** image paths are not configured or not available
- **THEN** image grid generation SHALL be skipped with a log message

### Requirement: Interactive Scatter with Images
The interactive visualization step SHALL generate a general interactive scatter plot with image hover when image paths are available.

#### Scenario: Image paths available
- **WHEN** image paths are configured and `interactive_viz.create_scatter_with_images` is enabled
- **THEN** the pipeline SHALL generate an interactive scatter HTML via `create_interactive_scatter_with_images()`
- **AND** the output SHALL be saved in the interactive_figures directory

### Requirement: HTML Image Viewer
The interactive visualization step SHALL generate an HTML page with click-to-view image panels when image paths are available.

#### Scenario: Image paths and interactive PCA available
- **WHEN** image paths are configured and interactive PCA plot exists
- **THEN** the pipeline SHALL generate an HTML image viewer via `create_html_with_image_viewer()`
- **AND** clicking any data point SHALL display the corresponding sample image

### Requirement: Interactive Image Gallery
The interactive visualization step SHALL generate a browsable HTML image gallery when image paths are available.

#### Scenario: Image paths available
- **WHEN** image paths are configured and `interactive_viz.create_image_gallery` is enabled
- **THEN** the pipeline SHALL generate an image gallery HTML via `create_interactive_image_gallery()`
- **AND** each image card SHALL show trait value tooltips on hover

### Requirement: Outlier Method Comparison Summary
The QC outlier visualization step SHALL generate a bar chart summarizing outlier counts per detection method.

#### Scenario: Multiple outlier methods run
- **WHEN** two or more outlier detection methods have been executed
- **THEN** the pipeline SHALL generate a bar chart showing the count of outliers detected by each method
- **AND** the chart SHALL include value labels on each bar
- **AND** the chart SHALL be saved alongside other outlier comparison figures

### Requirement: Interactive UMAP Genotype Coloring
The `GenerateInteractiveStep` SHALL generate interactive UMAP plots with data points colored by genotype, matching the style of interactive PCA plots.

#### Scenario: UMAP points colored by genotype
- **WHEN** interactive UMAP plot is generated with `interactive_viz.create_umap_plots` enabled
- **THEN** each data point SHALL be colored according to its genotype value
- **AND** colors SHALL be assigned consistently (same genotype = same color)
- **AND** a color legend SHALL be visible showing genotype names

#### Scenario: UMAP shows barcode on hover
- **WHEN** user hovers over a data point in the interactive UMAP plot
- **THEN** the Barcode value SHALL be displayed in the hover tooltip
- **AND** the Genotype value SHALL be displayed in the hover tooltip
- **AND** the UMAP coordinates (UMAP1, UMAP2) SHALL be displayed

#### Scenario: UMAP title shows parameters
- **WHEN** interactive UMAP plot is generated
- **THEN** the plot title SHALL include the UMAP parameters (n_neighbors, min_dist)
- **AND** the title SHALL be formatted consistently (e.g., "Interactive UMAP (n_neighbors=15, min_dist=0.1)")

### Requirement: Interactive UMAP Style Consistency
Interactive UMAP plots SHALL have visual consistency with interactive PCA plots for a coherent user experience.

#### Scenario: Consistent plot styling
- **WHEN** both interactive PCA and UMAP plots are generated
- **THEN** both SHALL use the same color palette for genotypes
- **AND** both SHALL use scatter plot mode with similar marker styling
- **AND** both SHALL include genotype information in hover data

### Requirement: Interactive UMAP Metadata Preservation
The `GenerateInteractiveStep` SHALL preserve all metadata from previous pipeline steps when generating UMAP plots.

#### Scenario: UMAP results preserved in output metadata
- **WHEN** `GenerateInteractiveStep` receives `umap_results` in `prev_result.metadata`
- **THEN** the step output metadata SHALL contain `umap_results`
- **AND** the embedding, clean_indices, and parameters SHALL be unchanged

#### Scenario: Image paths preserved through UMAP generation
- **WHEN** `GenerateInteractiveStep` receives `image_paths` in `prev_result.metadata`
- **THEN** the step output metadata SHALL contain `image_paths`
- **AND** image-dependent UMAP plots SHALL be generated if `show_images_on_hover` is enabled

#### Scenario: Trait names preserved through UMAP generation
- **WHEN** `GenerateInteractiveStep` receives `trait_names` in `prev_result.metadata`
- **THEN** the step output metadata SHALL contain `trait_names`
- **AND** the trait names list SHALL be unchanged

### Requirement: Interactive UMAP Data Alignment
The `GenerateInteractiveStep` SHALL correctly align DataFrame rows with UMAP embedding coordinates using `clean_indices`.

#### Scenario: DataFrame aligned with embedding using clean_indices
- **WHEN** UMAP results contain `clean_indices` (rows used after NaN removal)
- **THEN** the DataFrame SHALL be filtered to match those indices
- **AND** the Barcode values in hover data SHALL correspond to correct samples

#### Scenario: Full DataFrame used when no clean_indices
- **WHEN** UMAP results do not contain `clean_indices`
- **THEN** the full DataFrame SHALL be used for plotting
- **AND** embedding length SHALL match DataFrame row count

### Requirement: VIZ-OUTPUT-001 - Figures organized into subdirectories by plot type

All static and interactive figures MUST be saved to subdirectories within a single `figures/` directory, organized by plot type.

#### Scenario: PCA figures saved to figures/pca/
- **GIVEN** a viz pipeline run completes successfully
- **WHEN** PCA analysis is enabled
- **THEN** PCA plots (biplot, scree, loadings, contributions, pc_boxplots) are saved to `figures/pca/`
- **AND** no PCA plots exist in the root `figures/` directory

#### Scenario: Heritability figures saved to figures/heritability/
- **GIVEN** a viz pipeline run completes successfully
- **WHEN** heritability analysis produces paginated output
- **THEN** heritability plots are saved to `figures/heritability/`
- **AND** heritability plots are generated ONLY ONCE (not duplicated by multiple steps)

#### Scenario: Batched trait plots saved to dedicated subdirectories
- **GIVEN** a viz pipeline run with trait histograms and boxplots enabled
- **WHEN** batch generation completes
- **THEN** trait histograms are saved to `figures/trait_histograms/`
- **AND** trait boxplots are saved to `figures/trait_boxplots/`

#### Scenario: Interactive figures saved to figures/interactive/
- **GIVEN** a viz pipeline run with interactive visualization enabled
- **WHEN** interactive figures are generated
- **THEN** Plotly HTML files are saved to `figures/interactive/`
- **AND** no `interactive_figures/` directory is created at the run root

### Requirement: VIZ-OUTPUT-002 - No duplicate figure generation

Each figure type MUST be generated by exactly one pipeline step.

#### Scenario: Heritability plots not duplicated
- **GIVEN** a viz pipeline run completes
- **WHEN** counting heritability plot files
- **THEN** each heritability page exists exactly once (one PNG, optionally one PDF)
- **AND** the `statistical_analysis` step does not generate heritability plots

### Requirement: VIZ-OUTPUT-003 - Data outputs in data/ subdirectory

Analysis outputs (CSVs, JSONs) MUST be saved to a `data/` subdirectory, separate from figures.

#### Scenario: PCA data saved to data/pca/
- **GIVEN** a viz pipeline run with PCA analysis
- **WHEN** PCA analysis completes
- **THEN** PCA CSV outputs (components, loadings, variance) are saved to `data/pca/`
- **AND** no `pca/` directory exists at the run root

#### Scenario: Statistical analysis outputs saved to data/
- **GIVEN** a viz pipeline run with statistical analysis
- **WHEN** analysis completes
- **THEN** heritability_results.csv, anova_results.csv are saved to `data/`
- **AND** trait_statistics.json is saved to `data/`

### Requirement: Group-Based Visualization Execution

The visualization pipeline SHALL support the same group-by functionality as the QC pipeline for consistent per-group analysis.

#### Scenario: Viz pipeline groups like QC
- **GIVEN** a QC output with groups processed by plant_age_days
- **WHEN** viz pipeline runs with `group_by: "plant_age_days"`
- **THEN** viz SHALL create separate visualizations for each timepoint
- **AND** output structure SHALL mirror QC: `viz_<pipeline>_plant_age_days_<value>_<timestamp>/`

#### Scenario: PCA computed per group
- **GIVEN** a grouped viz pipeline (day_7, day_14, day_21)
- **WHEN** PCA analysis is performed
- **THEN** each group SHALL have independent PC loadings and variance explained
- **AND** PC1 for day_7 MAY differ from PC1 for day_14 (developmental differences)

#### Scenario: Interactive plots per group
- **GIVEN** a viz pipeline grouped by plant_age_days
- **WHEN** interactive PCA plots are generated
- **THEN** each group SHALL have its own `pca_interactive.html` file
- **AND** plot title SHALL indicate the group (e.g., "PCA: plant_age_days = 7")

#### Scenario: Statistical analysis per group
- **GIVEN** a viz pipeline grouped by experiment_id
- **WHEN** ANOVA and heritability are calculated
- **THEN** statistics SHALL be computed independently per group
- **AND** `08_heritability_results.csv` SHALL contain group-specific H² estimates

#### Scenario: Summary reports per group
- **GIVEN** a grouped viz pipeline
- **WHEN** summary markdown is generated
- **THEN** each group SHALL have its own `summary_report.md`
- **AND** the report SHALL document the group identifier and sample count

### Requirement: Boxplot Genotype Label Readability
Trait boxplots grouped by genotype SHALL produce readable, non-overlapping genotype labels regardless of genotype count or label length. Both vertical and horizontal orientations SHALL use a consistent visual style: unfilled outline boxes with blue outlines, green median lines, and gridlines enabled.

#### Scenario: Vertical boxplots with 10+ genotypes switch to horizontal
- **WHEN** generating trait boxplots with more than 8 genotypes
- **AND** orientation is set to "auto" (default)
- **THEN** the boxplots SHALL use horizontal orientation
- **AND** genotype names SHALL be displayed as y-axis labels (no rotation needed)
- **AND** boxes SHALL use unfilled outline style (matching vertical orientation)

#### Scenario: Vertical boxplots with 7 or fewer genotypes
- **WHEN** generating trait boxplots with 7 or fewer genotypes
- **AND** orientation is set to "auto" (default)
- **THEN** the boxplots SHALL use vertical orientation
- **AND** x-axis labels SHALL be rotated 90 degrees

#### Scenario: Consistent box style across orientations
- **WHEN** generating trait boxplots in either vertical or horizontal orientation
- **THEN** boxes SHALL use unfilled outline style (no fill color)
- **AND** box and whisker outlines SHALL be blue (`#1f77b4`)
- **AND** median lines SHALL be green (`#2ca02c`)
- **AND** gridlines SHALL be enabled
- **AND** the visual appearance SHALL be consistent regardless of orientation

#### Scenario: Subplot width scales with genotype count
- **WHEN** generating vertical trait boxplots with many genotypes
- **THEN** subplot width SHALL scale with the number of genotypes
- **AND** minimum subplot width SHALL be 4.0 inches
- **AND** width per genotype SHALL be at least 0.5 inches

#### Scenario: Subplot height scales with genotype count in horizontal orientation, bounded by a cap
- **WHEN** generating horizontal-orientation trait boxplots (`n_genotypes` above
  `horizontal_threshold`) via `create_trait_boxplots_by_genotype()` or
  `create_trait_boxplots_by_genotype_batched()`
- **THEN** subplot height SHALL scale with the number of genotypes at 0.3 inches per genotype,
  with a minimum of 4.0 inches
- **AND** subplot height SHALL NOT exceed 20.0 inches, regardless of genotype count
- **AND** this cap SHALL take effect at the point the figure is actually rendered (i.e. it SHALL
  NOT be silently discarded by an inner sizing recomputation)

#### Scenario: Horizontal subplot height is unaffected below the cap
- **WHEN** generating horizontal-orientation trait boxplots whose `n_genotypes * 0.3` is below
  20.0 inches
- **THEN** subplot height SHALL be computed exactly as before this change
  (`max(4.0, n_genotypes * 0.3)`), unchanged from current behavior

#### Scenario: Batched boxplots suptitle does not overlap subplots
- **WHEN** generating batched trait boxplots via `create_trait_boxplots_by_genotype_batched()`
- **THEN** the batch title (suptitle) SHALL NOT overlap the top row of subplots
- **AND** `tight_layout()` SHALL be called AFTER suptitle is set (not before)

#### Scenario: Label font size adapts to genotype count
- **WHEN** generating trait boxplots with many genotypes (>10)
- **THEN** x-axis tick label font size SHALL decrease to maintain readability
- **AND** font size SHALL not go below 6pt

#### Scenario: Backward compatibility with few genotypes
- **WHEN** generating trait boxplots with 5 or fewer genotypes
- **THEN** the plot appearance SHALL be visually similar to current behavior
- **AND** no layout changes SHALL be noticeable

#### Scenario: Zero genotypes or zero traits
- **WHEN** generating trait boxplots (batched or non-batched) with zero genotypes or an empty
  `trait_cols` list
- **THEN** the function SHALL return without error (an empty list for batched, a placeholder
  "no data"/"no traits" figure for non-batched)
- **AND** no cap or sizing calculation SHALL raise an exception on this input

### Requirement: Statistical Analysis Heritability Flag
The `StatisticalAnalysisStep` SHALL check `config.statistics.calculate_heritability`
before calling `calculate_heritability_estimates()`. When the flag is `False`, the step
SHALL skip heritability calculation entirely, set `heritability_results` to an empty dict
in output metadata, omit `08_heritability_results.csv`, and record
`"heritability_summary": {"skipped": true}` in the summary JSON.

#### Scenario: Heritability skipped when flag is disabled
- **WHEN** `config.statistics.calculate_heritability` is `False`
- **THEN** `StatisticalAnalysisStep` SHALL NOT call `calculate_heritability_estimates()`
- **AND** `heritability_results` in output metadata SHALL be an empty dict `{}`
- **AND** `08_heritability_results.csv` SHALL NOT be generated
- **AND** the summary JSON SHALL contain `"heritability_summary": {"skipped": true}`

#### Scenario: Heritability calculated when flag is enabled (default)
- **WHEN** `config.statistics.calculate_heritability` is `True` (default)
- **THEN** `StatisticalAnalysisStep` SHALL call `calculate_heritability_estimates()`
- **AND** `heritability_results` SHALL be populated in output metadata
- **AND** `08_heritability_results.csv` SHALL be generated
- **AND** behavior SHALL be identical to current implementation

#### Scenario: Downstream FilterHeritabilityStep handles skipped heritability
- **WHEN** `heritability_results` is an empty dict from a prior step
- **THEN** `FilterHeritabilityStep` SHALL skip filtering gracefully
- **AND** the pipeline SHALL continue without error

### Requirement: Boxplot Genotype Pagination
`create_trait_boxplots_by_genotype_batched()` SHALL split genotypes across multiple figures when
the genotype count exceeds what fits in one `max_subplot_height`-capped figure while remaining
readable at the standard per-genotype spacing (0.3"/genotype horizontal, 0.5"/genotype vertical),
rather than relying on the height cap alone, which is memory-safe but not necessarily readable at
extreme genotype counts.

#### Scenario: Genotypes are paginated across multiple figures when they exceed page capacity
- **WHEN** `create_trait_boxplots_by_genotype_batched()` generates boxplots for a genotype count
  above the per-page capacity (auto-derived from `max_subplot_height` and the per-genotype size for
  the resolved `actual_orientation`: `max_subplot_height // 0.3` ≈ 66 for horizontal,
  `max_subplot_height // 0.5` = 40 for vertical, unless `max_genotypes_per_page` is explicitly set)
- **THEN** genotypes SHALL be split into consecutive, alphabetically-sorted pages of at most
  `max_genotypes_per_page` genotypes each
- **AND** one figure SHALL be rendered per (trait batch, genotype page) combination
- **AND** every genotype SHALL appear in exactly one page's figure (no genotype dropped or
  duplicated across pages)
- **AND** each page's rendered subplot height SHALL use the pre-cap readable spacing
  (`page_genotype_count * per_genotype_size`), not the `max_subplot_height` cap, since pages are
  sized to stay under it by construction

#### Scenario: Pagination is a no-op at or below page capacity
- **WHEN** `create_trait_boxplots_by_genotype_batched()` generates boxplots for a genotype count at
  or below the per-page capacity (≤ 66 genotypes for horizontal orientation, ≤ 40 for vertical, by
  default)
- **THEN** exactly one genotype page SHALL be produced per trait batch (no behavior change from
  before pagination was introduced)

#### Scenario: Multi-page batch suptitle identifies the genotype range
- **WHEN** a trait batch is split into more than one genotype page
- **THEN** each page's figure `suptitle` SHALL include the genotype range and total genotype count
  for that page (e.g. "Genotypes 1-66 of 489"), in addition to the existing trait-range text

#### Scenario: Pagination orientation is consistent across all pages of a batch
- **WHEN** genotype pagination produces a small final page (e.g. below `horizontal_threshold`)
  alongside larger preceding pages within the same trait batch
- **THEN** every page SHALL use the same resolved orientation (derived from the full dataset's
  genotype count), not an orientation independently re-resolved from that page's own, possibly much
  smaller, genotype count

#### Scenario: Pagination handles a missing genotype column or NaN genotype values safely
- **WHEN** the DataFrame passed to `create_trait_boxplots_by_genotype_batched()` either has no
  `genotype_col` column, or has some rows with a NaN genotype value
- **THEN** pagination SHALL NOT raise an exception
- **AND** if `genotype_col` is absent, pagination SHALL be a no-op (one page per trait batch)
- **AND** if NaN genotype values are present, they SHALL be excluded from page assignment (dropped
  before sorting/paging), and every non-NaN genotype SHALL still appear in exactly one page

