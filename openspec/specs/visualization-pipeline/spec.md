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

#### Scenario: Pass parameters to PCA biplot
- **WHEN** generating PCA biplot via `create_pca_biplot`
- **THEN** the step SHALL pass `config.static_viz.genotypes_to_color` to the function
- **AND** the step SHALL pass `config.static_viz.highlight_genotypes` to the function

#### Scenario: Pass parameters to PC boxplots
- **WHEN** generating PC boxplots via `create_pc_genotype_boxplots`
- **THEN** the step SHALL pass `config.static_viz.highlight_genotypes` to the function
- **AND** the highlighted genotypes SHALL appear in gold with bold labels

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
The visualization pipeline SHALL manage matplotlib figure memory to prevent freezing or crashes during batch generation.

#### Scenario: Generating 50+ figures sequentially
- **WHEN** the pipeline generates 50+ batch figures (histograms, boxplots) in sequence
- **THEN** each figure SHALL be closed via `plt.close()` after saving
- **AND** memory SHALL be reclaimed periodically via garbage collection
- **AND** the pipeline SHALL NOT accumulate figures in memory

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

#### Scenario: Standard PCA analysis complete
- **WHEN** PCA results are available and `static_viz.create_pca_plots` is enabled
- **THEN** the pipeline SHALL generate a feature contribution bar chart via `create_feature_contribution_plot()`
- **AND** the chart SHALL be saved alongside other PCA figures (scree plot, biplot, heatmaps)

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

