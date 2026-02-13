## ADDED Requirements

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
