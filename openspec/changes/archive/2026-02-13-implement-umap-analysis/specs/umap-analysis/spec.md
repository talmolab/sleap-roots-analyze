## ADDED Requirements

### Requirement: UMAP Analysis Step Execution

The UMAPAnalysisStep SHALL perform UMAP dimensionality reduction on trait data when `config.umap.enabled` is true.

#### Scenario: UMAP analysis executes successfully
- **WHEN** UMAPAnalysisStep.execute() is called with valid trait data and `config.umap.enabled = true`
- **THEN** the step SHALL call `perform_umap_analysis()` with trait columns from metadata
- **AND** return a StepResult with umap_results in metadata

#### Scenario: UMAP analysis skipped when disabled
- **WHEN** UMAPAnalysisStep.execute() is called with `config.umap.enabled = false`
- **THEN** the step SHALL return early with `umap_status: "disabled"` in metadata
- **AND** preserve all previous metadata unchanged

#### Scenario: UMAP gracefully handles missing dependency
- **WHEN** umap-learn package is not installed
- **THEN** the step SHALL log a warning message
- **AND** return with `umap_status: "skipped_not_installed"` in metadata
- **AND** NOT raise an exception

### Requirement: UMAP Metadata Propagation

The UMAPAnalysisStep SHALL preserve all metadata from previous pipeline steps, ensuring downstream steps have access to required data.

#### Scenario: Previous metadata preserved
- **WHEN** UMAPAnalysisStep receives prev_result with metadata containing `image_paths`, `pca_results`, and `heritability_results`
- **THEN** the output metadata SHALL contain all these keys with their original values
- **AND** add `umap_results` to the metadata

#### Scenario: Image paths flow through to interactive step
- **WHEN** image_paths is set by LoadDataAndImagesStep and flows through StatisticalAnalysis → PCA → UMAP
- **THEN** GenerateInteractiveStep SHALL receive image_paths in its prev_result.metadata
- **AND** be able to create interactive UMAP plots with image hover

### Requirement: UMAP Results Structure

The umap_results dictionary in metadata SHALL contain all data needed for downstream visualization.

#### Scenario: UMAP results contain required keys
- **WHEN** UMAP analysis completes successfully
- **THEN** umap_results SHALL contain:
  - `embedding`: numpy array of shape (n_samples, 2)
  - `n_neighbors`: integer from config
  - `min_dist`: float from config

#### Scenario: UMAP embedding matches sample count
- **WHEN** UMAP analysis runs on N samples (after NaN removal)
- **THEN** umap_results["embedding"] SHALL have shape (N, 2)

### Requirement: UMAP Reproducibility

The UMAPAnalysisStep SHALL produce reproducible results when given the same data and configuration.

#### Scenario: Same random state produces identical results
- **WHEN** UMAP analysis runs twice with identical data and `config.umap.random_state = 42`
- **THEN** the resulting embeddings SHALL be identical

### Requirement: UMAP Artifact Export

The UMAPAnalysisStep SHALL export analysis artifacts for reproducibility and downstream analysis.

#### Scenario: UMAP artifacts saved to data directory
- **WHEN** UMAP analysis completes successfully
- **THEN** the step SHALL create `data/umap/` directory in run_dir
- **AND** save `umap_embedding.csv` with sample IDs and UMAP coordinates
- **AND** save `umap_parameters.json` with n_neighbors, min_dist, random_state

### Requirement: UMAP Static Visualizations

The GenerateStaticFiguresStep SHALL generate UMAP plots when UMAP results are available and `config.static_viz.create_umap_plots` is true.

#### Scenario: Basic UMAP scatter plot generated
- **WHEN** umap_results is in metadata and `create_umap_plots = true`
- **THEN** a UMAP scatter plot colored by genotype SHALL be saved to `figures/umap/`

#### Scenario: UMAP colored by top traits generated
- **WHEN** umap_results and pca_results are both in metadata
- **THEN** `create_umap_colored_by_top_traits()` SHALL be called
- **AND** the resulting figure SHALL be saved to `figures/umap/`

### Requirement: UMAP Interactive Visualizations

The GenerateInteractiveStep SHALL generate interactive UMAP plots when UMAP results are available and `config.interactive_viz.create_umap_plots` is true.

#### Scenario: Interactive UMAP with hover highlighting generated
- **WHEN** umap_results is in metadata and `create_umap_plots = true`
- **THEN** `create_interactive_umap_with_hover_highlight()` SHALL be called
- **AND** the result SHALL be saved as HTML to `figures/interactive/`

#### Scenario: Interactive UMAP with images generated
- **WHEN** umap_results and image_paths are both in metadata
- **AND** `show_images_on_hover = true`
- **THEN** `create_interactive_umap_with_images()` SHALL be called
- **AND** the result SHALL be saved as HTML to `figures/interactive/`
