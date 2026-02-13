## ADDED Requirements

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

### Requirement: Backward Compatibility
Existing visualization configs without genotype image grid configuration SHALL continue to function identically.

#### Scenario: Legacy config without image grid settings
- **WHEN** running a viz config without `genotype_image_grid_image_type` or `genotype_image_grid_trait_cols`
- **THEN** the default values SHALL be used ("features.png" and None respectively)
- **AND** existing behavior SHALL be preserved
- **AND** no validation errors SHALL occur
