# Visualization Pipeline - Genotype Highlighting

## ADDED Requirements

### Requirement: Genotype Highlighting Configuration
The `StaticVisualizationConfig` dataclass SHALL provide optional `genotypes_to_color` and `highlight_genotypes` parameters to enable selective genotype highlighting in PCA plots.

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
