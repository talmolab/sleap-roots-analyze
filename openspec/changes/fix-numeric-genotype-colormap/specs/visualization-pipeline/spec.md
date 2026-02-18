## MODIFIED Requirements

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
