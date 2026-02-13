## ADDED Requirements

### Requirement: Cross-Platform Plot Label Formatting via sanitize_trait_names
Cross-platform joint plots, boxplots, and heatmap axis labels SHALL use the existing `sanitize_trait_names()` function from `data_utils.py` for consistent trait name formatting, rather than ad-hoc string replacement.

#### Scenario: Joint plot axis labels
- **WHEN** generating a cross-platform joint plot with trait names as axis labels
- **THEN** labels SHALL be formatted using `sanitize_trait_names()` from `data_utils.py`
- **AND** ad-hoc `.replace('_', ' ').title()` calls SHALL be removed in favor of the shared utility
- **AND** labels SHALL match the formatting used in QC pipeline outputs

#### Scenario: Boxplot axis labels
- **WHEN** generating a cross-platform genotype boxplot with trait names
- **THEN** trait names SHALL be formatted using the shared `sanitize_trait_names()` utility
- **AND** formatting SHALL be consistent with QC and Viz pipeline label formatting

#### Scenario: Heatmap axis labels in trait clustering
- **WHEN** generating trait cluster heatmaps or dendrograms
- **THEN** trait labels on axes SHALL be formatted using the shared utility
- **AND** labels SHALL remain legible at the figure's native resolution
