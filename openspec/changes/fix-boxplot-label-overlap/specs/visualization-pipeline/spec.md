## ADDED Requirements

### Requirement: Boxplot Genotype Label Readability
Trait boxplots grouped by genotype SHALL produce readable, non-overlapping genotype labels regardless of genotype count or label length.

#### Scenario: Vertical boxplots with 10+ genotypes switch to horizontal
- **WHEN** generating trait boxplots with more than 8 genotypes
- **AND** orientation is set to "auto" (default)
- **THEN** the boxplots SHALL use horizontal orientation
- **AND** genotype names SHALL be displayed as y-axis labels (no rotation needed)

#### Scenario: Vertical boxplots with 7 or fewer genotypes
- **WHEN** generating trait boxplots with 7 or fewer genotypes
- **AND** orientation is set to "auto" (default)
- **THEN** the boxplots SHALL use vertical orientation
- **AND** x-axis labels SHALL be rotated 90 degrees

#### Scenario: Subplot width scales with genotype count
- **WHEN** generating vertical trait boxplots with many genotypes
- **THEN** subplot width SHALL scale with the number of genotypes
- **AND** minimum subplot width SHALL be 4.0 inches
- **AND** width per genotype SHALL be at least 0.5 inches

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
