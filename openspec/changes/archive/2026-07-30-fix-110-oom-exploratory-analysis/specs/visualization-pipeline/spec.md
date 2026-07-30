## MODIFIED Requirements

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

## ADDED Requirements

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
