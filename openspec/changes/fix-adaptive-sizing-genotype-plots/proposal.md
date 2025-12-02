# Fix Adaptive Sizing for Genotype and Variance Decomposition Plots

**Status**: 📋 PROPOSED
**Created**: 2025-12-01
**Type**: Bug Fix + Enhancement

## Why

The initial adaptive sizing implementation (add-qc-adaptive-sizing) only addressed 3 specific plots (correlation heatmap, heritability bar plot, variance decomposition grid). However, analysis revealed critical issues:

### Problems Identified:

1. **Genotype plots have NO adaptive sizing** - All plots with genotypes on X-axis use fixed figure sizes, causing severe crowding with 100+ genotypes despite rotated labels
2. **Variance decomposition doesn't actually use adaptive sizing** - Step 9 hardcodes figsize to (14, 10) instead of using the calculated adaptive size
3. **Variance decomposition Panel 4 has axis jittering** - Mixes numeric x positions with categorical labels, causing traits to appear "jittered around ticks"
4. **Wrong dimension scaling assumptions** - Need to scale WIDTH for X-axis item count, not HEIGHT

### User Impact:

- **150 genotype dataset**: Genotype boxplots have illegible, overlapping X-axis labels
- **Variance decomposition**: Fixed size regardless of trait count, plus visual artifacts
- **Poor user experience**: Users enable adaptive sizing but see no improvement in genotype plots

## What Changes

### 1. Fix Variance Decomposition (Step 9)

**File**: `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py`

**Current code (line 182-188)**:
```python
else:
    var_figsize = (14, 10)

fig_var = create_variance_decomposition_plot(
    comparison_df=comparison_df,
    figsize=var_figsize,
    output_path=None,  # Will save manually
)
```

**Fix**: Actually apply the adaptive sizing that was calculated:
```python
# Apply adaptive sizing to variance decomposition
if config.adaptive_sizing and config.adaptive_sizing.enabled:
    # 2x2 grid of panels, each showing traits on x-axis
    # Scale WIDTH based on trait count (traits are on x-axis)
    n_traits = len(comparison_df)
    var_figsize = calculate_barplot_size(
        n_items=n_traits,
        config=config.adaptive_sizing,
        orientation="vertical",  # Traits on X-axis
        as_subplot=True,  # Part of 2x2 grid
        n_subplots=4,
    )
else:
    var_figsize = (14, 10)
```

### 2. Fix Variance Decomposition Panel 4 Axis Jittering

**File**: `src/sleap_roots_analyze/visualization.py`

**Current code (lines 926-949)**: Uses numeric x_pos then manually sets trait labels
```python
x_pos = range(len(comparison_df))
# ... plotting code ...
ax.set_xticklabels(comparison_df["trait"], rotation=45, ha="right")
```

**Fix**: Use consistent categorical x-axis like Panels 1-3:
```python
# Use trait names directly as categorical x-axis (consistent with other panels)
comparison_df.plot(
    x="trait",
    y=["var_genetic", "var_residual"],
    kind="bar",
    stacked=True,
    ax=ax,
    color=colors,
    width=0.8,
    legend=False,
)
ax.set_xticklabels(comparison_df["trait"], rotation=45, ha="right")
```

### 3. Add Adaptive Sizing to Genotype Boxplots

**File**: `src/sleap_roots_analyze/visualization.py`

**Functions to update**:
- `create_trait_boxplots_by_genotype()` (lines 108-177)
- `create_trait_by_genotype_boxplots()` (lines 963-1061)

**Add optional `adaptive_config` parameter**:
```python
def create_trait_boxplots_by_genotype(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str,
    figsize: Tuple[int, int] = (16, 16),
    adaptive_config: Optional[AdaptiveSizingConfig] = None,  # NEW
) -> plt.Figure:
    """Create boxplots for each trait by genotype.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        genotype_col: Name of genotype column
        figsize: Figure size (only used if adaptive_config is None)
        adaptive_config: Optional adaptive sizing configuration
    """
    n_genotypes = df[genotype_col].nunique()

    # Calculate adaptive width for genotype-based plots
    if adaptive_config is not None:
        # Each subplot needs width based on genotype count
        subplot_width = calculate_barplot_size(
            n_items=n_genotypes,
            config=adaptive_config,
            orientation="vertical",  # Genotypes on X-axis
            as_subplot=True,
            n_subplots=len(trait_cols),
        )[0]  # Take width only

        # Calculate grid dimensions
        n_cols = min(3, len(trait_cols))
        n_rows = (len(trait_cols) + n_cols - 1) // n_cols

        # Total figure size
        fig_width = subplot_width * n_cols
        fig_height = adaptive_config.base_height * n_rows
        figsize = (
            min(adaptive_config.max_width, max(adaptive_config.min_width, fig_width)),
            min(adaptive_config.max_height, max(adaptive_config.min_height, fig_height)),
        )

    # Rest of function unchanged...
```

### 4. Add Adaptive Sizing to Sample Count Bar Plot

**File**: `src/sleap_roots_analyze/visualization.py` (lines 370-408)

**Current**: Fixed figsize (10, 6)

**Fix**: Add adaptive_config parameter and scale width based on genotype count:
```python
def create_exploratory_summary_plots(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str,
    replicate_col: Optional[str] = None,
    adaptive_config: Optional[AdaptiveSizingConfig] = None,  # NEW
) -> Dict[str, plt.Figure]:

    # Sample count bar plot
    n_genotypes = df[genotype_col].nunique()
    if adaptive_config is not None:
        sample_count_figsize = calculate_barplot_size(
            n_items=n_genotypes,
            config=adaptive_config,
            orientation="vertical",
        )
    else:
        sample_count_figsize = (10, 6)

    fig_sample_count, ax = plt.subplots(figsize=sample_count_figsize)
    # ... rest of plotting code
```

### 5. Update Pipeline Steps to Pass adaptive_config

**Files**:
- `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py` (Step 4)
- `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` (Step 9)

**Step 4 changes**:
```python
# Pass adaptive config to genotype-based plots
summary_figs = create_exploratory_summary_plots(
    df=df,
    trait_cols=trait_cols,
    genotype_col=config.columns.genotype,
    replicate_col=config.columns.replicate,
    adaptive_config=config.adaptive_sizing if config.adaptive_sizing.enabled else None,  # NEW
)

trait_boxplots_fig = create_trait_boxplots_by_genotype(
    df=df,
    trait_cols=trait_cols,
    genotype_col=config.columns.genotype,
    figsize=tuple(config.visualization.figsize),
    adaptive_config=config.adaptive_sizing if config.adaptive_sizing.enabled else None,  # NEW
)
```

**Step 9 changes**:
```python
# Pass adaptive config to removed traits boxplots
fig_box = create_trait_by_genotype_boxplots(
    df=df,
    traits=traits_to_plot,
    heritability_results=heritability_results,
    genotype_col=config.columns.genotype,
    output_path=None,
    adaptive_config=config.adaptive_sizing if config.adaptive_sizing.enabled else None,  # NEW
)
```

### 6. Add Helper Function for Subplot Sizing

**File**: `src/sleap_roots_analyze/viz_utils.py`

**Add new parameter to `calculate_barplot_size`**:
```python
def calculate_barplot_size(
    n_items: int,
    config: AdaptiveSizingConfig,
    orientation: str = "vertical",
    width_per_bar: float = 0.8,
    as_subplot: bool = False,  # NEW
    n_subplots: int = 1,  # NEW
) -> Tuple[float, float]:
    """Calculate adaptive figure size for bar plots.

    Args:
        n_items: Number of bars
        config: Adaptive sizing configuration
        orientation: "vertical" or "horizontal"
        width_per_bar: Width per bar in inches (for vertical bars)
        as_subplot: True if this is a subplot in a grid
        n_subplots: Total number of subplots in the figure

    Returns:
        (width, height) in inches
    """
    if orientation == "vertical":
        # Width scales with number of bars
        width = max(config.base_width, n_items * width_per_bar)
        height = config.base_height

        # If subplot, reduce width to account for multiple subplots
        if as_subplot and n_subplots > 1:
            width = width / math.sqrt(n_subplots)

    else:  # horizontal
        width = config.base_width
        height = max(config.base_height, n_items * width_per_bar)

        # If subplot, reduce height
        if as_subplot and n_subplots > 1:
            height = height / math.sqrt(n_subplots)

    # Apply bounds
    width = max(config.min_width, min(config.max_width, width))
    height = max(config.min_height, min(config.max_height, height))

    return (width, height)
```

## Impact

### Files Modified: 3-4 files
- `src/sleap_roots_analyze/visualization.py` - Add adaptive_config parameters
- `src/sleap_roots_analyze/viz_utils.py` - Add subplot support
- `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py` - Pass adaptive_config
- `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` - Pass adaptive_config and fix variance decomposition

### Backwards Compatibility
**YES** - All changes are backwards compatible:
- New parameters are optional with defaults
- Only activated when `adaptive_sizing.enabled = true`
- Existing configs with `enabled: false` behave identically

### Test Coverage
Need to update/add tests:
- `test_visualization.py` - Test adaptive sizing for genotype plots
- `test_step_exploratory_analysis.py` - Test adaptive config passing
- `test_step_filter_heritability.py` - Test variance decomposition adaptive sizing

### Benefits:
1. **Genotype plots scale properly** - 10 genotypes vs 150 genotypes → readable labels
2. **Variance decomposition actually works** - Was broken in initial implementation
3. **No axis jittering** - Consistent categorical x-axis across all panels
4. **Complete adaptive sizing** - All QC pipeline plots now responsive to data size

### Risks:
- **Medium risk** - Modifies visualization function signatures (but backwards compatible)
- **Subplot sizing complexity** - Need to test grid layouts with varying trait/genotype counts
- **Test maintenance** - More test scenarios to cover

## Implementation Checklist

- [ ] Add `as_subplot` parameter to `calculate_barplot_size()` in viz_utils.py
- [ ] Fix variance decomposition Panel 4 axis jittering in visualization.py
- [ ] Add `adaptive_config` parameter to `create_trait_boxplots_by_genotype()`
- [ ] Add `adaptive_config` parameter to `create_trait_by_genotype_boxplots()`
- [ ] Add `adaptive_config` parameter to `create_exploratory_summary_plots()`
- [ ] Update Step 4 to pass adaptive_config to genotype plots
- [ ] Update Step 9 to actually use calculated adaptive size for variance decomposition
- [ ] Update Step 9 to pass adaptive_config to removed traits boxplots
- [ ] Add/update tests for adaptive sizing with genotype plots
- [ ] Add/update tests for variance decomposition adaptive sizing
- [ ] Run full test suite
- [ ] Test with real data (150 genotypes, varying trait counts)

## Timeline

**Estimated**: 6-8 hours total
- Visualization function updates: 3 hours
- Pipeline step integration: 1 hour
- Testing: 2-3 hours
- Real data validation: 1-2 hours

## Related

- **Depends on**: add-qc-adaptive-sizing (completed)
- **Fixes**: Incomplete adaptive sizing implementation
- **Issue**: User-reported crowding in genotype plots and variance decomposition artifacts
