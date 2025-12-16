# Fix QC Pipeline Visualization Issues

**Status**: 📋 PROPOSED
**Created**: 2025-12-01
**Type**: Bug Fix + Enhancement

## Why

Multiple visualization issues were discovered during testing with the 150 genotype dataset:

### Problems Identified:

1. **Variance decomposition x-axis inconsistency** - Panels 1-3 use pandas `.plot(x="trait", ...)` which creates unexpected label placement, while Panel 4 uses numeric positions and looks correct
2. **Heritability threshold mismatch** - Variance decomposition plot hardcodes threshold to 0.3, but config specifies 0.40, creating visual inconsistency across plots
3. **Last batch subplot sizing** - Batched plots use fixed `figsize=(16, 16)` for all batches, causing last batch with fewer traits to have stretched/oversized subplots
4. **No batch plot configuration** - Batched plot generation is hardcoded (`len(trait_cols) > 16`, `batch_size=16`) with no user control

### User Impact:

- **Variance decomposition**: Confusing x-axis label placement in 3 out of 4 panels
- **Threshold confusion**: Different threshold lines (0.3 vs 0.40) across heritability-related plots
- **Last batch poor aesthetics**: Final batches with 6-10 traits look unprofessional with huge empty spaces
- **Limited control**: Cannot disable batching or adjust batch size for different workflows

## What Changes

### 1. Fix Variance Decomposition X-Axis Consistency

**File**: `src/sleap_roots_analyze/visualization.py`

**Current code (Panels 1-3, lines 940-967)**: Uses pandas plotting
```python
# Panel 1: Heritability bar chart
ax = axes[0]
comparison_df.plot(
    x="trait", y="heritability", kind="bar", ax=ax, legend=False, color="steelblue"
)
```

**Fix**: Use numeric positions consistently like Panel 4
```python
# Panel 1: Heritability bar chart
ax = axes[0]
x_pos = range(len(comparison_df))
ax.bar(x_pos, comparison_df["heritability"], color="steelblue", alpha=0.7)
ax.set_ylabel("Heritability (H²)")
ax.set_title("Heritability Estimates")
ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
ax.legend()
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df["trait"], rotation=45, ha="right")
ax.set_xlabel("")

# Panel 2: Variance components (stacked bar)
ax = axes[1]
x_pos = range(len(comparison_df))
ax.bar(x_pos, comparison_df["var_genetic"], label="Genetic (σ²_G)", color="steelblue", alpha=0.7)
ax.bar(x_pos, comparison_df["var_residual"], bottom=comparison_df["var_genetic"],
       label="Residual (σ²_E)", color="orange", alpha=0.7)
ax.set_ylabel("Variance")
ax.set_title("Genetic vs Residual Variance")
ax.legend()
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df["trait"], rotation=45, ha="right")
ax.set_xlabel("")

# Panel 3: Percentage between genotypes
ax = axes[2]
x_pos = range(len(comparison_df))
ax.bar(x_pos, comparison_df["pct_var_between"], color="green", alpha=0.7)
ax.set_ylabel("% of Total Variance")
ax.set_title("Percentage Variance Between Genotypes")
ax.axhline(y=50, color="r", linestyle="--", alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(comparison_df["trait"], rotation=45, ha="right")
ax.set_xlabel("")
```

### 2. Add Threshold Parameter to Variance Decomposition Plot

**File**: `src/sleap_roots_analyze/visualization.py`

**Current signature (line 897)**:
```python
def create_variance_decomposition_plot(
    comparison_df: pd.DataFrame,
    figsize: tuple = (14, 10),
    output_path: Optional[Path] = None,
) -> plt.Figure:
```

**Fix**: Add threshold parameter
```python
def create_variance_decomposition_plot(
    comparison_df: pd.DataFrame,
    figsize: tuple = (14, 10),
    output_path: Optional[Path] = None,
    threshold: float = 0.3,  # NEW - heritability threshold for reference lines
) -> plt.Figure:
    """Create 4-panel variance decomposition plot for heritability diagnostics.

    Args:
        comparison_df: DataFrame from compare_trait_heritabilities()
        figsize: Figure size (width, height) in inches
        output_path: Optional path to save figure
        threshold: Heritability threshold for reference lines (default: 0.3)

    Returns:
        matplotlib Figure object
    """
```

**Update Panel 1 (line 944)**: Use parameter instead of hardcoded value
```python
ax.axhline(y=threshold, color="r", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
```

### 3. Pass Config Threshold to Variance Decomposition

**File**: `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py`

**Current code (line 230)**:
```python
fig_var = create_variance_decomposition_plot(
    comparison_df=comparison_df,
    figsize=var_figsize,
    output_path=None,  # Will save manually
)
```

**Fix**: Pass threshold from config
```python
fig_var = create_variance_decomposition_plot(
    comparison_df=comparison_df,
    figsize=var_figsize,
    output_path=None,  # Will save manually
    threshold=config.heritability.threshold,  # NEW - use config threshold
)
```

### 4. Fix Last Batch Subplot Sizing

**File**: `src/sleap_roots_analyze/visualization.py`

**Update both batched functions** (lines 211-292):

**`create_trait_histograms_batched`**:
```python
def create_trait_histograms_batched(
    df: pd.DataFrame,
    trait_cols: List[str],
    batch_size: int = 16,
    n_cols: int = 4,
    figsize: Tuple[int, int] = (16, 16),
) -> List[plt.Figure]:
    """Create batched histogram plots for traits (multiple figures for many traits).

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        batch_size: Number of traits per figure (default: 16)
        n_cols: Number of columns in subplot grid
        figsize: Figure size for FULL batches (default: (16, 16))

    Returns:
        List of matplotlib figure objects (one per batch)
    """
    n_traits = len(trait_cols)
    if n_traits == 0:
        return []

    figures = []
    for batch_start in range(0, n_traits, batch_size):
        batch_end = min(batch_start + batch_size, n_traits)
        batch_traits = trait_cols[batch_start:batch_end]

        # Calculate adaptive figsize for this batch
        n_traits_in_batch = len(batch_traits)
        n_rows = (n_traits_in_batch + n_cols - 1) // n_cols

        # Scale figsize proportionally for partial batches
        if n_traits_in_batch < batch_size:
            # Calculate full batch dimensions
            full_n_rows = (batch_size + n_cols - 1) // n_cols
            # Scale height proportionally
            batch_figsize = (figsize[0], figsize[1] * (n_rows / full_n_rows))
        else:
            batch_figsize = figsize

        # Create figure for this batch
        fig = create_trait_histograms(df, batch_traits, n_cols=n_cols, figsize=batch_figsize)
        fig.suptitle(
            f"Trait Histograms (Traits {batch_start+1}-{batch_end} of {n_traits})",
            fontsize=14,
            y=0.995,
        )
        figures.append(fig)

    return figures
```

**`create_trait_boxplots_by_genotype_batched`**: Apply same logic

### 5. Add Batch Plot Configuration

**File**: `src/sleap_roots_analyze/pipeline/config/components.py`

**Add to VisualizationConfig**:
```python
@dataclass
class VisualizationConfig:
    """Visualization generation configuration."""

    # ... existing fields ...

    # Batched plot configuration
    enable_batched_plots: bool = True
    batched_plot_threshold: int = 16  # Create batches when > this many traits
    batch_size: int = 16  # Traits per batch figure
```

### 6. Update Step 4 to Use Batch Configuration

**File**: `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py`

**Current code (line 128)**:
```python
# 4. Add batched trait visualizations if comprehensive mode or many traits
# Generate batched plots if we have more than 16 traits
if len(trait_cols) > 16:
```

**Fix**: Use config parameters
```python
# 4. Add batched trait visualizations if enabled and threshold exceeded
if (
    config.visualization.enable_batched_plots
    and len(trait_cols) > config.visualization.batched_plot_threshold
):
    # Batched histograms
    hist_figs = create_trait_histograms_batched(
        df=df,
        trait_cols=trait_cols,
        batch_size=config.visualization.batch_size,
    )
    for i, fig in enumerate(hist_figs):
        all_figures[f"04_trait_histograms_batch_{i+1}"] = fig

    # Batched boxplots by genotype
    box_figs = create_trait_boxplots_by_genotype_batched(
        df=df,
        trait_cols=trait_cols,
        genotype_col=config.columns.genotype,
        batch_size=config.visualization.batch_size,
    )
    for i, fig in enumerate(box_figs):
        all_figures[f"04_trait_boxplots_batch_{i+1}"] = fig
```

### 7. Update Example Config

**File**: `configs/qc_turface_150genotypes.yaml`

**Add to visualization section**:
```yaml
visualization:
  dpi: 100
  figsize: [12, 8]

  # ... other fields ...

  # Batched plot configuration
  enable_batched_plots: true      # Set to false to disable batch plots
  batched_plot_threshold: 16      # Create batches when > this many traits
  batch_size: 16                  # Traits per batch figure
```

## Impact

### Files Modified: 4 files
1. `src/sleap_roots_analyze/visualization.py` - Fix variance decomposition, batch sizing
2. `src/sleap_roots_analyze/pipeline/steps/filter_heritability.py` - Pass threshold
3. `src/sleap_roots_analyze/pipeline/steps/exploratory_analysis.py` - Use batch config
4. `src/sleap_roots_analyze/pipeline/config/components.py` - Add batch config fields
5. `configs/qc_turface_150genotypes.yaml` - Document new options

### Backwards Compatibility
**YES** - All changes are backwards compatible:
- New threshold parameter has sensible default (0.3)
- Batch config fields have defaults matching current hardcoded behavior
- Existing configs work without modification

### Test Coverage
Need to update/add tests:
- `test_visualization.py` - Test variance decomposition with custom threshold
- `test_visualization.py` - Test batched plots with varying batch sizes
- `test_step_exploratory_analysis.py` - Test batch config parameters
- `test_step_filter_heritability.py` - Test threshold passing

### Benefits:

1. **Consistent x-axis rendering** - All 4 variance decomposition panels use same approach
2. **Threshold consistency** - Heritability threshold from config appears in all plots
3. **Professional last batches** - Partial batches sized appropriately, no stretched subplots
4. **User control** - Can disable batching or adjust batch size per workflow
5. **Better for diverse datasets** - Works well with 10 traits or 100 traits

### Risks:

- **Low risk** - Mostly parameter additions and proportional sizing
- **Testing needed** - Edge cases (batch_size=1, very small batches, etc.)
- **Visual changes** - Variance decomposition panels will look slightly different

## Implementation Checklist

- [ ] Add `threshold` parameter to `create_variance_decomposition_plot()`
- [ ] Fix variance decomposition Panels 1-3 to use numeric x positions
- [ ] Pass `config.heritability.threshold` in Step 9
- [ ] Add batch config fields to `VisualizationConfig`
- [ ] Fix `create_trait_histograms_batched()` last batch sizing
- [ ] Fix `create_trait_boxplots_by_genotype_batched()` last batch sizing
- [ ] Update Step 4 to use batch config parameters
- [ ] Update example config with batch settings
- [ ] Add/update tests for threshold parameter
- [ ] Add/update tests for batch sizing
- [ ] Run full test suite
- [ ] Test with real data (150 genotypes, varying trait counts)

## Timeline

**Estimated**: 4-6 hours total
- Variance decomposition fixes: 1.5 hours
- Batch sizing fixes: 1.5 hours
- Config parameter additions: 1 hour
- Testing: 2 hours

## Related

- **Related to**: fix-adaptive-sizing-genotype-plots (completed)
- **Fixes**: User-reported variance decomposition axis issues
- **Fixes**: User-reported last batch subplot stretching
- **Enhances**: User control over batch plot generation
