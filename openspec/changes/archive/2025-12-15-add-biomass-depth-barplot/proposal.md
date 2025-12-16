# Add Biomass Depth Barplot Visualization

## Why

Root core experiments measure biomass at multiple soil depths (e.g., 0-30cm, 30-60cm). Currently, the QC pipeline generates rotated line plots showing mean ± SE depth profiles, but lacks a grouped barplot comparing biomass across depth intervals by genotype. The biomass barplot provides a complementary view that:

1. **Better highlights depth-specific differences**: Bar heights make magnitude comparisons across depths more intuitive than line slopes
2. **Matches established analysis workflows**: The reference notebook (`second_step_root_biomass_plots_anovas_20240729.v000.ipynb`) demonstrates this is a critical visualization for biomass data
3. **Enables depth-layer comparisons**: Grouped bars (one per depth interval) make it easy to compare shallow vs deep rooting patterns across genotypes

This visualization is essential for biomass data but NOT needed for counting data (which uses continuous depth measurements in 5cm increments, making line plots more appropriate).

## What Changes

- **Add biomass-specific barplot function** to `depth_profile_plots.py`
  - Function: `plot_biomass_depth_barplot(df, depth_col, value_col, facet_col, output_path)`
  - Grouped barplot with depths on x-axis, one bar group per genotype
  - Supports optional stripplot overlay for individual plot-level data points
  - Standard error bars for each bar
  - Follows existing style: seaborn FacetGrid, rotated x-labels, dark grid theme

- **Integrate into VisualizeDepthProfilesStep (Step 00f)**
  - Only generate barplot for biomass data sources (check `data_type == "biomass"`)
  - Output: `00f_depth_profile_biomass_barplot.png`
  - Skip for counting data (continues using only line plots)

- **Update configuration schema**
  - No new config needed - uses existing `root_core.sources` to detect biomass vs counting

### Reference Implementation

From `second_step_root_biomass_plots_anovas_20240729.v000.ipynb` (cell 6):

```python
# Transform to long format for grouped barplot
df_melted = pd.melt(
    df_merged,
    id_vars=['salk_geno', 'Plot'],
    value_vars=['mean_0_30', 'mean_30_60'],
    var_name='Depth',
    value_name='Mean_Biomass'
)
df_melted['Depth'] = df_melted['Depth'].replace({
    'mean_0_30': '0-30cm',
    'mean_30_60': '30-60cm'
})

# Grouped barplot with stripplot overlay
sns.barplot(
    x='salk_geno',
    y='Mean_Biomass',
    hue='Depth',
    data=df_melted,
    dodge=True,
    order=custom_order  # Control first, then sorted by mean
)
sns.stripplot(
    x='salk_geno',
    y='Mean_Biomass',
    hue='Depth',
    data=df_melted,
    dodge=True,
    marker='o',
    alpha=0.5,
    jitter=True,
    palette='dark:k',
    order=custom_order
)
```

## Impact

**Affected specs:**
- `depth-profile-visualization` (NEW) - Add requirements for biomass barplot generation

**Affected code:**
- `src/sleap_roots_analyze/depth_profile_plots.py` - Add `plot_biomass_depth_barplot()` function
- `src/sleap_roots_analyze/pipeline/steps/visualize_depth_profiles.py` - Integrate barplot generation for biomass sources
- `tests/test_depth_profile_plots.py` (NEW) - Unit tests for barplot function

**Non-breaking:**
- Existing depth profile line plots unchanged
- New biomass barplot is additive (only generated for biomass sources)
- No API changes to existing functions