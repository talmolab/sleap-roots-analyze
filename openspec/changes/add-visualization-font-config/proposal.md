# Add Configurable Font Sizes and Publication Parameters to Visualization Config

**Status**: ✅ COMPLETED
**Created**: 2025-11-29
**Completed**: 2025-12-01
**Author**: AI Assistant (Claude)
**Type**: Enhancement

## Completion Summary

**All font configuration parameters have been implemented:**
- ✅ `title_fontsize` - Font size for plot titles (default: 14)
- ✅ `label_fontsize` - Font size for axis labels (default: 12)
- ✅ `tick_fontsize` - Font size for tick labels (default: 10)
- ✅ `legend_fontsize` - Font size for legend text (default: 10)
- ✅ `figure_format` - Output format (png, pdf, svg, eps)
- ✅ `bbox_inches` - Bounding box for saved figures
- ✅ `transparent` - Transparency for saved figures

**Implementation locations:**
- Config dataclasses: `src/sleap_roots_analyze/pipeline/config/components.py` (lines 403-406, 491-494)
- Both QC and Viz pipelines support font configuration
- Config files use parameters: `configs/qc_turface_150genotypes.yaml`

## Why

Currently, the `VisualizationConfig` dataclass only allows configuration of a limited set of parameters (DPI, figsize, plot types). However, for publication-quality figures, users need fine-grained control over:

1. **Font sizes** - Different journals require specific font sizes for titles, labels, ticks, and legends
2. **Figure formats** - Publications often require vector formats (PDF, SVG, EPS) instead of PNG
3. **Savefig parameters** - Control over bbox_inches, facecolor, edgecolor, and transparency

Currently these parameters are **hardcoded** throughout the codebase:
- All font sizes are hardcoded in individual plotting functions (title=14, labels=12, ticks=10, legend=10)
- All pipeline steps save as PNG only
- All savefig calls use hardcoded `bbox_inches="tight"`
- No control over transparency, facecolor, or edgecolor

This creates several problems:
- **Inconsistent configuration**: DPI is configurable but format is not
- **Poor user experience**: Users must edit source code to change font sizes
- **Publication blockers**: Cannot generate vector formats for journals without code changes
- **Reproducibility**: Font sizes and formats are implicit defaults, not explicit config

As the user stated: "i want everything explicitly set in the config. defaults are confusing"

## What

Add comprehensive publication-quality parameters to `VisualizationConfig`:

### Font Size Parameters
```python
title_fontsize: int = 14      # Main title font size
label_fontsize: int = 12      # Axis label font size  
tick_fontsize: int = 10       # Tick label font size
legend_fontsize: int = 10     # Legend font size
```

### Figure Format Parameters
```python
figure_format: str = "png"           # Output format: png, pdf, svg, eps
figure_formats: list[str] = ["png"]  # Multiple formats (for GenerateStaticFigures)
```

### Savefig Parameters
```python
bbox_inches: str = "tight"           # Bounding box mode ("tight" or None)
facecolor: Optional[str] = None      # Figure face color (None = transparent)
edgecolor: Optional[str] = None      # Figure edge color
transparent: bool = False            # Save with transparent background
```

### Apply Configuration Consistently

Update all plotting functions and pipeline steps to use these configured values instead of hardcoded constants:

**Pipeline Steps to Update:**
- `exploratory_analysis.py` - Apply font config to all EDA plots
- `visualize_outliers.py` - Apply to outlier detection plots
- `filter_heritability.py` - Apply to heritability threshold plot
- `generate_static_figures.py` - Apply format and savefig parameters

**Visualization Modules to Update:**
- `visualization.py` - ~50 hardcoded fontsize occurrences
- `outlier_visualization.py` - ~30 hardcoded fontsize occurrences
- `depth_profile_plots.py` - 2 hardcoded savefig calls

## Impact

### Benefits
1. **Full explicit control**: All visualization parameters in YAML config
2. **Publication-ready**: Generate PDF/SVG formats without code changes
3. **Journal compliance**: Adjust font sizes to meet specific requirements
4. **Better UX**: Consistent configuration interface (all via YAML)
5. **Reproducibility**: All parameters documented in config file
6. **Backward compatible**: All parameters have sensible defaults

### Breaking Changes
**None** - This is a backward-compatible enhancement. All new parameters have defaults matching current hardcoded values.

### Migration Path
Existing configs work unchanged. Users can optionally add new parameters:

```yaml
# Before (still works)
visualization:
  dpi: 100
  figsize: [10, 8]

# After (with new parameters)
visualization:
  dpi: 100
  figsize: [10, 8]
  title_fontsize: 14
  label_fontsize: 12
  tick_fontsize: 10
  legend_fontsize: 10
  figure_format: "pdf"
  bbox_inches: "tight"
  transparent: false
```

### Effort Estimate
- **Config changes**: 1-2 hours (add fields to VisualizationConfig)
- **Pipeline updates**: 2-3 hours (4 pipeline steps)
- **Visualization updates**: 4-5 hours (3 modules, ~80 hardcoded values)
- **Testing**: 2-3 hours (update existing tests, add new tests)
- **Documentation**: 1 hour (update YAML examples)

**Total**: ~10-15 hours

### Testing Strategy
1. Update existing tests to verify defaults match old hardcoded values
2. Add parametric tests for each new config parameter
3. Test multiple format generation (PNG + PDF + SVG)
4. Verify backward compatibility (old configs still work)
5. Integration test with Turface QC config

## Dependencies

- Depends on: None
- Blocks: None
- Related: Turface QC pipeline replication (wants explicit config)

## Alternatives Considered

1. **Keep hardcoded values** - Rejected: Poor UX, not publication-ready
2. **Add only font sizes** - Rejected: Incomplete, format control also needed
3. **Separate FontConfig dataclass** - Rejected: Over-engineering for 4 fields
4. **Global matplotlib rcParams** - Rejected: Not thread-safe, affects all plots

## References

- Turface QC notebook parameters: TITLE_FONTSIZE=14, LABEL_FONTSIZE=12, etc.
- Current VisualizationConfig: `src/sleap_roots_analyze/pipeline/config/components.py`
- Hardcoded values: `visualization.py:86,212,256,743,812,813,836,837,1050,1101,1292,1293,1296,1304,1768,1777,1901,1905,1906,1918,1934,2031,2033,2034,2037,2039,2078,2079,2080,2186,2199,2300,2320,2815,2816,2838,2839,2967,3021,3243,3244,3247,3249`
- Matplotlib savefig docs: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
