# Visualization Configuration Design

## VisualizationConfig Schema

```python
@dataclass
class VisualizationConfig:
    """Visualization generation configuration.
    
    Controls all aspects of plot generation including which plots to create,
    appearance parameters, and output formats.
    """
    
    # Plot type flags (existing)
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    create_outlier_plots: bool = True
    interactive: bool = False
    
    # Figure size and resolution (existing)
    dpi: int = 300
    figsize: tuple[int, int] = (10, 8)
    
    # Font sizes (NEW)
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    
    # Figure format (NEW)
    figure_format: str = "png"  # Primary format for pipeline steps
    figure_formats: list[str] = field(default_factory=lambda: ["png"])  # For batch generation
    
    # Savefig parameters (NEW)
    bbox_inches: Optional[str] = "tight"  # "tight" or None
    facecolor: Optional[str] = None  # None = transparent
    edgecolor: Optional[str] = None  # None = no edge
    transparent: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate font sizes are positive
        for name in ['title_fontsize', 'label_fontsize', 'tick_fontsize', 'legend_fontsize']:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        
        # Validate figure format
        valid_formats = {'png', 'pdf', 'svg', 'eps'}
        if self.figure_format not in valid_formats:
            raise ValueError(
                f"figure_format must be one of {valid_formats}, got {self.figure_format}"
            )
        
        # Validate figure_formats list
        for fmt in self.figure_formats:
            if fmt not in valid_formats:
                raise ValueError(
                    f"All figure_formats must be in {valid_formats}, got {fmt}"
                )
        
        # Validate bbox_inches
        if self.bbox_inches not in [None, "tight"]:
            raise ValueError(
                f"bbox_inches must be 'tight' or None, got {self.bbox_inches}"
            )
        
        # Validate DPI is positive
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")
```

## YAML Configuration Example

```yaml
visualization:
  # Plot types to generate
  create_pca_plots: true
  create_umap_plots: false
  create_cluster_plots: false
  create_outlier_plots: true
  interactive: false
  
  # Figure size and resolution
  dpi: 300
  figsize: [10, 8]
  
  # Font sizes (NEW)
  title_fontsize: 14
  label_fontsize: 12
  tick_fontsize: 10
  legend_fontsize: 10
  
  # Figure format (NEW)
  figure_format: "pdf"
  
  # Savefig parameters (NEW)
  bbox_inches: "tight"
  transparent: false
```

## Function Signature Changes

All visualization functions will accept optional font size parameters:

```python
def create_trait_histograms_batched(
    df: pd.DataFrame,
    trait_cols: list,
    batch_size: int = 9,
    figsize: tuple = (15, 10),
    title_fontsize: int = 14,  # NEW
    label_fontsize: int = 12,  # NEW
    tick_fontsize: int = 10,   # NEW
) -> dict:
    """Create batched histogram plots."""
    # Use configured font sizes instead of hardcoded values
    axes[i].set_title(f"{trait}\n(n={len(data)})", fontsize=tick_fontsize)
    fig.suptitle(
        f"Trait Histograms (Traits {batch_start+1}-{batch_end} of {n_traits})",
        fontsize=title_fontsize,
        y=0.995,
    )
    # ...
```

## Pipeline Step Integration

Pipeline steps will pass config parameters to visualization functions:

```python
# exploratory_analysis.py
def execute(self, data, config, run_dir, prev_result):
    # ...
    figs = create_trait_histograms_batched(
        df,
        trait_cols,
        batch_size=9,
        title_fontsize=config.visualization.title_fontsize,
        label_fontsize=config.visualization.label_fontsize,
        tick_fontsize=config.visualization.tick_fontsize,
    )
    
    # Save with configured format and parameters
    for fig_name, fig in figs.items():
        fig_path = figures_dir / f"{fig_name}.{config.visualization.figure_format}"
        fig.savefig(
            fig_path,
            dpi=config.visualization.dpi,
            bbox_inches=config.visualization.bbox_inches,
            facecolor=config.visualization.facecolor,
            edgecolor=config.visualization.edgecolor,
            transparent=config.visualization.transparent,
        )
```

## Configuration Validation

All validation happens in `VisualizationConfig.__post_init__()`:

1. Font sizes > 0
2. Format in {'png', 'pdf', 'svg', 'eps'}
3. bbox_inches in {None, "tight"}
4. DPI > 0

## Test Coverage

Required tests:

1. **Unit tests** for VisualizationConfig validation
2. **Integration tests** verifying configured values are applied
3. **Backward compatibility tests** ensuring old configs still work
4. **Format tests** verifying each supported format generates valid files
5. **Font tests** verifying font sizes are actually applied in plots

## Acceptance Criteria

- [ ] All hardcoded font sizes removed from codebase
- [ ] All pipeline steps use configured font sizes
- [ ] Can generate figures in PNG, PDF, SVG, EPS formats
- [ ] Existing configs work without modification
- [ ] All 1006 existing tests pass
- [ ] New tests cover all new parameters
- [ ] Documentation updated with examples
- [ ] Turface QC config runs with explicit font/format config

## Migration Guide

### For Users

**No action required for existing configs.** Optionally, add new parameters:

```yaml
# Add font customization
visualization:
  title_fontsize: 16  # Larger titles
  label_fontsize: 14  # Larger labels
  
# Add format control  
visualization:
  figure_format: "pdf"  # Vector format
```

### For Developers

**When creating new visualization functions:**

1. Add optional font size parameters with defaults
2. Use configured values instead of hardcoded numbers
3. Accept format/savefig parameters if saving directly

```python
def create_new_plot(
    data,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
):
    ax.set_title("My Plot", fontsize=title_fontsize)
    ax.set_xlabel("X", fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
```

## References

- Matplotlib savefig documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
- Current VisualizationConfig: `src/sleap_roots_analyze/pipeline/config/components.py:52-60`
- Example configs: `configs/qc_turface_150genotypes.yaml`
