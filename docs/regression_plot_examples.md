# Regression Plot Examples

This document provides example code for using `create_regression_plot()` for publication figures.

## Basic Usage

### Simple Regression Plot

```python
from sleap_roots_analyze import create_regression_plot, load_trait_data, save_figure_with_unique_name
from pathlib import Path

# Load your trait data
df_traits = load_trait_data("path/to/cleaned_traits.csv")

# Create regression plot
fig = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    title='Root Biomass vs Surface Area'
)

# Display in Jupyter
fig.show()

# Save for publication
save_figure_with_unique_name(fig, Path("publication_figures"), "regression_biomass_surface")
# Or save directly
fig.savefig('regression_biomass_surface.png', dpi=300, bbox_inches='tight')
```

### Regression with Genotype Coloring

```python
# Color points by genotype while fitting single regression line
fig = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)',
    color_by='Genotype',
    title='Root vs Shoot Biomass',
    figsize=(10, 8)
)

fig.savefig('regression_root_shoot_by_genotype.png', dpi=300, bbox_inches='tight')
```

## Complete Notebook Example

```python
# ==============================================
# IMPORTS
# ==============================================
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sleap_roots_analyze import (
    create_regression_plot,
    load_trait_data,
    get_trait_columns,
    save_figure_with_unique_name,
)

# ==============================================
# CONFIGURATION
# ==============================================
CLEANED_DATA_PATH = "C:/repos/runs/run_20251105_120437/cleaned_traits.csv"
PUBLICATION_DIR = Path("publication_figures")
PUBLICATION_DIR.mkdir(exist_ok=True)

BARCODE_COL = "Barcode"
GENOTYPE_COL = "Genotype"
REPLICATE_COL = "Replicate"

# ==============================================
# LOAD DATA
# ==============================================
df_traits = load_trait_data(
    csv_path=CLEANED_DATA_PATH,
    barcode_col=BARCODE_COL,
    genotype_col=GENOTYPE_COL,
    replicate_col=REPLICATE_COL,
)

trait_cols = get_trait_columns(
    df=df_traits,
    barcode_col=BARCODE_COL,
    genotype_col=GENOTYPE_COL,
    replicate_col=REPLICATE_COL,
)

print(f"Loaded {len(df_traits)} samples with {len(trait_cols)} traits")

# ==============================================
# REGRESSION PLOTS FOR PAPER
# ==============================================

# Figure 1: Root Biomass vs Surface Area
print("Creating Figure 1: Root Biomass vs Surface Area")
fig1 = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    title='Root Biomass vs Surface Area',
    figsize=(8, 8)
)
fig1.savefig(PUBLICATION_DIR / 'Figure1_biomass_surface_regression.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved to {PUBLICATION_DIR / 'Figure1_biomass_surface_regression.png'}")
plt.show()

# Figure 2: Root vs Shoot Biomass
print("\\nCreating Figure 2: Root vs Shoot Biomass")
fig2 = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)',
    title='Root vs Shoot Biomass',
    figsize=(8, 8)
)
fig2.savefig(PUBLICATION_DIR / 'Figure2_root_shoot_regression.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved to {PUBLICATION_DIR / 'Figure2_root_shoot_regression.png'}")
plt.show()

# Figure 3 (Supplemental): Root vs Shoot with Genotype Coloring
print("\\nCreating Figure S1: Root vs Shoot by Genotype")
fig3 = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)',
    color_by=GENOTYPE_COL,
    title='Root vs Shoot Biomass (by Genotype)',
    figsize=(10, 8)
)
fig3.savefig(PUBLICATION_DIR / 'FigureS1_root_shoot_by_genotype.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved to {PUBLICATION_DIR / 'FigureS1_root_shoot_by_genotype.png'}")
plt.show()

print("\\n✅ All regression figures generated successfully!")
```

## Custom Styling

### Custom Point and Line Appearance

```python
# Custom scatter point styling
fig = create_regression_plot(
    df_traits,
    x_col='Total Root Length (mm)',
    y_col='Root Biomass (mg)',
    scatter_kws={'s': 80, 'alpha': 0.5},  # Larger, more transparent points
    line_kws={'color': 'red', 'linewidth': 3}  # Red, thicker line
)
```

### Larger Figure for Presentations

```python
# Presentation-ready figure (larger text, bigger points)
fig = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    figsize=(12, 10),
    scatter_kws={'s': 100, 'alpha': 0.6}
)

# Increase font sizes for presentation
ax = fig.axes[0]
ax.set_xlabel(ax.get_xlabel(), fontsize=18)
ax.set_ylabel(ax.get_ylabel(), fontsize=18)
ax.set_title(ax.get_title(), fontsize=20)
ax.tick_params(labelsize=14)

fig.savefig('presentation_regression.png', dpi=150, bbox_inches='tight')
```

## Tips for Publication Figures

1. **Always save at high DPI** (300 for print, 150 for web)
2. **Use `bbox_inches='tight'`** to avoid cutting off labels
3. **Check color-blind friendliness** if using color_by
4. **Verify statistical annotations are readable** at final size
5. **Consider figure dimensions** - most journals prefer square or near-square

## Statistical Annotations

The function automatically includes:
- **Pearson R**: Correlation coefficient (-1 to 1)
- **R²**: Coefficient of determination (0 to 1, proportion of variance explained)
- **p-value**: Statistical significance (p < 0.001 shown for very small p)
- **Regression equation**: y = mx + b format
- **Sample size (n)**: Number of valid samples used (after NaN removal)

These annotations appear in a text box positioned to avoid data overlap.
