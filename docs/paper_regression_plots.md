# Paper Regression Plots - Quick Start

This is ready-to-use code for creating the regression plots for your paper.

## Notebook Cell - Copy & Paste Ready

```python
# ==============================================
# LINEAR REGRESSION PLOTS FOR PAPER
# ==============================================

from sleap_roots_analyze import create_regression_plot
import matplotlib.pyplot as plt

print("Creating regression plots for paper...")

# Figure 1: Root Biomass vs Surface Area
print("\n📊 Creating Figure 1: Root Biomass vs Surface Area")
fig1 = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    title='Root Biomass vs Surface Area',
    figsize=(8, 8)
)

# Save for publication
fig1.savefig(PUBLICATION_DIR / 'Figure_Regression_Biomass_Surface.png',
             dpi=300, bbox_inches='tight')
print(f"✅ Saved: Figure_Regression_Biomass_Surface.png")
plt.show()

# Figure 2: Root vs Shoot Biomass
print("\n📊 Creating Figure 2: Root vs Shoot Biomass")
fig2 = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)',
    title='Root vs Shoot Biomass',
    figsize=(8, 8)
)

# Save for publication
fig2.savefig(PUBLICATION_DIR / 'Figure_Regression_Root_Shoot.png',
             dpi=300, bbox_inches='tight')
print(f"✅ Saved: Figure_Regression_Root_Shoot.png")
plt.show()

print("\n✨ All regression figures created successfully!")
print(f"📁 Figures saved to: {PUBLICATION_DIR}")
```

## What You Get

Each plot automatically includes:
- ✅ **Scatter points** for all samples
- ✅ **Linear regression line** (least squares fit)
- ✅ **95% confidence interval** (shaded region)
- ✅ **Statistical annotations**:
  - Pearson R (correlation coefficient)
  - R² (proportion of variance explained)
  - p-value (statistical significance)
  - Regression equation (y = mx + b)
  - Sample size (n)
- ✅ **Publication-ready quality** (300 DPI, proper sizing)

## Optional: Add to Existing Notebook

If you already have `trait_viz_turface_20251105.ipynb` open, just add this cell at the end:

```python
## Linear Regression Analysis

from sleap_roots_analyze import create_regression_plot

# Root Biomass vs Surface Area
fig_reg1 = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)'
)
save_figure_with_unique_name(fig_reg1, PUBLICATION_DIR, "regression_biomass_surface")

# Root vs Shoot Biomass
fig_reg2 = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)'
)
save_figure_with_unique_name(fig_reg2, PUBLICATION_DIR, "regression_root_shoot")
```

## Customization Options

### Larger figure for presentation
```python
fig = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    figsize=(12, 10)  # Larger for slides
)
```

### Color by genotype
```python
fig = create_regression_plot(
    df_traits,
    x_col='Shoot Biomass (mg)',
    y_col='Root Biomass (mg)',
    color_by='Genotype',  # Color points by group
    title='Root vs Shoot by Genotype'
)
```

### Custom styling
```python
fig = create_regression_plot(
    df_traits,
    x_col='Surface Area (mm²)',
    y_col='Root Biomass (mg)',
    scatter_kws={'s': 80, 'alpha': 0.5},  # Bigger, more transparent points
    line_kws={'color': 'red', 'linewidth': 3}  # Red regression line
)
```

That's it! The function handles all the statistical calculations and formatting automatically.
