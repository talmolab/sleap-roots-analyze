"""Visualization functions for depth profile data from root core experiments.

This module provides plotting utilities for visualizing root count vs depth data,
including faceted plots showing mean trends and spaghetti plots showing individual
biological replicates.

Typical workflow:
    1. Process data using root_core_analysis module
    2. Create faceted mean±error plots by genotype
    3. Create spaghetti plots showing replicate variability

Example:
    >>> import pandas as pd
    >>> from sleap_roots_analyze import (
    ...     melt_depth_data,
    ...     aggregate_by_replicate
    ... )
    >>> from sleap_roots_analyze.depth_profile_plots import (
    ...     plot_depth_profile_faceted,
    ...     plot_depth_profile_replicates
    ... )
    >>>
    >>> # After processing data (melting and aggregating)
    >>> df_agg = aggregate_by_replicate(
    ...     df_melted,
    ...     group_cols=['plot_rep', 'geno', 'Depth_cm'],
    ...     value_col='Root_Count'
    ... )
    >>>
    >>> # Create faceted mean plot
    >>> fig1 = plot_depth_profile_faceted(
    ...     df_agg,
    ...     x='Depth_cm',
    ...     y='Root_Count',
    ...     facet_col='geno',
    ...     errorbar='se',
    ...     output_path='depth_profile_mean.png'
    ... )
    >>>
    >>> # Create replicate spaghetti plot
    >>> fig2 = plot_depth_profile_replicates(
    ...     df_agg,
    ...     x='Depth_cm',
    ...     y='Root_Count',
    ...     facet_col='geno',
    ...     hue='plot_rep',
    ...     output_path='depth_profile_reps.png'
    ... )
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional


def plot_depth_profile_faceted(
    df: pd.DataFrame,
    x: str = "Depth_cm",
    y: str = "Root_Count",
    facet_col: str = "geno",
    errorbar: Optional[str] = "se",
    col_wrap: int = 4,
    height: int = 4,
    output_path: Optional[Path | str] = None,
    **kwargs,
) -> plt.Figure:
    """Create faceted line plots showing mean root count vs depth by genotype.

    Creates a grid of subplots, one per value in facet_col, with lines showing
    mean values and optional error bars (standard error, standard deviation, or
    confidence intervals).

    Args:
        df: DataFrame with aggregated depth profile data
        x: Column name for x-axis (depth in cm), default: 'Depth_cm'
        y: Column name for y-axis (root count), default: 'Root_Count'
        facet_col: Column to facet by (creates one subplot per unique value),
            default: 'geno'
        errorbar: Error bar type - 'se' (standard error), 'sd' (standard deviation),
            'ci' (confidence interval), or None (no error bars), default: 'se'
        col_wrap: Number of columns in facet grid, default: 4
        height: Height of each subplot in inches, default: 4
        output_path: Optional path to save figure. If provided, saves to file
        **kwargs: Additional arguments passed to sns.lineplot (e.g., lw, color)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_depth_profile_faceted(
        ...     df_agg,
        ...     facet_col='geno',
        ...     errorbar='se',
        ...     col_wrap=3,
        ...     height=5,
        ...     output_path='depth_profiles.png'
        ... )
    """
    # Set seaborn style
    sns.set_theme(style="dark")
    sns.set_style("darkgrid")

    # Create FacetGrid
    g = sns.FacetGrid(
        df,
        col=facet_col,
        col_wrap=col_wrap,
        height=height,
        sharex=False,
        sharey=True,
    )

    # Map lineplot with error bars
    # Set default lw if not provided in kwargs
    if "lw" not in kwargs and "linewidth" not in kwargs:
        kwargs["lw"] = 2
    g.map_dataframe(sns.lineplot, x=x, y=y, errorbar=errorbar, **kwargs)

    # Set axis labels
    g.set_axis_labels(
        f"{x.replace('_', ' ').title()}", f"{y.replace('_', ' ').title()}"
    )

    # Set titles
    g.set_titles("{col_name}")

    # Add grid and rotate x-axis labels
    for ax in g.axes.flat:
        ax.grid(True)
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=300)

    return g.fig


def plot_depth_profile_replicates(
    df: pd.DataFrame,
    x: str = "Depth_cm",
    y: str = "Root_Count",
    facet_col: str = "geno",
    hue: str = "plot_rep",
    col_wrap: int = 4,
    height: int = 4,
    alpha: float = 0.6,
    output_path: Optional[Path | str] = None,
    **kwargs,
) -> plt.Figure:
    """Create spaghetti plots showing individual biological replicate depth profiles.

    Creates a grid of subplots with individual lines for each replicate, useful for
    visualizing variability within genotypes.

    Args:
        df: DataFrame with replicate-level depth profile data
        x: Column name for x-axis (depth in cm), default: 'Depth_cm'
        y: Column name for y-axis (root count), default: 'Root_Count'
        facet_col: Column to facet by, default: 'geno'
        hue: Column to color lines by (typically plot_rep), default: 'plot_rep'
        col_wrap: Number of columns in facet grid, default: 4
        height: Height of each subplot in inches, default: 4
        alpha: Transparency of lines (0-1), default: 0.6 for overlapping visibility
        output_path: Optional path to save figure
        **kwargs: Additional arguments passed to sns.lineplot (e.g., lw)

    Returns:
        matplotlib Figure object

    Example:
        >>> fig = plot_depth_profile_replicates(
        ...     df_agg,
        ...     facet_col='geno',
        ...     hue='plot_rep',
        ...     alpha=0.5,
        ...     output_path='depth_profiles_reps.png'
        ... )
    """
    # Set seaborn style
    sns.set_theme(style="dark")
    sns.set_style("darkgrid")

    # Create FacetGrid
    g2 = sns.FacetGrid(
        df, col=facet_col, col_wrap=col_wrap, height=height, sharex=False, sharey=True
    )

    # Map lineplot with individual lines (no aggregation)
    g2.map_dataframe(
        sns.lineplot,
        x=x,
        y=y,
        hue=hue,
        estimator=None,  # Show individual lines, no aggregation
        lw=1.5,
        alpha=alpha,
        legend=False,  # Suppress legend (too many replicates)
        **kwargs,
    )

    # Set axis labels
    g2.set_axis_labels(
        f"{x.replace('_', ' ').title()}", f"{y.replace('_', ' ')} (Replicates)"
    )

    # Set titles
    g2.set_titles("{col_name}")

    # Add grid and rotate x-axis labels
    for ax in g2.axes.flat:
        ax.grid(True)
        for label in ax.get_xticklabels():
            label.set_rotation(90)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        plt.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=300)

    return g2.fig
