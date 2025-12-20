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
from PIL import Image


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
    # Save current style and set whitegrid for this plot only
    original_style = plt.rcParams.copy()
    sns.set_style("whitegrid")

    try:
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

        # Save and rotate if output path provided
        if output_path:
            output_path = Path(output_path)
            temp_path = output_path.with_suffix(".temp.png")

            # Save the plot to temporary file
            g.fig.savefig(temp_path, bbox_inches="tight", facecolor="white", dpi=300)

            # Close matplotlib to release the file
            plt.close(g.fig)

            # Rotate the image 90 degrees clockwise to make depth intuitive (going down)
            with Image.open(temp_path) as img:
                rotated_img = img.rotate(-90, expand=True)
                rotated_img.save(output_path)

            # Clean up temp file
            temp_path.unlink()

            return None  # Figure is closed, can't return it

        return g.fig
    finally:
        # Restore original matplotlib style to avoid polluting global state
        plt.rcParams.update(original_style)


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
    # Save current style and set whitegrid for this plot only
    original_style = plt.rcParams.copy()
    sns.set_style("whitegrid")

    try:
        # Create FacetGrid
        g2 = sns.FacetGrid(
            df,
            col=facet_col,
            col_wrap=col_wrap,
            height=height,
            sharex=False,
            sharey=True,
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

        # Save and rotate if output path provided
        if output_path:
            output_path = Path(output_path)
            temp_path = output_path.with_suffix(".temp.png")

            # Save the plot to temporary file
            g2.fig.savefig(temp_path, bbox_inches="tight", facecolor="white", dpi=300)

            # Close matplotlib to release the file
            plt.close(g2.fig)

            # Rotate the image 90 degrees clockwise to make depth intuitive (going down)
            with Image.open(temp_path) as img:
                rotated_img = img.rotate(-90, expand=True)
                rotated_img.save(output_path)

            # Clean up temp file
            temp_path.unlink()

            return None  # Figure is closed, can't return it

        return g2.fig
    finally:
        # Restore original matplotlib style to avoid polluting global state
        plt.rcParams.update(original_style)


def plot_biomass_depth_barplot(
    df: pd.DataFrame,
    depth_col: str = "Depth_cm",
    value_col: str = "Root_DW_g",
    genotype_col: str = "geno",
    include_points: bool = False,
    estimator: str = "median",
    output_path: Optional[Path | str] = None,
    **kwargs,
) -> Optional[plt.Figure]:
    """Create grouped barplot showing mean biomass at discrete depth intervals by genotype.

    This function generates a barplot visualization for biomass depth profile data, where
    each genotype has grouped bars representing different depth intervals (e.g., 0-30cm,
    30-60cm). The function automatically orders genotypes with Control first (if present),
    followed by ascending order based on mean biomass in the shallowest depth interval.

    Args:
        df: DataFrame with biomass depth profile data containing columns for genotype,
            depth, and biomass values.
        depth_col: Column name for depth values in cm, default: 'Depth_cm'.
        value_col: Column name for biomass values (e.g., Root_DW_g), default: 'Root_DW_g'.
        genotype_col: Column name for genotype identifiers, default: 'geno'.
        include_points: If True, overlay individual plot-level data points as stripplot
            with jitter and transparency, default: False.
        estimator: Statistical estimator for aggregating data, default: 'median'.
            Options: 'mean', 'median'. Median is more robust to outliers.
        output_path: Optional path to save figure. If provided, figure is saved and
            function returns None (figure is closed).
        **kwargs: Additional arguments passed to sns.barplot (e.g., palette, capsize).

    Returns:
        matplotlib Figure object if output_path is None, otherwise None (figure saved
        and closed).

    Raises:
        ValueError: If DataFrame is empty or missing required columns.

    Example:
        >>> # After aggregating biomass data from root core experiment
        >>> fig = plot_biomass_depth_barplot(
        ...     df=biomass_agg,
        ...     depth_col="Depth_cm",
        ...     value_col="Root_DW_g",
        ...     genotype_col="geno",
        ...     include_points=True,
        ...     output_path="biomass_barplot.png"
        ... )

    Notes:
        - Barplot only makes sense for biomass data with discrete depth intervals
          (e.g., 0-30cm, 30-60cm), not continuous counting data.
        - Genotypes are ordered: Control first (if present), then ascending by
          median/mean biomass in shallowest depth interval (matches estimator).
        - Error bars show standard error across biological replicates.
        - Median estimator is default (more robust to outliers), matching typical
          root core aggregation strategies.
        - Styling follows existing depth profile plots: dark grid theme, rotated labels.
    """
    # Validate inputs
    if df.empty:
        raise ValueError("DataFrame is empty")

    required_cols = [depth_col, value_col, genotype_col]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Get unique depths and generate interval labels
    unique_depths = sorted(df[depth_col].unique())

    # Generate depth interval labels (e.g., "0-30cm", "30-60cm")
    # Strategy: Use midpoint between consecutive depths as interval boundaries
    depth_labels = {}
    if len(unique_depths) == 1:
        # Single depth - just use the depth value
        depth_labels = {unique_depths[0]: f"{int(unique_depths[0])}cm"}
    else:
        # Multiple depths - create interval labels using midpoints
        for i, depth in enumerate(unique_depths):
            if i == 0:
                # First interval: 0 to midpoint to next depth
                upper = int((unique_depths[i] + unique_depths[i + 1]) / 2)
                depth_labels[depth] = f"0-{upper}cm"
            elif i == len(unique_depths) - 1:
                # Last interval: midpoint from previous to 2x current depth
                lower = int((unique_depths[i - 1] + unique_depths[i]) / 2)
                upper = int(depth * 2)
                depth_labels[depth] = f"{lower}-{upper}cm"
            else:
                # Middle intervals: midpoint to midpoint
                lower = int((unique_depths[i - 1] + unique_depths[i]) / 2)
                upper = int((unique_depths[i] + unique_depths[i + 1]) / 2)
                depth_labels[depth] = f"{lower}-{upper}cm"

    # Add depth label column for hue
    df = df.copy()
    df["Depth_Label"] = df[depth_col].map(depth_labels)

    # Determine genotype ordering: Control first, then sorted by shallow-layer biomass
    shallow_depth = min(unique_depths)
    shallow_data = df[df[depth_col] == shallow_depth]

    # Use same estimator for ordering as for plotting
    if estimator == "median":
        genotype_stats = (
            shallow_data.groupby(genotype_col)[value_col].median().sort_values()
        )
    else:
        genotype_stats = (
            shallow_data.groupby(genotype_col)[value_col].mean().sort_values()
        )

    # Build custom order
    has_control = "Control" in genotype_stats.index
    if has_control:
        custom_order = ["Control"] + [g for g in genotype_stats.index if g != "Control"]
    else:
        custom_order = list(genotype_stats.index)

    # Save current style and set whitegrid for this plot only
    original_style = plt.rcParams.copy()
    sns.set_style("whitegrid")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))

        # Create grouped barplot
        import numpy as np

        estimator_func = np.median if estimator == "median" else "mean"

        sns.barplot(
            x=genotype_col,
            y=value_col,
            hue="Depth_Label",
            data=df,
            dodge=True,
            estimator=estimator_func,
            errorbar="se",  # Standard error
            order=custom_order,
            ax=ax,
            **kwargs,
        )

        # Add stripplot overlay if requested
        if include_points:
            sns.stripplot(
                x=genotype_col,
                y=value_col,
                hue="Depth_Label",
                data=df,
                dodge=True,
                marker="o",
                alpha=0.5,
                jitter=True,
                palette="dark:k",
                order=custom_order,
                ax=ax,
                legend=False,  # Don't duplicate legend
            )

        # Clean up legend (remove duplicate from stripplot)
        handles, labels = ax.get_legend_handles_labels()
        n_depths = len(unique_depths)
        ax.legend(handles[:n_depths], labels[:n_depths], title="Depth")

        # Set labels and title
        estimator_label = "Median" if estimator == "median" else "Mean"
        ax.set_ylabel(f"{value_col.replace('_', ' ').title()} ({estimator_label} ± SE)")
        ax.set_xlabel("Genotype")
        ax.set_title(
            f"{estimator_label} Biomass per Plot at Different Depths by Genotype"
        )

        # Add grid and rotate x-labels
        ax.grid(True, axis="y")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

        plt.tight_layout()

        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            return None

        return fig
    finally:
        # Restore original matplotlib style to avoid polluting global state
        plt.rcParams.update(original_style)
