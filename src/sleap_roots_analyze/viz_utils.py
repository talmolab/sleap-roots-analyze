"""Adaptive figure sizing utilities for dynamic plot dimensions.

This module provides utilities for automatically calculating appropriate figure
sizes based on the number of items to display, solving the issue of figures
being too small when there are many traits or genotypes.
"""

from __future__ import annotations

import math
from typing import Tuple

from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig


def calculate_grid_dimensions(n_items: int, max_cols: int = 4) -> Tuple[int, int]:
    """Calculate optimal grid dimensions for a given number of items.

    Args:
        n_items: Number of items to display in grid.
        max_cols: Maximum number of columns (default: 4).

    Returns:
        Tuple of (n_rows, n_cols) for the grid layout.

    Example:
        >>> calculate_grid_dimensions(10, max_cols=4)
        (3, 4)  # 3 rows x 4 cols for 10 items
        >>> calculate_grid_dimensions(3, max_cols=4)
        (1, 3)  # 1 row x 3 cols for 3 items
    """
    if n_items <= 0:
        return (0, 0)

    if n_items <= max_cols:
        return (1, n_items)

    n_cols = min(max_cols, n_items)
    n_rows = math.ceil(n_items / n_cols)
    return (n_rows, n_cols)


def calculate_figure_size(
    n_items: int,
    config: AdaptiveSizingConfig,
    layout: str = "grid",
    max_cols: int = 4,
) -> Tuple[float, float]:
    """Calculate adaptive figure size based on number of items.

    Args:
        n_items: Number of items to display.
        config: Adaptive sizing configuration.
        layout: Layout type ("grid", "horizontal", "vertical", "single").
        max_cols: Maximum columns for grid layout (default: 4).

    Returns:
        Tuple of (width, height) in inches.

    Example:
        >>> from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig
        >>> config = AdaptiveSizingConfig()
        >>> calculate_figure_size(10, config, layout="grid")
        (16.0, 12.0)  # Width and height for 10 items in grid
    """
    if not config.enabled:
        # Return base size if adaptive sizing is disabled
        return (config.base_width, config.base_height)

    if layout == "single":
        # Single item - use base size
        width = config.base_width
        height = config.base_height

    elif layout == "horizontal":
        # Horizontal layout (e.g., multiple plots side by side)
        width = config.base_width + (n_items - 1) * config.width_per_item
        height = config.base_height

    elif layout == "vertical":
        # Vertical layout (e.g., stacked plots)
        width = config.base_width
        height = config.base_height + (n_items - 1) * config.height_per_item

    elif layout == "grid":
        # Grid layout - calculate based on rows and columns
        n_rows, n_cols = calculate_grid_dimensions(n_items, max_cols)
        width = config.base_width + (n_cols - 1) * config.width_per_item
        height = config.base_height + (n_rows - 1) * config.height_per_item

    else:
        raise ValueError(
            f"Invalid layout type '{layout}'. "
            f"Must be one of: single, horizontal, vertical, grid"
        )

    # Clamp to min/max bounds
    width = max(config.min_width, min(config.max_width, width))
    height = max(config.min_height, min(config.max_height, height))

    return (width, height)


def calculate_subplot_grid_size(
    n_traits: int,
    config: AdaptiveSizingConfig,
    max_cols: int = 4,
    width_per_subplot: float = 4.0,
    height_per_subplot: float = 3.0,
) -> Tuple[float, float]:
    """Calculate figure size for a subplot grid.

    This is a specialized version for creating grids of subplots where each
    subplot needs a specific minimum size (e.g., histograms, scatter plots).

    Args:
        n_traits: Number of subplots (traits) to display.
        config: Adaptive sizing configuration.
        max_cols: Maximum number of columns (default: 4).
        width_per_subplot: Minimum width per subplot in inches (default: 4.0).
        height_per_subplot: Minimum height per subplot in inches (default: 3.0).

    Returns:
        Tuple of (total_width, total_height) in inches.

    Example:
        >>> from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig
        >>> config = AdaptiveSizingConfig()
        >>> calculate_subplot_grid_size(12, config, max_cols=4)
        (16.0, 9.0)  # 4 cols x 3 rows
    """
    if not config.enabled:
        # Return fixed size if adaptive sizing disabled
        n_rows, n_cols = calculate_grid_dimensions(n_traits, max_cols)
        return (n_cols * width_per_subplot, n_rows * height_per_subplot)

    # Calculate grid dimensions
    n_rows, n_cols = calculate_grid_dimensions(n_traits, max_cols)

    # Calculate total size based on subplot dimensions
    total_width = n_cols * width_per_subplot
    total_height = n_rows * height_per_subplot

    # Clamp to configured bounds
    total_width = max(config.min_width, min(config.max_width, total_width))
    total_height = max(config.min_height, min(config.max_height, total_height))

    return (total_width, total_height)


def calculate_correlation_matrix_size(
    n_traits: int,
    config: AdaptiveSizingConfig,
    min_size_per_trait: float = 0.5,
) -> Tuple[float, float]:
    """Calculate figure size for correlation matrix heatmap.

    Args:
        n_traits: Number of traits in the correlation matrix.
        config: Adaptive sizing configuration.
        min_size_per_trait: Minimum size (width and height) per trait in inches (default: 0.5).

    Returns:
        Tuple of (width, height) in inches for square correlation matrix.

    Example:
        >>> from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig
        >>> config = AdaptiveSizingConfig()
        >>> calculate_correlation_matrix_size(20, config)
        (15.0, 15.0)  # Square matrix with room for labels
    """
    if not config.enabled:
        return (config.base_width, config.base_height)

    # Calculate size based on number of traits
    # Add extra space for labels and colorbar
    matrix_size = n_traits * min_size_per_trait
    total_size = matrix_size + 2.0  # Add 2 inches for labels and colorbar

    # Clamp to bounds (use same for width and height since it's square)
    total_size = max(config.min_width, min(config.max_width, total_size))

    return (total_size, total_size)


def calculate_barplot_size(
    n_items: int,
    config: AdaptiveSizingConfig,
    orientation: str = "vertical",
    width_per_bar: float = 0.8,
    height_per_bar: float = 0.6,
) -> Tuple[float, float]:
    """Calculate figure size for bar plots.

    Args:
        n_items: Number of bars to display.
        config: Adaptive sizing configuration.
        orientation: Bar orientation ("vertical" or "horizontal").
        width_per_bar: Width per bar for vertical plots (default: 0.8).
        height_per_bar: Height per bar for horizontal plots (default: 0.6).

    Returns:
        Tuple of (width, height) in inches.

    Example:
        >>> from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig
        >>> config = AdaptiveSizingConfig()
        >>> calculate_barplot_size(30, config, orientation="horizontal")
        (10.0, 16.0)  # Horizontal bar plot for 30 items
    """
    if not config.enabled:
        return (config.base_width, config.base_height)

    if orientation == "vertical":
        # Width scales with number of bars
        width = max(config.base_width, n_items * width_per_bar)
        height = config.base_height
    elif orientation == "horizontal":
        # Height scales with number of bars
        width = config.base_width
        height = max(config.base_height, n_items * height_per_bar)
    else:
        raise ValueError(
            f"Invalid orientation '{orientation}'. Must be 'vertical' or 'horizontal'"
        )

    # Clamp to bounds
    width = max(config.min_width, min(config.max_width, width))
    height = max(config.min_height, min(config.max_height, height))

    return (width, height)


def suggest_layout_params(
    n_items: int,
    figure_size: Tuple[float, float],
    max_cols: int = 4,
) -> dict:
    """Suggest matplotlib layout parameters for a given figure size.

    Args:
        n_items: Number of items to display.
        figure_size: Figure size (width, height) in inches.
        max_cols: Maximum number of columns for grid layout.

    Returns:
        Dictionary with layout parameters (nrows, ncols, figsize, and spacing hints).

    Example:
        >>> suggest_layout_params(10, (16.0, 12.0), max_cols=4)
        {
            'nrows': 3,
            'ncols': 4,
            'figsize': (16.0, 12.0),
            'hspace': 0.3,
            'wspace': 0.3
        }
    """
    n_rows, n_cols = calculate_grid_dimensions(n_items, max_cols)

    # Suggest spacing based on figure size and number of subplots
    # More subplots = need more spacing for labels
    base_spacing = 0.2
    spacing_factor = 1.0 + (n_items / 20.0)  # Increase spacing for many items
    suggested_spacing = min(0.5, base_spacing * spacing_factor)

    return {
        "nrows": n_rows,
        "ncols": n_cols,
        "figsize": figure_size,
        "hspace": suggested_spacing,  # Vertical spacing
        "wspace": suggested_spacing,  # Horizontal spacing
    }
