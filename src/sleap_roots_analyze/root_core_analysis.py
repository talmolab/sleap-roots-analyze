"""Root core analysis functions for processing depth-stratified root count data.

This module provides utilities for processing root core experiments where multiple
soil cores are extracted from each plot at different depths, with each depth segment
counted for root intersections.

Typical workflow:
    1. Load raw core count data (wide format)
    2. Create unique sample identifiers
    3. Validate identifier uniqueness
    4. Melt to long format with depth calculation
    5. Aggregate across technical replicates (cores) to biological level (plots)
    6. Visualize depth profiles (see depth_profile_plots module)

Example:
    >>> import pandas as pd
    >>> from sleap_roots_analyze.root_core_analysis import (
    ...     create_sample_identifier,
    ...     validate_unique_identifiers,
    ...     melt_depth_data,
    ...     aggregate_by_replicate
    ... )
    >>>
    >>> # Load raw core data
    >>> df = pd.read_csv("root_counting.csv")
    >>>
    >>> # Create sample identifiers
    >>> df['sample_id'] = create_sample_identifier(df)
    >>>
    >>> # Validate uniqueness
    >>> is_unique, dupes = validate_unique_identifiers(df, 'sample_id')
    >>> if not is_unique:
    ...     print(f"Duplicate samples: {dupes}")
    >>>
    >>> # Melt to long format with depth calculation
    >>> df_melted = melt_depth_data(
    ...     df,
    ...     id_vars=['sample_id', 'Plot', 'geno', 'Rep', 'core_n'],
    ...     depth_prefix='c_',
    ...     parse_depth=True
    ... )
    >>>
    >>> # Create plot-replicate identifier for aggregation
    >>> df_melted['plot_rep'] = (
    ...     'plot' + df_melted['Plot'].astype(str) +
    ...     '_rep' + df_melted['Rep'].astype(str)
    ... )
    >>>
    >>> # Aggregate cores within each plot-replicate
    >>> df_agg = aggregate_by_replicate(
    ...     df_melted,
    ...     group_cols=['plot_rep', 'geno', 'Depth_cm'],
    ...     value_col='Root_Count',
    ...     agg_func='mean'
    ... )
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import re
from typing import Optional, List, Tuple, Callable


def create_sample_identifier(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    prefix_map: Optional[dict[str, str]] = None,
) -> pd.Series:
    """Create unique sample identifiers from experimental metadata columns.

    Combines multiple metadata columns into a single unique identifier string.
    Default format: plot{Plot}_rep{Rep}_{geno}_core{core_n}

    Args:
        df: DataFrame containing metadata columns
        columns: List of column names to include in identifier.
            Default: ['Plot', 'Rep', 'geno', 'core_n']
        prefix_map: Optional dictionary mapping column names to custom prefixes.
            Example: {'Plot': 'p', 'Rep': 'r'} -> 'p1_r1_...'
            If None, uses lowercase column name as prefix.

    Returns:
        pd.Series: Sample identifiers, one per row

    Raises:
        ValueError: If any specified column is missing from DataFrame

    Example:
        >>> df = pd.DataFrame({
        ...     'Plot': [1, 1, 2],
        ...     'Rep': [1, 1, 1],
        ...     'geno': ['GH_7386', 'GH_7386', 'GH_7418'],
        ...     'core_n': [1, 2, 1]
        ... })
        >>> create_sample_identifier(df)
        0    plot1_rep1_GH_7386_core1
        1    plot1_rep1_GH_7386_core2
        2    plot2_rep1_GH_7418_core1
        dtype: object
    """
    if columns is None:
        columns = ["Plot", "Rep", "geno", "core_n"]

    # Check all columns exist
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    # Build default prefix map for specific columns
    if prefix_map is None:
        # Only add prefixes for specific columns, not for all
        default_prefixes = {
            "Plot": "plot",
            "Rep": "rep",
            "core_n": "core",
            "Ent": "ent",
            "Sub": "sub",
        }
        prefix_map = {col: default_prefixes.get(col, "") for col in columns}

    # Create identifier parts
    parts = []
    for col in columns:
        prefix = prefix_map.get(col, "")
        
        # Convert to appropriate type before string conversion
        # For numeric columns (Plot, Rep, core_n), convert to int to avoid ".0"
        if col in ["Plot", "Rep", "core_n", "Ent", "Sub"] and pd.api.types.is_numeric_dtype(df[col]):
            # Convert float to int (handles 1.0 -> 1)
            col_str = df[col].astype(int).astype(str)
        else:
            col_str = df[col].astype(str)
        
        if prefix:
            parts.append(prefix + col_str)
        else:
            # No prefix for this column (e.g., 'geno')
            parts.append(col_str)

    # Join with underscores
    return parts[0].str.cat(parts[1:], sep="_")


def validate_unique_identifiers(df: pd.DataFrame, id_column: str) -> Tuple[bool, List]:
    """Validate that identifier column contains unique values for all samples.

    Args:
        df: DataFrame containing identifier column
        id_column: Name of column to validate for uniqueness

    Returns:
        Tuple of (is_unique, duplicates) where:
            - is_unique: True if all values are unique, False otherwise
            - duplicates: List of duplicate values (empty if all unique)

    Example:
        >>> df = pd.DataFrame({'sample_id': ['s1', 's2', 's1', 's3']})
        >>> is_unique, dupes = validate_unique_identifiers(df, 'sample_id')
        >>> is_unique
        False
        >>> dupes
        ['s1']
    """
    if len(df) == 0:
        return True, []

    # Count unique values
    unique_count = df[id_column].nunique()
    total_count = len(df)

    if unique_count == total_count:
        return True, []

    # Find duplicates
    duplicated_mask = df.duplicated(subset=[id_column], keep=False)
    duplicates = df.loc[duplicated_mask, id_column].unique().tolist()

    return False, duplicates


def melt_depth_data(
    df: pd.DataFrame,
    id_vars: List[str],
    depth_prefix: str = "c_",
    parse_depth: bool = True,
) -> pd.DataFrame:
    """Convert wide-format depth data to long format with automatic depth calculation.

    Transforms columns matching pattern `c_<start>_<end>_<subcore>` (e.g., c_0_10_1)
    into long format with optional depth calculation.

    Depth calculation formula:
        Depth_cm = start + (subcore_index - 1) * (end - start) / 2

    Examples:
        - c_0_10_1 (first subcore 0-10cm)  -> Depth_cm = 0 + (1-1) * 10/2 = 0.0
        - c_0_10_2 (second subcore 0-10cm) -> Depth_cm = 0 + (2-1) * 10/2 = 5.0
        - c_50_60_1 (first subcore 50-60cm) -> Depth_cm = 50 + (1-1) * 10/2 = 50.0

    Args:
        df: Wide-format DataFrame with depth columns
        id_vars: List of ID columns to preserve (e.g., ['Plot', 'geno', 'Rep'])
        depth_prefix: Prefix identifying depth columns (default: 'c_')
        parse_depth: If True, calculate Depth_cm from column names (default: True)

    Returns:
        Long-format DataFrame with columns:
            - All id_vars columns
            - Depth: Original column name (e.g., 'c_0_10_1')
            - Root_Count: The measured value
            - Depth_cm: Calculated depth in cm (if parse_depth=True)

    Example:
        >>> df = pd.DataFrame({
        ...     'Plot': [1, 2],
        ...     'c_0_10_1': [78, 120],
        ...     'c_0_10_2': [62, 134]
        ... })
        >>> melt_depth_data(df, id_vars=['Plot'], parse_depth=True)
           Plot     Depth  Root_Count  Depth_cm
        0     1  c_0_10_1        78.0       0.0
        1     2  c_0_10_1       120.0       0.0
        2     1  c_0_10_2        62.0       5.0
        3     2  c_0_10_2       134.0       5.0
    """
    # Find columns matching depth pattern
    value_vars = [col for col in df.columns if col.startswith(depth_prefix)]

    # Melt to long format
    df_melted = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="Depth",
        value_name="Root_Count",
    )

    if not parse_depth:
        return df_melted

    # Parse depth from column names using pattern: c_<start>_<end>_<subcore>
    pattern = rf"{re.escape(depth_prefix)}(\d+)_(\d+)_(\d+)"
    depth_parts = df_melted["Depth"].str.extract(pattern).astype(float)
    depth_parts.columns = ["core_start", "core_end", "subcore_index"]

    # Calculate depth in cm
    # Formula: start + (subcore_index - 1) * (end - start) / 2
    core_thickness = depth_parts["core_end"] - depth_parts["core_start"]
    subcore_thickness = core_thickness / 2
    df_melted["Depth_cm"] = (
        depth_parts["core_start"]
        + (depth_parts["subcore_index"] - 1) * subcore_thickness
    )

    return df_melted


def aggregate_by_replicate(
    df_melted: pd.DataFrame,
    group_cols: List[str],
    value_col: str = "Root_Count",
    agg_func: str | Callable = "mean",
) -> pd.DataFrame:
    """Aggregate measurements across technical replicates to biological replicate level.

    Typically used to average root counts across multiple cores within the same
    plot-replicate at each depth.

    Args:
        df_melted: Long-format DataFrame (output from melt_depth_data)
        group_cols: Columns to group by (e.g., ['plot_rep', 'geno', 'Depth_cm'])
        value_col: Column name containing values to aggregate (default: 'Root_Count')
        agg_func: Aggregation function - 'mean', 'median', 'sum', etc., or callable
            (default: 'mean')

    Returns:
        DataFrame with one row per unique combination of group columns,
        with aggregated values

    Example:
        >>> # After melting, aggregate cores within each plot-replicate
        >>> df_agg = aggregate_by_replicate(
        ...     df_melted,
        ...     group_cols=['plot_rep', 'geno', 'Depth_cm'],
        ...     value_col='Root_Count',
        ...     agg_func='mean'
        ... )
    """
    result = df_melted.groupby(group_cols, as_index=False)[value_col].agg(agg_func)
    return result
