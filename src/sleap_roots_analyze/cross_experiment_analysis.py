"""Cross-experiment correlation analysis functions.

This module provides functions for analyzing correlations between traits
measured in different experimental modalities (e.g., cylinder vs turface).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr


def _calculate_correlations(
    values1: np.ndarray, values2: np.ndarray
) -> Tuple[float, float, float, float]:
    """Helper function to calculate both Pearson and Spearman correlations.

    Args:
        values1: First set of values
        values2: Second set of values

    Returns:
        Tuple of (pearson_r, pearson_p, spearman_r, spearman_p)
    """
    pearson_r, pearson_p = stats.pearsonr(values1, values2)
    spearman_r, spearman_p = spearmanr(values1, values2)
    return pearson_r, pearson_p, spearman_r, spearman_p


def _prepare_aligned_values(
    exp1_data: pd.DataFrame,
    exp2_data: pd.DataFrame,
    exp1_col: str,
    exp2_col: str,
    min_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Helper to align values from two experiments.

    Args:
        exp1_data: Data from experiment 1 (with index as genotypes)
        exp2_data: Data from experiment 2 (with index as genotypes)
        exp1_col: Column name from exp1
        exp2_col: Column name from exp2
        min_samples: Minimum samples required per genotype

    Returns:
        Tuple of (values1, values2, valid_genotypes) with NaN removed
    """
    # Find common genotypes
    valid_genotypes = list(set(exp1_data.index) & set(exp2_data.index))

    # Filter by sample count if needed
    if (
        min_samples > 0
        and "n_samples" in exp1_data.columns
        and "n_samples" in exp2_data.columns
    ):
        valid_genotypes = [
            g
            for g in valid_genotypes
            if exp1_data.loc[g, "n_samples"] >= min_samples
            and exp2_data.loc[g, "n_samples"] >= min_samples
        ]

    if len(valid_genotypes) < 3:
        return np.array([]), np.array([]), []

    # Get aligned values
    values1 = exp1_data.loc[valid_genotypes, exp1_col].values
    values2 = exp2_data.loc[valid_genotypes, exp2_col].values

    # Remove NaN pairs
    valid_mask = ~(np.isnan(values1) | np.isnan(values2))
    if valid_mask.sum() < 3:
        return np.array([]), np.array([]), []

    return (
        values1[valid_mask],
        values2[valid_mask],
        [g for i, g in enumerate(valid_genotypes) if valid_mask[i]],
    )


def load_and_align_experiments(
    exp1_path: Path | str,
    exp2_path: Path | str,
    genotype_col1: str = "Geno",
    genotype_col2: str = "geno",
    rep_col1: str = "Rep",
    rep_col2: str = "rep",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Load and align two experimental datasets by genotype.

    Args:
        exp1_path: Path to first experiment data
        exp2_path: Path to second experiment data
        genotype_col1: Genotype column name in first dataset
        genotype_col2: Genotype column name in second dataset
        rep_col1: Replicate column name in first dataset
        rep_col2: Replicate column name in second dataset

    Returns:
        Tuple of (exp1_data, exp2_data, common_genotypes)
    """
    exp1_path = Path(exp1_path)
    exp2_path = Path(exp2_path)

    # Load data
    exp1_df = pd.read_csv(exp1_path)
    exp2_df = pd.read_csv(exp2_path)

    # Standardize column names
    exp1_df = exp1_df.rename(columns={genotype_col1: "genotype", rep_col1: "replicate"})
    exp2_df = exp2_df.rename(columns={genotype_col2: "genotype", rep_col2: "replicate"})

    # Find common genotypes
    genotypes1 = set(exp1_df["genotype"].unique())
    genotypes2 = set(exp2_df["genotype"].unique())
    common_genotypes = sorted(list(genotypes1.intersection(genotypes2)))

    # Filter to common genotypes
    exp1_df = exp1_df[exp1_df["genotype"].isin(common_genotypes)]
    exp2_df = exp2_df[exp2_df["genotype"].isin(common_genotypes)]

    print(f"Found {len(common_genotypes)} common genotypes between experiments")
    print(f"Experiment 1: {len(exp1_df)} samples, {len(exp1_df.columns)} traits")
    print(f"Experiment 2: {len(exp2_df)} samples, {len(exp2_df.columns)} traits")

    return exp1_df, exp2_df, common_genotypes


def calculate_genotype_means(
    df: pd.DataFrame, trait_cols: List[str], genotype_col: str = "genotype"
) -> pd.DataFrame:
    """Calculate mean trait values per genotype.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait columns to aggregate
        genotype_col: Name of genotype column

    Returns:
        DataFrame with mean trait values per genotype
    """
    # Group by genotype and calculate means
    genotype_means = df.groupby(genotype_col)[trait_cols].mean()

    # Add sample counts
    genotype_means["n_samples"] = df.groupby(genotype_col).size()

    return genotype_means


def calculate_genotype_statistics(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "genotype",
    statistics: List[str] = ["mean", "median", "min", "max", "std"],
) -> Dict[str, pd.DataFrame]:
    """Calculate multiple statistics per genotype for all traits.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait columns to aggregate
        genotype_col: Name of genotype column
        statistics: List of statistics to compute

    Returns:
        Dictionary with statistic name as key and DataFrame as value
    """
    results = {}

    # Group by genotype
    grouped = df.groupby(genotype_col)[trait_cols]

    # Calculate each statistic
    for stat in statistics:
        if stat == "mean":
            stat_df = grouped.mean()
        elif stat == "median":
            stat_df = grouped.median()
        elif stat == "min":
            stat_df = grouped.min()
        elif stat == "max":
            stat_df = grouped.max()
        elif stat == "std":
            stat_df = grouped.std()
        elif stat == "count":
            stat_df = grouped.count()
        else:
            continue

        # Add sample counts
        stat_df["n_samples"] = df.groupby(genotype_col).size()
        results[stat] = stat_df

    return results


def calculate_cross_experiment_correlations_extended(
    exp1_stats: Dict[str, pd.DataFrame],
    exp2_stats: Dict[str, pd.DataFrame],
    exp1_traits: List[str],
    exp2_traits: List[str],
    min_samples: int = 3,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Calculate correlations between traits using all statistic combinations.

    Args:
        exp1_stats: Dictionary of statistics DataFrames for experiment 1
        exp2_stats: Dictionary of statistics DataFrames for experiment 2
        exp1_traits: Trait columns from experiment 1
        exp2_traits: Trait columns from experiment 2
        min_samples: Minimum samples required per genotype
        top_n: If specified, only return top N correlations

    Returns:
        DataFrame with correlation results for all statistic combinations
    """
    results = []

    # Iterate through all statistic combinations
    for stat1_name, exp1_df in exp1_stats.items():
        for stat2_name, exp2_df in exp2_stats.items():
            # Find common genotypes with sufficient samples
            valid_genotypes = list(set(exp1_df.index) & set(exp2_df.index))

            if min_samples > 0:
                valid_genotypes = [
                    g
                    for g in valid_genotypes
                    if exp1_df.loc[g, "n_samples"] >= min_samples
                    and exp2_df.loc[g, "n_samples"] >= min_samples
                ]

            if len(valid_genotypes) < 3:
                continue

            # Calculate correlations for this statistic combination
            for trait1 in exp1_traits:
                for trait2 in exp2_traits:
                    # Get aligned and cleaned values
                    values1_clean, values2_clean, _ = _prepare_aligned_values(
                        exp1_df, exp2_df, trait1, trait2, min_samples=0
                    )

                    if len(values1_clean) < 3:
                        continue

                    # Calculate correlations
                    pearson_corr, pearson_p, spearman_corr, spearman_p = (
                        _calculate_correlations(values1_clean, values2_clean)
                    )

                    results.append(
                        {
                            "exp1_trait": trait1,
                            "exp2_trait": trait2,
                            "exp1_statistic": stat1_name,
                            "exp2_statistic": stat2_name,
                            "pearson_r": pearson_corr,
                            "pearson_p": pearson_p,
                            "spearman_r": spearman_corr,
                            "spearman_p": spearman_p,
                            "n_genotypes": len(values1_clean),
                            "abs_pearson": abs(pearson_corr),
                            "abs_spearman": abs(spearman_corr),
                        }
                    )

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        return results_df

    # Add significance flags
    results_df["pearson_significant"] = results_df["pearson_p"] < 0.05
    results_df["spearman_significant"] = results_df["spearman_p"] < 0.05

    # Sort by absolute correlation
    results_df = results_df.sort_values("abs_pearson", ascending=False)

    # Return top N if specified
    if top_n is not None and len(results_df) > top_n:
        return results_df.head(top_n)

    return results_df


def summarize_statistic_combinations(
    extended_corr_df: pd.DataFrame, metric: str = "pearson_r"
) -> pd.DataFrame:
    """Summarize which statistic combinations yield the strongest correlations.

    Args:
        extended_corr_df: DataFrame from calculate_cross_experiment_correlations_extended
        metric: Which correlation metric to use ('pearson_r' or 'spearman_r')

    Returns:
        DataFrame summarizing best statistic combinations
    """
    if len(extended_corr_df) == 0:
        return pd.DataFrame()

    abs_metric = f"abs_{metric.split('_')[0]}"

    # Group by statistic combination
    stat_summary = (
        extended_corr_df.groupby(["exp1_statistic", "exp2_statistic"])
        .agg(
            {
                abs_metric: ["mean", "max", "std", "count"],
                f"{metric.split('_')[0]}_significant": "sum",
            }
        )
        .round(3)
    )

    # Flatten column names
    stat_summary.columns = [
        "_".join(col).strip() for col in stat_summary.columns.values
    ]
    stat_summary = stat_summary.reset_index()

    # Sort by mean absolute correlation
    stat_summary = stat_summary.sort_values(f"{abs_metric}_mean", ascending=False)

    return stat_summary


def create_statistic_combination_heatmap(
    extended_corr_df: pd.DataFrame,
    trait1: str,
    trait2: str,
    metric: str = "pearson_r",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Create heatmap showing correlations across all statistic combinations for a trait pair.

    Args:
        extended_corr_df: DataFrame from calculate_cross_experiment_correlations_extended
        trait1: Trait from experiment 1
        trait2: Trait from experiment 2
        metric: Which correlation metric to plot
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    # Handle empty dataframe
    if len(extended_corr_df) == 0 or "exp1_trait" not in extended_corr_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            fontsize=14,
        )
        return fig

    # Filter to specific trait pair
    trait_data = extended_corr_df[
        (extended_corr_df["exp1_trait"] == trait1)
        & (extended_corr_df["exp2_trait"] == trait2)
    ]

    if len(trait_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No data for this trait pair",
            ha="center",
            va="center",
            fontsize=14,
        )
        return fig

    # Pivot for heatmap
    heatmap_data = trait_data.pivot(
        index="exp1_statistic", columns="exp2_statistic", values=metric
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": f"{metric.replace('_', ' ').title()}"},
    )

    ax.set_title(
        f"Correlation Across Statistic Combinations\n{trait1} vs {trait2}", fontsize=12
    )
    ax.set_xlabel("Experiment 2 Statistic", fontsize=10)
    ax.set_ylabel("Experiment 1 Statistic", fontsize=10)

    plt.tight_layout()

    return fig


def calculate_cross_experiment_correlations(
    exp1_means: pd.DataFrame,
    exp2_means: pd.DataFrame,
    exp1_traits: List[str],
    exp2_traits: List[str],
    min_samples: int = 3,
) -> pd.DataFrame:
    """Calculate correlations between traits from two experiments.

    Args:
        exp1_means: Mean trait values from experiment 1
        exp2_means: Mean trait values from experiment 2
        exp1_traits: Trait columns from experiment 1
        exp2_traits: Trait columns from experiment 2
        min_samples: Minimum samples required per genotype

    Returns:
        DataFrame with correlation results
    """
    # Find common genotypes with sufficient samples
    valid_genotypes = list(set(exp1_means.index) & set(exp2_means.index))

    if min_samples > 0:
        valid_genotypes = [
            g
            for g in valid_genotypes
            if exp1_means.loc[g, "n_samples"] >= min_samples
            and exp2_means.loc[g, "n_samples"] >= min_samples
        ]

    print(f"Using {len(valid_genotypes)} genotypes with >= {min_samples} samples each")

    # Calculate correlations
    results = []
    for trait1 in exp1_traits:
        for trait2 in exp2_traits:
            # Get aligned and cleaned values
            values1_clean, values2_clean, _ = _prepare_aligned_values(
                exp1_means, exp2_means, trait1, trait2, min_samples=0
            )

            if len(values1_clean) < 3:
                continue

            # Calculate correlation (only Pearson for backward compatibility)
            corr, pval, _, _ = _calculate_correlations(values1_clean, values2_clean)

            results.append(
                {
                    "exp1_trait": trait1,
                    "exp2_trait": trait2,
                    "correlation": corr,
                    "p_value": pval,
                    "n_genotypes": len(values1_clean),
                    "abs_correlation": abs(corr),
                }
            )

    results_df = pd.DataFrame(results)

    # Add significance flags if there are results
    if len(results_df) > 0:
        results_df["significant"] = results_df["p_value"] < 0.05
        results_df["highly_significant"] = results_df["p_value"] < 0.01
        # Sort by absolute correlation
        results_df = results_df.sort_values("abs_correlation", ascending=False)
    else:
        # Create empty dataframe with expected columns
        results_df = pd.DataFrame(
            columns=[
                "exp1_trait",
                "exp2_trait",
                "correlation",
                "p_value",
                "n_genotypes",
                "abs_correlation",
                "significant",
                "highly_significant",
            ]
        )

    return results_df


def create_cross_experiment_heatmap(
    correlation_df: pd.DataFrame,
    top_n_traits: int = 20,
    figsize: Tuple[int, int] = (14, 12),
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> plt.Figure:
    """Create heatmap of cross-experiment correlations.

    Args:
        correlation_df: DataFrame with correlation results
        top_n_traits: Number of top traits to show from each experiment
        figsize: Figure size
        cmap: Colormap for heatmap
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale

    Returns:
        Matplotlib figure
    """
    # Handle empty dataframe
    if len(correlation_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No correlations to display",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.set_title("Cross-Experiment Trait Correlations")
        return fig

    # Get top traits from each experiment
    top_exp1_traits = (
        correlation_df.groupby("exp1_trait")["abs_correlation"]
        .max()
        .nlargest(top_n_traits)
        .index.tolist()
    )
    top_exp2_traits = (
        correlation_df.groupby("exp2_trait")["abs_correlation"]
        .max()
        .nlargest(top_n_traits)
        .index.tolist()
    )

    # Filter to top traits
    filtered_df = correlation_df[
        (correlation_df["exp1_trait"].isin(top_exp1_traits))
        & (correlation_df["exp2_trait"].isin(top_exp2_traits))
    ]

    # Pivot to matrix form
    corr_matrix = filtered_df.pivot(
        index="exp1_trait", columns="exp2_trait", values="correlation"
    )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=False,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    )

    ax.set_title(
        "Cross-Experiment Trait Correlations\n(Top Correlated Traits)", fontsize=14
    )
    ax.set_xlabel("Experiment 2 Traits", fontsize=12)
    ax.set_ylabel("Experiment 1 Traits", fontsize=12)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()

    return fig


def create_top_correlations_plot(
    correlation_df: pd.DataFrame, top_n: int = 30, figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """Create bar plot of top correlations between experiments.

    Args:
        correlation_df: DataFrame with correlation results
        top_n: Number of top correlations to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Get top correlations
    top_corr = correlation_df.nlargest(top_n, "abs_correlation").copy()

    # Create trait pair labels
    top_corr["trait_pair"] = (
        top_corr["exp1_trait"].str[:20] + " vs " + top_corr["exp2_trait"].str[:20]
    )

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot absolute correlations
    colors = ["red" if x < 0 else "blue" for x in top_corr["correlation"].values]
    bars1 = ax1.barh(
        range(len(top_corr)),
        top_corr["abs_correlation"].values,
        color=colors,
        alpha=0.7,
    )
    ax1.set_yticks(range(len(top_corr)))
    ax1.set_yticklabels(top_corr["trait_pair"].values, fontsize=8)
    ax1.set_xlabel("Absolute Correlation", fontsize=10)
    ax1.set_title(f"Top {top_n} Cross-Experiment Correlations", fontsize=12)
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # Add significance markers
    for i, (corr, sig) in enumerate(
        zip(top_corr["abs_correlation"], top_corr["highly_significant"])
    ):
        if sig:
            ax1.text(corr + 0.01, i, "**", va="center", fontsize=8)
        elif top_corr.iloc[i]["significant"]:
            ax1.text(corr + 0.01, i, "*", va="center", fontsize=8)

    # Plot signed correlations
    bars2 = ax2.barh(
        range(len(top_corr)), top_corr["correlation"].values, color=colors, alpha=0.7
    )
    ax2.set_yticks(range(len(top_corr)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("Correlation (Pearson r)", fontsize=10)
    ax2.set_title("Signed Correlations", fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis="x", alpha=0.3)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", alpha=0.7, label="Positive"),
        Patch(facecolor="red", alpha=0.7, label="Negative"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.suptitle("Cross-Experiment Trait Correlations Analysis", fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def create_scatter_plot_grid(
    exp1_means: pd.DataFrame,
    exp2_means: pd.DataFrame,
    top_pairs: List[Tuple[str, str]],
    figsize: Tuple[int, int] = (16, 12),
    n_cols: int = 4,
) -> plt.Figure:
    """Create grid of scatter plots for top correlated trait pairs.

    Args:
        exp1_means: Mean trait values from experiment 1
        exp2_means: Mean trait values from experiment 2
        top_pairs: List of (exp1_trait, exp2_trait) tuples
        figsize: Figure size
        n_cols: Number of columns in grid

    Returns:
        Matplotlib figure
    """
    n_pairs = len(top_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_pairs > 1 else [axes]

    # Find common genotypes
    common_genotypes = list(set(exp1_means.index) & set(exp2_means.index))

    for idx, (trait1, trait2) in enumerate(top_pairs):
        ax = axes[idx]

        # Get aligned values
        x_vals = exp1_means.loc[common_genotypes, trait1].values
        y_vals = exp2_means.loc[common_genotypes, trait2].values

        # Remove NaNs
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]

        # Create scatter plot
        ax.scatter(x_vals, y_vals, alpha=0.6, s=30)

        # Add regression line
        if len(x_vals) > 2:
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, p(x_line), "r-", alpha=0.5, linewidth=2)

            # Calculate correlation
            corr, pval, _, _ = _calculate_correlations(x_vals, y_vals)
            ax.text(
                0.05,
                0.95,
                f"r={corr:.3f}\np={pval:.3e}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # Labels
        ax.set_xlabel(f"Exp1: {trait1[:25]}", fontsize=8)
        ax.set_ylabel(f"Exp2: {trait2[:25]}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Scatter Plots of Top Correlated Trait Pairs", fontsize=14)
    plt.tight_layout()

    return fig


def calculate_per_trait_correlations(
    exp1_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    exp1_trait: str,
    exp2_trait: str,
    genotype_col: str = "genotype",
) -> Dict:
    """Calculate correlation between specific traits using individual samples.

    Args:
        exp1_df: DataFrame from experiment 1 with individual samples
        exp2_df: DataFrame from experiment 2 with individual samples
        exp1_trait: Trait column from experiment 1
        exp2_trait: Trait column from experiment 2
        genotype_col: Name of genotype column

    Returns:
        Dictionary with correlation statistics
    """
    # Get genotype means for each trait
    exp1_means = exp1_df.groupby(genotype_col)[exp1_trait].mean()
    exp2_means = exp2_df.groupby(genotype_col)[exp2_trait].mean()

    # Find common genotypes
    common_genos = list(set(exp1_means.index) & set(exp2_means.index))

    if len(common_genos) < 3:
        return {
            "n_genotypes": len(common_genos),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "valid": False,
        }

    # Align values
    values1 = exp1_means.loc[common_genos].values
    values2 = exp2_means.loc[common_genos].values

    # Remove NaN pairs
    valid_mask = ~(np.isnan(values1) | np.isnan(values2))
    if valid_mask.sum() < 3:
        return {
            "n_genotypes": valid_mask.sum(),
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
            "valid": False,
        }

    values1_clean = values1[valid_mask]
    values2_clean = values2[valid_mask]

    # Calculate correlations
    pearson_r, pearson_p, spearman_r, spearman_p = _calculate_correlations(
        values1_clean, values2_clean
    )

    return {
        "n_genotypes": len(values1_clean),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "values_exp1": values1_clean,
        "values_exp2": values2_clean,
        "genotypes": [
            common_genos[i] for i in range(len(common_genos)) if valid_mask[i]
        ],
        "valid": True,
    }


def create_joint_plot(
    exp1_means: pd.DataFrame,
    exp2_means: pd.DataFrame,
    exp1_trait: str,
    exp2_trait: str,
    exp1_name: str = "Experiment 1",
    exp2_name: str = "Experiment 2",
    figsize: Tuple[int, int] = (10, 10),
    color: str = "#4CB391",
    line_color: str = "#2E6E73",
) -> plt.Figure:
    """Create joint plot for two traits with regression line and marginal distributions.

    Args:
        exp1_means: Genotype means from experiment 1
        exp2_means: Genotype means from experiment 2
        exp1_trait: Trait column from experiment 1
        exp2_trait: Trait column from experiment 2
        exp1_name: Name of experiment 1 for labeling
        exp2_name: Name of experiment 2 for labeling
        figsize: Figure size
        color: Color for scatter points
        line_color: Color for regression line

    Returns:
        Matplotlib figure
    """
    import seaborn as sns

    # Get common genotypes
    common_genos = list(set(exp1_means.index) & set(exp2_means.index))

    # Create DataFrame for plotting
    plot_df = pd.DataFrame(
        {
            exp1_trait: exp1_means.loc[common_genos, exp1_trait],
            exp2_trait: exp2_means.loc[common_genos, exp2_trait],
        }
    )

    # Remove NaN pairs
    plot_df = plot_df.dropna()

    if len(plot_df) < 3:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "Insufficient data for plot",
            ha="center",
            va="center",
            fontsize=14,
        )
        return fig

    # Calculate correlations
    pearson_r, pearson_p, spearman_r, spearman_p = _calculate_correlations(
        plot_df[exp1_trait].values, plot_df[exp2_trait].values
    )

    # Create joint plot
    g = sns.jointplot(
        data=plot_df,
        x=exp1_trait,
        y=exp2_trait,
        kind="reg",
        scatter_kws={"s": 50, "alpha": 0.8, "color": color},
        line_kws={"color": line_color},
        height=figsize[0],
    )

    # Add labels
    g.set_axis_labels(
        f"{exp1_name}: {exp1_trait.replace('_', ' ').title()}",
        f"{exp2_name}: {exp2_trait.replace('_', ' ').title()}",
        fontsize=12,
    )

    # Add title
    g.figure.suptitle(
        f"Cross-Experiment Trait Correlation\n{exp1_name} vs {exp2_name}", fontsize=14
    )

    # Add correlation text
    corr_text = f"Pearson r = {pearson_r:.3f} (p = {pearson_p:.3g})\n"
    corr_text += f"Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.3g})\n"
    corr_text += f"n = {len(plot_df)} genotypes"

    g.ax_joint.annotate(
        corr_text,
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=1),
    )

    g.figure.tight_layout()
    g.figure.subplots_adjust(top=0.93)

    return g.figure


def identify_significant_correlations(
    correlation_df: pd.DataFrame,
    p_threshold: float = 0.05,
    r_threshold: float = 0.5,
    use_fdr: bool = True,
) -> pd.DataFrame:
    """Identify statistically significant correlations with FDR correction.

    Args:
        correlation_df: DataFrame with correlation results
        p_threshold: P-value threshold for significance
        r_threshold: Minimum absolute correlation threshold
        use_fdr: Whether to use FDR correction for multiple testing

    Returns:
        DataFrame with significant correlations only
    """
    from statsmodels.stats.multitest import multipletests

    # Filter by absolute correlation threshold
    strong_corr = correlation_df[
        correlation_df["abs_correlation"] >= r_threshold
    ].copy()

    if len(strong_corr) == 0:
        return pd.DataFrame()

    # Apply multiple testing correction if requested
    if use_fdr and len(strong_corr) > 1:
        # Benjamini-Hochberg FDR correction
        reject, pvals_corrected, _, _ = multipletests(
            strong_corr["p_value"], alpha=p_threshold, method="fdr_bh"
        )
        strong_corr["p_value_corrected"] = pvals_corrected
        strong_corr["significant_fdr"] = reject

        # Filter to significant after correction
        significant = strong_corr[strong_corr["significant_fdr"]].copy()
    else:
        # Simple p-value threshold
        significant = strong_corr[strong_corr["p_value"] < p_threshold].copy()
        significant["p_value_corrected"] = significant["p_value"]
        significant["significant_fdr"] = True

    return significant.sort_values("abs_correlation", ascending=False)


def calculate_correlation_confidence_intervals(
    correlation_df: pd.DataFrame, n_genotypes: int, confidence: float = 0.95
) -> pd.DataFrame:
    """Calculate confidence intervals for correlations using Fisher's z-transformation.

    Args:
        correlation_df: DataFrame with correlation results
        n_genotypes: Number of genotypes used in correlation
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        DataFrame with confidence intervals added
    """
    from scipy import stats as scipy_stats

    df = correlation_df.copy()

    # Fisher's z-transformation
    z_scores = np.arctanh(df["correlation"])

    # Standard error
    se = 1 / np.sqrt(n_genotypes - 3)

    # Z critical value
    z_crit = scipy_stats.norm.ppf((1 + confidence) / 2)

    # Confidence intervals in z-space
    z_lower = z_scores - z_crit * se
    z_upper = z_scores + z_crit * se

    # Transform back to correlation space
    df["ci_lower"] = np.tanh(z_lower)
    df["ci_upper"] = np.tanh(z_upper)
    df["ci_width"] = df["ci_upper"] - df["ci_lower"]

    return df


def summarize_correlation_results(
    correlation_df: pd.DataFrame,
    exp1_name: str = "Experiment 1",
    exp2_name: str = "Experiment 2",
) -> Dict:
    """Summarize cross-experiment correlation results.

    Args:
        correlation_df: DataFrame with correlation results
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "experiment_1": exp1_name,
        "experiment_2": exp2_name,
        "total_correlations": len(correlation_df),
        "significant_correlations": correlation_df["significant"].sum(),
        "highly_significant_correlations": correlation_df["highly_significant"].sum(),
        "max_positive_correlation": correlation_df["correlation"].max(),
        "max_negative_correlation": correlation_df["correlation"].min(),
        "mean_abs_correlation": correlation_df["abs_correlation"].mean(),
        "median_abs_correlation": correlation_df["abs_correlation"].median(),
        "n_exp1_traits": correlation_df["exp1_trait"].nunique(),
        "n_exp2_traits": correlation_df["exp2_trait"].nunique(),
    }

    # Get top correlated traits from each experiment
    top_exp1 = (
        correlation_df.groupby("exp1_trait")["abs_correlation"]
        .max()
        .nlargest(5)
        .to_dict()
    )
    top_exp2 = (
        correlation_df.groupby("exp2_trait")["abs_correlation"]
        .max()
        .nlargest(5)
        .to_dict()
    )

    summary["top_exp1_traits"] = top_exp1
    summary["top_exp2_traits"] = top_exp2

    # Get top trait pairs
    top_pairs = correlation_df.nlargest(10, "abs_correlation")[
        ["exp1_trait", "exp2_trait", "correlation"]
    ].to_dict("records")
    summary["top_trait_pairs"] = top_pairs

    return summary


def create_genotype_boxplots(
    exp1_df: pd.DataFrame,
    exp2_df: pd.DataFrame,
    exp1_trait: str,
    exp2_trait: str,
    genotype_col: str = "genotype",
    exp1_name: str = "Experiment 1",
    exp2_name: str = "Experiment 2",
    figsize: Tuple[float, float] = (14, 6),
    colors: Tuple[str, str] = ("#E8998D", "#2E6E73"),
) -> plt.Figure:
    """Create side-by-side boxplots grouped by genotype for two traits.

    Args:
        exp1_df: First experiment data with genotype and trait columns
        exp2_df: Second experiment data with genotype and trait columns
        exp1_trait: Trait name from first experiment
        exp2_trait: Trait name from second experiment
        genotype_col: Column name for genotype
        exp1_name: Name of first experiment for labeling
        exp2_name: Name of second experiment for labeling
        figsize: Figure size (width, height)
        colors: Colors for exp1 and exp2 boxplots

    Returns:
        Figure with genotype-grouped boxplots

    Raises:
        ValueError: If no common genotypes found or no data available
    """
    # Get common genotypes
    common_genotypes = sorted(
        set(exp1_df[genotype_col].unique()) & set(exp2_df[genotype_col].unique())
    )

    if len(common_genotypes) == 0:
        raise ValueError("No common genotypes found between experiments")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Prepare data for boxplots
    exp1_data = [
        exp1_df[exp1_df[genotype_col] == g][exp1_trait].dropna().values
        for g in common_genotypes
    ]
    exp2_data = [
        exp2_df[exp2_df[genotype_col] == g][exp2_trait].dropna().values
        for g in common_genotypes
    ]

    # Filter out empty groups
    valid_indices = [
        i
        for i, (d1, d2) in enumerate(zip(exp1_data, exp2_data))
        if len(d1) > 0 and len(d2) > 0
    ]

    if len(valid_indices) == 0:
        raise ValueError("No genotypes with data in both experiments")

    exp1_data = [exp1_data[i] for i in valid_indices]
    exp2_data = [exp2_data[i] for i in valid_indices]
    valid_genotypes = [common_genotypes[i] for i in valid_indices]

    # Sort genotypes by median of first experiment
    medians = [np.median(d) for d in exp1_data]
    sorted_indices = np.argsort(medians)
    exp1_data = [exp1_data[i] for i in sorted_indices]
    exp2_data = [exp2_data[i] for i in sorted_indices]
    valid_genotypes = [valid_genotypes[i] for i in sorted_indices]

    # Create boxplots for experiment 1
    bp1 = ax1.boxplot(
        exp1_data,
        tick_labels=valid_genotypes,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )

    for patch in bp1["boxes"]:
        patch.set_facecolor(colors[0])
        patch.set_alpha(0.7)

    ax1.set_xlabel("Genotype", fontsize=10)
    ax1.set_ylabel(exp1_trait[:50], fontsize=9)
    ax1.set_title(f"{exp1_name}: {exp1_trait[:60]}", fontsize=11)
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    ax1.grid(True, alpha=0.3, axis="y")

    # Add sample sizes
    sample_sizes = [len(d) for d in exp1_data]
    ax1.set_xticklabels(
        [f"{g}\n(n={n})" for g, n in zip(valid_genotypes, sample_sizes)],
        rotation=45,
        ha="right",
        fontsize=7,
    )

    # Create boxplots for experiment 2
    bp2 = ax2.boxplot(
        exp2_data,
        tick_labels=valid_genotypes,
        patch_artist=True,
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=4, alpha=0.5),
    )

    for patch in bp2["boxes"]:
        patch.set_facecolor(colors[1])
        patch.set_alpha(0.7)

    ax2.set_xlabel("Genotype", fontsize=10)
    ax2.set_ylabel(exp2_trait[:50], fontsize=9)
    ax2.set_title(f"{exp2_name}: {exp2_trait[:60]}", fontsize=11)
    ax2.tick_params(axis="x", rotation=45, labelsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add sample sizes
    sample_sizes = [len(d) for d in exp2_data]
    ax2.set_xticklabels(
        [f"{g}\n(n={n})" for g, n in zip(valid_genotypes, sample_sizes)],
        rotation=45,
        ha="right",
        fontsize=7,
    )

    plt.tight_layout()
    return fig
