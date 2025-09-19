"""Visualization utilities for trait analysis.

This module provides basic static visualization functions including:
- Trait distribution plots (histograms, boxplots)
- Correlation analysis
- PCA and UMAP visualization functions
- Extreme phenotype identification
- Publication-ready figure generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns

from datetime import datetime
import logging

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

logger = logging.getLogger(__name__)

from sleap_roots_analyze.data_cleanup import (
    apply_data_cleanup_filters,
)


def create_trait_histograms(
    df: pd.DataFrame,
    trait_cols: List[str],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Create histogram plots for all traits.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        n_cols: Number of columns in subplot grid
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    n_traits = len(trait_cols)
    if n_traits == 0:
        # Handle empty case
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No traits to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    n_rows = (n_traits + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row case
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, trait in enumerate(trait_cols):
        if trait in df.columns:
            data = df[trait].dropna()

            if len(data) > 0:
                axes[i].hist(data, bins=30, alpha=0.7, edgecolor="black")
                axes[i].set_title(f"{trait}\n(n={len(data)})", fontsize=10)
                axes[i].set_xlabel("Value")
                axes[i].set_ylabel("Frequency")
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(trait)

    # Hide empty subplots
    for i in range(n_traits, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def create_trait_boxplots_by_genotype(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    n_cols: int = 3,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Create boxplots for traits grouped by genotype.

    Args:
        df: DataFrame with trait and genotype data
        trait_cols: List of trait column names
        genotype_col: Name of genotype column
        n_cols: Number of columns in subplot grid
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    n_traits = len(trait_cols)
    if n_traits == 0:
        # Handle empty case
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.text(
            0.5,
            0.5,
            "No traits to plot",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    n_rows = (n_traits + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Handle single row case
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, trait in enumerate(trait_cols):
        if trait in df.columns and genotype_col in df.columns:
            # Create boxplot
            df_plot = df[[trait, genotype_col]].dropna()

            if len(df_plot) > 0:
                df_plot.boxplot(column=trait, by=genotype_col, ax=axes[i])
                axes[i].set_title(f"{trait}")
                axes[i].set_xlabel("Genotype")
                axes[i].set_ylabel(trait)
                plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(trait)

    # Hide empty subplots
    for i in range(n_traits, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig


def create_correlation_heatmap(
    df: pd.DataFrame, trait_cols: List[str], figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Create correlation heatmap for traits.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Calculate correlation matrix
    trait_data = df[trait_cols]
    corr_matrix = trait_data.corr()

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_title("Trait Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def save_figure_with_unique_name(
    fig: plt.Figure, run_dir: Path, base_name: str, dpi: int = 300, format: str = "png"
) -> Path:
    """Save figure with unique timestamped name to prevent overwrites.

    Args:
        fig: Matplotlib figure object
        run_dir: Directory to save the figure
        base_name: Base name for the file
        dpi: Resolution for saved plot
        format: File format (png, pdf, svg)

    Returns:
        Path to saved figure
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create unique filename with timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    filename = f"{base_name}_{timestamp}.{format}"
    plot_path = run_dir / filename

    # Ensure uniqueness even with same timestamp
    counter = 1
    while plot_path.exists():
        filename = f"{base_name}_{timestamp}_{counter:02d}.{format}"
        plot_path = run_dir / filename
        counter += 1

    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    return plot_path


def create_exploratory_summary_plots(
    df: pd.DataFrame, trait_cols: List[str], genotype_col: str = "geno"
) -> Dict[str, plt.Figure]:
    """Create comprehensive exploratory data analysis plots.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        genotype_col: Name of genotype column

    Returns:
        Dictionary of plot names to figure objects
    """
    figures = {}

    # 1. Trait distribution summary
    if len(trait_cols) > 0:
        n_traits_to_show = min(16, len(trait_cols))
        fig = create_trait_histograms(df, trait_cols[:n_traits_to_show], n_cols=4)
        figures["trait_distributions"] = fig

    # 2. Missing data heatmap
    if len(trait_cols) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        missing_data = df[trait_cols].isna()
        sns.heatmap(missing_data.T, cbar=True, ax=ax, cmap="RdYlBu_r")
        ax.set_title("Missing Data Pattern")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Traits")
        figures["missing_data_pattern"] = fig

    # 3. Trait value ranges (box plots)
    if len(trait_cols) > 0:
        n_traits_box = min(12, len(trait_cols))
        fig = create_trait_boxplots_by_genotype(
            df, trait_cols[:n_traits_box], genotype_col
        )
        figures["trait_ranges_by_genotype"] = fig

    # 4. Sample size per genotype
    if genotype_col in df.columns:
        genotype_counts = df[genotype_col].value_counts()
        if len(genotype_counts) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            genotype_counts.plot(kind="bar", ax=ax)
            ax.set_title("Sample Size per Genotype")
            ax.set_xlabel("Genotype")
            ax.set_ylabel("Number of Samples")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            figures["samples_per_genotype"] = fig

    # 5. Trait correlation overview (for subset of traits)
    if len(trait_cols) > 1:
        n_traits_corr = min(25, len(trait_cols))
        fig = create_correlation_heatmap(df, trait_cols[:n_traits_corr])
        figures["trait_correlations"] = fig

    return figures


def create_trait_eda_plots(
    df: pd.DataFrame,
    trait_cols: List[str],
    thresholds: Dict[str, float],
    cleanup_log: Optional[Dict] = None,
    min_samples_per_trait: int = 10,
) -> Dict[str, plt.Figure]:
    """Create comprehensive trait EDA plots using apply_data_cleanup_filters for consistency.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names
        thresholds: Dictionary with nan and zero thresholds (outlier ignored as it's not used for trait removal)
        cleanup_log: Optional cleanup log from apply_data_cleanup_filters with actual removed traits
        min_samples_per_trait: Minimum number of valid samples required per trait

    Returns:
        Dictionary of plot names to figure objects
    """
    figures = {}

    # Calculate EDA metrics
    eda_metrics = {
        "Trait": [],
        "Num_NaNs": [],
        "Num_Zeros": [],
        "Num_Outliers": [],
        "Variance": [],
        "Fraction_NaNs": [],
        "Fraction_Zeros": [],
        "Fraction_Outliers": [],
    }

    for col in trait_cols:
        if col in df.columns:
            # Count NaNs and zeros
            n_nans = df[col].isna().sum()
            n_zeros = (df[col] == 0).sum()

            # Count outliers using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            n_outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

            # Calculate variance
            variance = df[col].var()

            # Store metrics
            eda_metrics["Trait"].append(col)
            eda_metrics["Num_NaNs"].append(n_nans)
            eda_metrics["Num_Zeros"].append(n_zeros)
            eda_metrics["Num_Outliers"].append(n_outliers)
            eda_metrics["Variance"].append(variance)
            eda_metrics["Fraction_NaNs"].append(n_nans / len(df))
            eda_metrics["Fraction_Zeros"].append(n_zeros / len(df))
            eda_metrics["Fraction_Outliers"].append(n_outliers / len(df))

    eda_df = pd.DataFrame(eda_metrics)

    # Add trait prefix for grouping
    eda_df["Prefix"] = eda_df["Trait"].apply(
        lambda x: x.split("_")[0] if "_" in x else "NoPrefix"
    )

    # 1. Trait overview plot (similar to plot_eda_summary)
    fig, axes = plt.subplots(3, 1, figsize=(18, 14), sharex=True)

    # NaN fraction
    sns.barplot(x="Trait", y="Fraction_NaNs", hue="Prefix", data=eda_df, ax=axes[0])
    axes[0].axhline(
        y=thresholds.get("nan", 0.3),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({thresholds.get('nan', 0.3)})",
    )
    axes[0].set_title("Fraction of NaN Values per Trait")
    axes[0].tick_params(labelbottom=False)
    axes[0].legend()

    # Zero fraction
    sns.barplot(x="Trait", y="Fraction_Zeros", hue="Prefix", data=eda_df, ax=axes[1])
    axes[1].axhline(
        y=thresholds.get("zero", 0.5),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({thresholds.get('zero', 0.5)})",
    )
    axes[1].set_title("Fraction of Zero Values per Trait")
    axes[1].tick_params(labelbottom=False)
    axes[1].legend()

    # Outlier fraction
    sns.barplot(x="Trait", y="Fraction_Outliers", hue="Prefix", data=eda_df, ax=axes[2])
    axes[2].axhline(
        y=thresholds.get("outlier", 0.1),
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({thresholds.get('outlier', 0.1)})",
    )
    axes[2].set_title("Fraction of IQR Outliers per Trait")
    axes[2].set_xlabel("Trait")
    axes[2].tick_params(axis="x", rotation=90)
    axes[2].legend(title="Prefix", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    figures["trait_eda_overview"] = fig

    # 2. Traits Actually Removed (if cleanup_log provided)
    # Use actual removed traits from cleanup log if available
    actual_removed_traits = []
    removal_reasons_dict = {}

    if cleanup_log and "removed_traits" in cleanup_log:
        for trait_info in cleanup_log["removed_traits"]:
            if isinstance(trait_info, dict):
                trait_name = trait_info.get("trait", "")
                if trait_name:
                    actual_removed_traits.append(trait_name)
                    # Get the actual removal reason from the cleanup log
                    reason = trait_info.get("reason", "Unknown")
                    if reason == "too_many_zeros":
                        removal_reasons_dict[trait_name] = (
                            f"High Zeros ({trait_info.get('zero_fraction', 0):.2%})"
                        )
                    elif reason == "too_many_nans":
                        removal_reasons_dict[trait_name] = (
                            f"High NaNs ({trait_info.get('nan_fraction', 0):.2%})"
                        )
                    elif reason == "insufficient_samples":
                        removal_reasons_dict[trait_name] = (
                            f"Insufficient samples ({trait_info.get('valid_samples', 0)})"
                        )
                    else:
                        removal_reasons_dict[trait_name] = reason

    # If we have actually removed traits, show them
    if actual_removed_traits:
        fig, ax = plt.subplots(figsize=(12, max(8, len(actual_removed_traits) * 0.4)))

        # Filter eda_df to only include actually removed traits
        removed_df = eda_df[eda_df["Trait"].isin(actual_removed_traits)].copy()

        # Add the actual removal reasons
        removed_df["Removal_Reason"] = removed_df["Trait"].map(removal_reasons_dict)

        # Plot (only NaN and Zero fractions since outliers don't affect trait removal)
        y_pos = np.arange(len(removed_df))
        ax.barh(y_pos, removed_df["Fraction_NaNs"], label="NaN Fraction", alpha=0.7)
        ax.barh(
            y_pos,
            removed_df["Fraction_Zeros"],
            left=removed_df["Fraction_NaNs"],
            label="Zero Fraction",
            alpha=0.7,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(removed_df["Trait"])
        ax.set_xlabel("Fraction")
        ax.set_title(f"Traits Actually Removed ({len(actual_removed_traits)} traits)")
        ax.legend()

        # Add actual removal reasons as text
        for i, (idx, row) in enumerate(removed_df.iterrows()):
            ax.text(1.02, i, row["Removal_Reason"], va="center", fontsize=8)

        plt.tight_layout()
        figures["traits_actually_removed"] = fig

    # If no cleanup_log provided, use apply_data_cleanup_filters to determine what WOULD be removed
    # This ensures consistency with the actual pipeline behavior
    if not cleanup_log:
        # Run apply_data_cleanup_filters to see what would be removed
        _, simulated_log = apply_data_cleanup_filters(
            df.copy(),
            trait_cols,
            max_zeros_per_trait=thresholds.get("zero", 0.5),
            max_nans_per_trait=thresholds.get("nan", 0.3),
            min_samples_per_trait=min_samples_per_trait,
        )

        # Extract traits that would be removed
        hypothetical_removals = []
        hypothetical_reasons = {}
        for trait_info in simulated_log.get("removed_traits", []):
            if isinstance(trait_info, dict):
                trait_name = trait_info.get("trait", "")
                if trait_name:
                    hypothetical_removals.append(trait_name)
                    reason = trait_info.get("reason", "Unknown")
                    if reason == "too_many_zeros":
                        hypothetical_reasons[trait_name] = (
                            f"High Zeros ({trait_info.get('zero_fraction', 0):.2%})"
                        )
                    elif reason == "too_many_nans":
                        hypothetical_reasons[trait_name] = (
                            f"High NaNs ({trait_info.get('nan_fraction', 0):.2%})"
                        )
                    elif reason == "insufficient_samples":
                        hypothetical_reasons[trait_name] = (
                            f"Insufficient samples ({trait_info.get('valid_samples', 0)})"
                        )
                    else:
                        hypothetical_reasons[trait_name] = reason
    else:
        hypothetical_removals = []
        hypothetical_reasons = {}

    # Show traits that would be removed but weren't (shouldn't happen if cleanup_log is from same parameters)
    traits_exceeding_thresholds = [
        t for t in hypothetical_removals if t not in actual_removed_traits
    ]

    if traits_exceeding_thresholds:
        fig, ax = plt.subplots(
            figsize=(12, max(8, len(traits_exceeding_thresholds) * 0.4))
        )
        exceed_df = eda_df[eda_df["Trait"].isin(traits_exceeding_thresholds)].copy()

        # Add the simulated removal reasons
        exceed_df["Threshold_Exceeded"] = exceed_df["Trait"].map(hypothetical_reasons)

        # Plot (only NaN and Zero fractions since outliers don't affect trait removal)
        y_pos = np.arange(len(exceed_df))
        ax.barh(y_pos, exceed_df["Fraction_NaNs"], label="NaN Fraction", alpha=0.7)
        ax.barh(
            y_pos,
            exceed_df["Fraction_Zeros"],
            left=exceed_df["Fraction_NaNs"],
            label="Zero Fraction",
            alpha=0.7,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(exceed_df["Trait"])
        ax.set_xlabel("Fraction")
        ax.set_title(
            f"Traits Exceeding Cleanup Thresholds But Not Removed ({len(traits_exceeding_thresholds)} traits)"
        )
        ax.legend()

        # Add threshold info as text
        for i, (idx, row) in enumerate(exceed_df.iterrows()):
            ax.text(1.02, i, row["Threshold_Exceeded"], va="center", fontsize=8)

        plt.tight_layout()
        figures["traits_exceeding_thresholds"] = fig

    # 3. Variance distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_vars = eda_df[eda_df["Variance"] > 0]["Variance"]
    if len(valid_vars) > 0:
        ax.hist(np.log10(valid_vars + 1e-10), bins=30, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Log10(Variance)")
        ax.set_ylabel("Number of Traits")
        ax.set_title("Distribution of Trait Variances (log scale)")
    plt.tight_layout()
    figures["variance_distribution"] = fig

    return figures


def create_heritability_plot(
    heritability_results: Dict,
    threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Create bar plot of heritability estimates.

    Args:
        heritability_results: Results from heritability analysis
        threshold: Threshold line for high heritability
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    # Extract valid heritability values
    traits = []
    h2_values = []

    for trait, results in heritability_results.items():
        if isinstance(results, dict) and "heritability" in results:
            h2_value = results["heritability"]
            # Skip None or invalid values
            if (
                h2_value is not None
                and isinstance(h2_value, (int, float))
                and 0 <= h2_value <= 1
            ):
                traits.append(trait)
                h2_values.append(h2_value)

    if not traits:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No heritability data available", ha="center", va="center")
        ax.set_title("Heritability Estimates")
        return fig

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Color bars based on threshold
    colors = ["green" if h2 >= threshold else "orange" for h2 in h2_values]

    bars = ax.bar(range(len(traits)), h2_values, color=colors, alpha=0.7)

    # Add threshold line
    ax.axhline(
        y=threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold ({threshold})",
    )

    # Customize plot
    ax.set_xlabel("Traits")
    ax.set_ylabel("Heritability (H²)")
    ax.set_title("Broad-sense Heritability Estimates")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, h2 in zip(bars, h2_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{h2:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def create_heritability_threshold_plot(
    threshold_analysis: Dict[str, np.ndarray],
    current_threshold: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create plot showing trait retention vs heritability threshold.

    Args:
        threshold_analysis: Results from analyze_heritability_thresholds
        current_threshold: Current threshold to highlight (optional)
        figsize: Figure size

    Returns:
        Figure with threshold analysis plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])

    thresholds = threshold_analysis["thresholds"]
    traits_retained = threshold_analysis["traits_retained"]
    fraction_retained = threshold_analysis["fraction_retained"]
    total_traits = threshold_analysis["total_traits"]

    # Top plot: Number of traits retained
    ax1.plot(thresholds, traits_retained, "b-", linewidth=2, label="Traits retained")
    ax1.fill_between(thresholds, 0, traits_retained, alpha=0.3, color="blue")

    # Add reference lines
    ax1.axhline(
        y=total_traits * 0.5,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="50% of traits",
    )
    ax1.axhline(
        y=total_traits * 0.75,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="75% of traits",
    )

    # Highlight current threshold
    if current_threshold is not None:
        idx = np.argmin(np.abs(thresholds - current_threshold))
        ax1.axvline(
            x=current_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Current: {current_threshold}",
        )
        ax1.plot(current_threshold, traits_retained[idx], "ro", markersize=8)
        ax1.text(
            current_threshold + 0.02,
            traits_retained[idx],
            f"{int(traits_retained[idx])} traits",
            verticalalignment="center",
        )

    ax1.set_ylabel("Number of Traits Retained", fontsize=12)
    ax1.set_title("Trait Retention vs Heritability Threshold", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, total_traits * 1.05)

    # Bottom plot: Fraction retained
    ax2.plot(thresholds, fraction_retained * 100, "g-", linewidth=2)
    ax2.fill_between(thresholds, 0, fraction_retained * 100, alpha=0.3, color="green")

    if current_threshold is not None:
        ax2.axvline(x=current_threshold, color="red", linestyle="--", alpha=0.7)
        idx = np.argmin(np.abs(thresholds - current_threshold))
        ax2.plot(current_threshold, fraction_retained[idx] * 100, "ro", markersize=8)
        ax2.text(
            current_threshold + 0.02,
            fraction_retained[idx] * 100,
            f"{fraction_retained[idx]*100:.1f}%",
            verticalalignment="center",
        )

    ax2.set_xlabel("Heritability Threshold (H²)", fontsize=12)
    ax2.set_ylabel("Traits Retained (%)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 105)

    # Add some common threshold annotations
    for threshold, label in [(0.3, "Low"), (0.5, "Moderate"), (0.7, "High")]:
        ax2.axvline(x=threshold, color="gray", linestyle=":", alpha=0.3)
        ax2.text(threshold, 102, label, ha="center", fontsize=9, alpha=0.7)

    plt.tight_layout()
    return fig


def identify_extreme_samples_in_pc_space(
    pca_results: Dict,
    df: pd.DataFrame,
    n_components: int = 3,
    n_std: float = 2.0,
    sample_id_col: str = "Barcode",
) -> pd.DataFrame:
    """Identify samples with extreme values in PC space.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with sample metadata.
        n_components: Number of PC components to consider.
        n_std: Number of standard deviations to consider extreme.
        sample_id_col: Column name containing sample IDs.

    Returns:
        DataFrame with extreme samples and their PC scores.
    """
    X_pca = pca_results["transformed_data"]
    n_components = min(n_components, X_pca.shape[1])

    extreme_samples = []

    # Check each PC individually
    for pc_idx in range(n_components):
        pc_scores = X_pca[:, pc_idx]
        pc_mean = np.mean(pc_scores)
        pc_std = np.std(pc_scores, ddof=1)

        # Guard against zero/near-zero std
        if pc_std <= 0:
            continue

        # Find extreme samples (both high and low)
        extreme_mask = np.abs(pc_scores - pc_mean) > n_std * pc_std
        extreme_indices = np.where(extreme_mask)[0]

        for idx in extreme_indices:
            sample_id = df.iloc[idx][sample_id_col]
            z_score = (pc_scores[idx] - pc_mean) / pc_std
            extreme_type = "high" if pc_scores[idx] > pc_mean else "low"

            extreme_samples.append(
                {
                    sample_id_col: sample_id,
                    "pc_component": f"PC{pc_idx + 1}",
                    "pc_score": pc_scores[idx],
                    "z_score": z_score,
                    "extreme_type": extreme_type,
                    "explained_variance_ratio": pca_results["explained_variance_ratio"][
                        pc_idx
                    ],
                }
            )

    # Optional: Check for samples extreme in multiple PCs using Hotelling's T²
    # This is less interpretable but provides a global outlier statistic
    if n_components > 1 and "eigenvalues" in pca_results:
        # Calculate Hotelling's T² statistic (variance-standardized radius)
        # T² = sum((score_k)² / eigenvalue_k)
        eigenvalues = np.asarray(pca_results["eigenvalues"][:n_components])
        eigenvalues = np.maximum(eigenvalues, 1e-12)  # Protect against tiny eigenvalues
        t_squared = np.sum((X_pca[:, :n_components] ** 2) / eigenvalues, axis=1)

        # For multivariate normal, T² follows chi-square distribution
        # Use chi-square critical value for threshold
        from scipy.stats import chi2

        chi2_threshold = chi2.ppf(1 - 0.05, df=n_components)  # 95% confidence

        extreme_t2_mask = t_squared > chi2_threshold
        extreme_t2_indices = np.where(extreme_t2_mask)[0]

        # Calculate empirical z-score on radius for better interpretability
        r = np.sqrt(t_squared)
        r_mean = float(np.mean(r))
        r_std = float(np.std(r, ddof=1) or 1e-12)  # Guard against zero std

        for idx in extreme_t2_indices:
            sample_id = df.iloc[idx][sample_id_col]
            # Use empirical z-score on radius
            z_score = (r[idx] - r_mean) / r_std

            extreme_samples.append(
                {
                    sample_id_col: sample_id,
                    "pc_component": "Hotelling T²",
                    "pc_score": t_squared[idx],
                    "z_score": z_score,
                    "extreme_type": "multi-pc",
                    "explained_variance_ratio": np.sum(
                        pca_results["explained_variance_ratio"][:n_components]
                    ),
                }
            )

    extreme_df = pd.DataFrame(extreme_samples)

    # Remove duplicates, keeping the most extreme case (by absolute z-score)
    if not extreme_df.empty:
        extreme_df["abs_z_score"] = extreme_df["z_score"].abs()
        extreme_df = extreme_df.sort_values("abs_z_score", ascending=False)
        extreme_df = extreme_df.drop_duplicates(subset=[sample_id_col], keep="first")
        extreme_df = extreme_df.drop(columns=["abs_z_score"])

    return extreme_df


def create_pca_scree_plot(
    pca_results: Dict,
    variance_threshold: float = 0.95,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Create a combined scree plot with variance threshold indicator.

    Args:
        pca_results: Results from perform_pca_analysis.
        variance_threshold: Cumulative variance threshold to highlight.
        figsize: Figure size.

    Returns:
        Enhanced scree plot with combined individual and cumulative variance.
    """
    fig, ax = plt.subplots(figsize=figsize)

    explained_var = pca_results["explained_variance_ratio"]
    cumulative_var = pca_results["cumulative_variance_ratio"]
    n_components = len(explained_var)

    # Find the number of components for threshold
    n_threshold = np.argmax(cumulative_var >= variance_threshold) + 1

    # Individual variance bars with color coding
    colors = [
        "darkblue" if i < n_threshold else "lightblue" for i in range(n_components)
    ]
    bars = ax.bar(
        range(1, n_components + 1),
        explained_var * 100,
        color=colors,
        alpha=0.7,
        label="Individual variance",
    )

    # Add percentage labels on bars
    for i, (bar, var) in enumerate(zip(bars, explained_var)):
        if i < 10:  # Only label first 10 for readability
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{var * 100:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Cumulative variance line on same axis
    ax.plot(
        range(1, n_components + 1),
        cumulative_var * 100,
        color="darkred",
        marker="o",
        markersize=4,
        linewidth=2,
        label="Cumulative variance",
    )

    # Add threshold line
    ax.axhline(
        y=variance_threshold * 100,
        color="green",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"{variance_threshold * 100:.0f}% threshold",
    )

    # Find and mark threshold crossing point
    if n_threshold <= n_components:
        ax.plot(
            n_threshold,
            cumulative_var[n_threshold - 1] * 100,
            "o",
            color="green",
            markersize=8,
            zorder=5,
        )

        # Add vertical line at threshold
        ax.axvline(
            x=n_threshold,
            color="green",
            linestyle=":",
            alpha=0.5,
        )

        # Add simple text annotation with exact variance
        ax.text(
            n_threshold,
            cumulative_var[n_threshold - 1] * 100 + 2,
            f"{cumulative_var[n_threshold - 1] * 100:.1f}%",
            ha="center",
            color="green",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_xticks(range(1, min(21, n_components + 1)))
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)

    # Custom legend explaining the color coding
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="darkblue",
            alpha=0.7,
            label=f"First {n_threshold} PCs (meets threshold)",
        ),
        Line2D(
            [0], [0], color="darkred", lw=2, marker="o", label="Cumulative variance"
        ),
        Line2D(
            [0],
            [0],
            color="green",
            lw=2,
            linestyle="--",
            label=f"{variance_threshold * 100:.0f}% threshold",
        ),
    ]
    # Only add light blue legend if there are actually visible bars beyond threshold
    if n_threshold < n_components and any(
        explained_var[n_threshold:] * 100 > 1
    ):  # Only if some remaining PCs > 1%
        legend_elements.insert(
            1,
            Patch(
                facecolor="lightblue", alpha=0.7, label=f"Remaining PCs (low variance)"
            ),
        )

    ax.legend(handles=legend_elements, loc="center right", framealpha=0.9)

    # Title with feature count
    n_features = pca_results.get(
        "n_features", len(pca_results.get("feature_names", []))
    )
    title = f"Enhanced PCA Variance Analysis"
    if n_features:
        title += f" (Total features: {n_features})"
    plt.title(title, fontsize=14, pad=20)

    plt.tight_layout()
    return fig


def create_feature_contribution_plot(
    pca_results: Dict,
    trait_names: List[str],
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    top_n: int = 20,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Create a plot showing feature contributions across selected PCs.

    This function can use pre-calculated contributions from run_pca_and_export_artifacts
    (if available in pca_results["trait_contrib_df"]) or calculate them on the fly.

    Args:
        pca_results: Results from perform_pca_analysis or run_pca_and_export_artifacts.
            If it contains "trait_contrib_df", those pre-calculated contributions will be used.
        trait_names: Names of traits/features.
        n_components: Number of PCs to consider. If None, use variance threshold.
        variance_threshold: Cumulative variance threshold for PC selection.
        top_n: Number of top contributing features to show.
        figsize: Figure size.

    Returns:
        Feature contribution plot.
    """
    # Check if pre-calculated contributions are available
    if (
        "trait_contrib_df" in pca_results
        or "feature_importance_consistent" in pca_results
    ):
        # Use pre-calculated contributions
        trait_contrib_df = pca_results.get(
            "trait_contrib_df", pca_results.get("feature_importance_consistent")
        )

        # Determine number of components
        if n_components is None:
            cumulative_var = pca_results["cumulative_variance_ratio"]
            n_components = np.argmax(cumulative_var >= variance_threshold) + 1

        # Get available PC columns from the DataFrame
        pc_contrib_cols = [
            col
            for col in trait_contrib_df.columns
            if col.startswith("PC") and col.endswith("_variance_contrib")
        ]
        available_pcs = min(len(pc_contrib_cols), n_components)

        # Use the first n_components PCs
        pc_cols_to_use = [f"PC{i+1}_variance_contrib" for i in range(available_pcs)]

        # Get top contributors (already sorted in trait_contrib_df)
        top_features_df = trait_contrib_df.head(min(top_n, len(trait_contrib_df)))

        # Extract data for plotting
        top_traits = top_features_df["trait"].tolist()
        contributions = top_features_df[pc_cols_to_use].values
        total_contributions = top_features_df["trait_total_variance_contrib"].values

    else:
        # Calculate contributions on the fly (backward compatibility)
        # Determine number of components
        if n_components is None:
            cumulative_var = pca_results["cumulative_variance_ratio"]
            n_components = np.argmax(cumulative_var >= variance_threshold) + 1

        n_components = min(
            n_components,
            pca_results.get(
                "n_components_selected", len(pca_results["explained_variance_ratio"])
            ),
        )

        # Calculate variance-weighted contributions
        loadings = pca_results["loadings"][:, :n_components]
        eigenvalues = pca_results["eigenvalues"][:n_components]

        # Calculate contribution of each feature to each PC
        contributions = np.zeros((len(trait_names), n_components))
        for i in range(n_components):
            contributions[:, i] = eigenvalues[i] * loadings[:, i] ** 2

        # Sum across selected PCs
        total_contributions = np.sum(contributions, axis=1)

        # Sort features by total contribution
        sorted_indices = np.argsort(total_contributions)[::-1]

        # Limit to available features
        actual_top_n = min(top_n, len(trait_names))
        top_indices = sorted_indices[:actual_top_n]

        # Get top features data
        top_traits = [trait_names[i] for i in top_indices]
        contributions = contributions[top_indices]
        total_contributions = total_contributions[top_indices]
        available_pcs = n_components

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create stacked bar chart
    # Reverse the order so highest contribution is at top
    top_traits = top_traits[::-1]
    contributions = contributions[::-1]
    total_contributions = total_contributions[::-1]
    y_pos = np.arange(len(top_traits))

    # Colors for different PCs
    colors = plt.cm.tab20(np.linspace(0, 1, available_pcs))

    # Plot stacked bars
    left = np.zeros(len(top_traits))
    for i in range(available_pcs):
        ax.barh(
            y_pos,
            contributions[:, i],
            left=left,
            label=f"PC{i+1}",
            color=colors[i],
            edgecolor="black",
            linewidth=0.5,
        )
        left += contributions[:, i]

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_traits)
    ax.set_xlabel("Variance Contribution", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title(
        f"Top {len(top_traits)} Feature Contributions to First {available_pcs} PCs",
        fontsize=14,
    )

    # Add legend
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        ncol=min(available_pcs, 5),
        fontsize=10,
    )

    # Add grid
    ax.grid(True, axis="x", alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    return fig


def create_pca_biplot(
    pca_results: Dict,
    df: pd.DataFrame,
    trait_names: List[str],
    color_by: Optional[str] = None,
    pc_x: int = 1,
    pc_y: int = 2,
    top_n_features: int = 10,
    figsize: Tuple[float, float] = (10, 8),
    alpha: float = 0.6,
    arrow_scale: float = 4.0,
) -> plt.Figure:
    """Create a decluttered PCA biplot showing only top contributing features.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with metadata for coloring.
        trait_names: Names of traits/features.
        color_by: Column name to color points by.
        pc_x: PC for x-axis (1-indexed).
        pc_y: PC for y-axis (1-indexed).
        top_n_features: Number of top contributing features to show.
        figsize: Figure size.
        alpha: Transparency for scatter points.
        arrow_scale: Scaling factor for feature arrows.

    Returns:
        Decluttered biplot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    X_pca = pca_results["transformed_data"]
    loadings = pca_results["loadings"]
    explained_var = pca_results["explained_variance_ratio"]

    # Calculate variance-weighted contributions for feature selection
    eigenvalues = pca_results["eigenvalues"]
    pc_indices = [pc_x - 1, pc_y - 1]

    # Ensure we handle the correct number of features
    n_features = min(len(trait_names), loadings.shape[0])
    contributions = np.zeros(n_features)

    for pc_idx in pc_indices:
        contributions += eigenvalues[pc_idx] * loadings[:n_features, pc_idx] ** 2

    # Get top contributing features
    top_indices = np.argsort(contributions)[::-1][:top_n_features]

    # Plot samples
    if color_by and color_by in df.columns:
        # Handle categorical coloring
        if df[color_by].dtype == "object" or isinstance(
            df[color_by].dtype, pd.CategoricalDtype
        ):
            # Get unique categories
            categories = df[color_by].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

            # Plot each category separately
            for i, cat in enumerate(categories):
                mask = df[color_by] == cat
                ax.scatter(
                    X_pca[mask, pc_x - 1],
                    X_pca[mask, pc_y - 1],
                    c=[colors[i]],
                    label=cat,
                    alpha=alpha,
                    s=50,
                    edgecolors="none",
                )
            ax.legend(title=color_by, bbox_to_anchor=(1.05, 1), loc="upper left")
        else:
            # Numeric coloring
            scatter = ax.scatter(
                X_pca[:, pc_x - 1],
                X_pca[:, pc_y - 1],
                c=df[color_by],
                alpha=alpha,
                s=50,
                edgecolors="none",
                cmap="viridis",
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
    else:
        # No coloring
        ax.scatter(
            X_pca[:, pc_x - 1],
            X_pca[:, pc_y - 1],
            c="blue",
            alpha=alpha,
            s=50,
            edgecolors="none",
        )

    # Plot feature vectors (only top contributors)
    for idx in top_indices:
        # Skip if index is out of bounds for trait_names
        if idx >= len(trait_names):
            continue
        # Scale loadings by explained variance for visibility
        x_load = loadings[idx, pc_x - 1] * arrow_scale
        y_load = loadings[idx, pc_y - 1] * arrow_scale

        # Draw arrow
        ax.arrow(
            0,
            0,
            x_load,
            y_load,
            head_width=0.05,
            head_length=0.05,
            fc="red",
            ec="red",
            alpha=0.8,
            linewidth=1.5,
        )

        # Add label with smart positioning to avoid overlaps
        angle = np.arctan2(y_load, x_load)

        # Create a pattern of offsets to spread out labels
        idx_pos = list(top_indices).index(idx)
        offsets = [-0.3, 0, 0.3, -0.2, 0.2, -0.4, 0.1, 0.4, -0.1, 0.35]
        angle_offset = offsets[idx_pos % len(offsets)]

        # Vary radius to create layers
        radius_mult = 1.15 + (idx_pos % 4) * 0.1
        radius = np.sqrt(x_load**2 + y_load**2) * radius_mult

        label_x = radius * np.cos(angle + angle_offset)
        label_y = radius * np.sin(angle + angle_offset)

        # Adjust text position to avoid overlap with arrow
        ha = "left" if label_x > 0 else "right"
        va = "bottom" if label_y > 0 else "top"

        ax.text(
            label_x,
            label_y,
            trait_names[idx],
            fontsize=9,
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        )

    # Set axis labels and title
    ax.set_xlabel(f"PC{pc_x} ({explained_var[pc_x - 1] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC{pc_y} ({explained_var[pc_y - 1] * 100:.1f}% variance)")
    ax.set_title(
        f"PCA Biplot - Top {top_n_features} Contributing Features", fontsize=14
    )

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Set equal aspect ratio for better interpretation
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    return fig


def create_umap_colored_by_top_traits(
    umap_results: np.ndarray,
    df: pd.DataFrame,
    trait_columns: List[str],
    trait_names: List[str],
    pca_results: Dict,
    n_traits: int = 6,
    variance_threshold: Optional[float] = None,
    figsize: Tuple[float, float] = (15, 10),
) -> plt.Figure:
    """Create UMAP plots colored by top contributing traits.

    Args:
        umap_results: 2D UMAP embedding.
        df: Original dataframe with trait values.
        trait_columns: Column names of traits.
        trait_names: Display names of traits.
        pca_results: PCA results for determining trait importance.
        n_traits: Number of top traits to plot.
        variance_threshold: Cumulative variance threshold for PC selection.
            If None, use the same threshold as perform_pca_analysis.
        figsize: Figure size.

    Returns:
        UMAP plots colored by traits.
    """
    # Calculate trait contributions
    loadings = pca_results["loadings"]
    eigenvalues = pca_results["eigenvalues"]

    # Use the same PCA threshold/components as perform_pca_analysis by default
    cumulative_var = pca_results["cumulative_variance_ratio"]

    # Check if pca_results has the selected components info
    if "n_components_selected" in pca_results:
        n_pcs = pca_results["n_components_selected"]
    elif variance_threshold is not None:
        # Use provided threshold
        n_pcs = np.argmax(cumulative_var >= variance_threshold) + 1
    else:
        # Default to 95% variance if not specified
        n_pcs = np.argmax(cumulative_var >= 0.95) + 1

    # Clamp n_pcs to available data
    n_pcs = min(n_pcs, loadings.shape[1], len(eigenvalues))

    # Ensure we handle the correct number of features
    n_features = min(len(trait_columns), loadings.shape[0])
    contributions = np.zeros(n_features)
    for i in range(n_pcs):
        contributions += eigenvalues[i] * loadings[:n_features, i] ** 2

    # Get top contributing traits
    top_indices = np.argsort(contributions)[::-1][:n_traits]

    # Create subplots
    n_cols = 3
    n_rows = (n_traits + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else axes

    for i, trait_idx in enumerate(top_indices):
        ax = axes[i]
        # Skip if index is out of bounds
        if trait_idx >= len(trait_columns) or trait_idx >= len(trait_names):
            continue
        trait_col = trait_columns[trait_idx]
        trait_name = trait_names[trait_idx]
        trait_values = df[trait_col].values

        # Create scatter plot
        scatter = ax.scatter(
            umap_results[:, 0],
            umap_results[:, 1],
            c=trait_values,
            cmap="viridis",
            s=30,
            alpha=0.7,
            edgecolors="none",
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(trait_name, fontsize=9)
        cbar.ax.tick_params(labelsize=8)

        # Set labels and title
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_title(
            f"{trait_name}\n(Contribution: {contributions[trait_idx]:.3f})",
            fontsize=10,
        )
        ax.tick_params(labelsize=8)

    # Remove empty subplots
    for i in range(n_traits, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle(f"UMAP Colored by Top {n_traits} Contributing Traits", fontsize=14)
    plt.tight_layout()

    return fig


def identify_extreme_genotypes_by_pc(
    pca_results: Dict,
    df: pd.DataFrame,
    genotype_col: str = "geno",
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    n_extreme: int = 3,
) -> pd.DataFrame:
    """Identify extreme genotypes based on their median PC scores.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with genotype information.
        genotype_col: Column name containing genotype information.
        n_components: Number of PCs to consider. If None, use variance threshold.
        variance_threshold: Cumulative variance threshold for PC selection.
        n_extreme: Number of extreme genotypes to identify per PC (both high and low).

    Returns:
        DataFrame with extreme genotypes, their median PC scores, and rankings.
    """
    if genotype_col not in df.columns:
        raise ValueError(f"Genotype column '{genotype_col}' not found in dataframe")

    X_pca = pca_results["transformed_data"]

    # Determine number of components
    if n_components is None:
        cumulative_var = pca_results["cumulative_variance_ratio"]
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1

    n_components = min(n_components, X_pca.shape[1])

    # Create dataframe with PC scores and genotypes
    pc_df = pd.DataFrame()
    pc_df[genotype_col] = df[genotype_col]

    for i in range(n_components):
        pc_df[f"PC{i+1}"] = X_pca[:, i]

    # Calculate median PC scores by genotype
    median_scores = pc_df.groupby(genotype_col).median()
    counts = pc_df.groupby(genotype_col).size()

    extreme_genotypes = []

    # For each PC, identify extreme genotypes
    for i in range(n_components):
        pc_col = f"PC{i+1}"
        pc_medians = median_scores[pc_col].sort_values()

        # Get n_extreme lowest and highest genotypes
        low_genotypes = pc_medians.head(n_extreme)
        high_genotypes = pc_medians.tail(n_extreme)

        # Add low extremes
        for rank, (geno, median_score) in enumerate(low_genotypes.items(), 1):
            extreme_genotypes.append(
                {
                    genotype_col: geno,
                    "pc_component": pc_col,
                    "median_pc_score": median_score,
                    "direction": "low",
                    "rank": rank,
                    "n_samples": counts[geno],
                    "explained_variance_ratio": pca_results["explained_variance_ratio"][
                        i
                    ],
                }
            )

        # Add high extremes
        for rank, (geno, median_score) in enumerate(
            high_genotypes.iloc[::-1].items(), 1
        ):
            extreme_genotypes.append(
                {
                    genotype_col: geno,
                    "pc_component": pc_col,
                    "median_pc_score": median_score,
                    "direction": "high",
                    "rank": rank,
                    "n_samples": counts[geno],
                    "explained_variance_ratio": pca_results["explained_variance_ratio"][
                        i
                    ],
                }
            )

    return pd.DataFrame(extreme_genotypes)


def create_pc_genotype_boxplots(
    pca_results: Dict,
    df: pd.DataFrame,
    genotype_col: str = "geno",
    n_components: Optional[int] = None,
    variance_threshold: float = 0.95,
    highlight_extreme: int = 3,
    figsize: Tuple[float, float] = (16, 10),
) -> plt.Figure:
    """Create boxplots showing PC score distributions by genotype.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with genotype information.
        genotype_col: Column name containing genotype information.
        n_components: Number of PCs to plot. If None, use variance threshold.
        variance_threshold: Cumulative variance threshold for PC selection.
        highlight_extreme: Number of extreme genotypes to highlight per PC.
        figsize: Figure size.

    Returns:
        Boxplot figure.
    """
    if genotype_col not in df.columns:
        raise ValueError(f"Genotype column '{genotype_col}' not found in dataframe")

    X_pca = pca_results["transformed_data"]

    # Determine number of components
    if n_components is None:
        cumulative_var = pca_results["cumulative_variance_ratio"]
        n_components = np.argmax(cumulative_var >= variance_threshold) + 1

    n_components = min(
        n_components, X_pca.shape[1]
    )  # Use exact number from variance threshold

    # Create dataframe with PC scores and genotypes
    pc_df = pd.DataFrame()
    pc_df[genotype_col] = df[genotype_col]

    for i in range(n_components):
        pc_df[f"PC{i+1}"] = X_pca[:, i]

    # Create subplots
    n_cols = min(3, n_components)
    n_rows = (n_components + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_components == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    # Get extreme genotypes
    extreme_df = identify_extreme_genotypes_by_pc(
        pca_results,
        df,
        genotype_col,
        n_components,
        variance_threshold,
        highlight_extreme,
    )

    for i in range(n_components):
        ax = axes[i] if n_components > 1 else axes[0]
        pc_col = f"PC{i+1}"

        # Get data for this PC
        pc_data = []
        labels = []
        colors = []

        # Get extreme genotypes for this PC
        pc_extremes = extreme_df[extreme_df["pc_component"] == pc_col]
        extreme_genos_high = set(
            pc_extremes[pc_extremes["direction"] == "high"][genotype_col]
        )
        extreme_genos_low = set(
            pc_extremes[pc_extremes["direction"] == "low"][genotype_col]
        )

        # Sort genotypes by median PC score
        genotype_medians = pc_df.groupby(genotype_col)[pc_col].median().sort_values()

        for geno in genotype_medians.index:
            geno_data = pc_df[pc_df[genotype_col] == geno][pc_col].values
            pc_data.append(geno_data)
            labels.append(geno)

            # Color extreme genotypes
            if geno in extreme_genos_high:
                colors.append("darkred")
            elif geno in extreme_genos_low:
                colors.append("darkblue")
            else:
                colors.append("gray")

        # Create boxplot (use tick_labels for newer matplotlib)
        try:
            bp = ax.boxplot(pc_data, tick_labels=labels, patch_artist=True)
        except TypeError:
            # Fallback for older matplotlib versions
            bp = ax.boxplot(pc_data, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # Styling
        ax.set_ylabel(f"{pc_col} Score")
        ax.set_xlabel("Genotype")
        ax.set_title(
            f'{pc_col} ({pca_results["explained_variance_ratio"][i]:.1%} variance)',
            fontsize=12,
        )
        ax.tick_params(axis="x", rotation=90)
        ax.grid(axis="y", alpha=0.3)

        # Add legend for first subplot
        if i == 0:
            legend_elements = [
                Patch(facecolor="darkred", alpha=0.6, label=f"Top {highlight_extreme}"),
                Patch(
                    facecolor="darkblue", alpha=0.6, label=f"Bottom {highlight_extreme}"
                ),
                Patch(facecolor="gray", alpha=0.6, label="Other"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    # Remove empty subplots
    for i in range(n_components, len(axes)):
        fig.delaxes(axes[i])

    # Main title
    fig.suptitle(
        f"PC Score Distributions by Genotype\n"
        + f"(Using {n_components} PCs explaining {pca_results['cumulative_variance_ratio'][n_components-1]:.1%} variance)",
        fontsize=14,
    )

    plt.tight_layout()
    return fig


def create_feature_contribution_heatmap(
    pca_results: Dict,
    n_components: int = 5,
    n_features: int = 20,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Create a heatmap showing feature contributions to principal components.

    Args:
        pca_results: Results from perform_pca_analysis containing 'feature_importance'
            DataFrame.
        n_components: Number of components to show.
        n_features: Number of top features to show.
        figsize: Figure size.

    Returns:
        The generated figure.
    """
    # Get feature importance - check multiple possible keys
    feature_importance = None
    for key in [
        "feature_importance",
        "feature_importance_consistent",
        "trait_ev_df",
        "trait_contrib_df",
    ]:
        if key in pca_results:
            feature_importance = pca_results[key]
            break

    if feature_importance is None:
        raise ValueError(
            "pca_results must contain feature importance data. "
            "Expected one of: 'feature_importance', 'feature_importance_consistent', "
            "'trait_ev_df', or 'trait_contrib_df'"
        )

    # Get top features by total contribution
    # Check for different column naming conventions
    if "total_contribution" in feature_importance.columns:
        sort_col = "total_contribution"
    elif "trait_total_variance_contrib" in feature_importance.columns:
        sort_col = "trait_total_variance_contrib"
    else:
        # Calculate total contribution if not present
        pc_cols = [col for col in feature_importance.columns if col.startswith("PC")]
        top_features = feature_importance.copy()
        top_features["total_contribution"] = np.abs(top_features[pc_cols]).sum(axis=1)
        sort_col = "total_contribution"

    # Sort by total contribution (already sorted if from run_pca_and_export_artifacts)
    if sort_col in feature_importance.columns:
        top_features = feature_importance.nlargest(n_features, sort_col)
    else:
        top_features = feature_importance.head(n_features)

    # Determine number of components to show
    n_comp_available = pca_results.get(
        "n_components_selected",
        len([c for c in feature_importance.columns if c.startswith("PC")]),
    )
    n_comp_to_show = min(n_components, n_comp_available)

    # Select PC columns for heatmap
    # First try standard PC column names
    pc_cols = [f"PC{i+1}" for i in range(n_comp_to_show)]
    available_pc_cols = [col for col in pc_cols if col in top_features.columns]

    if not available_pc_cols:
        # Try variance contribution column names (from run_pca_and_export_artifacts)
        pc_cols = [f"PC{i+1}_variance_contrib" for i in range(n_comp_to_show)]
        available_pc_cols = [col for col in pc_cols if col in top_features.columns]

    if not available_pc_cols:
        # Fall back to any PC-prefixed columns
        available_pc_cols = [
            col for col in top_features.columns if col.startswith("PC")
        ][:n_comp_to_show]

    heatmap_data = top_features[available_pc_cols]

    # Set row labels to trait names if available
    if "trait" in top_features.columns:
        heatmap_data.index = top_features["trait"].values
    elif top_features.index.name == "trait":
        # Index might already be trait names
        pass
    else:
        # Use existing index
        pass

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    # Determine appropriate label based on column names
    if available_pc_cols and "variance_contrib" in available_pc_cols[0]:
        cbar_label = "Variance Contribution"
    else:
        cbar_label = "Loading"

    sns.heatmap(
        heatmap_data,
        cmap="RdBu_r",
        center=0,
        fmt=".3f",
        cbar_kws={"label": cbar_label},
        ax=ax,
        annot=True,
        annot_kws={"size": 8},
    )

    ax.set_title(
        f"Top {n_features} Feature Contributions to First {n_comp_to_show} PCs"
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    return fig


def create_publication_figure(
    fig: Union[plt.Figure, "go.Figure"],
    output_path: Union[str, Path],
    dpi: int = 300,
    format: str = "pdf",
    transparent: bool = False,
    bbox_inches: str = "tight",
) -> None:
    """Save a figure in publication-ready format.

    Args:
        fig: Figure to save (matplotlib or plotly).
        output_path: Output file path.
        dpi: Resolution for raster formats.
        format: Output format ('pdf', 'eps', 'png', 'svg').
        transparent: Whether to use transparent background.
        bbox_inches: Bbox setting for matplotlib figures.

    Raises:
        ValueError: If figure type is not supported.
    """
    output_path = Path(output_path)

    if isinstance(fig, plt.Figure):
        # Save matplotlib figure
        fig.savefig(
            output_path,
            dpi=dpi,
            format=format,
            transparent=transparent,
            bbox_inches=bbox_inches,
        )
    elif go is not None and hasattr(fig, "write_image"):
        # Save plotly figure if plotly is available
        if format == "pdf":
            fig.write_image(str(output_path), format="pdf")
        elif format == "png":
            fig.write_image(str(output_path), format="png", scale=dpi / 100)
        elif format == "svg":
            fig.write_image(str(output_path), format="svg")
        else:
            fig.write_html(str(output_path.with_suffix(".html")))
    else:
        raise ValueError("Unsupported figure type")


def identify_extreme_phenotypes(
    df: pd.DataFrame,
    trait_cols: List[str],
    group_col: str = "geno",
    n_std: float = 2.0,
    min_samples_per_group: int = 3,
) -> Dict[str, pd.DataFrame]:
    """Identify genotypes with extreme phenotypes for each trait.

    Args:
        df: DataFrame containing trait data.
        trait_cols: List of trait columns to analyze.
        group_col: Column to group by (e.g., 'geno').
        n_std: Number of standard deviations to consider extreme.
        min_samples_per_group: Minimum samples required per group.

    Returns:
        Dictionary mapping trait names to DataFrames of extreme genotypes.
        Each DataFrame contains columns: mean, std, count, deviation, direction.
    """
    extreme_phenotypes = {}

    # Return empty dict for empty DataFrame
    if df.empty or not trait_cols:
        return extreme_phenotypes

    for trait in trait_cols:
        if trait not in df.columns:
            continue

        # Calculate group means
        group_stats = df.groupby(group_col)[trait].agg(["mean", "std", "count"])

        # Filter groups with enough samples
        valid_groups = group_stats[group_stats["count"] >= min_samples_per_group]

        if len(valid_groups) > 0:
            # Calculate overall mean and std
            overall_mean = df[trait].mean()
            overall_std = df[trait].std()

            if pd.notna(overall_std) and overall_std > 0:
                # Identify extreme groups
                high_threshold = overall_mean + n_std * overall_std
                low_threshold = overall_mean - n_std * overall_std

                extreme_groups = valid_groups[
                    (valid_groups["mean"] > high_threshold)
                    | (valid_groups["mean"] < low_threshold)
                ].copy()

                if len(extreme_groups) > 0:
                    extreme_groups["deviation"] = (
                        extreme_groups["mean"] - overall_mean
                    ) / overall_std
                    extreme_groups["direction"] = extreme_groups["deviation"].apply(
                        lambda x: "high" if x > 0 else "low"
                    )
                    extreme_groups = extreme_groups.sort_values(
                        "deviation", key=lambda x: abs(x), ascending=False
                    )
                    extreme_phenotypes[trait] = extreme_groups

    return extreme_phenotypes


def create_phenotype_variation_plot(
    df: pd.DataFrame,
    trait: str,
    group_col: str = "geno",
    highlight_extreme: bool = True,
    n_std: float = 2.0,
    point_size: float = 50,
    figsize: Tuple[float, float] = (12, 8),
    output_csv_path: Optional[Path] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Create a box plot with jittered points showing phenotypic variation across groups.

    Args:
        df: DataFrame containing trait data.
        trait: Trait column name.
        group_col: Column to group by.
        highlight_extreme: Whether to highlight extreme phenotypes.
        n_std: Number of standard deviations for extreme threshold.
        point_size: Size of the jittered points.
        figsize: Figure size.
        output_csv_path: If provided, save plot data to this CSV file.

    Returns:
        Tuple of (Figure, DataFrame with plot data).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Prepare data
    df_plot = df[[group_col, trait]].dropna()

    # Calculate group statistics for ordering
    group_stats = df_plot.groupby(group_col)[trait].agg(["mean", "std", "count"])
    group_stats = group_stats.sort_values("mean")
    group_order = group_stats.index.tolist()

    # Create position mapping
    positions = {g: i for i, g in enumerate(group_order)}

    # Prepare data for box plot
    plot_data = [df_plot[df_plot[group_col] == g][trait].values for g in group_order]

    # Create box plot
    bp = ax.boxplot(
        plot_data,
        positions=list(range(len(group_order))),
        widths=0.6,
        patch_artist=True,
        showfliers=False,  # Don't show outliers as we'll plot all points
    )

    # Style the box plot
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    # Add jittered points
    np.random.seed(42)  # For reproducibility
    for i, group in enumerate(group_order):
        group_data = df_plot[df_plot[group_col] == group][trait].values
        n_points = len(group_data)

        # Create jitter
        jitter = np.random.uniform(-0.2, 0.2, n_points)
        x_positions = np.full(n_points, i) + jitter

        # Plot points
        ax.scatter(
            x_positions,
            group_data,
            alpha=0.6,
            s=point_size,
            color="darkblue",
            edgecolors="black",
            linewidth=0.5,
            zorder=10,
        )

    # Add mean ± std lines
    for i, (group, stats) in enumerate(group_stats.iterrows()):
        mean = stats["mean"]
        std = stats["std"]

        # Plot mean line
        ax.plot([i - 0.25, i + 0.25], [mean, mean], "r-", linewidth=2, zorder=15)

        # Plot std range if not NaN
        if pd.notna(std):
            ax.plot([i, i], [mean - std, mean + std], "r-", linewidth=1.5, zorder=15)
            ax.plot(
                [i - 0.1, i + 0.1],
                [mean + std, mean + std],
                "r-",
                linewidth=1.5,
                zorder=15,
            )
            ax.plot(
                [i - 0.1, i + 0.1],
                [mean - std, mean - std],
                "r-",
                linewidth=1.5,
                zorder=15,
            )

    # Calculate overall statistics
    overall_mean = df_plot[trait].mean()
    overall_std = df_plot[trait].std()

    # Highlight extreme phenotypes
    if highlight_extreme and pd.notna(overall_std) and overall_std > 0:
        high_threshold = overall_mean + n_std * overall_std
        low_threshold = overall_mean - n_std * overall_std

        # Color boxes based on mean values
        for i, (group, stats) in enumerate(group_stats.iterrows()):
            if stats["mean"] > high_threshold:
                bp["boxes"][i].set_facecolor("red")
                bp["boxes"][i].set_alpha(0.6)
            elif stats["mean"] < low_threshold:
                bp["boxes"][i].set_facecolor("blue")
                bp["boxes"][i].set_alpha(0.6)

        # Add threshold lines
        ax.axhline(
            y=high_threshold,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"+{n_std} SD",
        )
        ax.axhline(
            y=low_threshold,
            color="blue",
            linestyle="--",
            alpha=0.5,
            label=f"-{n_std} SD",
        )
        ax.axhline(
            y=overall_mean,
            color="black",
            linestyle="-",
            alpha=0.3,
            label="Overall mean",
        )

    # Set labels and title
    ax.set_xticks(list(range(len(group_order))))
    ax.set_xticklabels(group_order, rotation=45, ha="right")
    ax.set_xlabel(group_col.capitalize())
    ax.set_ylabel(trait)

    # Add custom legend
    legend_elements = [
        Line2D([0], [0], color="r", linewidth=2, label="Mean ± SD"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="darkblue",
            markersize=8,
            alpha=0.6,
            label="Individual observations",
        ),
    ]

    if highlight_extreme and pd.notna(overall_std) and overall_std > 0:
        legend_elements.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"+{n_std} SD",
                ),
                Line2D(
                    [0],
                    [0],
                    color="blue",
                    linestyle="--",
                    alpha=0.5,
                    label=f"-{n_std} SD",
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    alpha=0.3,
                    label="Overall mean",
                ),
            ]
        )

    ax.legend(handles=legend_elements, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    # Leave space at top for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Create DataFrame with plot data
    plot_data_list = []
    for group in group_order:
        group_data = df_plot[df_plot[group_col] == group][trait].values
        for value in group_data:
            plot_data_list.append(
                {
                    group_col: group,
                    trait: value,
                    f"{trait}_mean": group_stats.loc[group, "mean"],
                    f"{trait}_std": group_stats.loc[group, "std"],
                    f"{trait}_count": group_stats.loc[group, "count"],
                }
            )

    plot_df = pd.DataFrame(plot_data_list)

    # Add overall statistics
    plot_df[f"{trait}_overall_mean"] = overall_mean
    plot_df[f"{trait}_overall_std"] = overall_std

    if highlight_extreme and pd.notna(overall_std) and overall_std > 0:
        plot_df[f"{trait}_high_threshold"] = overall_mean + n_std * overall_std
        plot_df[f"{trait}_low_threshold"] = overall_mean - n_std * overall_std

    # Save to CSV if path provided
    if output_csv_path:
        output_csv_path = Path(output_csv_path)
        plot_df.to_csv(output_csv_path, index=False)

    return fig, plot_df
