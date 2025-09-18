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
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
