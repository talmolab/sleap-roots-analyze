"""Publication-quality figures for cross-platform PC correlations.

Figure helpers consume the tidy correlation table from
:func:`sleap_roots_analyze.pc_correlations.correlate.calculate_all_platform_correlations`.
The ``fdr_method`` argument selects which significance column to read (e.g.
``"combined_fdr_by"`` -> ``"significant_combined_fdr_by"``).

These helpers create figures but do not select a matplotlib backend; callers
running headless (tests, the workflow orchestrator) should use the ``Agg``
backend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_correlation_heatmap(
    correlation_df: pd.DataFrame,
    platform1: str,
    platform2: str,
    fdr_method: str = "fdr_by",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Create a PC-by-PC correlation heatmap for one platform pair.

    Args:
        correlation_df: Tidy cross-platform correlation table.
        platform1: First platform name.
        platform2: Second platform name.
        fdr_method: Significance column suffix (e.g. ``"combined_fdr_by"``).
        figsize: Figure size.
        cmap: Diverging colormap.

    Returns:
        The matplotlib figure.
    """
    mask = (correlation_df["platform1"] == platform1) & (
        correlation_df["platform2"] == platform2
    )
    if not mask.any():
        mask = (correlation_df["platform1"] == platform2) & (
            correlation_df["platform2"] == platform1
        )
        if mask.any():
            platform1, platform2 = platform2, platform1

    df = correlation_df[mask].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"No data for {platform1} vs {platform2}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return fig

    corr_matrix = df.pivot(
        index="platform1_pc", columns="platform2_pc", values="spearman_r"
    )
    sig_col = f"significant_{fdr_method}"
    sig_matrix = df.pivot(index="platform1_pc", columns="platform2_pc", values=sig_col)

    annot = corr_matrix.copy().astype(str)
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            val = corr_matrix.loc[i, j]
            sig = bool(sig_matrix.loc[i, j])
            if pd.isna(val):
                annot.loc[i, j] = ""
            elif sig:
                annot.loc[i, j] = f"{val:.2f}*"
            else:
                annot.loc[i, j] = f"{val:.2f}"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Spearman r", "shrink": 0.8},
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(
        f"PC Correlations: {platform1} vs {platform2}\n"
        f"(* = FDR significant, {fdr_method})"
    )
    ax.set_xlabel(f"{platform2} PCs")
    ax.set_ylabel(f"{platform1} PCs")
    plt.tight_layout()
    return fig


def create_correlation_summary_figure(
    correlation_df: pd.DataFrame,
    fdr_method: str = "fdr_by",
    figsize: tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Create a four-panel summary (histogram, volcano, top +/- correlations).

    Args:
        correlation_df: Tidy cross-platform correlation table.
        fdr_method: Significance column suffix.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    sig_col = f"significant_{fdr_method}"

    ax = axes[0, 0]
    ax.hist(
        correlation_df["spearman_r"].dropna(), bins=20, edgecolor="black", alpha=0.7
    )
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Spearman r")
    ax.set_ylabel("Count")
    ax.set_title("A) Distribution of PC Correlations")

    ax = axes[0, 1]
    x = correlation_df["spearman_r"]
    y = -np.log10(correlation_df["spearman_p"])
    colors = correlation_df[sig_col].map({True: "red", False: "gray"})
    ax.scatter(x, y, c=colors, alpha=0.7, edgecolors="black", linewidth=0.5)
    ax.axhline(
        -np.log10(0.05), color="blue", linestyle="--", linewidth=1, label="p=0.05"
    )
    ax.set_xlabel("Spearman r")
    ax.set_ylabel("-log10(p)")
    ax.set_title(f"B) Volcano Plot (red = {fdr_method} significant)")
    ax.legend()

    for ax, ascending, panel in (
        (axes[1, 0], False, "C) Top 10 Positive Correlations"),
        (axes[1, 1], True, "D) Top 10 Negative Correlations"),
    ):
        top = (
            correlation_df.nsmallest(10, "spearman_r")
            if ascending
            else correlation_df.nlargest(10, "spearman_r")
        )
        labels = [
            f"{row['platform1_pc']}-{row['platform2_pc']}\n"
            f"({row['platform1'][:3]}-{row['platform2'][:3]})"
            for _, row in top.iterrows()
        ]
        colors = ["red" if sig else "steelblue" for sig in top[sig_col]]
        ax.barh(range(len(top)), top["spearman_r"], color=colors, edgecolor="black")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Spearman r")
        ax.set_title(panel)
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def create_scatter_plot(
    platform1_pcs: pd.DataFrame,
    platform2_pcs: pd.DataFrame,
    pc1: str,
    pc2: str,
    platform1_name: str,
    platform2_name: str,
    correlation_stats: dict | None = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Create a genotype-level scatter plot for a single PC pair.

    Args:
        platform1_pcs: Genotype-level PC scores for platform 1.
        platform2_pcs: Genotype-level PC scores for platform 2.
        pc1: PC column from platform 1.
        pc2: PC column from platform 2.
        platform1_name: Display name for platform 1.
        platform2_name: Display name for platform 2.
        correlation_stats: Optional dict with ``spearman_r`` / ``spearman_p``.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    common = sorted(set(platform1_pcs.index) & set(platform2_pcs.index))
    x = platform1_pcs.loc[common, pc1]
    y = platform2_pcs.loc[common, pc2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, s=80, alpha=0.7, edgecolors="black", linewidth=1)
    for geno in common:
        ax.annotate(
            str(geno).replace("GH_", ""),
            (x[geno], y[geno]),
            fontsize=7,
            ha="center",
            va="bottom",
            xytext=(0, 3),
            textcoords="offset points",
        )

    if len(common) >= 2:
        z = np.polyfit(x, y, 1)
        poly = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, poly(x_line), "r--", linewidth=2, alpha=0.7)

    ax.set_xlabel(f"{platform1_name} {pc1}")
    ax.set_ylabel(f"{platform2_name} {pc2}")
    if correlation_stats:
        ax.set_title(
            f"{platform1_name} {pc1} vs {platform2_name} {pc2}\n"
            f"r = {correlation_stats.get('spearman_r', float('nan')):.3f}, "
            f"p = {correlation_stats.get('spearman_p', float('nan')):.4f}"
        )
    else:
        ax.set_title(f"{platform1_name} {pc1} vs {platform2_name} {pc2}")
    plt.tight_layout()
    return fig


def create_sensitivity_analysis_figure(
    correlation_df: pd.DataFrame,
    methods: list[str] | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Compare significant-correlation counts across FDR methods and scopes.

    Args:
        correlation_df: Tidy cross-platform correlation table.
        methods: FDR methods to compare. Defaults to BY/BH/Bonferroni.
        figsize: Figure size.

    Returns:
        The matplotlib figure.
    """
    if methods is None:
        methods = ["fdr_by", "fdr_bh", "bonferroni"]

    fig, ax = plt.subplots(figsize=figsize)
    has_combined = any(
        c.startswith("significant_combined_") for c in correlation_df.columns
    )
    has_per_pair = any(
        c.startswith("significant_per_pair_") for c in correlation_df.columns
    )

    if has_combined and has_per_pair:
        x = np.arange(len(methods))
        width = 0.35
        combined_counts = [
            int(
                correlation_df.get(
                    f"significant_combined_{m}", pd.Series(dtype=bool)
                ).sum()
            )
            for m in methods
        ]
        per_pair_counts = [
            int(
                correlation_df.get(
                    f"significant_per_pair_{m}", pd.Series(dtype=bool)
                ).sum()
            )
            for m in methods
        ]
        bars1 = ax.bar(
            x - width / 2,
            combined_counts,
            width,
            label="Combined",
            color="steelblue",
            edgecolor="black",
        )
        bars2 = ax.bar(
            x + width / 2,
            per_pair_counts,
            width,
            label="Per-pair",
            color="coral",
            edgecolor="black",
        )
        for bars, counts in ((bars1, combined_counts), (bars2, per_pair_counts)):
            for bar, count in zip(bars, counts):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    str(int(count)),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )
        ax.set_title("Sensitivity Analysis: FDR Scope Comparison")
        ax.legend(loc="upper right")
        ax.set_xticks(x)
    else:
        counts = [
            int(correlation_df.get(f"significant_{m}", pd.Series(dtype=bool)).sum())
            for m in methods
        ]
        x = np.arange(len(methods))
        bars = ax.bar(x, counts, color="steelblue", edgecolor="black")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(int(count)),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
        ax.set_title(
            f"Sensitivity Analysis: Significant Correlations by Method "
            f"(n = {len(correlation_df)} tests)"
        )
        ax.set_xticks(x)

    ax.set_xticklabels([m.replace("_", " ").upper() for m in methods])
    ax.set_ylabel("Number of Significant Correlations")
    n_tests = len(correlation_df)
    ax.axhline(
        n_tests * 0.05,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Expected false positives at alpha=0.05 ({n_tests * 0.05:.1f})",
    )
    ax.legend()
    plt.tight_layout()
    return fig


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    formats: list[str] | None = None,
    dpi: int = 300,
) -> list[Path]:
    """Save a figure in one or more formats.

    Args:
        fig: The figure to save.
        output_dir: Output directory (created if missing).
        filename: Base filename without extension.
        formats: Formats to write. Defaults to ``["png", "pdf"]``.
        dpi: Resolution for raster formats.

    Returns:
        List of written file paths.
    """
    if formats is None:
        formats = ["png", "pdf"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths
