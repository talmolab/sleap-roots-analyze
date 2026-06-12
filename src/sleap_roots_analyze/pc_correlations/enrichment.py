"""Binomial enrichment test for cross-platform trait correlations.

Tests whether the number of nominally significant (``p < alpha``) trait-level
correlations deviates from chance. Under the null each test is significant with
probability ``alpha``, so the significant count follows ``Binomial(n_tests,
alpha)``; this module tests for enrichment (more than expected) or depletion
(fewer than expected).

This is the trait-level enrichment DAG; it consumes existing
``cross_platform_correlations.csv`` files and is independent of the PC-level
correlation workflow.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


@dataclass
class EnrichmentResult:
    """Result of a single binomial enrichment test.

    Attributes:
        platform_pair: ``"Combined"`` or a ``"Platform1 vs Platform2"`` label.
        n_tests: Total number of correlation tests.
        n_significant: Number of tests with ``p < alpha``.
        n_expected: Expected significant count under the null (``n_tests * alpha``).
        fold_enrichment: Observed / expected ratio.
        p_value_two_sided: Two-sided binomial p-value.
        p_value_enrichment: One-sided p-value for enrichment (greater).
        p_value_depletion: One-sided p-value for depletion (less).
        proportion_observed: ``n_significant / n_tests``.
        proportion_ci_low: Lower bound of the proportion confidence interval.
        proportion_ci_high: Upper bound of the proportion confidence interval.
        interpretation: ``"enriched"``, ``"depleted"``, or ``"null"``.
    """

    platform_pair: str
    n_tests: int
    n_significant: int
    n_expected: float
    fold_enrichment: float
    p_value_two_sided: float
    p_value_enrichment: float
    p_value_depletion: float
    proportion_observed: float
    proportion_ci_low: float
    proportion_ci_high: float
    interpretation: str

    def to_dict(self) -> dict:
        """Return the result as a plain dict (for CSV/JSON export)."""
        return asdict(self)


def calculate_enrichment_test(
    n_significant: int,
    n_tests: int,
    platform_pair: str = "Combined",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> EnrichmentResult:
    """Run a binomial enrichment test for nominally significant correlations.

    Note:
        The ``interpretation`` label (enriched/depleted/null) uses a fixed 0.05
        threshold on the one-sided binomial p-values. This is the *meta-test*
        threshold and is intentionally independent of ``confidence_level`` (which
        only governs the proportion CI) and of ``alpha`` (which is the per-test
        null proportion).

    Args:
        n_significant: Number of tests with ``p < alpha``.
        n_tests: Total number of correlation tests.
        platform_pair: Label for this comparison.
        alpha: Nominal significance threshold (the null proportion).
        confidence_level: Confidence level for the proportion interval.

    Returns:
        An :class:`EnrichmentResult`.

    Raises:
        ValueError: If counts are out of range or ``alpha`` is not in ``(0, 1)``.

    Example:
        >>> result = calculate_enrichment_test(n_significant=26, n_tests=1331)
        >>> round(result.fold_enrichment, 2)
        0.39
    """
    if n_tests <= 0:
        raise ValueError(f"n_tests must be positive, got {n_tests}")
    if n_significant < 0:
        raise ValueError(f"n_significant must be non-negative, got {n_significant}")
    if n_significant > n_tests:
        raise ValueError(
            f"n_significant ({n_significant}) cannot exceed n_tests ({n_tests})"
        )
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    n_expected = n_tests * alpha
    fold_enrichment = n_significant / n_expected if n_expected > 0 else float("inf")

    result_two_sided = binomtest(
        k=n_significant, n=n_tests, p=alpha, alternative="two-sided"
    )
    result_enrichment = binomtest(
        k=n_significant, n=n_tests, p=alpha, alternative="greater"
    )
    result_depletion = binomtest(
        k=n_significant, n=n_tests, p=alpha, alternative="less"
    )

    ci = result_two_sided.proportion_ci(
        confidence_level=confidence_level, method="exact"
    )

    if result_enrichment.pvalue < 0.05:
        interpretation = "enriched"
    elif result_depletion.pvalue < 0.05:
        interpretation = "depleted"
    else:
        interpretation = "null"

    return EnrichmentResult(
        platform_pair=platform_pair,
        n_tests=n_tests,
        n_significant=n_significant,
        n_expected=n_expected,
        fold_enrichment=fold_enrichment,
        p_value_two_sided=result_two_sided.pvalue,
        p_value_enrichment=result_enrichment.pvalue,
        p_value_depletion=result_depletion.pvalue,
        proportion_observed=n_significant / n_tests,
        proportion_ci_low=ci.low,
        proportion_ci_high=ci.high,
        interpretation=interpretation,
    )


def load_cross_platform_correlations(
    correlation_csv_path: Path,
    p_value_column: str = "spearman_p",
) -> pd.DataFrame:
    """Load a trait-level cross-platform correlation CSV.

    Args:
        correlation_csv_path: Path to a ``cross_platform_correlations.csv``.
        p_value_column: Column expected to hold p-values.

    Returns:
        The loaded DataFrame.

    Raises:
        ValueError: If ``p_value_column`` is absent.
    """
    df = pd.read_csv(correlation_csv_path)
    if p_value_column not in df.columns:
        raise ValueError(
            f"Column {p_value_column!r} not found in {correlation_csv_path}"
        )
    return df


def count_significant(
    df: pd.DataFrame,
    p_value_column: str = "spearman_p",
    alpha: float = 0.05,
) -> tuple[int, int]:
    """Count nominally significant and total correlations.

    Args:
        df: Correlation results.
        p_value_column: Column holding p-values.
        alpha: Significance threshold.

    Returns:
        Tuple of ``(n_significant, n_tests)``. ``n_tests`` counts only rows with
        a non-NaN p-value: a degenerate pair (NaN p) is neither significant nor a
        valid test, so including it in the denominator would bias the binomial
        toward "depleted".
    """
    valid = df[p_value_column].notna()
    n_tests = int(valid.sum())
    n_significant = int((df.loc[valid, p_value_column] < alpha).sum())
    return n_significant, n_tests


def calculate_all_enrichment_tests(
    correlation_files: dict[str, Path],
    alpha: float = 0.05,
    p_value_column: str = "spearman_p",
    confidence_level: float = 0.95,
) -> list[EnrichmentResult]:
    """Run enrichment tests for each platform pair plus a pooled combined test.

    Args:
        correlation_files: Mapping of platform-pair label to correlation CSV path.
        alpha: Significance threshold.
        p_value_column: Column holding p-values.
        confidence_level: Confidence level for the proportion intervals.

    Returns:
        List of :class:`EnrichmentResult`, with the ``Combined`` result first.
    """
    results: list[EnrichmentResult] = []
    combined_significant = 0
    combined_tests = 0

    for pair_name, csv_path in correlation_files.items():
        df = load_cross_platform_correlations(csv_path, p_value_column)
        n_sig, n_tests = count_significant(df, p_value_column, alpha)
        results.append(
            calculate_enrichment_test(
                n_significant=n_sig,
                n_tests=n_tests,
                platform_pair=pair_name,
                alpha=alpha,
                confidence_level=confidence_level,
            )
        )
        combined_significant += n_sig
        combined_tests += n_tests

    combined_result = calculate_enrichment_test(
        n_significant=combined_significant,
        n_tests=combined_tests,
        platform_pair="Combined",
        alpha=alpha,
        confidence_level=confidence_level,
    )
    results.insert(0, combined_result)

    return results


def results_to_dataframe(results: list[EnrichmentResult]) -> pd.DataFrame:
    """Convert a list of :class:`EnrichmentResult` to a DataFrame for export.

    Args:
        results: Enrichment results.

    Returns:
        DataFrame with one row per result.
    """
    return pd.DataFrame([r.to_dict() for r in results])


def create_enrichment_figure(
    results: list[EnrichmentResult],
    figsize: tuple[int, int] = (10, 6),
    title: str = "Trait-Level Cross-Platform Correlation Enrichment",
) -> plt.Figure:
    """Create an observed-vs-expected enrichment summary figure.

    Args:
        results: Enrichment results.
        figsize: Figure size.
        title: Figure title.

    Returns:
        The matplotlib figure.
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=figsize)

    labels = [r.platform_pair for r in results]
    observed = [r.n_significant for r in results]
    expected = [r.n_expected for r in results]

    ci_low = [r.proportion_ci_low * r.n_tests for r in results]
    ci_high = [r.proportion_ci_high * r.n_tests for r in results]
    errors_low = [obs - low for obs, low in zip(observed, ci_low)]
    errors_high = [high - obs for obs, high in zip(observed, ci_high)]

    x = np.arange(len(labels))
    width = 0.6

    color_map = {"enriched": "#2ca02c", "depleted": "#d62728", "null": "#7f7f7f"}
    colors = [color_map[r.interpretation] for r in results]

    ax.bar(x, observed, width, color=colors, edgecolor="black", linewidth=1)
    ax.errorbar(
        x,
        observed,
        yerr=[errors_low, errors_high],
        fmt="none",
        color="black",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    for i, exp in enumerate(expected):
        ax.hlines(
            exp,
            x[i] - width / 2,
            x[i] + width / 2,
            colors="navy",
            linestyles="--",
            linewidth=2,
        )

    for i, r in enumerate(results):
        if r.interpretation == "enriched":
            p = r.p_value_enrichment
        elif r.interpretation == "depleted":
            p = r.p_value_depletion
        else:
            p = r.p_value_two_sided
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if stars:
            y_pos = max(observed[i], expected[i]) + errors_high[i] + 5
            ax.text(
                x[i],
                y_pos,
                stars,
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_xlabel("Platform Comparison", fontsize=12)
    ax.set_ylabel("Number of Significant Correlations (p < 0.05)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")

    legend_elements = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#d62728",
            edgecolor="black",
            label="Depleted (p < 0.05)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#7f7f7f", edgecolor="black", label="Null"
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#2ca02c",
            edgecolor="black",
            label="Enriched (p < 0.05)",
        ),
        Line2D(
            [0],
            [0],
            color="navy",
            linestyle="--",
            linewidth=2,
            label="Expected (n x 0.05)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    return fig
