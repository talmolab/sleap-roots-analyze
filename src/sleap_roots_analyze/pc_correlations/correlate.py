"""Cross-platform PC correlations with multi-scope FDR correction.

These nodes correlate genotype-mean PC scores between platforms and apply
multiple-testing correction at two scopes:

- ``combined``: one correction across every cross-platform PC test.
- ``per_pair``: a separate correction within each platform pair.

Confidence intervals, achieved power, and the minimum detectable correlation are
reused from :mod:`sleap_roots_analyze.cross_experiment_analysis` (the single
source of truth for those statistics) rather than re-implemented here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sleap_roots_analyze.cross_experiment_analysis import (
    achieved_power,
    calculate_correlation_ci,
    minimum_detectable_correlation,
)

DEFAULT_FDR_METHODS = ["fdr_by", "fdr_bh", "bonferroni"]


def calculate_correlation(
    x: np.ndarray, y: np.ndarray, method: str = "spearman"
) -> tuple[float, float]:
    """Calculate a correlation coefficient and its p-value.

    Args:
        x: First array.
        y: Second array.
        method: ``"spearman"``, ``"pearson"``, or ``"kendall"``.

    Returns:
        Tuple of ``(correlation_coefficient, p_value)``.

    Raises:
        ValueError: If ``method`` is not recognised.
    """
    if method == "spearman":
        r, p = stats.spearmanr(x, y)
    elif method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "kendall":
        r, p = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return float(r), float(p)


def calculate_cross_platform_pc_correlations(
    platform1_pcs: pd.DataFrame,
    platform2_pcs: pd.DataFrame,
    platform1_name: str = "Platform1",
    platform2_name: str = "Platform2",
    platform1_n_pcs: int | None = None,
    platform2_n_pcs: int | None = None,
    fdr_methods: list[str] | None = None,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Correlate every PC of one platform against every PC of another.

    Args:
        platform1_pcs: Genotype-level PC scores for platform 1 (index = genotype).
        platform2_pcs: Genotype-level PC scores for platform 2 (index = genotype).
        platform1_name: Display name for platform 1.
        platform2_name: Display name for platform 2.
        platform1_n_pcs: Number of PCs to retain from platform 1 (default: all).
        platform2_n_pcs: Number of PCs to retain from platform 2 (default: all).
        fdr_methods: Correction methods to apply directly to this pair. Defaults
            to no correction (the orchestrator applies pooled/per-pair FDR).
        alpha: Significance level for power and (optional) correction.
        confidence_level: Confidence level for the correlation interval.

    Returns:
        Tidy DataFrame with one row per PC-by-PC test.
    """
    if fdr_methods is None:
        fdr_methods = []

    pc1_cols = [c for c in platform1_pcs.columns if c.startswith("PC")]
    pc2_cols = [c for c in platform2_pcs.columns if c.startswith("PC")]
    if platform1_n_pcs is not None:
        pc1_cols = pc1_cols[:platform1_n_pcs]
    if platform2_n_pcs is not None:
        pc2_cols = pc2_cols[:platform2_n_pcs]

    common_genotypes = sorted(set(platform1_pcs.index) & set(platform2_pcs.index))
    n_genotypes = len(common_genotypes)
    p1 = platform1_pcs.loc[common_genotypes]
    p2 = platform2_pcs.loc[common_genotypes]

    results: list[dict[str, Any]] = []
    for pc1 in pc1_cols:
        for pc2 in pc2_cols:
            x = p1[pc1].to_numpy()
            y = p2[pc2].to_numpy()
            if n_genotypes >= 2 and np.std(x) > 0 and np.std(y) > 0:
                spearman_r, spearman_p = calculate_correlation(x, y, method="spearman")
                pearson_r, pearson_p = calculate_correlation(x, y, method="pearson")
            else:
                # Degenerate panel: report the pair without a spurious value.
                spearman_r = spearman_p = pearson_r = pearson_p = np.nan

            ci_lower, ci_upper = calculate_correlation_ci(
                spearman_r, n_genotypes, confidence_level
            )
            power = achieved_power(spearman_r, n_genotypes, alpha)

            results.append(
                {
                    "platform1": platform1_name,
                    "platform2": platform2_name,
                    "platform1_pc": pc1,
                    "platform2_pc": pc2,
                    "spearman_r": spearman_r,
                    "spearman_p": spearman_p,
                    "pearson_r": pearson_r,
                    "pearson_p": pearson_p,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "power": power,
                    "n_genotypes": n_genotypes,
                }
            )

    df = pd.DataFrame(results)

    for method in fdr_methods:
        df = _apply_multipletests(df, df.index, method, alpha, suffix=method)

    return df


def _apply_multipletests(
    df: pd.DataFrame,
    idx: pd.Index,
    method: str,
    alpha: float,
    suffix: str,
) -> pd.DataFrame:
    """Apply ``multipletests`` to ``df.loc[idx, 'spearman_p']`` in place.

    Rows with NaN p-values are skipped (their corrected p-value stays NaN and
    they are not flagged significant), so a degenerate test never corrupts the
    correction for the rest of the family.
    """
    p_adj_col = f"p_adj_{suffix}"
    sig_col = f"significant_{suffix}"
    if p_adj_col not in df.columns:
        df[p_adj_col] = np.nan
        df[sig_col] = False

    sub = df.loc[idx]
    valid = sub["spearman_p"].notna()
    valid_idx = sub.index[valid]
    if len(valid_idx) > 0:
        reject, p_adj, _, _ = multipletests(
            df.loc[valid_idx, "spearman_p"].to_numpy(), alpha=alpha, method=method
        )
        df.loc[valid_idx, p_adj_col] = p_adj
        df.loc[valid_idx, sig_col] = reject
    return df


def calculate_all_platform_correlations(
    aligned_data: dict[str, pd.DataFrame],
    platform_n_pcs: dict[str, int],
    fdr_methods: list[str] | None = None,
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    fdr_scope: str = "both",
) -> pd.DataFrame:
    """Correlate PCs across every platform pair with multi-scope FDR.

    Args:
        aligned_data: Mapping of platform name to genotype-level PC scores.
        platform_n_pcs: Mapping of platform name to the number of PCs to retain.
        fdr_methods: Correction methods. Defaults to ``DEFAULT_FDR_METHODS``.
        alpha: Significance level.
        confidence_level: Confidence level for correlation intervals.
        fdr_scope: ``"combined"``, ``"per_pair"``, or ``"both"`` (default).

    Returns:
        Tidy DataFrame over all platform pairs. For ``fdr_scope="both"`` the
        correction columns are prefixed ``combined_`` and ``per_pair_``; for a
        single scope they are unprefixed.
    """
    if fdr_methods is None:
        fdr_methods = list(DEFAULT_FDR_METHODS)

    platforms = list(aligned_data.keys())
    all_results: list[pd.DataFrame] = []
    for i, p1_name in enumerate(platforms):
        for p2_name in platforms[i + 1 :]:
            result = calculate_cross_platform_pc_correlations(
                platform1_pcs=aligned_data[p1_name],
                platform2_pcs=aligned_data[p2_name],
                platform1_name=p1_name,
                platform2_name=p2_name,
                platform1_n_pcs=platform_n_pcs.get(p1_name),
                platform2_n_pcs=platform_n_pcs.get(p2_name),
                fdr_methods=[],  # FDR applied below at the requested scope(s)
                alpha=alpha,
                confidence_level=confidence_level,
            )
            all_results.append(result)

    combined = (
        pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    )
    if combined.empty:
        return combined

    if fdr_scope in ("combined", "both"):
        prefix = "combined_" if fdr_scope == "both" else ""
        for method in fdr_methods:
            combined = _apply_multipletests(
                combined, combined.index, method, alpha, suffix=f"{prefix}{method}"
            )

    if fdr_scope in ("per_pair", "both"):
        prefix = "per_pair_" if fdr_scope == "both" else ""
        for method in fdr_methods:
            combined[f"p_adj_{prefix}{method}"] = np.nan
            combined[f"significant_{prefix}{method}"] = False
        for _, group in combined.groupby(["platform1", "platform2"]):
            for method in fdr_methods:
                combined = _apply_multipletests(
                    combined, group.index, method, alpha, suffix=f"{prefix}{method}"
                )

    return combined


def summarize_correlations(
    correlation_df: pd.DataFrame,
    primary_method: str = "fdr_by",
    fdr_scope: str = "both",
) -> dict[str, Any]:
    """Summarise a cross-platform correlation table.

    Args:
        correlation_df: Output of :func:`calculate_all_platform_correlations`.
        primary_method: FDR method used for the headline significance counts.
        fdr_scope: Scope whose columns to read (``"combined"``, ``"per_pair"``,
            or ``"both"``).

    Returns:
        Dict of summary statistics (test counts, per-pair counts, significance
        counts, power, and the minimum detectable correlation at 80% power).
    """
    n_tests = len(correlation_df)
    n_genotypes = int(correlation_df["n_genotypes"].iloc[0]) if n_tests > 0 else 0

    sig_counts: dict[str, int] = {}
    for col in correlation_df.columns:
        if col.startswith("significant_"):
            sig_counts[col.replace("significant_", "")] = int(correlation_df[col].sum())

    tests_per_pair: dict[str, int] = {}
    if n_tests > 0:
        for (p1, p2), group in correlation_df.groupby(["platform1", "platform2"]):
            tests_per_pair[f"{p1}_vs_{p2}"] = len(group)

    min_detectable_r = minimum_detectable_correlation(
        n_genotypes, alpha=0.05, power=0.80
    )
    mean_power = float(correlation_df["power"].mean()) if n_tests > 0 else float("nan")

    return {
        "n_tests": n_tests,
        "n_genotypes": n_genotypes,
        "tests_per_pair": tests_per_pair,
        "significant_counts": sig_counts,
        "primary_method": primary_method,
        "fdr_scope": fdr_scope,
        "n_significant_combined": sig_counts.get(
            f"combined_{primary_method}", sig_counts.get(primary_method, 0)
        ),
        "n_significant_per_pair": sig_counts.get(f"per_pair_{primary_method}", 0),
        "min_detectable_r_80_power": min_detectable_r,
        "mean_achieved_power": mean_power,
    }
