"""Public cross-platform PC-correlation workflow (issue #119).

The wheat EDPIE analysis correlates principal-component scores across phenotyping
platforms. This module exposes that workflow as a single public function,
:func:`cross_platform_pc_correlations`, returning a typed
:class:`CrossPlatformPCResult`.

Workflow order (maintainer-confirmed; **not** average-then-PCA):

1. Per platform, fit PCA on the *sample-level* trait matrix.
2. Aggregate the resulting sample-level PC scores to *genotype means*.
3. Correlate every PC of one platform against every PC of another, over the
   genotypes common to all platforms.
4. Pool every cross-platform PC test into one family and apply a single
   multiple-testing correction; attach Fisher-z confidence intervals and power.

It composes existing building blocks (:func:`perform_pca_analysis`,
:func:`calculate_genotype_means`, :func:`calculate_correlation_ci`,
:func:`achieved_power`, :func:`minimum_detectable_correlation`, and
``statsmodels`` ``multipletests``) rather than reimplementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from sleap_roots_analyze.cross_experiment_analysis import (
    achieved_power,
    calculate_correlation_ci,
    calculate_genotype_means,
    minimum_detectable_correlation,
)
from sleap_roots_analyze.pca import perform_pca_analysis

__all__ = ["CrossPlatformPCResult", "cross_platform_pc_correlations"]


@dataclass(frozen=True)
class CrossPlatformPCResult:
    """Bundled result of :func:`cross_platform_pc_correlations`.

    Attributes:
        pca: Mapping of platform name to its full :func:`perform_pca_analysis`
            result dict (loadings, explained variance, transformed sample scores, …).
        pc_scores: Mapping of platform name to a genotype-indexed DataFrame of
            mean PC scores (columns ``PC1``…``PCk``), before cross-platform
            alignment.
        common_genotypes: Genotypes present in every platform, used for all
            pairwise correlations.
        correlations: Tidy table with one row per cross-platform PC test, with
            columns ``platform1``, ``platform2``, ``pc1``, ``pc2``, ``r``,
            ``p_value``, ``p_value_fdr``, ``ci_low``, ``ci_high``, ``power``,
            ``n_genotypes`` and ``significant_fdr`` (pivot for a per-pair matrix).
        significant: Subset of ``correlations`` that survives FDR correction.
        summary: Headline numbers — ``n_tests``, ``n_genotypes``,
            ``n_fdr_significant``, ``tests_per_pair``, ``method``,
            ``correction_method``, ``mean_power`` and
            ``min_detectable_r_80_power``.
    """

    pca: dict[str, dict[str, Any]]
    pc_scores: dict[str, pd.DataFrame]
    common_genotypes: list[str]
    correlations: pd.DataFrame
    significant: pd.DataFrame
    summary: dict[str, Any]


def _correlate(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float]:
    """Return ``(r, p_value)`` for two 1-D arrays using ``method``."""
    if method == "spearman":
        result = stats.spearmanr(x, y)
        return float(result.statistic), float(result.pvalue)
    if method == "pearson":
        result = stats.pearsonr(x, y)
        return float(result.statistic), float(result.pvalue)
    if method == "kendall":
        result = stats.kendalltau(x, y)
        return float(result.statistic), float(result.pvalue)
    raise ValueError(
        f"Unknown correlation method {method!r}; expected 'spearman', "
        "'pearson', or 'kendall'."
    )


def _platform_pc_means(
    df: pd.DataFrame,
    trait_cols: list[str],
    n_components: int,
    genotype_col: str,
    random_state: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit PCA on sample traits, then average the PC scores per genotype.

    Returns the full PCA result and a genotype-indexed DataFrame of mean PC
    scores (columns ``PC1``…``PC{n_components}``).
    """
    traits = df[trait_cols]
    # perform_pca_analysis drops NaN rows internally and returns an index-less
    # array, so drop here too and keep the genotype labels aligned by position.
    keep = traits.notna().all(axis=1).to_numpy()
    traits_clean = traits.loc[keep]
    genotypes = df.loc[keep, genotype_col].to_numpy()

    if traits_clean.empty:
        raise ValueError(
            "No complete-case samples remain after dropping rows with NaN trait "
            "values. PCA needs complete cases; pass trait_cols that are free of "
            "NaN (e.g. drop NaN-bearing columns) for this platform."
        )

    pca_result = perform_pca_analysis(
        traits_clean,
        standardize=True,
        n_components=n_components,
        random_state=random_state,
        include_feature_metrics=False,
    )

    scores = np.asarray(pca_result["transformed_data"])[:, :n_components]
    if scores.shape[0] != genotypes.shape[0]:
        # Standardization only drops zero-variance columns, never rows; guard in
        # case that ever changes so genotype alignment can't silently corrupt.
        raise ValueError(
            "PCA changed the sample count; cannot align PC scores to genotypes."
        )

    pc_cols = [f"PC{i + 1}" for i in range(scores.shape[1])]
    sample_scores = pd.DataFrame(scores, columns=pc_cols)
    sample_scores[genotype_col] = genotypes
    means = calculate_genotype_means(sample_scores, pc_cols, genotype_col=genotype_col)
    return pca_result, means[pc_cols]


def cross_platform_pc_correlations(
    platforms: dict[str, pd.DataFrame],
    trait_cols: dict[str, list[str]],
    n_components: dict[str, int],
    *,
    genotype_col: str = "genotype",
    method: str = "spearman",
    alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    confidence_level: float = 0.95,
    random_state: Optional[int] = None,
) -> CrossPlatformPCResult:
    """Correlate per-platform PC scores across platforms in one call.

    For each platform, PCA is fit on the sample-level trait matrix and the
    resulting sample PC scores are averaged per genotype. PCs are then correlated
    across every unordered platform pair over the genotypes common to all
    platforms, and all tests are pooled into a single FDR family.

    Args:
        platforms: Mapping of platform name to its sample-level trait DataFrame
            (one row per sample, including ``genotype_col``).
        trait_cols: Mapping of platform name to the trait columns to feed PCA.
        n_components: Mapping of platform name to the number of PCs to retain and
            correlate.
        genotype_col: Name of the genotype column in every platform DataFrame.
        method: Correlation method — ``"spearman"`` (default, as in the wheat
            EDPIE paper), ``"pearson"`` or ``"kendall"``.
        alpha: Significance level for FDR and power.
        correction_method: Multiple-testing method passed to
            ``statsmodels.stats.multitest.multipletests`` (default ``"fdr_bh"``,
            per the issue #119 signature; note ``CrossPlatformConfig`` defaults to
            ``"fdr_by"`` for the trait-level pipeline).
        confidence_level: Confidence level for the Fisher-z correlation interval.
        random_state: Seed forwarded to PCA for reproducibility (``None`` uses the
            ``perform_pca_analysis`` default of 42).

    Returns:
        A :class:`CrossPlatformPCResult`.

    Raises:
        ValueError: If fewer than two platforms are given, a platform is missing
            its ``trait_cols``/``n_components`` entry, or ``method`` is unknown.
    """
    if len(platforms) < 2:
        raise ValueError("At least two platforms are required to correlate.")

    pca_seed = 42 if random_state is None else random_state

    # 1-2. Per platform: sample-level PCA -> genotype-mean PC scores.
    pca_results: dict[str, dict[str, Any]] = {}
    pc_scores: dict[str, pd.DataFrame] = {}
    for name, df in platforms.items():
        if name not in trait_cols:
            raise ValueError(f"Missing trait_cols for platform {name!r}.")
        if name not in n_components:
            raise ValueError(f"Missing n_components for platform {name!r}.")
        pca_results[name], pc_scores[name] = _platform_pc_means(
            df, trait_cols[name], n_components[name], genotype_col, pca_seed
        )

    # 3. Align on genotypes common to every platform.
    common = sorted(set.intersection(*(set(s.index) for s in pc_scores.values())))
    n = len(common)
    aligned = {name: s.loc[common] for name, s in pc_scores.items()}

    # 4. Pairwise PC x PC correlations over the common genotypes.
    rows: list[dict[str, Any]] = []
    for p1, p2 in combinations(platforms.keys(), 2):
        s1, s2 = aligned[p1], aligned[p2]
        for pc1 in s1.columns:
            for pc2 in s2.columns:
                x = s1[pc1].to_numpy()
                y = s2[pc2].to_numpy()
                if n >= 2 and np.std(x) > 0 and np.std(y) > 0:
                    r, p_value = _correlate(x, y, method)
                else:
                    # Too few/degenerate genotypes: report the pair without a
                    # spurious correlation instead of raising.
                    r, p_value = np.nan, np.nan
                ci_low, ci_high = calculate_correlation_ci(r, n, confidence_level)
                rows.append(
                    {
                        "platform1": p1,
                        "platform2": p2,
                        "pc1": pc1,
                        "pc2": pc2,
                        "r": r,
                        "p_value": p_value,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "power": achieved_power(r, n, alpha),
                        "n_genotypes": n,
                    }
                )

    correlations = pd.DataFrame(
        rows,
        columns=[
            "platform1",
            "platform2",
            "pc1",
            "pc2",
            "r",
            "p_value",
            "ci_low",
            "ci_high",
            "power",
            "n_genotypes",
        ],
    )

    # 5. Pool every test into one family and FDR-correct together.
    valid = correlations["p_value"].notna().to_numpy()
    p_fdr = np.full(len(correlations), np.nan)
    significant = np.zeros(len(correlations), dtype=bool)
    if valid.any():
        reject, p_adj, _, _ = multipletests(
            correlations.loc[valid, "p_value"].to_numpy(),
            alpha=alpha,
            method=correction_method,
        )
        p_fdr[valid] = p_adj
        significant[valid] = reject
    correlations["p_value_fdr"] = p_fdr
    correlations["significant_fdr"] = significant

    significant_df = correlations[correlations["significant_fdr"]].copy()

    tests_per_pair = {
        f"{p1}_vs_{p2}": int(
            (
                (correlations["platform1"] == p1) & (correlations["platform2"] == p2)
            ).sum()
        )
        for p1, p2 in combinations(platforms.keys(), 2)
    }

    summary = {
        "n_tests": int(len(correlations)),
        "n_genotypes": n,
        "n_fdr_significant": int(correlations["significant_fdr"].sum()),
        "tests_per_pair": tests_per_pair,
        "method": method,
        "correction_method": correction_method,
        "alpha": alpha,
        "mean_power": (
            float(correlations["power"].mean()) if len(correlations) else float("nan")
        ),
        "min_detectable_r_80_power": (
            float(minimum_detectable_correlation(n, alpha=alpha, power=0.80))
            if n >= 4
            else float("nan")
        ),
    }

    return CrossPlatformPCResult(
        pca=pca_results,
        pc_scores=pc_scores,
        common_genotypes=common,
        correlations=correlations,
        significant=significant_df,
        summary=summary,
    )
