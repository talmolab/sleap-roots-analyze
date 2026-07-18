"""Prediction/permutation figure content: 3-panel plotting functions (Tier 4, #200).

Parallel to ``cross_experiment_analysis.py``'s home for
``create_correlation_summary_plot`` etc. -- plotting logic as plain,
independently-testable functions returning a ``matplotlib.Figure``, no file
I/O. ``pipeline/steps/visualize_prediction.py`` (a different module, despite
the shared basename -- see proposal.md's naming-collision note) is the only
caller that saves the returned figure to disk.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from sleap_roots_analyze.result_types import PermutationResult, TargetPrediction

_OBSERVED_COLOR = "#2E6E73"
_NULL_COLOR = "#B0B0B0"


def _pc1_scatter_panel(
    ax: plt.Axes, target_predictions: Sequence[TargetPrediction]
) -> None:
    """Panel 1: observed-vs-predicted scatter, PC1 target only."""
    pc1 = next((tp for tp in target_predictions if tp.target_name == "PC1"), None)
    if pc1 is None:
        raise ValueError(
            "target_predictions must include a target named 'PC1' for the "
            "obs-vs-pred scatter panel"
        )
    ax.scatter(pc1.y_true, pc1.y_pred, color=_OBSERVED_COLOR, alpha=0.8, zorder=2)
    combined = list(pc1.y_true) + list(pc1.y_pred)
    lo, hi = min(combined), max(combined)
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Observed PC1")
    ax.set_ylabel("Predicted PC1 (LOGO-CV)")
    ax.set_title("Observed vs. Predicted (PC1)")


def _r2_violin_panel(
    ax: plt.Axes, permutation_results: Sequence[PermutationResult]
) -> None:
    """Panel 2: every target's observed R^2 vs. the pooled all-targets null R^2.

    Pooling every target's null distribution into one violin assumes every
    target within a pair shares the same genotype count -- true here because
    all targets are drawn from one shared, already-column-filtered
    ``target_clean`` matrix (see ``predict_cross_platform.py``'s
    ``dropna(axis=1, how="any")``, which never changes the row/genotype
    count).
    """
    pooled_null_r2 = np.concatenate(
        [np.asarray(pr.null_r2, dtype=float) for pr in permutation_results]
    )
    observed_r2 = [pr.observed_r2 for pr in permutation_results]

    ax.violinplot([pooled_null_r2], positions=[1], showmedians=True)
    # Seed is jitter-only (horizontal scatter position for readability) --
    # does not affect any reported statistic.
    rng = np.random.default_rng(0)
    x_jitter = 1 + rng.uniform(-0.05, 0.05, size=len(observed_r2))
    ax.scatter(x_jitter, observed_r2, color=_OBSERVED_COLOR, zorder=3, label="Observed")
    ax.set_xticks([1])
    ax.set_xticklabels(["All targets"])
    ax.set_ylabel("R^2")
    ax.set_title("Observed R^2 vs. Permutation Null")
    ax.legend(loc="best", fontsize="small")


def _top_quartile_bar_panel(
    ax: plt.Axes, permutation_results: Sequence[PermutationResult]
) -> None:
    """Panel 3: mean observed vs. mean null top-quartile recovery, across all targets."""
    mean_observed = float(
        np.mean([pr.observed_top_quartile_recovery for pr in permutation_results])
    )
    pooled_null = np.concatenate(
        [
            np.asarray(pr.null_top_quartile_recovery, dtype=float)
            for pr in permutation_results
        ]
    )
    mean_null = float(np.mean(pooled_null))
    ax.bar(
        ["Observed", "Null"],
        [mean_observed, mean_null],
        color=[_OBSERVED_COLOR, _NULL_COLOR],
    )
    ax.set_ylabel("Mean top-quartile recovery")
    ax.set_title("Top-Quartile Recovery: Observed vs. Null")


def create_prediction_figure(
    target_predictions: Sequence[TargetPrediction],
    permutation_results: Sequence[PermutationResult],
    figsize: tuple = (18, 6),
) -> plt.Figure:
    """Build the 3-panel prediction/permutation summary figure for one directed pair.

    Uses only the primary ``reduction_method``'s results (design.md Decision
    9) -- callers pass the primary method's observed
    ``CrossPlatformPredictionResult.predictions`` and
    ``CrossPlatformPermutationResult.predictions``, not any
    ``comparison_methods`` entry.

    Args:
        target_predictions: The primary method's observed per-target
            predictions (Tier 3.5's ``TargetPrediction``, from
            ``CrossPlatformPredictionResult.predictions``) -- must include
            one entry named ``"PC1"``, used for panel 1's scatter.
        permutation_results: The primary method's per-target permutation
            results (``PermutationResult``, from
            ``CrossPlatformPermutationResult.predictions``) -- used for
            panels 2 and 3, pooling every target's null distribution.
        figsize: Figure size as ``(width, height)``.

    Returns:
        A 3-panel ``matplotlib.Figure``: (1) observed-vs-predicted scatter
        (PC1 only), (2) observed R^2 per target vs. pooled all-targets null
        R^2 violin, (3) mean observed vs. mean null top-quartile recovery bar
        chart.

    Raises:
        ValueError: If ``target_predictions`` or ``permutation_results`` is
            empty, or if no target named ``"PC1"`` is present in
            ``target_predictions``.
    """
    if not target_predictions:
        raise ValueError("target_predictions must be non-empty")
    if not permutation_results:
        raise ValueError("permutation_results must be non-empty")

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    try:
        _pc1_scatter_panel(axes[0], target_predictions)
        _r2_violin_panel(axes[1], permutation_results)
        _top_quartile_bar_panel(axes[2], permutation_results)
    except Exception:
        # _pc1_scatter_panel's missing-"PC1" ValueError fires after
        # plt.subplots() has already allocated `fig` -- close it here so a
        # raised panel error never leaks a figure (round-2 /review-pr on PR
        # #201, Behavioural Correctness).
        plt.close(fig)
        raise
    fig.tight_layout()
    return fig
