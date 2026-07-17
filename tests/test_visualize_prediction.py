"""Tests for create_prediction_figure() (Tier 4, #200, tasks.md Section 6).

Distinct from ``tests/test_step_visualize_prediction.py`` (Section 7's
``VisualizePredictionStep`` wiring tests) -- see proposal.md's naming-
collision note: this repo has two new ``visualize_prediction.py`` source
files (this one, and ``pipeline/steps/visualize_prediction.py``), sharing a
basename in two different subpackages.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleap_roots_analyze.result_types import PermutationResult, TargetPrediction
from sleap_roots_analyze.visualize_prediction import create_prediction_figure


def _target_prediction(target_name, y_true, y_pred, r2=0.8, rmse=1.0, rho=0.7):
    n = len(y_true)
    return TargetPrediction(
        target_name=target_name,
        r2=r2,
        rmse=rmse,
        spearman_rho=rho,
        spearman_p=0.05,
        genotype_names=[f"g{i}" for i in range(n)],
        y_true=list(y_true),
        y_pred=list(y_pred),
    )


def _permutation_result(
    target_name, observed_r2, null_r2, observed_top_quartile_recovery, null_tqr
):
    n = len(null_r2)
    return PermutationResult(
        target_name=target_name,
        observed_r2=observed_r2,
        observed_rmse=1.0,
        observed_spearman_rho=0.7,
        observed_top_quartile_recovery=observed_top_quartile_recovery,
        null_r2=list(null_r2),
        null_rmse=[1.0] * n,
        null_spearman_rho=[0.0] * n,
        null_top_quartile_recovery=list(null_tqr),
        p_value_r2=0.05,
        p_value_rmse=0.5,
        p_value_spearman_rho=0.05,
        n_permutations=n,
    )


def test_create_prediction_figure_scatter_panel_uses_pc1_target_only():
    """The obs-vs-pred scatter panel's points correspond only to the PC1 target."""
    pc1_y_true = [1.0, 2.0, 3.0, 4.0]
    pc1_y_pred = [1.1, 1.9, 3.2, 3.8]
    target_predictions = [
        _target_prediction(
            "trait_a", [10.0, 20.0, 30.0, 40.0], [11.0, 19.0, 29.0, 42.0]
        ),
        _target_prediction("PC1", pc1_y_true, pc1_y_pred),
    ]
    permutation_results = [
        _permutation_result("trait_a", 0.7, [0.1, 0.2, 0.3], 0.5, [0.2, 0.3, 0.4]),
        _permutation_result("PC1", 0.8, [0.1, 0.2, 0.3], 0.6, [0.2, 0.3, 0.4]),
    ]

    fig = create_prediction_figure(target_predictions, permutation_results)

    scatter_ax = fig.axes[0]
    offsets = scatter_ax.collections[0].get_offsets()
    plotted_x = sorted(offsets[:, 0].tolist())
    plotted_y = sorted(offsets[:, 1].tolist())
    assert plotted_x == sorted(pc1_y_true)
    assert plotted_y == sorted(pc1_y_pred)


def test_create_prediction_figure_violin_panel_pools_all_targets_nulls():
    """The violin panel's null data is every target's null_r2 concatenated."""
    target_predictions = [
        _target_prediction("trait_a", [1.0, 2.0], [1.1, 1.9]),
        _target_prediction("PC1", [3.0, 4.0], [3.1, 3.9]),
    ]
    null_a = [0.1, 0.2, 0.3]
    null_pc1 = [0.4, 0.5, 0.6]
    permutation_results = [
        _permutation_result("trait_a", 0.7, null_a, 0.5, [0.2, 0.3, 0.4]),
        _permutation_result("PC1", 0.8, null_pc1, 0.6, [0.2, 0.3, 0.4]),
    ]

    fig = create_prediction_figure(target_predictions, permutation_results)

    violin_ax = fig.axes[1]
    # The violin body's vertices span the pooled null distribution's range.
    violin_body = violin_ax.collections[0]
    all_y = np.concatenate([path.vertices[:, 1] for path in violin_body.get_paths()])
    pooled_null = np.array(null_a + null_pc1)
    assert all_y.min() <= pooled_null.min()
    assert all_y.max() >= pooled_null.max()
    # Observed points: one per target.
    observed_scatter = violin_ax.collections[-1]
    observed_y = sorted(observed_scatter.get_offsets()[:, 1].tolist())
    assert observed_y == sorted([0.7, 0.8])


def test_create_prediction_figure_bar_chart_shows_observed_vs_null_mean():
    """The bar chart's two bars equal mean observed and mean null top-quartile recovery."""
    target_predictions = [
        _target_prediction("trait_a", [1.0, 2.0], [1.1, 1.9]),
        _target_prediction("PC1", [3.0, 4.0], [3.1, 3.9]),
    ]
    tqr_null_a = [0.2, 0.3, 0.4]
    tqr_null_pc1 = [0.5, 0.6, 0.7]
    permutation_results = [
        _permutation_result("trait_a", 0.7, [0.1, 0.2, 0.3], 0.5, tqr_null_a),
        _permutation_result("PC1", 0.8, [0.1, 0.2, 0.3], 0.9, tqr_null_pc1),
    ]

    fig = create_prediction_figure(target_predictions, permutation_results)

    bar_ax = fig.axes[2]
    bar_heights = [patch.get_height() for patch in bar_ax.patches]
    expected_observed_mean = np.mean([0.5, 0.9])
    expected_null_mean = np.mean(tqr_null_a + tqr_null_pc1)
    assert bar_heights[0] == pytest.approx(expected_observed_mean)
    assert bar_heights[1] == pytest.approx(expected_null_mean)


def test_create_prediction_figure_returns_a_figure_with_three_axes():
    """create_prediction_figure returns a matplotlib.Figure with exactly 3 axes."""
    target_predictions = [_target_prediction("PC1", [1.0, 2.0], [1.1, 1.9])]
    permutation_results = [
        _permutation_result("PC1", 0.8, [0.1, 0.2, 0.3], 0.6, [0.2, 0.3, 0.4])
    ]

    fig = create_prediction_figure(target_predictions, permutation_results)

    assert len(fig.axes) == 3


def test_create_prediction_figure_handles_single_target():
    """A single (PC1-only) target still builds successfully with all 3 panels."""
    target_predictions = [_target_prediction("PC1", [1.0, 2.0, 3.0], [1.1, 1.9, 3.2])]
    permutation_results = [
        _permutation_result("PC1", 0.8, [0.1, 0.2, 0.3, 0.4], 0.6, [0.2, 0.3, 0.4, 0.5])
    ]

    fig = create_prediction_figure(target_predictions, permutation_results)

    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert ax is not None
