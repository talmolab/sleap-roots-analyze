"""Tests for cross-platform genotype-effect prediction (Tier 3, #194).

See openspec/changes/add-cross-platform-prediction/ for the full design and
acceptance-criteria oracles this test suite implements against.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from sleap_roots_analyze.cross_platform_prediction import fit_pca_on_fold


class TestFitPcaOnFold:
    """Tests for fit_pca_on_fold (theory.md Section 5's contract)."""

    def test_fit_pca_on_fold_fits_on_train_only(self):
        """Projection depends only on X_train, not X_test."""
        rng = np.random.default_rng(0)
        X_train = rng.standard_normal((18, 5))
        X_test_a = rng.standard_normal((1, 5))
        X_test_b = rng.standard_normal((1, 5))

        out_a = fit_pca_on_fold(X_train, X_test_a, n_components=1)
        out_b = fit_pca_on_fold(X_train, X_test_b, n_components=1)

        expected_pca = PCA(n_components=1).fit(X_train)
        np.testing.assert_allclose(out_a, expected_pca.transform(X_test_a))
        np.testing.assert_allclose(out_b, expected_pca.transform(X_test_b))

    def test_fit_pca_on_fold_output_shape(self):
        """Output shape is (n_test, n_components)."""
        rng = np.random.default_rng(1)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((3, 5))

        for n_components in (1, 2):
            out = fit_pca_on_fold(X_train, X_test, n_components=n_components)
            assert out.shape == (3, n_components)

    def test_fit_pca_on_fold_raises_when_n_traits_less_than_n_components(self):
        """Raises ValueError before calling sklearn when n_traits < n_components."""
        rng = np.random.default_rng(2)
        X_train = rng.standard_normal((18, 2))
        X_test = rng.standard_normal((1, 2))

        with patch("sleap_roots_analyze.cross_platform_prediction.PCA") as mock_pca:
            with pytest.raises(ValueError):
                fit_pca_on_fold(X_train, X_test, n_components=3)
            mock_pca.assert_not_called()

    def test_fit_pca_on_fold_deterministic(self):
        """Calling twice with identical inputs returns identical output."""
        rng = np.random.default_rng(3)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((1, 5))

        out1 = fit_pca_on_fold(X_train, X_test, n_components=1)
        out2 = fit_pca_on_fold(X_train, X_test, n_components=1)

        np.testing.assert_array_equal(out1, out2)

    def test_fit_pca_on_fold_does_not_mutate_inputs(self):
        """X_train and X_test are unchanged after the call."""
        rng = np.random.default_rng(4)
        X_train = rng.standard_normal((18, 5))
        X_test = rng.standard_normal((1, 5))
        X_train_copy = X_train.copy()
        X_test_copy = X_test.copy()

        fit_pca_on_fold(X_train, X_test, n_components=1)

        np.testing.assert_array_equal(X_train, X_train_copy)
        np.testing.assert_array_equal(X_test, X_test_copy)
