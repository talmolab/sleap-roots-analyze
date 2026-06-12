"""Tests for the serializable ``PCAResult`` dataclass and its adapter (#127).

These cover the serializable-result-types contract: a JSON round-trip that
yields native Python types (the reproducibility CI gate), faithful mapping from
the legacy ``perform_pca_analysis`` dict, ordering/name fidelity of feature
contributions, determinism, provenance stamping, public export, and the
non-breaking / non-mutating guarantee.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from sleap_roots_analyze.pca import perform_pca_analysis
from sleap_roots_analyze.result_types import FeatureContribution, PCAResult


class TestPCAResultJSONRoundTrip:
    """The clean view must serialize to JSON as native Python types."""

    def test_roundtrip_native_types(self, pca_3d_data):
        """json.dumps(asdict(...)) succeeds and round-trips to native types."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, standardize=True)
        result = PCAResult.from_pca_dict(d)

        dumped = json.dumps(dataclasses.asdict(result))
        parsed = json.loads(dumped)

        assert type(parsed["n_components"]) is int
        assert type(parsed["standardized"]) is bool
        for key in (
            "explained_variance_ratio",
            "eigenvalues",
            "cumulative_variance_ratio",
        ):
            assert all(type(v) is float for v in parsed[key]), key
        for matrix_key in ("loadings", "scores"):
            assert all(
                type(v) is float for row in parsed[matrix_key] for v in row
            ), matrix_key

    def test_no_sklearn_objects_in_view(self, pca_3d_data):
        """The clean view drops sklearn objects and redundant frames."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, standardize=True)
        result = PCAResult.from_pca_dict(d)

        view = dataclasses.asdict(result)
        for forbidden in ("pca", "scaler", "feature_metrics_df"):
            assert forbidden not in view

    def test_to_dict_matches_asdict(self, pca_3d_data):
        """``to_dict`` is a convenience over ``dataclasses.asdict``."""
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df))
        assert result.to_dict() == dataclasses.asdict(result)


class TestPCAResultAdapter:
    """``from_pca_dict`` maps the legacy dict faithfully."""

    def test_core_field_shapes(self, pca_3d_data):
        """Arrays are truncated/copied to (n_components) and 2-D nested."""
        df, _ = pca_3d_data  # 3 features
        # Force 2 retained components so the shape checks are non-trivial.
        d = perform_pca_analysis(df, standardize=True, n_components=2)
        result = PCAResult.from_pca_dict(d)

        n = result.n_components
        assert n == int(d["n_components_selected"])
        assert n < df.shape[1]  # truncation is non-trivial

        assert len(result.explained_variance_ratio) == n
        assert len(result.eigenvalues) == n
        assert len(result.cumulative_variance_ratio) == n

        # scores come from transformed_data, shape (n_samples, n_components)
        assert len(result.scores) == df.shape[0]
        assert all(len(row) == n for row in result.scores)
        np.testing.assert_allclose(
            np.asarray(result.scores), d["transformed_data"][:, :n]
        )

        # loadings shape (n_features, n_components)
        assert len(result.loadings) == len(result.feature_names)
        assert all(len(row) == n for row in result.loadings)

    def test_feature_contributions_order_and_names(self, pca_3d_data):
        """Contributions keep DataFrame order and name↔value correspondence."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, standardize=True)
        result = PCAResult.from_pca_dict(d)

        fc_df = d["feature_contributions"]
        # Order: descending by total_contribution, matching the source frame.
        totals = [c.total_contribution for c in result.feature_contributions]
        assert totals == sorted(totals, reverse=True)
        assert [c.feature for c in result.feature_contributions] == list(fc_df.index)
        for c in result.feature_contributions:
            assert isinstance(c, FeatureContribution)
            assert c.total_contribution == pytest.approx(
                fc_df.loc[c.feature, "total_contribution"]
            )
            assert c.fractional_contribution == pytest.approx(
                fc_df.loc[c.feature, "fractional_contribution"]
            )

    @pytest.mark.parametrize("standardize", [True, False])
    def test_standardized_flag(self, pca_3d_data, standardize):
        """``standardized`` reflects whether a scaler was fitted."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, standardize=standardize)
        result = PCAResult.from_pca_dict(d)

        assert result.standardized is standardize
        # serializes cleanly in both branches
        json.dumps(dataclasses.asdict(result))

    def test_n_components_one_keeps_nested_shape(self, pca_variance_threshold_data):
        """A single retained component stays 2-D nested, not flattened."""
        df = pca_variance_threshold_data["one_component"]
        d = perform_pca_analysis(df, standardize=True)
        result = PCAResult.from_pca_dict(d)

        assert result.n_components == 1
        assert all(len(row) == 1 for row in result.loadings)
        assert all(len(row) == 1 for row in result.scores)

    def test_provenance_args_stamped(self, pca_3d_data):
        """random_state / threshold are stamped from adapter args."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, random_state=42)

        stamped = PCAResult.from_pca_dict(
            d, random_state=42, explained_variance_threshold=0.95
        )
        assert stamped.random_state == 42
        assert stamped.explained_variance_threshold == 0.95

        bare = PCAResult.from_pca_dict(d)
        assert bare.random_state is None
        assert bare.explained_variance_threshold is None


class TestPCAResultProperties:
    """Derived properties."""

    def test_cumulative_variance(self, pca_3d_data):
        """``cumulative_variance`` sums retained ratios as a native float."""
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df))

        cv = result.cumulative_variance
        assert type(cv) is float
        assert cv == pytest.approx(sum(result.explained_variance_ratio))
        assert 0.0 < cv <= 1.0 + 1e-9


class TestPCAResultDeterminism:
    """Same seed → identical serialized result (epic #118)."""

    def test_same_random_state_identical(self, pca_3d_data):
        """Identical seeds produce byte-identical serialized results."""
        df, _ = pca_3d_data
        r1 = PCAResult.from_pca_dict(perform_pca_analysis(df, random_state=42))
        r2 = PCAResult.from_pca_dict(perform_pca_analysis(df, random_state=42))

        assert dataclasses.asdict(r1) == dataclasses.asdict(r2)
        assert json.dumps(dataclasses.asdict(r1)) == json.dumps(dataclasses.asdict(r2))


class TestPCAResultExport:
    """Public API surface."""

    def test_importable_from_package_root(self):
        """Both result types are importable and listed once in __all__."""
        import sleap_roots_analyze as sra

        assert sra.PCAResult is PCAResult
        assert sra.FeatureContribution is FeatureContribution
        for name in ("PCAResult", "FeatureContribution"):
            assert name in sra.__all__
        assert len(sra.__all__) == len(set(sra.__all__))


class TestPCAResultNonBreaking:
    """The legacy dict return is preserved and not mutated by the adapter."""

    def test_dict_keys_unchanged_and_nonmutating(self, pca_3d_data):
        """Legacy dict keeps its keys and is not mutated by the adapter."""
        df, _ = pca_3d_data
        d = perform_pca_analysis(df, standardize=True)

        expected_keys = {
            "loadings",
            "transformed_data",
            "explained_variance_ratio",
            "eigenvalues",
            "cumulative_variance_ratio",
            "feature_names",
            "feature_contributions",
            "scaler",
            "pca",
            "n_components_selected",
        }
        assert expected_keys.issubset(d.keys())

        before = set(d.keys())
        PCAResult.from_pca_dict(d)
        assert set(d.keys()) == before
