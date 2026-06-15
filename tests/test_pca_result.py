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
import pandas as pd
import pytest

from sleap_roots_analyze.pca import perform_pca_analysis
from sleap_roots_analyze.result_types import FeatureContribution, PCAResult


class TestPCAResultJSONRoundTrip:
    """The clean view must serialize to JSON as native Python types."""

    def test_fields_are_native_types_pre_serialization(self, pca_3d_data):
        """Float fields are native ``float`` on the dataclass, not laundered by JSON.

        ``np.float64`` is a subclass of ``float``, so ``json.dumps`` silently
        converts a leaked ``np.float64`` into a native float before any assertion on
        the parsed output — making post-round-trip float checks vacuous. Assert on
        the dataclass fields themselves, where a leak would survive as ``np.float64``.
        """
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df, standardize=True))

        assert type(result.n_components) is int
        assert type(result.standardized) is bool
        for v in (
            *result.explained_variance_ratio,
            *result.eigenvalues,
            *result.cumulative_variance_ratio,
            *(v for row in result.loadings for v in row),
            *(v for row in result.scores for v in row),
        ):
            assert type(v) is float
        for c in result.feature_contributions:
            assert type(c.total_contribution) is float
            assert type(c.fractional_contribution) is float

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


class TestPCAResultJSONBoundaryContract:
    """`to_json` enforces strict, finite-floats-only JSON (the bloom-mcp contract)."""

    def test_to_json_roundtrips_finite_data(self, pca_3d_data):
        """On finite data, to_json emits strict JSON that parses back to to_dict."""
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df, standardize=True))

        # allow_nan=False is the default; parsing back must equal the dict view.
        parsed = json.loads(result.to_json())
        assert parsed == json.loads(json.dumps(result.to_dict()))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_to_json_rejects_non_finite_floats(self, pca_3d_data, bad):
        """A non-finite field raises at to_json instead of emitting invalid JSON.

        Reachable in practice via degenerate PCA (e.g. all-zero loadings →
        ``NaN`` fractional contribution). The plain ``json.dumps`` default would
        silently emit ``NaN``/``Infinity``, which a strict consumer rejects.
        """
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df, standardize=True))
        # frozen dataclass: inject the non-finite value via replace.
        tainted = dataclasses.replace(result, eigenvalues=[bad])

        with pytest.raises(ValueError, match="not JSON compliant"):
            tainted.to_json()

        # And the unguarded default would have laundered it into invalid JSON.
        assert json.dumps(tainted.to_dict()) != ""  # plain dumps does NOT raise

    def test_to_json_forwards_kwargs(self, pca_3d_data):
        """Extra kwargs reach json.dumps (e.g. indent) without breaking the contract."""
        df, _ = pca_3d_data
        result = PCAResult.from_pca_dict(perform_pca_analysis(df, standardize=True))
        assert "\n" in result.to_json(indent=2)


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
        """Legacy dict keeps its keys *and values* — the adapter never mutates it.

        A keys-only check would pass an in-place value/array edit; snapshot the
        actual values (arrays, frame, scalars) and assert deep equality afterwards.
        """
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

        # Deep snapshot of every value, using type-appropriate equality afterwards.
        array_keys = [
            "loadings",
            "transformed_data",
            "explained_variance_ratio",
            "eigenvalues",
            "cumulative_variance_ratio",
        ]
        before_keys = set(d.keys())
        before_arrays = {k: np.asarray(d[k]).copy() for k in array_keys}
        before_fc = d["feature_contributions"].copy(deep=True)
        before_names = list(d["feature_names"])
        before_n = d["n_components_selected"]
        scaler_id, pca_id = id(d["scaler"]), id(d["pca"])

        PCAResult.from_pca_dict(d)

        assert set(d.keys()) == before_keys
        for k in array_keys:
            np.testing.assert_array_equal(np.asarray(d[k]), before_arrays[k])
        pd.testing.assert_frame_equal(d["feature_contributions"], before_fc)
        assert list(d["feature_names"]) == before_names
        assert d["n_components_selected"] == before_n
        assert id(d["scaler"]) == scaler_id and id(d["pca"]) == pca_id


# Numeric tolerance per docs/reproducibility.md (#118), matching test_pipeline_reproduction.
_RTOL = 1e-6
_ATOL = 1e-9
_REPRO_PLATFORMS = ["turface_19", "turface_150", "cylinder", "root_core"]


class TestPCAResultReproductionGolden:
    """PCAResult carries the #120 golden science (epic #130 acceptance).

    Lives with the PCAResult type (#127) rather than the cluster PR — it validates
    PCAResult against the #120 reproduction goldens, using the session reproduction
    fixtures defined in tests/fixtures.py.
    """

    @pytest.mark.parametrize("platform", _REPRO_PLATFORMS)
    def test_viz_pca_typed_view_golden(
        self, final_data_by_platform, viz_pca_by_platform, platform
    ):
        """The typed view reproduces the golden explained variance and round-trips."""
        golden = viz_pca_by_platform[platform]
        n = golden["n_pca_components"]
        res = perform_pca_analysis(
            final_data_by_platform[platform][golden["trait_cols"]],
            standardize=True,
            explained_variance_threshold=0.95,
            random_state=42,
        )
        result = PCAResult.from_pca_dict(
            res, random_state=42, explained_variance_threshold=0.95
        )

        # Golden explained variance asserted via the typed field. The typed view
        # retains the 0.95-threshold components (>= the golden count); summing its
        # first ``n`` ratios reproduces the golden value (matching the legacy-dict test).
        assert result.n_components >= n
        reproduced = float(np.sum(result.explained_variance_ratio[:n]))
        assert np.isclose(
            reproduced, golden["pca_explained_variance"], rtol=_RTOL, atol=_ATOL
        )

        # Serializable with no custom encoder — the whole point of the typed view.
        restored = json.loads(json.dumps(dataclasses.asdict(result)))
        assert restored["explained_variance_ratio"] == result.explained_variance_ratio
        assert restored["n_components"] == result.n_components
