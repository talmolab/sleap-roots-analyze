"""Tests for the serializable UMAP result dataclass and adapter (#180).

Covers the serializable-result-types contract for UMAP: native-type JSON
round-trips (the reproducibility CI gate), adapter field mapping (embedding shape,
``n_components``/``n_samples`` derivation, ``feature_names``, ``standardized``), the
``random_state`` argument/dict resolution, the additive ``feature_names``/``random_state``
producer keys, determinism via the typed view, public export, and the non-breaking /
non-mutating guarantee. Sibling to ``tests/test_cluster_result.py`` (#129).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

# Import fixtures
from tests.fixtures import traits_summary_sample  # noqa: F401

from sleap_roots_analyze.result_types import UMAPResult
from sleap_roots_analyze.umap import perform_umap_analysis

# The four clean numeric feature columns used throughout tests/test_umap.py.
FEATURE_COLS = [
    "network_length_mean",
    "network_length_max",
    "chull_max_width_mean",
    "crown_lengths_mean_mean",
]


@pytest.fixture
def umap_dict(traits_summary_sample):
    """A real ``perform_umap_analysis`` dict (10 samples, 2 components, seed 42)."""
    return perform_umap_analysis(
        traits_summary_sample,
        FEATURE_COLS,
        n_neighbors=5,  # small value for the 10-row sample
        min_dist=0.1,
        n_components=2,
        random_state=42,
    )


class TestUMAPResultJSON:
    """UMAP clean view serializes to native Python types."""

    def test_fields_are_native_types_pre_serialization(self, umap_dict):
        """Fields are native types on the dataclass, not laundered by JSON.

        ``np.float64`` is a subclass of ``float``, so a JSON round-trip silently
        casts a leak to native float before any assertion — assert on the fields.
        """
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)

        assert type(result.n_neighbors) is int
        assert type(result.n_components) is int
        assert type(result.n_samples) is int
        assert type(result.min_dist) is float
        assert type(result.standardized) is bool
        assert type(result.random_state) is int
        assert all(type(v) is float for row in result.embedding for v in row)
        assert all(type(v) is str for v in result.feature_names)

    def test_json_roundtrip_native_types(self, umap_dict):
        """UMAP view round-trips to native types with values preserved."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)
        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert type(parsed["n_neighbors"]) is int
        assert type(parsed["n_components"]) is int
        assert type(parsed["n_samples"]) is int
        assert type(parsed["min_dist"]) is float
        assert type(parsed["standardized"]) is bool
        assert all(type(v) is float for row in parsed["embedding"] for v in row)
        assert all(type(v) is str for v in parsed["feature_names"])
        np.testing.assert_allclose(
            np.asarray(parsed["embedding"]),
            np.asarray(dataclasses.asdict(result)["embedding"]),
        )

    def test_no_sklearn_object_in_clean_view(self, umap_dict):
        """The fitted reducer/scaler are excluded from the clean view."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)
        d = dataclasses.asdict(result)

        assert "reducer" not in d
        assert "scaler" not in d

    def test_n_samples_and_n_components_are_materialized_fields(self, umap_dict):
        """n_samples / n_components are stored fields, present in the payload."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)
        d = dataclasses.asdict(result)

        assert result.n_samples == len(result.embedding)
        assert result.n_components == len(result.embedding[0])
        assert "n_samples" in d
        assert "n_components" in d

    def test_to_json_roundtrips_finite_data(self, umap_dict):
        """On finite data, to_json emits strict JSON that parses back to to_dict."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)
        assert json.loads(result.to_json()) == json.loads(json.dumps(result.to_dict()))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_to_json_rejects_non_finite_embedding(self, umap_dict, bad):
        """A non-finite embedding value raises at to_json vs emitting invalid JSON."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=42)
        tainted_embedding = [list(row) for row in result.embedding]
        tainted_embedding[0][0] = bad
        tainted = dataclasses.replace(result, embedding=tainted_embedding)
        with pytest.raises(ValueError, match="not JSON compliant"):
            tainted.to_json()


class TestUMAPAdapter:
    """from_umap_dict maps the legacy dict faithfully."""

    def test_adapter_maps_core_fields(self, umap_dict):
        """Adapter maps embedding shape, counts, and scalars by value."""
        d = umap_dict
        result = UMAPResult.from_umap_dict(d)

        n_samples, n_components = np.asarray(d["embedding"]).shape
        assert result.n_samples == n_samples
        assert result.n_components == n_components
        assert len(result.embedding) == n_samples
        assert all(len(row) == n_components for row in result.embedding)
        assert result.n_neighbors == int(d["n_neighbors"])
        assert result.min_dist == pytest.approx(float(d["min_dist"]))
        assert result.feature_names == [str(c) for c in FEATURE_COLS]
        np.testing.assert_allclose(
            np.asarray(result.embedding), np.asarray(d["embedding"])
        )
        assert result.standardized is True

    def test_n_components_one_preserves_nested_shape(self, traits_summary_sample):
        """n_components == 1 keeps (n_samples, 1) — inner rows are one-element lists."""
        d = perform_umap_analysis(
            traits_summary_sample,
            FEATURE_COLS,
            n_neighbors=5,
            n_components=1,
            random_state=42,
        )
        result = UMAPResult.from_umap_dict(d)

        assert result.n_components == 1
        assert all(len(row) == 1 for row in result.embedding)

    def test_standardized_false_when_scaler_none(self):
        """Adapter derives standardized from scaler presence (guards a hardcoded True)."""
        d = {
            "embedding": [[0.1, 0.2], [0.3, 0.4]],
            "n_neighbors": 1,
            "min_dist": 0.1,
            "feature_names": ["a", "b"],
            "scaler": None,
        }
        result = UMAPResult.from_umap_dict(d)

        assert result.standardized is False
        result.to_json()  # finite -> succeeds

    def test_random_state_explicit_arg_wins(self, umap_dict):
        """An explicit random_state overrides the dict's echoed seed."""
        result = UMAPResult.from_umap_dict(umap_dict, random_state=7)
        assert result.random_state == 7

    def test_random_state_falls_back_to_dict(self, umap_dict):
        """With no argument, the seed comes from the dict's echoed random_state."""
        result = UMAPResult.from_umap_dict(umap_dict)
        assert result.random_state == 42

    def test_random_state_explicit_arg_wins_with_zero(self, umap_dict):
        """An explicit random_state=0 is preserved, not treated as falsy/absent.

        Guards a regression to the idiomatic-but-wrong ``random_state or
        d.get("random_state")`` pattern, which would silently discard an explicit
        seed of ``0`` and fall through to the dict's echoed seed instead.
        """
        result = UMAPResult.from_umap_dict(umap_dict, random_state=0)
        assert result.random_state == 0
        assert type(result.random_state) is int

    def test_random_state_falls_back_to_dict_with_zero(self):
        """With no argument, an echoed dict seed of 0 is preserved, not treated as absent.

        Guards a regression to a truthiness-based fallback (e.g. ``d.get("random_state")
        or None``), which would silently collapse a legitimate seed of ``0`` to ``None``.
        """
        d = {
            "embedding": [[0.1, 0.2], [0.3, 0.4]],
            "n_neighbors": 1,
            "min_dist": 0.1,
            "feature_names": ["a", "b"],
            "scaler": None,
            "random_state": 0,
        }
        result = UMAPResult.from_umap_dict(d)

        assert result.random_state == 0
        assert type(result.random_state) is int

    def test_random_state_none_when_absent(self):
        """With no argument and no echoed seed, random_state is None -> JSON null."""
        d = {
            "embedding": [[0.1, 0.2]],
            "n_neighbors": 1,
            "min_dist": 0.1,
            "feature_names": ["a", "b"],
            "scaler": None,
        }
        result = UMAPResult.from_umap_dict(d)

        assert result.random_state is None
        assert json.loads(result.to_json())["random_state"] is None

    def test_adapter_does_not_mutate_dict(self, umap_dict):
        """Adapter does not mutate its input dict (keys, objects, and values).

        The dict carries the fitted ``reducer``/``scaler`` (no ``__eq__``), so a
        deep-copy value compare would spuriously differ — check object identity for
        those and value-equality for the serializable members instead.
        """
        d = umap_dict
        keys_before = set(d.keys())
        reducer_before, scaler_before = d["reducer"], d["scaler"]
        embedding_before = np.array(d["embedding"], copy=True)
        scalars_before = {
            k: d[k]
            for k in ("n_neighbors", "min_dist", "feature_names", "random_state")
        }

        UMAPResult.from_umap_dict(d, random_state=42)

        assert set(d.keys()) == keys_before
        assert d["reducer"] is reducer_before
        assert d["scaler"] is scaler_before
        np.testing.assert_array_equal(np.asarray(d["embedding"]), embedding_before)
        for k, v in scalars_before.items():
            assert d[k] == v


class TestUMAPDeterminism:
    """Same seed -> identical embedding via the typed view."""

    def test_same_seed_identical_embedding(self, traits_summary_sample):
        """Re-running UMAP with the same seed yields an identical embedding."""
        r1 = UMAPResult.from_umap_dict(
            perform_umap_analysis(
                traits_summary_sample,
                FEATURE_COLS,
                n_neighbors=5,
                n_components=2,
                random_state=42,
            )
        )
        r2 = UMAPResult.from_umap_dict(
            perform_umap_analysis(
                traits_summary_sample,
                FEATURE_COLS,
                n_neighbors=5,
                n_components=2,
                random_state=42,
            )
        )
        assert r1.embedding == r2.embedding


class TestUMAPProducerEnrichment:
    """perform_umap_analysis additively carries feature_names + random_state."""

    def test_producer_returns_feature_names_and_random_state(
        self, traits_summary_sample
    ):
        """Existing keys preserved; new feature_names/random_state keys added."""
        d = perform_umap_analysis(
            traits_summary_sample,
            FEATURE_COLS,
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=42,
        )

        for key in ("embedding", "reducer", "scaler", "n_neighbors", "min_dist"):
            assert key in d  # non-breaking
        assert d["feature_names"] == FEATURE_COLS
        assert d["random_state"] == 42

    def test_end_to_end_from_producer(self, traits_summary_sample):
        """Producer dict -> UMAPResult with no explicit seed (resolved from the dict)."""
        d = perform_umap_analysis(
            traits_summary_sample,
            FEATURE_COLS,
            n_neighbors=5,
            min_dist=0.1,
            n_components=2,
            random_state=42,
        )
        result = UMAPResult.from_umap_dict(d)

        assert len(result.embedding) == len(traits_summary_sample)
        assert result.feature_names == FEATURE_COLS
        assert result.n_components == 2
        assert result.n_samples == len(traits_summary_sample)
        assert result.n_neighbors == 5
        assert result.min_dist == pytest.approx(0.1)
        assert result.random_state == 42  # from the echoed dict key
        result.to_json()  # succeeds

    def test_n_neighbors_clamped_value_reflected(self, traits_summary_sample):
        """The effective (clamped) n_neighbors is what the result captures."""
        df = traits_summary_sample  # 10 rows
        d = perform_umap_analysis(
            df,
            FEATURE_COLS,
            n_neighbors=len(df) + 50,  # forces clamp to n_samples - 1
            n_components=2,
            random_state=42,
        )
        result = UMAPResult.from_umap_dict(d)

        assert result.n_neighbors == len(df) - 1
        assert type(result.n_neighbors) is int


class TestUMAPResultExport:
    """Public API surface."""

    def test_importable_from_package_root(self):
        """UMAPResult is importable from the package root and in __all__."""
        import sleap_roots_analyze as sra

        assert sra.UMAPResult is UMAPResult
        assert "UMAPResult" in sra.__all__
        assert len(sra.__all__) == len(set(sra.__all__))

    def test_listed_in_result_types_all(self):
        """UMAPResult is listed in result_types.__all__."""
        from sleap_roots_analyze import result_types

        assert "UMAPResult" in result_types.__all__
