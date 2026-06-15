"""Tests for the serializable ``HeritabilityResult`` dataclass and adapter (#128).

Covers the serializable-result-types contract for heritability: a native-type
JSON round-trip (the reproducibility CI gate), threshold classification,
``mean_h2`` / ``n_above_threshold`` derivations, separation of failed traits,
determinism, public export, and the non-breaking / non-mutating guarantee.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from sleap_roots_analyze.result_types import HeritabilityResult, TraitHeritability
from sleap_roots_analyze.statistics import calculate_heritability_estimates

TRAIT_COLS = ["trait_high_h2", "trait_moderate_h2", "trait_low_h2"]


def _run(df):
    """Compute the legacy heritability dict for the known-H2 fixture."""
    return calculate_heritability_estimates(
        df, trait_cols=TRAIT_COLS, genotype_col="geno", replicate_col="rep"
    )


class TestHeritabilityResultJSON:
    """The clean view must serialize to JSON as native Python types."""

    def test_fields_are_native_types_pre_serialization(
        self, heritability_data_known_h2
    ):
        """Float fields are native ``float`` on the dataclass, not laundered by JSON.

        ``np.float64`` is a subclass of ``float``, so ``json.dumps`` silently casts
        a leaked ``np.float64`` to native float before any assertion on the parsed
        output. Assert on the dataclass fields themselves, where a leak survives.
        """
        df, _ = heritability_data_known_h2
        result = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)

        assert type(result.threshold) is float
        assert result.per_trait, "expected at least one successful trait"
        for t in result.per_trait:
            assert type(t.h2) is float
            assert type(t.var_genetic) is float
            assert type(t.var_residual) is float
            assert type(t.passed_threshold) is bool
            assert type(t.n_genotypes) is int
            assert type(t.n_observations) is int

    def test_json_roundtrip_native_types(self, heritability_data_known_h2):
        """json.dumps(asdict(...)) round-trips to native Python types."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        parsed = json.loads(json.dumps(dataclasses.asdict(result)))

        assert type(parsed["threshold"]) is float
        assert parsed["per_trait"], "expected at least one successful trait"
        for t in parsed["per_trait"]:
            assert type(t["h2"]) is float
            assert type(t["passed_threshold"]) is bool
            assert type(t["n_genotypes"]) is int
            assert type(t["n_observations"]) is int

    def test_to_json_roundtrips_finite_data(self, heritability_data_known_h2):
        """On finite data, to_json emits strict JSON that parses back to to_dict."""
        df, _ = heritability_data_known_h2
        result = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)
        assert json.loads(result.to_json()) == json.loads(json.dumps(result.to_dict()))

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_to_json_rejects_non_finite_h2(self, heritability_data_known_h2, bad):
        """A non-finite h2 raises at to_json instead of emitting invalid JSON."""
        df, _ = heritability_data_known_h2
        result = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)
        tainted = dataclasses.replace(
            result,
            per_trait=[dataclasses.replace(result.per_trait[0], h2=bad)],
        )
        with pytest.raises(ValueError, match="not JSON compliant"):
            tainted.to_json()

    def test_to_dict_matches_asdict(self, heritability_data_known_h2):
        """``to_dict`` is a convenience over ``dataclasses.asdict``."""
        df, _ = heritability_data_known_h2
        result = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)
        assert result.to_dict() == dataclasses.asdict(result)


class TestHeritabilityResultAdapter:
    """``from_heritability_dict`` maps the legacy dict faithfully."""

    def test_method_and_metadata_handling(self, heritability_data_known_h2):
        """Method is read from metadata; the metadata key is not a trait."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        assert result.method == "mixed_model"
        traits = {t.trait for t in result.per_trait}
        assert "__calculation_metadata__" not in traits
        assert traits <= set(TRAIT_COLS)

    def test_maps_all_trait_fields_from_real_entry(self, heritability_data_known_h2):
        """Every TraitHeritability field is value-asserted against the source entry.

        Guards against a swapped or wrong-key mapping (e.g. var_genetic↔var_residual)
        that h2/passed_threshold-only checks would miss.
        """
        df, _ = heritability_data_known_h2
        d = _run(df)
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        for t in result.per_trait:
            src = d[t.trait]
            assert t.h2 == pytest.approx(float(src["heritability"]))
            assert t.var_genetic == pytest.approx(float(src["var_genetic"]))
            assert t.var_residual == pytest.approx(float(src["var_residual"]))
            assert t.n_genotypes == int(src["n_genotypes"])
            assert t.n_observations == int(src["n_observations"])
            assert t.model_type == str(src["model_type"])
            # var_genetic and var_residual are distinct keys, not swapped.
            assert t.var_genetic != t.var_residual

    def test_run_level_error_surfaced_not_a_fake_trait(self):
        """A lone {"error": ...} run failure populates `error`, not failed_traits."""
        msg = "Missing required columns: ['geno', 'rep']"
        result = HeritabilityResult.from_heritability_dict(
            {"error": msg}, threshold=0.3
        )

        assert result.error == msg
        assert result.per_trait == []
        assert result.failed_traits == []  # NOT ["error"]
        assert "error" not in result.failed_traits
        assert result.method == "unknown"

    def test_per_trait_error_still_a_failed_trait(self, heritability_data_known_h2):
        """A per-trait {"error": ...} entry remains a failed trait, not run-level."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        d["trait_bad"] = {"error": "Trait column 'trait_bad' not found"}

        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        assert result.error is None  # not a run-level failure
        assert "trait_bad" in result.failed_traits

    def test_threshold_classification(self, heritability_data_known_h2):
        """passed_threshold is consistent and preserves H² ordering."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        by_trait = {t.trait: t for t in result.per_trait}
        # The H² formula divides σ²_E by mean reps, so estimates run high but
        # preserve the genetic-variance ordering high > moderate > low.
        assert (
            by_trait["trait_high_h2"].h2
            > by_trait["trait_moderate_h2"].h2
            > by_trait["trait_low_h2"].h2
        )
        for t in result.per_trait:
            assert isinstance(t, TraitHeritability)
            assert 0.0 <= t.h2 <= 1.0
            assert t.passed_threshold == (t.h2 >= result.threshold)
            assert t.passed_threshold is True  # all exceed 0.3

    def test_mean_h2_and_n_above_threshold(self, heritability_data_known_h2):
        """mean_h2 and n_above_threshold derive from successful traits."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        # Discriminating threshold: only the high-H² trait (~0.96) clears 0.9.
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.9)

        mean = result.mean_h2
        assert type(mean) is float
        assert mean == pytest.approx(
            sum(t.h2 for t in result.per_trait) / len(result.per_trait)
        )
        assert result.n_above_threshold == sum(
            t.passed_threshold for t in result.per_trait
        )
        assert result.n_above_threshold == 1  # only trait_high_h2 clears 0.9

    def test_failed_traits_separated(self, heritability_data_known_h2):
        """Error trait entries land in failed_traits, not per_trait."""
        df, _ = heritability_data_known_h2
        d = _run(df)
        # Inject an error trait the way the legacy function would.
        d["trait_missing"] = {"error": "Trait column 'trait_missing' not found"}

        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)

        assert "trait_missing" in result.failed_traits
        assert "trait_missing" not in {t.trait for t in result.per_trait}

    def test_empty_per_trait_mean_is_zero(self):
        """mean_h2 is 0.0 and counts are 0 when no trait succeeds."""
        d = {
            "__calculation_metadata__": {
                "method_used_for_all_traits": "mixed_model",
                "method_consistency": True,
            },
            "t1": {"error": "boom"},
        }
        result = HeritabilityResult.from_heritability_dict(d, threshold=0.3)
        assert result.per_trait == []
        assert result.mean_h2 == 0.0
        assert result.n_above_threshold == 0


class TestHeritabilityResultDeterminism:
    """Same input → identical serialized result."""

    def test_deterministic(self, heritability_data_known_h2):
        """Same input yields an identical serialized result."""
        df, _ = heritability_data_known_h2
        r1 = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)
        r2 = HeritabilityResult.from_heritability_dict(_run(df), threshold=0.3)
        assert dataclasses.asdict(r1) == dataclasses.asdict(r2)


class TestHeritabilityResultExport:
    """Public API surface."""

    def test_importable_from_package_root(self):
        """Both result types are importable and listed once in __all__."""
        import sleap_roots_analyze as sra

        assert sra.HeritabilityResult is HeritabilityResult
        assert sra.TraitHeritability is TraitHeritability
        for name in ("HeritabilityResult", "TraitHeritability"):
            assert name in sra.__all__
        assert len(sra.__all__) == len(set(sra.__all__))


class TestHeritabilityResultNonBreaking:
    """The legacy dict is preserved and not mutated by the adapter."""

    def test_dict_unchanged_and_nonmutating(self, heritability_data_known_h2):
        """Legacy dict keeps its keys *and values* — the adapter never mutates it.

        A keys-only check would pass an in-place value edit; deep-copy the dict and
        assert full value equality after the adapter runs.
        """
        import copy

        df, _ = heritability_data_known_h2
        d = _run(df)

        assert "__calculation_metadata__" in d
        before = copy.deepcopy(d)
        HeritabilityResult.from_heritability_dict(d, threshold=0.3)
        assert d == before
