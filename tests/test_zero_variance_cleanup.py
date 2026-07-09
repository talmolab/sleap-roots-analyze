"""Tests for the configurable zero-variance / constant-trait cleanup filter (#177).

Covers the new granular filter ``remove_zero_variance_traits`` and its wiring into
``apply_data_cleanup_filters`` (final step, post-sample-removal), the
``clean_traits_for_analysis`` re-check after its own ``dropna``, the
``CleanupConfig.min_variance`` config surface, and ``CleanupTraitsStep`` forwarding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.data_cleanup import (
    apply_data_cleanup_filters,
    clean_traits_for_analysis,
    remove_zero_variance_traits,
)


def _wide_frame(traits: dict, n: int) -> pd.DataFrame:
    """Build a metadata + trait wide frame of ``n`` rows."""
    data = {
        "Barcode": [f"b{i}" for i in range(n)],
        "geno": ["A", "B"] * (n // 2),
        "rep": list(range(n)),
    }
    data.update(traits)
    return pd.DataFrame(data)


class TestRemoveZeroVarianceTraits:
    """Unit tests for the granular ``remove_zero_variance_traits`` filter."""

    def test_constant_dropped_varying_kept(self):
        """A constant trait is dropped; a varying trait and metadata are kept."""
        df = pd.DataFrame(
            {
                "Barcode": ["a", "b", "c", "d"],
                "const": [5.0, 5.0, 5.0, 5.0],
                "vary": [1.0, 2.0, 3.0, 4.0],
            }
        )
        filtered, remaining, details = remove_zero_variance_traits(
            df, ["const", "vary"]
        )
        # Return shape mirrors the sibling filters.
        assert "const" not in filtered.columns
        assert "vary" in filtered.columns
        assert "Barcode" in filtered.columns  # metadata preserved
        assert remaining == ["vary"]
        assert details["const"] == {
            "reason": "zero_variance",
            "variance": 0.0,
            "threshold": 0.0,
        }
        assert "vary" not in details

    def test_near_constant_kept_at_zero_threshold(self):
        """A trait with tiny but non-zero variance survives at ``min_variance=0``."""
        df = pd.DataFrame({"t": [1.0, 1.0, 1.0, 1.0000001]})
        _, remaining, details = remove_zero_variance_traits(df, ["t"])
        assert remaining == ["t"]
        assert details == {}

    def test_threshold_boundary_is_inclusive(self):
        """The ``<=`` boundary drops a trait whose variance equals the threshold."""
        # var(ddof=0) of [0, 2] == 1.0 exactly.
        df = pd.DataFrame({"t": [0.0, 2.0]})
        # At the boundary (<=) the trait is dropped.
        _, remaining_drop, details_drop = remove_zero_variance_traits(
            df, ["t"], min_variance=1.0
        )
        assert remaining_drop == []
        assert details_drop["t"]["variance"] == 1.0
        assert details_drop["t"]["threshold"] == 1.0
        # One notch below the variance, the trait is kept.
        _, remaining_keep, details_keep = remove_zero_variance_traits(
            df, ["t"], min_variance=0.9
        )
        assert remaining_keep == ["t"]
        assert details_keep == {}

    def test_uses_population_variance_ddof0(self):
        """Population variance (``ddof=0``) is used, not the sample variance."""
        # [0, 2]: var(ddof=0)=1.0, var(ddof=1)=2.0. min_variance=1.0 drops iff ddof=0.
        df = pd.DataFrame({"t": [0.0, 2.0]})
        _, remaining, _ = remove_zero_variance_traits(df, ["t"], min_variance=1.0)
        assert remaining == []  # dropped => population variance (ddof=0) was used

    def test_empty_frame_flags_nothing(self):
        """An empty frame yields ``var == NaN`` so nothing is flagged."""
        df = pd.DataFrame({"t": pd.Series([], dtype=float)})
        _, remaining, details = remove_zero_variance_traits(df, ["t"])
        assert remaining == ["t"]  # var == NaN, and NaN <= x is False
        assert details == {}

    def test_single_row_is_constant(self):
        """A single-row frame yields ``var == 0`` and the trait is flagged."""
        df = pd.DataFrame({"t": [3.0]})
        _, remaining, details = remove_zero_variance_traits(df, ["t"])
        assert remaining == []
        assert details["t"]["reason"] == "zero_variance"
        assert details["t"]["variance"] == 0.0

    def test_missing_column_is_skipped(self):
        """A trait name absent from the frame is skipped without error."""
        df = pd.DataFrame({"t": [1.0, 2.0]})
        _, remaining, details = remove_zero_variance_traits(df, ["t", "absent"])
        assert remaining == ["t", "absent"]  # matches sibling-filter contract
        assert details == {}

    def test_negative_threshold_disables(self):
        """A negative ``min_variance`` keeps even an exactly-constant trait."""
        df = pd.DataFrame({"const": [5.0, 5.0, 5.0]})
        _, remaining, details = remove_zero_variance_traits(
            df, ["const"], min_variance=-1.0
        )
        assert remaining == ["const"]
        assert details == {}

    def test_all_nan_column_flags_nothing(self):
        """An all-NaN column yields ``var == NaN`` so it is not flagged (kept)."""
        df = pd.DataFrame({"t": [np.nan, np.nan, np.nan]})
        _, remaining, details = remove_zero_variance_traits(df, ["t"])
        assert remaining == ["t"]  # var == NaN, and NaN <= x is False
        assert details == {}

    def test_non_numeric_column_is_skipped(self):
        """A non-numeric (object) trait column is skipped, not raised on (#177 review)."""
        df = pd.DataFrame({"s": ["x", "y", "z"], "vary": [1.0, 2.0, 3.0]})
        # Must not raise a TypeError (var() on strings) — the sibling filters tolerate
        # object dtype, and this filter skips it instead of crashing public callers.
        filtered, remaining, details = remove_zero_variance_traits(df, ["s", "vary"])
        assert "s" not in details  # non-numeric never flagged
        assert "s" in filtered.columns  # and never dropped
        assert remaining == ["s", "vary"]

    def test_apply_filters_tolerates_non_numeric_trait_col(self):
        """apply_data_cleanup_filters does not crash on a hand-built non-numeric col."""
        n = 20
        df = _wide_frame(
            {"vary": [float(i) for i in range(n)], "label": ["a", "b"] * (n // 2)}, n
        )
        # A direct caller passing an object column in trait_cols used to hit a TypeError
        # in the zero-variance step; it should now pass through untouched.
        clean, log = apply_data_cleanup_filters(
            df, ["vary", "label"], replicate_col="rep"
        )
        assert "label" in clean.columns
        assert not [e for e in log["removed_traits"] if e["trait"] == "label"]


class TestApplyDataCleanupFiltersZeroVariance:
    """Wiring of the filter into the cleanup orchestrator (final step)."""

    def test_constant_trait_dropped_and_logged(self):
        """A constant trait is dropped at the final step and logged fully."""
        n = 20
        df = _wide_frame({"vary": [float(i) for i in range(n)], "const": [7.0] * n}, n)
        clean, log = apply_data_cleanup_filters(
            df, ["vary", "const"], replicate_col="rep"
        )
        assert "const" not in clean.columns
        assert "vary" in clean.columns
        zv = [e for e in log["removed_traits"] if e["reason"] == "zero_variance"]
        assert len(zv) == 1
        assert zv[0]["trait"] == "const"
        assert zv[0]["variance"] == 0.0
        assert zv[0]["threshold"] == 0.0
        steps = [
            s
            for s in log["cleanup_steps"]
            if s["step"] == "remove_zero_variance_traits"
        ]
        assert len(steps) == 1
        assert steps[0]["traits_removed"] == 1
        assert steps[0]["remaining_traits"] == 1
        assert log["final_traits"] == 1

    def test_variance_evaluated_after_sample_removal(self):
        """A trait made constant only by sample removal is dropped at the end."""
        # `t` varies only on the two rows that are dropped as NaN-heavy (NaN in `g`),
        # so it is constant on the post-sample-removal frame.
        n = 20
        g = [10.0 + i for i in range(n)]
        t = [5.0] * n
        for r in (0, 1):
            g[r] = np.nan  # row removed by per-sample NaN filter (default 0.0)
            t[r] = 99.0
        df = _wide_frame({"g": g, "t": t}, n)
        clean, log = apply_data_cleanup_filters(df, ["g", "t"], replicate_col="rep")
        assert "t" not in clean.columns  # constant after the 2 rows are dropped
        assert "g" in clean.columns
        assert any(
            e["trait"] == "t" and e["reason"] == "zero_variance"
            for e in log["removed_traits"]
        )

    def test_negative_threshold_disables_in_orchestrator(self):
        """A negative ``min_variance`` disables the orchestrator's final step."""
        n = 20
        df = _wide_frame({"vary": [float(i) for i in range(n)], "const": [7.0] * n}, n)
        clean, log = apply_data_cleanup_filters(
            df, ["vary", "const"], replicate_col="rep", min_variance=-1.0
        )
        assert "const" in clean.columns
        assert not [e for e in log["removed_traits"] if e["reason"] == "zero_variance"]


class TestCleanTraitsForAnalysisZeroVariance:
    """Constant-free guarantee at the analysis-ready entry point."""

    def test_constant_trait_dropped_and_named(self):
        """A constant trait is absent from the output and named in the log."""
        n = 20
        df = _wide_frame({"vary": [float(i) for i in range(n)], "const": [7.0] * n}, n)
        clean_df, surviving, log = clean_traits_for_analysis(
            df, trait_cols=["vary", "const"], replicate_col="rep"
        )
        assert "const" not in surviving
        assert "const" not in clean_df.columns
        assert "vary" in surviving
        assert any(
            e["trait"] == "const" and e["reason"] == "zero_variance"
            for e in log["removed_traits"]
        )

    def test_recheck_after_entry_point_dropna(self):
        """A trait made constant by the entry point's own dropna is re-caught."""
        # With a loosened per-sample NaN budget, the orchestrator retains a row that
        # carries a residual NaN; the entry point's own dropna then removes it, which
        # turns `t` constant — caught by the post-dropna re-check.
        n = 20
        g = [10.0 + i for i in range(n)]
        t = [5.0] * n
        g[0] = np.nan
        t[0] = 42.0  # the only varying value of `t`, on the row dropna will remove
        df = _wide_frame({"g": g, "t": t}, n)
        clean_df, surviving, log = clean_traits_for_analysis(
            df,
            trait_cols=["g", "t"],
            replicate_col="rep",
            max_nans_per_sample=0.5,  # 1/2 NaN == 0.5, not > 0.5 -> row retained
        )
        assert "t" not in surviving
        assert "t" not in clean_df.columns
        assert "g" in surviving
        assert any(
            e["trait"] == "t" and e["reason"] == "zero_variance"
            for e in log["removed_traits"]
        )
        # The re-check keeps the summary fields self-consistent (I1): both final_traits
        # and traits_retained_fraction reflect the extra drop, not just final_traits.
        assert log["original_traits"] == 2
        assert log["final_traits"] == 1
        assert log["traits_retained_fraction"] == 0.5

    def test_all_constant_still_raises_non_constant_guard(self):
        """All-constant input still raises the existing check-(4) guard."""
        n = 20
        df = _wide_frame({"c1": [1.0] * n, "c2": [2.0] * n}, n)
        with pytest.raises(ValueError, match="non-constant"):
            clean_traits_for_analysis(df, trait_cols=["c1", "c2"], replicate_col="rep")

    def test_effective_thresholds_records_min_variance(self):
        """The entry point records ``min_variance`` in effective_thresholds."""
        n = 12
        df = _wide_frame(
            {
                "trait1": [float(i + 1) for i in range(n)],
                "trait2": [float(2 * i + 1) for i in range(n)],
            },
            n,
        )
        _, _, log = clean_traits_for_analysis(
            df, trait_cols=["trait1", "trait2"], replicate_col="rep"
        )
        assert log["effective_thresholds"]["min_variance"] == 0.0


class TestCleanupConfigMinVariance:
    """Config surface + pipeline-step forwarding."""

    def test_cleanup_config_default(self):
        """``CleanupConfig().min_variance`` defaults to ``0.0``."""
        from sleap_roots_analyze.pipeline import CleanupConfig

        assert CleanupConfig().min_variance == 0.0

    def test_cleanup_step_forwards_min_variance(self, tmp_path):
        """``CleanupTraitsStep`` forwards ``min_variance`` so constants are dropped."""
        from sleap_roots_analyze.pipeline import (
            CleanupConfig,
            ColumnConfig,
            DataConfig,
            QCPipelineConfig,
        )
        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import CleanupTraitsStep

        n = 20
        df = pd.DataFrame(
            {
                "Barcode": [f"p{i}" for i in range(n)],
                "geno": ["A", "B"] * (n // 2),
                "rep": [1, 2] * (n // 2),
                "good": np.linspace(1.0, 20.0, n),
                "const_trait": [3.0] * n,
            }
        )
        config = QCPipelineConfig(
            pipeline_name="t",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            cleanup=CleanupConfig(
                max_zeros_per_trait=0.5,
                max_nans_per_trait=0.3,
                max_nan_fraction=0.5,
                min_samples_per_trait=10,
            ),
        )
        prev = StepResult(
            data=df,
            metadata={
                "trait_column_names": ["good", "const_trait"],
                "metadata_column_names": ["Barcode", "geno", "rep"],
            },
            files_generated=[],
        )
        result = CleanupTraitsStep().execute(df, config, tmp_path, prev)

        log = result.metadata["cleanup_log"]
        zv = [t for t in log["removed_traits"] if t.get("reason") == "zero_variance"]
        assert len(zv) == 1  # the constant trait was dropped by the step
        assert result.metadata["traits_final"] == 1  # only `good` survives

    def test_cleanup_step_min_variance_negative_disables(self, tmp_path):
        """A negative ``cleanup.min_variance`` keeps constant traits through the step."""
        from sleap_roots_analyze.pipeline import (
            CleanupConfig,
            ColumnConfig,
            DataConfig,
            QCPipelineConfig,
        )
        from sleap_roots_analyze.pipeline.core import StepResult
        from sleap_roots_analyze.pipeline.steps import CleanupTraitsStep

        n = 20
        df = pd.DataFrame(
            {
                "Barcode": [f"p{i}" for i in range(n)],
                "geno": ["A", "B"] * (n // 2),
                "rep": [1, 2] * (n // 2),
                "good": np.linspace(1.0, 20.0, n),
                "const_trait": [3.0] * n,
            }
        )
        config = QCPipelineConfig(
            pipeline_name="t",
            columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
            data=DataConfig(csv_path="dummy.csv"),
            cleanup=CleanupConfig(
                max_zeros_per_trait=0.5,
                max_nans_per_trait=0.3,
                max_nan_fraction=0.5,
                min_samples_per_trait=10,
                min_variance=-1.0,  # disable the zero-variance filter
            ),
        )
        prev = StepResult(
            data=df,
            metadata={
                "trait_column_names": ["good", "const_trait"],
                "metadata_column_names": ["Barcode", "geno", "rep"],
            },
            files_generated=[],
        )
        result = CleanupTraitsStep().execute(df, config, tmp_path, prev)

        log = result.metadata["cleanup_log"]
        assert not [
            t for t in log["removed_traits"] if t.get("reason") == "zero_variance"
        ]
        assert result.metadata["traits_final"] == 2  # constant trait retained
