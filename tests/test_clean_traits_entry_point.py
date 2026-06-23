"""Tests for the public minimal-QC entry point ``clean_traits_for_analysis``.

Covers the composition (cleanup -> validate), the analysis-readiness gates, the
public API surface, and the single-source-of-truth refactor of QC step 03.
See OpenSpec change ``add-clean-traits-entry-point`` (issue #164).
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

import sleap_roots_analyze as sra
from sleap_roots_analyze.data_cleanup import (
    apply_data_cleanup_filters,
    build_clean_validation_report,
    clean_traits_for_analysis,
    validate_clean_traits,
)
from sleap_roots_analyze.pca import perform_pca_analysis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def nan_heavy_data():
    """20 samples: two clean traits + one NaN-heavy trait.

    The NaN-heavy trait exceeds max_nans_per_trait (so it is dropped, saving its
    rows), while the clean traits clear min_samples_per_trait. Naive dropna()
    would drop every row that has a NaN in the NaN-heavy trait.
    """
    np.random.seed(42)
    nan_heavy = np.arange(20, dtype=float)
    nan_heavy[:10] = np.nan  # 50% NaN -> dropped at max_nans_per_trait=0.3
    return pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(20)],
            "geno": ["G1"] * 10 + ["G2"] * 10,
            "rep": list(range(1, 11)) * 2,
            "trait_good1": np.random.randn(20) + 10,
            "trait_good2": np.random.randn(20) + 5,
            "trait_nan_heavy": nan_heavy,
        }
    )


@pytest.fixture
def quarter_nan_data():
    """A trait at 25% NaN: survives default (0.3) but dropped at 0.1."""
    np.random.seed(7)
    quarter = np.random.randn(20) + 3
    quarter[:5] = np.nan  # 25% NaN
    return pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(20)],
            "geno": ["G1"] * 10 + ["G2"] * 10,
            "rep": list(range(1, 11)) * 2,
            "trait_good": np.random.randn(20) + 10,
            "trait_quarter_nan": quarter,
        }
    )


# ---------------------------------------------------------------------------
# Composition + return shape (spec: Analysis-Ready Cleanup Entry Point)
# ---------------------------------------------------------------------------
def test_returns_three_tuple_and_survivor_derivation(nan_heavy_data):
    """Returns a 3-tuple; surviving traits are those still present as columns."""
    input_traits = ["trait_good1", "trait_good2", "trait_nan_heavy"]
    clean_df, trait_cols, cleanup_log = clean_traits_for_analysis(
        nan_heavy_data, trait_cols=input_traits, min_samples_per_trait=2
    )
    assert isinstance(clean_df, pd.DataFrame)
    assert isinstance(cleanup_log, dict)
    # Survivor derivation: exactly the input traits still present as columns.
    assert trait_cols == [c for c in input_traits if c in clean_df.columns]
    assert "trait_nan_heavy" not in trait_cols  # dropped by cleanup


def test_no_nan_in_output_and_pca_dropna_is_noop(nan_heavy_data):
    """Output has no NaNs and PCA's internal row dropna removes nothing."""
    clean_df, trait_cols, _ = clean_traits_for_analysis(
        nan_heavy_data, min_samples_per_trait=2
    )
    assert clean_df[trait_cols].isna().sum().sum() == 0
    # Row dropna would remove nothing.
    assert len(clean_df[trait_cols].dropna()) == len(clean_df)
    # And PCA runs, using every surviving row.
    result = perform_pca_analysis(clean_df[trait_cols])
    assert result["data_processed"].shape[0] == len(clean_df)


def test_sample_loss_minimized_vs_naive_dropna(nan_heavy_data):
    """Retains more samples than a naive df.dropna()."""
    clean_df, _, _ = clean_traits_for_analysis(nan_heavy_data, min_samples_per_trait=2)
    assert len(clean_df) > len(nan_heavy_data.dropna())


def test_caller_supplied_trait_cols_bypasses_inference(nan_heavy_data):
    """Explicit trait_cols bypass get_trait_columns inference."""
    # Only declare one trait; the others must be treated as non-traits.
    clean_df, trait_cols, _ = clean_traits_for_analysis(
        nan_heavy_data, trait_cols=["trait_good1"], min_samples_per_trait=2
    )
    assert trait_cols == ["trait_good1"]


def test_cleanup_kwargs_pass_through_and_effective_thresholds_recorded(
    quarter_nan_data,
):
    """Threshold kwargs reach the cleanup function and are recorded in the log."""
    # Default (0.3): the 25%-NaN trait survives.
    _, traits_default, log_default = clean_traits_for_analysis(
        quarter_nan_data, min_samples_per_trait=2
    )
    assert "trait_quarter_nan" in traits_default
    assert log_default["effective_thresholds"]["max_nans_per_trait"] == 0.3

    # Tightened (0.1): it is dropped.
    _, traits_tight, log_tight = clean_traits_for_analysis(
        quarter_nan_data, min_samples_per_trait=2, max_nans_per_trait=0.1
    )
    assert "trait_quarter_nan" not in traits_tight
    assert log_tight["effective_thresholds"]["max_nans_per_trait"] == 0.1


def test_default_column_names_and_optional_replicate():
    """Default geno/rep names work and replicate_col=None is honored."""
    np.random.seed(1)
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(6)],
            "geno": ["A"] * 3 + ["B"] * 3,
            "trait_a": np.random.randn(6) + 1,
            "trait_b": np.random.randn(6) + 2,
        }
    )
    # replicate_col=None must be honored (no "rep" column present).
    clean_df, trait_cols, _ = clean_traits_for_analysis(
        df, replicate_col=None, min_samples_per_trait=2
    )
    assert set(trait_cols) == {"trait_a", "trait_b"}
    # Metadata columns are excluded from traits and preserved in output.
    assert "geno" not in trait_cols and "geno" in clean_df.columns


def test_validation_summary_recorded(nan_heavy_data):
    """cleanup_log carries a validation_summary block."""
    _, trait_cols, log = clean_traits_for_analysis(
        nan_heavy_data, min_samples_per_trait=2
    )
    summary = log["validation_summary"]
    assert summary["n_surviving_traits"] == len(trait_cols)
    assert summary["n_nonconstant_traits"] >= 1
    assert summary["n_samples"] == 20


# ---------------------------------------------------------------------------
# Analysis-readiness validation (ordered, distinct errors)
# ---------------------------------------------------------------------------
def test_raises_on_empty_input_before_delegating():
    """Empty input raises the entry point's own error before delegating."""
    with pytest.raises(ValueError, match="no rows"):
        clean_traits_for_analysis(pd.DataFrame())


def test_raises_when_no_trait_columns():
    """Raises when no trait columns can be resolved."""
    df = pd.DataFrame({"Barcode": ["a", "b"], "geno": ["G1", "G2"], "rep": [1, 2]})
    with pytest.raises(ValueError, match="no trait columns"):
        clean_traits_for_analysis(df)


def test_raises_when_fewer_than_two_samples():
    """Raises, naming the count, when fewer than 2 samples survive."""
    df = pd.DataFrame(
        {
            "Barcode": ["BC0"],
            "geno": ["G1"],
            "rep": [1],
            "trait_a": [1.0],
            "trait_b": [2.0],
        }
    )
    with pytest.raises(ValueError, match=r"only 1 sample"):
        clean_traits_for_analysis(df, min_samples_per_trait=1)


def test_raises_when_only_constant_trait_survives():
    """Raises when the only surviving trait is constant (var(ddof=0)==0)."""
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(6)],
            "geno": ["A"] * 3 + ["B"] * 3,
            "rep": [1, 2, 3, 1, 2, 3],
            "trait_const": [5.0] * 6,  # var(ddof=0) == 0
        }
    )
    with pytest.raises(ValueError, match="non-constant"):
        clean_traits_for_analysis(df, min_samples_per_trait=2)


def test_succeeds_when_one_trait_varies_among_several():
    """Passes when at least one trait varies among several."""
    np.random.seed(3)
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(6)],
            "geno": ["A"] * 3 + ["B"] * 3,
            "rep": [1, 2, 3, 1, 2, 3],
            "trait_const": [5.0] * 6,
            "trait_varying": np.random.randn(6) + 1,
        }
    )
    clean_df, trait_cols, _ = clean_traits_for_analysis(df, min_samples_per_trait=2)
    # A constant trait alongside a varying one does not fail the gate.
    assert "trait_varying" in trait_cols
    assert len(clean_df) == 6


# ---------------------------------------------------------------------------
# Public API surface (#116 pattern)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "apply_data_cleanup_filters",
        "validate_clean_traits",
        "build_clean_validation_report",
        "clean_traits_for_analysis",
    ],
)
def test_public_api_exposed(name):
    """Each new function is in __all__, callable, identity-equal, type-hint-resolvable."""
    assert name in sra.__all__
    assert sra.__all__.count(name) == 1
    fn = getattr(sra, name)
    assert callable(fn)
    # Identity-equal to the module definition.
    assert fn is getattr(
        __import__("sleap_roots_analyze.data_cleanup", fromlist=[name]), name
    )
    # Type hints resolve (downstream tool-schema path).
    typing.get_type_hints(fn)


def test_star_import_binds_new_names():
    """Star import binds the four new public names."""
    ns: dict = {}
    exec("from sleap_roots_analyze import *", ns)
    for name in (
        "apply_data_cleanup_filters",
        "validate_clean_traits",
        "build_clean_validation_report",
        "clean_traits_for_analysis",
    ):
        assert name in ns


# ---------------------------------------------------------------------------
# Single source of truth: validate_clean_traits / build_clean_validation_report
# ---------------------------------------------------------------------------
def test_validate_clean_traits_passes_on_clean_data():
    """validate_clean_traits returns a passing report on clean data."""
    df = pd.DataFrame({"t1": [1.0, 2.0], "t2": [3.0, 4.0]})
    report = validate_clean_traits(df, ["t1", "t2"])
    assert bool(report["validation_passed"]) is True
    assert report["nan_values_in_traits"] == 0


def test_validate_clean_traits_byte_exact_error_message():
    """validate_clean_traits raises the canonical byte-exact message."""
    df = pd.DataFrame({"t1": [1.0, np.nan], "t2": [3.0, 4.0]})
    expected = (
        "Validation failed: 1 NaN values found in trait columns!\n"
        "Affected traits: ['t1']"
    )
    with pytest.raises(ValueError) as exc:
        validate_clean_traits(df, ["t1", "t2"])
    assert str(exc.value) == expected


def test_build_report_keys_match_step_contract():
    """build_clean_validation_report returns the step-03 report keys."""
    df = pd.DataFrame({"Barcode": ["a", "b"], "t1": [1.0, np.nan], "t2": [3.0, 4.0]})
    report = build_clean_validation_report(df, ["t1", "t2"])
    assert set(report) == {
        "validation_passed",
        "total_samples",
        "total_trait_columns",
        "total_metadata_columns",
        "nan_values_in_traits",
        "nan_values_in_metadata",
        "trait_nan_counts",
        "metadata_nan_counts",
    }
    assert report["trait_nan_counts"] == {"t1": 1}


def test_entry_point_and_validate_share_nan_message():
    """clean_traits_for_analysis surfaces the canonical validate message verbatim.

    Force a residual NaN by declaring a metadata-NaN column as a trait so cleanup
    keeps it, proving the entry point delegates to validate_clean_traits.
    """
    # Build a frame where the "trait" has a single NaN that cleanup will not
    # remove because the sample is otherwise fine and the trait is below the NaN
    # threshold -- handled by remove_nan_samples only if it exceeds per-sample
    # fraction; here we keep it via a high max_nans_per_sample so a NaN remains.
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(10)],
            "geno": ["A"] * 5 + ["B"] * 5,
            "rep": list(range(5)) * 2,
            "trait_a": [np.nan] + [float(i) for i in range(1, 10)],
            "trait_b": [float(i) for i in range(10)],
        }
    )
    # Keep the NaN row (max_nans_per_sample=1.0) and the trait
    # (max_nans_per_trait=1.0) so a residual NaN reaches validation.
    with pytest.raises(ValueError, match="Validation failed:"):
        clean_traits_for_analysis(
            df,
            min_samples_per_trait=2,
            max_nans_per_trait=1.0,
            max_nans_per_sample=1.0,
        )
