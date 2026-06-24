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
    nan_heavy[:10] = np.nan  # 50% NaN -> exceeds the default max_nans_per_trait (0.2)
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
    """A trait at 25% NaN: dropped at the QC-canonical default (0.2); kept if loosened to 0.3."""
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
    # QC-canonical default (0.2): the 25%-NaN trait is dropped.
    _, traits_default, log_default = clean_traits_for_analysis(
        quarter_nan_data, min_samples_per_trait=2
    )
    assert "trait_quarter_nan" not in traits_default
    assert log_default["effective_thresholds"]["max_nans_per_trait"] == 0.2
    assert log_default["effective_thresholds"]["max_nans_per_sample"] == 0.0

    # Loosened (0.3): the trait survives -> caller kwargs override the default.
    _, traits_loose, log_loose = clean_traits_for_analysis(
        quarter_nan_data, min_samples_per_trait=2, max_nans_per_trait=0.3
    )
    assert "trait_quarter_nan" in traits_loose
    assert log_loose["effective_thresholds"]["max_nans_per_trait"] == 0.3


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


# ---------------------------------------------------------------------------
# Default-threshold behavior: residual NaN rows are dropped, not raised
# ---------------------------------------------------------------------------
def test_default_thresholds_deliver_clean_frame_on_sparse_data():
    """Ordinary sparse-missing data returns a clean frame (no raise) on defaults.

    A benign 12x5 frame with a single residual NaN cleans to a NaN-free frame
    instead of raising; only the one offending row is lost.
    """
    np.random.seed(11)
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i:02d}" for i in range(12)],
            "geno": ["A"] * 6 + ["B"] * 6,
            "rep": list(range(6)) * 2,
            **{f"trait_{j}": np.random.randn(12) + j for j in range(5)},
        }
    )
    # One NaN in trait_0 (8% of its column, kept as a trait). Its row has a NaN in a
    # surviving trait, so the QC-canonical max_nans_per_sample=0.0 drops just that row.
    df.loc[0, "trait_0"] = np.nan
    clean_df, trait_cols, _ = clean_traits_for_analysis(df)  # all defaults
    assert clean_df[trait_cols].isna().sum().sum() == 0
    # Only the single offending row is dropped; everything else is retained.
    assert len(clean_df) == 11


def test_residual_nan_rows_dropped_keeps_other_rows():
    """Residual NaN rows in surviving traits are dropped; clean rows are kept."""
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(10)],
            "geno": ["A"] * 5 + ["B"] * 5,
            "rep": list(range(5)) * 2,
            "trait_a": [np.nan] + [float(i) for i in range(1, 10)],
            "trait_b": [float(i) for i in range(10)],
        }
    )
    clean_df, trait_cols, _ = clean_traits_for_analysis(
        df, min_samples_per_trait=2, max_nans_per_trait=1.0, max_nans_per_sample=1.0
    )
    assert clean_df[trait_cols].isna().sum().sum() == 0
    assert len(clean_df) == 9  # only the single NaN row dropped


def test_cleanup_path_reduces_to_below_two_samples():
    """The cleanup path itself (not a pre-shrunk input) can trip the >=2 gate.

    All but one row carry a NaN in the trait; dropping residual NaN rows leaves a
    single sample, so the >=2-samples gate fires.
    """
    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i}" for i in range(6)],
            "geno": ["A"] * 3 + ["B"] * 3,
            "rep": [1, 2, 3, 1, 2, 3],
            "trait_a": [1.0] + [np.nan] * 5,
            "trait_b": [2.0] + [np.nan] * 5,
        }
    )
    with pytest.raises(ValueError, match=r"only 1 sample"):
        clean_traits_for_analysis(
            df, min_samples_per_trait=1, max_nans_per_trait=1.0, max_nans_per_sample=1.0
        )


# ---------------------------------------------------------------------------
# Misuse -> actionable errors
# ---------------------------------------------------------------------------
def test_explicit_missing_trait_col_raises_actionable_error():
    """An explicit trait_cols name absent from df raises an actionable error."""
    df = pd.DataFrame({"Barcode": ["a", "b"], "trait_a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="not found in dataframe"):
        clean_traits_for_analysis(df, trait_cols=["trait_a", "does_not_exist"])


def test_explicit_non_numeric_trait_col_raises():
    """A non-numeric explicit trait column raises before PCA would break."""
    df = pd.DataFrame(
        {
            "Barcode": ["a", "b", "c"],
            "trait_a": [1.0, 2.0, 3.0],
            "label": ["x", "y", "z"],
        }
    )
    with pytest.raises(ValueError, match="must be numeric"):
        clean_traits_for_analysis(df, trait_cols=["trait_a", "label"])


def test_duplicate_column_names_raise():
    """Duplicate column names raise a clear error, not 'no trait columns found'."""
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["trait_a", "trait_a", "geno"]
    )
    with pytest.raises(ValueError, match="duplicate column names"):
        clean_traits_for_analysis(df)


# ---------------------------------------------------------------------------
# p > n warning
# ---------------------------------------------------------------------------
def test_warns_in_p_greater_than_n_regime():
    """A UserWarning is emitted when surviving traits outnumber samples."""
    np.random.seed(5)
    df = pd.DataFrame(
        {
            "Barcode": ["BC0", "BC1"],
            "geno": ["A", "B"],
            "rep": [1, 1],
            **{f"trait_{j}": [float(j), float(j) + 1] for j in range(5)},
        }
    )
    with pytest.warns(UserWarning, match="p > n"):
        clean_traits_for_analysis(df, min_samples_per_trait=2)


# ---------------------------------------------------------------------------
# Step 02 -> 03 pipeline regression (the extracted functions in the real steps)
# ---------------------------------------------------------------------------
def test_step02_to_step03_uses_shared_functions_and_passes():
    """CleanupTraitsStep -> ValidateCleanStep run on the extracted functions.

    Exercises the transparent refactor through the real step objects (not just the
    unit-level helpers): step 03 must report validation_passed with no trait NaNs.
    """
    import numpy as _np
    from sleap_roots_analyze.pipeline import ColumnConfig, DataConfig, QCPipelineConfig
    from sleap_roots_analyze.pipeline.core import StepResult
    from sleap_roots_analyze.pipeline.steps import CleanupTraitsStep, ValidateCleanStep

    _np.random.seed(0)
    df = pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(12)],
            "geno": ["A"] * 6 + ["B"] * 6,
            "rep": [1, 2, 3, 4, 5, 6] * 2,
            "trait1": _np.random.randn(12) * 10 + 50,
            "trait2": _np.random.randn(12) * 5 + 25,
            "trait3": _np.random.randn(12) * 3 + 15,
        }
    )
    config = QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path="dummy.csv"),
    )
    trait_cols = ["trait1", "trait2", "trait3"]
    load_result = StepResult(
        data=df,
        metadata={
            "trait_column_names": trait_cols,
            "metadata_column_names": ["Barcode", "geno", "rep"],
        },
    )

    def _run(step, data, prev, tmp):
        return step.execute(data=data, config=config, run_dir=tmp, prev_result=prev)

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as d:
        tmp = _Path(d)
        cleaned = _run(CleanupTraitsStep(), df, load_result, tmp)
        validated = _run(ValidateCleanStep(), cleaned.data, cleaned, tmp)

    assert validated.metadata["validation_passed"] is True
    assert validated.metadata["total_nans_in_traits"] == 0
