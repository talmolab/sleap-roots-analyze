"""Tests for the public outlier-removal entry point ``remove_outlier_samples``.

Covers the composition (detect -> remove), the clean-input + unique-index
preconditions, the analysis-readiness gates and warnings, the auditable
JSON-serializable report, determinism, and the public API surface. See OpenSpec
change ``add-outlier-removal-entry-point`` (issue #165).

Fixture note: the canonical fixture (``clean_frame``) is pinned so that the default
``detect_outliers_mahalanobis(chi2_percentile=97.5)`` flags *exactly* the injected
indices — verified during authoring against the real detector (deterministic,
exact-SVD path). The distance-threshold constants used by the readiness /
over-removal / p>n fixtures were likewise tuned against the real detector.
"""

from __future__ import annotations

import inspect
import json
import typing
import warnings

import numpy as np
import pandas as pd
import pytest

import sleap_roots_analyze as sra
from sleap_roots_analyze import outlier_removal
from sleap_roots_analyze.outlier_removal import (
    OutlierRemovalError,
    _assert_output_analysis_ready,
    remove_outlier_samples,
)
from sleap_roots_analyze.pca import perform_pca_analysis

TRAITS = [f"trait_{j}" for j in range(5)]
INJECTED = [5, 17, 28]  # the indices pushed far out in the canonical fixture


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _build_outlier_frame(
    n=40,
    n_traits=5,
    inject=(5, 17, 28),
    sd=8.0,
    n_inject_traits=3,
    seed=42,
):
    """Build a clean (NaN-free) trait frame with a few injected outlier rows.

    Args:
        n: Number of samples.
        n_traits: Number of numeric trait columns.
        inject: Row indices to push out as outliers.
        sd: Standard-deviation offset applied to injected rows.
        n_inject_traits: How many leading traits to perturb per injected row.
        seed: RNG seed (deterministic, exact).

    Returns:
        A DataFrame with Barcode/geno/rep metadata plus ``n_traits`` numeric
        traits, default RangeIndex.
    """
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f"trait_{j}": rng.randn(n) for j in range(n_traits)})
    for idx in inject:
        for j in range(n_inject_traits):
            df.loc[idx, f"trait_{j}"] += sd
    df.insert(0, "Barcode", [f"BC{i:03d}" for i in range(n)])
    df.insert(1, "geno", ["G1"] * (n // 2) + ["G2"] * (n - n // 2))
    df.insert(2, "rep", list(range(1, n // 2 + 1)) * 2)
    return df


@pytest.fixture
def clean_frame():
    """Canonical 40-sample frame; default Mahalanobis flags exactly INJECTED."""
    return _build_outlier_frame()


@pytest.fixture
def no_detect(monkeypatch):
    """Make both detectors explode if reached (proves a guard fired first)."""

    def _boom(*args, **kwargs):
        raise AssertionError("detector should not run; a guard must fire first")

    monkeypatch.setattr(outlier_removal, "detect_outliers_mahalanobis", _boom)
    monkeypatch.setattr(outlier_removal, "detect_outliers_isolation_forest", _boom)


# ---------------------------------------------------------------------------
# 1.2 / 1.3 / 1.4 — composition, return shape, alignment
# ---------------------------------------------------------------------------
def test_returns_tuple_and_drops_rows_not_columns(clean_frame):
    """Returns (trimmed_df, report); rows dropped by n_outliers, columns intact."""
    trimmed, report = remove_outlier_samples(clean_frame)
    assert isinstance(trimmed, pd.DataFrame)
    assert isinstance(report, dict)
    assert len(trimmed) == len(clean_frame) - report["n_outliers"]
    # Outlier removal drops rows, never columns.
    assert list(trimmed.columns) == list(clean_frame.columns)


def test_default_mahalanobis_flags_exact_injected(clean_frame):
    """Default Mahalanobis removes exactly the injected rows."""
    trimmed, report = remove_outlier_samples(clean_frame)
    assert sorted(report["outlier_indices"]) == sorted(INJECTED)
    assert not set(INJECTED) & set(trimmed.index)


def test_return_detector_result_flag(clean_frame):
    """return_detector_result=True adds the raw detector dict as a 3rd element."""
    # Default contract: a 2-tuple with a compact report (no per-sample arrays).
    two = remove_outlier_samples(clean_frame)
    assert len(two) == 2
    assert "mahalanobis_distances" not in two[1]

    trimmed, report, detector_result = remove_outlier_samples(
        clean_frame, return_detector_result=True
    )
    assert isinstance(detector_result, dict)
    # The raw dict carries the per-sample arrays the compact report omits.
    assert "mahalanobis_distances" in detector_result
    assert set(detector_result["outlier_indices"]) == set(report["outlier_indices"])


def test_isolation_forest_is_superset_not_exact(clean_frame):
    """Isolation forest (a contamination quota) flags at least the injected rows."""
    _, report = remove_outlier_samples(clean_frame, method="isolation_forest")
    assert set(INJECTED) <= set(report["outlier_indices"])


def test_indices_align_with_index_and_barcodes(clean_frame):
    """Every flagged label is in the input index; barcodes list removed rows."""
    _, report = remove_outlier_samples(clean_frame)
    assert all(label in clean_frame.index for label in report["outlier_indices"])
    expected_barcodes = clean_frame.loc[report["outlier_indices"], "Barcode"].tolist()
    assert report["outlier_barcodes"] == expected_barcodes


# ---------------------------------------------------------------------------
# 1.5 — method selection
# ---------------------------------------------------------------------------
def test_default_method_is_mahalanobis(clean_frame):
    """The dispatch key recorded for the default path is 'mahalanobis'."""
    _, report = remove_outlier_samples(clean_frame)
    assert report["method"] == "mahalanobis"


def test_isolation_forest_dispatch_key_and_params(clean_frame):
    """Method is the dispatch key (not 'IsolationForest'); params echo contamination."""
    _, report = remove_outlier_samples(
        clean_frame, method="isolation_forest", contamination=0.2
    )
    assert report["method"] == "isolation_forest"
    assert report["method_params"]["contamination"] == 0.2


def test_mahalanobis_kwargs_echoed(clean_frame):
    """A forwarded Mahalanobis kwarg is recorded in method_params."""
    _, report = remove_outlier_samples(clean_frame, chi2_percentile=99.0)
    assert report["method_params"]["chi2_percentile"] == 99.0


def test_unknown_method_rejected_before_detection(clean_frame, no_detect):
    """An unknown method raises (naming the supported set) before any detection."""
    with pytest.raises(ValueError, match=r"mahalanobis.*isolation_forest|unknown"):
        remove_outlier_samples(clean_frame, method="not_a_method")


def test_unknown_detect_kwarg_rejected_before_detection(clean_frame, no_detect):
    """An unknown / cross-method detect_kwarg raises a prefixed ValueError.

    Not a bare TypeError — the message names the unrecognized key and the supported
    set, and ``no_detect`` proves the guard fires before any detector runs.
    """
    # isolation-forest-only knob passed to the default Mahalanobis method
    with pytest.raises(ValueError, match=r"unknown detect_kwargs.*contamination"):
        remove_outlier_samples(clean_frame, contamination=0.2)
    # Mahalanobis knob passed to isolation forest
    with pytest.raises(ValueError, match=r"unknown detect_kwargs.*chi2_percentile"):
        remove_outlier_samples(
            clean_frame, method="isolation_forest", chi2_percentile=99.0
        )
    # a plain typo is caught too (and the supported set is named)
    with pytest.raises(ValueError, match=r"unknown detect_kwargs.*chi2_percentil\b"):
        remove_outlier_samples(clean_frame, chi2_percentil=99.0)


# ---------------------------------------------------------------------------
# 1.6 — clean-input precondition (NaN-free, wrapped message)
# ---------------------------------------------------------------------------
def test_nan_input_rejected_with_pointer(clean_frame, no_detect):
    """A NaN in a trait raises, naming the trait AND pointing to the sibling."""
    dirty = clean_frame.copy()
    dirty.loc[0, "trait_1"] = np.nan
    with pytest.raises(ValueError, match=r"clean_traits_for_analysis") as exc:
        remove_outlier_samples(dirty)
    assert "trait_1" in str(exc.value)  # wrapped validator message names the trait


# ---------------------------------------------------------------------------
# 1.7 — unique-index precondition
# ---------------------------------------------------------------------------
def test_non_unique_index_rejected_before_detection(clean_frame, no_detect):
    """A duplicate-label index raises, before any detector runs."""
    dup = clean_frame.copy()
    dup.index = [0] * len(dup)
    with pytest.raises(ValueError, match=r"unique"):
        remove_outlier_samples(dup)


def test_default_rangeindex_accepted(clean_frame):
    """A default unique RangeIndex passes the uniqueness check."""
    trimmed, _ = remove_outlier_samples(clean_frame)
    assert len(trimmed) == len(clean_frame) - len(INJECTED)


# ---------------------------------------------------------------------------
# 1.8 — output analysis-readiness
# ---------------------------------------------------------------------------
def test_readiness_under_two_samples_raises_and_carries_report(clean_frame):
    """<2 survivors raises, names the count, and the error carries the report."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # over-removal warning also fires here
        with pytest.raises(OutlierRemovalError, match=r"only 0 sample") as exc:
            remove_outlier_samples(
                clean_frame,
                method="mahalanobis",
                use_chi_squared=False,
                distance_threshold=0.0,
            )
    assert hasattr(exc.value, "outlier_report")
    assert exc.value.outlier_report["n_outliers"] == len(clean_frame)


def test_readiness_constant_trait_gate_whitebox():
    """White-box: the constant-only branch of the readiness helper raises.

    A "trait varies in clean_df but is constant only after the detector removes
    exactly the varying rows" fixture is not reliably achievable through real
    detection, so the gate is exercised directly on the helper.
    """
    const_only = pd.DataFrame({"trait_a": [5.0, 5.0, 5.0]})
    with pytest.raises(ValueError, match=r"non-constant"):
        _assert_output_analysis_ready(const_only, ["trait_a"], {"n_outliers": 0})


def test_trimmed_frame_runs_pca_without_dropping_rows(clean_frame):
    """The trimmed frame flows straight into PCA with no row loss."""
    trimmed, _ = remove_outlier_samples(clean_frame)
    result = perform_pca_analysis(trimmed[TRAITS])
    assert result["data_processed"].shape[0] == len(trimmed)


def test_constant_trait_alongside_varying_passes(clean_frame):
    """A constant trait surviving with a varying one does not fail the gate."""
    frame = clean_frame.copy()
    frame["trait_const"] = 5.0
    trimmed, _ = remove_outlier_samples(frame, trait_cols=TRAITS + ["trait_const"])
    assert "trait_const" in trimmed.columns
    assert len(trimmed) == len(clean_frame) - len(INJECTED)


# ---------------------------------------------------------------------------
# 1.9 — p > n warning
# ---------------------------------------------------------------------------
def test_p_greater_than_n_warns_without_over_removal():
    """Trimming into p > n warns specifically about p > n, not via over-removal.

    The input is already p > n (6 samples, 8 traits) and clean, so the default trim
    removes ~0 rows — the p > n warning fires while the over-removal rail does not, so
    the assertion cannot pass on the wrong warning.
    """
    frame = _build_outlier_frame(n=6, n_traits=8, inject=())
    with pytest.warns(UserWarning) as record:
        trimmed, _ = remove_outlier_samples(frame)
    msgs = [str(w.message) for w in record]
    assert any("p > n" in m for m in msgs)
    assert not any("large removal fraction" in m for m in msgs)
    assert len(trimmed) < 8  # survivors fewer than traits


# ---------------------------------------------------------------------------
# 1.10 — over-removal safety rail
# ---------------------------------------------------------------------------
def test_over_removal_warns_and_still_returns():
    """>50% removed (>=2 survive) warns about the large removal fraction, returns."""
    frame = _build_outlier_frame(n=30)
    with pytest.warns(UserWarning, match=r"large removal fraction"):
        trimmed, report = remove_outlier_samples(
            frame,
            method="mahalanobis",
            use_chi_squared=False,
            distance_threshold=1.1,
        )
    assert report["removal_fraction"] > 0.5
    assert len(trimmed) >= 2


def test_over_removal_warning_precedes_readiness_failure(clean_frame):
    """When over-removal also breaches readiness, BOTH warn and raise occur."""
    with pytest.warns(UserWarning, match=r"large removal fraction"):
        with pytest.raises(OutlierRemovalError):
            remove_outlier_samples(
                clean_frame,
                method="mahalanobis",
                use_chi_squared=False,
                distance_threshold=0.0,
            )


# ---------------------------------------------------------------------------
# Mahalanobis scientific-quality signals (small-n, goodness-of-fit)
# ---------------------------------------------------------------------------
def test_small_n_mahalanobis_warns():
    """The Mahalanobis path warns when the sample count is small (< 30)."""
    frame = _build_outlier_frame(n=20, inject=())  # clean, 5 traits, n < 30
    with pytest.warns(UserWarning, match=r"statistically fragile at this size"):
        trimmed, _ = remove_outlier_samples(frame)
    assert len(trimmed) >= 2


def test_goodness_of_fit_violation_warns():
    """A poor chi-squared fit (heavy-tailed data) warns about the assumption.

    n=50 keeps it off the small-n path, isolating the goodness-of-fit signal: the
    squared Mahalanobis distances of exp^2 data do not match a chi-squared tail, so
    the detector reports ``distributional_assumption_valid=False`` and the entry
    point surfaces it rather than trimming ~2.5% silently.
    """
    rng = np.random.RandomState(1)
    df = pd.DataFrame({f"trait_{j}": rng.exponential(2.0, 50) ** 2 for j in range(4)})
    df.insert(0, "Barcode", [f"BC{i:03d}" for i in range(50)])
    with pytest.warns(UserWarning, match=r"chi-squared assumption"):
        trimmed, _ = remove_outlier_samples(df, replicate_col=None)
    assert len(trimmed) >= 2


def test_no_quality_warnings_on_clean_large_sample(clean_frame):
    """A clean n=40 frame trims without any small-n / goodness-of-fit warning."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        remove_outlier_samples(clean_frame)
    msgs = [str(w.message) for w in record]
    assert not any("statistically fragile" in m for m in msgs)
    assert not any("chi-squared assumption" in m for m in msgs)


# ---------------------------------------------------------------------------
# 1.11 — report schema + JSON adversarial
# ---------------------------------------------------------------------------
def test_report_counts_and_fraction_consistent(clean_frame):
    """Counts reconcile and removal_fraction matches n_outliers / n_input."""
    _, report = remove_outlier_samples(clean_frame)
    assert (
        report["n_input_samples"] == report["n_outliers"] + report["n_output_samples"]
    )
    assert report["removal_fraction"] == pytest.approx(
        report["n_outliers"] / report["n_input_samples"]
    )


def test_mahalanobis_report_threshold_fields_populated(clean_frame):
    """Mahalanobis report carries threshold / n_components / goodness-of-fit."""
    _, report = remove_outlier_samples(clean_frame)
    assert report["threshold_type"] is not None
    assert report["threshold_value"] is not None
    assert report["n_components"] is not None
    assert report["goodness_of_fit"] is not None


def test_isolation_forest_report_threshold_fields_none(clean_frame):
    """Isolation forest has no distance threshold -> those four fields are None."""
    _, report = remove_outlier_samples(clean_frame, method="isolation_forest")
    assert report["threshold_type"] is None
    assert report["threshold_value"] is None
    assert report["n_components"] is None
    assert report["goodness_of_fit"] is None


def test_outlier_barcodes_none_without_barcode_column(clean_frame):
    """outlier_barcodes is None when the frame has no barcode column."""
    no_bc = clean_frame.drop(columns=["Barcode"])
    _, report = remove_outlier_samples(no_bc)
    assert report["outlier_barcodes"] is None


def test_report_is_json_serializable_with_plain_types(clean_frame):
    """JSON adversarial: plain int/str indices, float threshold, json.dumps works."""
    _, report = remove_outlier_samples(clean_frame)
    assert all(type(i) in (int, str) for i in report["outlier_indices"])
    assert type(report["threshold_value"]) is float
    assert json.dumps(report)  # must not raise


# ---------------------------------------------------------------------------
# 1.12 — determinism
# ---------------------------------------------------------------------------
def test_mahalanobis_deterministic_records_seed(clean_frame):
    """Two seeded Mahalanobis runs are identical; the seed is recorded."""
    t1, r1 = remove_outlier_samples(clean_frame, random_state=7)
    t2, r2 = remove_outlier_samples(clean_frame, random_state=7)
    assert r1["outlier_indices"] == r2["outlier_indices"]
    pd.testing.assert_frame_equal(t1, t2)
    assert r1["random_state"] == 7


def test_isolation_forest_seed_is_load_bearing(clean_frame):
    """The seed is actually threaded: same seed -> identical indices (IF path)."""
    _, a = remove_outlier_samples(
        clean_frame, method="isolation_forest", random_state=3
    )
    _, b = remove_outlier_samples(
        clean_frame, method="isolation_forest", random_state=3
    )
    assert a["outlier_indices"] == b["outlier_indices"]


# ---------------------------------------------------------------------------
# 1.13 — detector-failure surfacing
# ---------------------------------------------------------------------------
def test_detector_error_is_surfaced_not_swallowed(clean_frame, monkeypatch):
    """A detector returning an error raises, not a silent (clean_df, n_outliers=0)."""
    monkeypatch.setattr(
        outlier_removal,
        "detect_outliers_mahalanobis",
        lambda *a, **k: {"method": "Mahalanobis", "error": "degenerate PCA"},
    )
    with pytest.raises(ValueError, match=r"degenerate PCA|detection failed"):
        remove_outlier_samples(clean_frame)


def test_detector_missing_indices_is_surfaced(clean_frame, monkeypatch):
    """A detector result lacking outlier_indices raises rather than swallowing."""
    monkeypatch.setattr(
        outlier_removal,
        "detect_outliers_mahalanobis",
        lambda *a, **k: {"method": "Mahalanobis"},
    )
    with pytest.raises(ValueError, match=r"detection failed|no outlier_indices"):
        remove_outlier_samples(clean_frame)


# ---------------------------------------------------------------------------
# 1.14 — input misuse diagnostics
# ---------------------------------------------------------------------------
def test_empty_input_rejected(no_detect):
    """An empty table raises an actionable error before delegating."""
    with pytest.raises(ValueError, match=r"no rows"):
        remove_outlier_samples(pd.DataFrame())


def test_duplicate_column_names_rejected(no_detect):
    """Duplicate column names raise a clear error."""
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=["trait_a", "trait_a", "geno"]
    )
    with pytest.raises(ValueError, match=r"duplicate column names"):
        remove_outlier_samples(df)


def test_explicit_missing_trait_col_rejected(clean_frame, no_detect):
    """An explicit trait_cols name absent from the frame raises (not KeyError)."""
    with pytest.raises(ValueError, match=r"not found in dataframe"):
        remove_outlier_samples(clean_frame, trait_cols=["trait_0", "nope"])


def test_explicit_non_numeric_trait_col_rejected(clean_frame, no_detect):
    """A non-numeric explicit trait column raises."""
    with pytest.raises(ValueError, match=r"must be numeric"):
        remove_outlier_samples(clean_frame, trait_cols=["trait_0", "geno"])


# ---------------------------------------------------------------------------
# 1.15 — trait_cols / replicate_col ergonomics
# ---------------------------------------------------------------------------
def test_explicit_trait_cols_bypass_inference(clean_frame, monkeypatch):
    """Explicit trait_cols bypass get_trait_columns entirely."""

    def _boom(*args, **kwargs):
        raise AssertionError("get_trait_columns must not run when trait_cols given")

    monkeypatch.setattr(outlier_removal, "get_trait_columns", _boom)
    trimmed, _ = remove_outlier_samples(clean_frame, trait_cols=TRAITS)
    assert len(trimmed) == len(clean_frame) - len(INJECTED)


def test_replicate_col_none_on_frame_without_replicate():
    """replicate_col=None is honored on a frame that has no replicate column."""
    frame = _build_outlier_frame().drop(columns=["rep"])
    trimmed, report = remove_outlier_samples(frame, replicate_col=None)
    assert report["n_outliers"] == len(INJECTED)
    assert "rep" not in trimmed.columns


# ---------------------------------------------------------------------------
# 1.16 — single source of truth
# ---------------------------------------------------------------------------
def test_composes_detect_and_remove_primitives(clean_frame, monkeypatch):
    """The entry point calls the public detect + remove primitives."""
    calls = {"detect": 0, "remove": 0}
    real_detect = outlier_removal.detect_outliers_mahalanobis
    real_remove = outlier_removal.remove_outliers_from_data

    def spy_detect(*args, **kwargs):
        calls["detect"] += 1
        return real_detect(*args, **kwargs)

    def spy_remove(*args, **kwargs):
        calls["remove"] += 1
        return real_remove(*args, **kwargs)

    monkeypatch.setattr(outlier_removal, "detect_outliers_mahalanobis", spy_detect)
    monkeypatch.setattr(outlier_removal, "remove_outliers_from_data", spy_remove)
    remove_outlier_samples(clean_frame)
    assert calls["detect"] == 1
    assert calls["remove"] == 1


def test_detector_receives_trait_subframe_and_forwarded_kwargs(
    clean_frame, monkeypatch
):
    """The detector is invoked on exactly clean_df[trait_cols] with forwarded kwargs.

    A spy on the call args (not just the echoed method_params, which would look right
    even if the kwarg were dropped before the call) confirms the trait subframe and
    the threaded random_state actually reach the detector.
    """
    captured: dict = {}
    real = outlier_removal.detect_outliers_mahalanobis

    def spy(data, **kwargs):
        captured["columns"] = list(data.columns)
        captured["index"] = data.index
        captured["kwargs"] = dict(kwargs)
        return real(data, **kwargs)

    monkeypatch.setattr(outlier_removal, "detect_outliers_mahalanobis", spy)
    remove_outlier_samples(clean_frame, chi2_percentile=99.0, random_state=7)
    # Detection operates over exactly the trait columns (no metadata), same index.
    assert captured["columns"] == TRAITS
    assert captured["index"].equals(clean_frame.index)
    # The forwarded kwarg and the threaded seed reach the detector unchanged.
    assert captured["kwargs"]["chi2_percentile"] == 99.0
    assert captured["kwargs"]["random_state"] == 7


def test_source_has_no_independent_thresholding_logic():
    """The source defines no alternative distance/score thresholding of its own."""
    src = inspect.getsource(remove_outlier_samples)
    for forbidden in (
        "calculate_mahalanobis_distances",
        "IsolationForest(",
        "chi2.ppf",
        "np.percentile",
    ):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# 1.17 — public API surface
# ---------------------------------------------------------------------------
def test_public_api_surface():
    """Importable, in __all__ once, identity-equal, type-hints resolve."""
    assert "remove_outlier_samples" in sra.__all__
    assert sra.__all__.count("remove_outlier_samples") == 1
    assert sra.remove_outlier_samples is remove_outlier_samples
    typing.get_type_hints(remove_outlier_samples)


def test_composed_primitives_still_exported():
    """The composed primitives remain in __all__ (no duplicate entries)."""
    for name in (
        "detect_outliers_mahalanobis",
        "detect_outliers_isolation_forest",
        "remove_outliers_from_data",
    ):
        assert name in sra.__all__
        assert sra.__all__.count(name) == 1
