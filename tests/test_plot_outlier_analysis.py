"""Tests for the public outlier-plotting entry point ``plot_outlier_analysis``.

Covers the composition over the existing ``create_*_outlier`` figure functions, the
method-appropriate figure set (mahalanobis / isolation_forest only), ``which``
selection, deterministic re-detection matching ``remove_outlier_samples``, the
clean-input + unique-index preconditions, detector-failure surfacing, IO-freeness,
and the public API surface. See OpenSpec change ``add-outlier-plotting-entry-point``
(issue #173).

The fixture mirrors ``tests/test_remove_outlier_samples.py`` so the default
Mahalanobis re-detection flags exactly the injected indices.
"""

from __future__ import annotations

import inspect
import typing

import matplotlib

matplotlib.use("Agg")  # headless; no display needed for figure objects
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import sleap_roots_analyze as sra  # noqa: E402
from sleap_roots_analyze import outlier_visualization  # noqa: E402
from sleap_roots_analyze.outlier_removal import remove_outlier_samples  # noqa: E402
from sleap_roots_analyze.outlier_visualization import (  # noqa: E402
    _select_outlier_figures,
    plot_outlier_analysis,
)

INJECTED = [5, 17, 28]
MAHAL_KEYS = {
    "mahalanobis_outlier_detection",
    "mahalanobis_pc_analysis",
    "mahalanobis_threshold_analysis",
}


def _build_outlier_frame(n=40, n_traits=5, inject=(5, 17, 28), sd=8.0, seed=42):
    """Clean (NaN-free) trait frame with injected outlier rows (mirrors #165)."""
    rng = np.random.RandomState(seed)
    df = pd.DataFrame({f"trait_{j}": rng.randn(n) for j in range(n_traits)})
    for idx in inject:
        for j in range(3):
            df.loc[idx, f"trait_{j}"] += sd
    df.insert(0, "Barcode", [f"BC{i:03d}" for i in range(n)])
    df.insert(1, "geno", ["G1"] * (n // 2) + ["G2"] * (n - n // 2))
    df.insert(2, "rep", list(range(1, n // 2 + 1)) * 2)
    return df


@pytest.fixture
def clean_frame():
    """Canonical 40-sample frame; default Mahalanobis flags exactly INJECTED."""
    return _build_outlier_frame()


@pytest.fixture(autouse=True)
def _close_figs():
    """Close any figures a test leaves open (tests own the returned figures)."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Return shape / composition
# ---------------------------------------------------------------------------
def test_returns_dict_of_figures(clean_frame):
    """Returns a dict of matplotlib Figures; writes no files."""
    figs = plot_outlier_analysis(clean_frame)
    assert isinstance(figs, dict)
    assert figs, "expected a non-empty figure set"
    assert all(isinstance(f, plt.Figure) for f in figs.values())


def test_delegates_to_public_figure_function(clean_frame, monkeypatch):
    """The composer invokes the public create_* function, not its own drawing."""
    called = {}

    def _spy(df, mahal_results):
        called["hit"] = True
        return {"mahalanobis_outlier_detection": plt.figure()}

    monkeypatch.setattr(outlier_visualization, "create_mahalanobis_outlier_plots", _spy)
    plot_outlier_analysis(clean_frame, method="mahalanobis", which=None)
    assert called.get("hit") is True


# ---------------------------------------------------------------------------
# Method-appropriate figure set
# ---------------------------------------------------------------------------
def test_mahalanobis_figure_set(clean_frame):
    """Mahalanobis returns its figures + per-genotype; no pca_outlier figure."""
    figs = plot_outlier_analysis(clean_frame, method="mahalanobis")
    assert MAHAL_KEYS <= set(figs)
    assert "outliers_per_genotype" in figs
    assert "pca_outlier" not in figs
    assert "isolation_forest_analysis" not in figs


def test_isolation_forest_figure_set(clean_frame):
    """Isolation forest returns its figure + per-genotype; no mahalanobis keys."""
    figs = plot_outlier_analysis(clean_frame, method="isolation_forest")
    assert "isolation_forest_analysis" in figs
    assert "outliers_per_genotype" in figs
    assert not (MAHAL_KEYS & set(figs))


def test_genotype_figure_only_when_column_present(clean_frame):
    """The per-genotype figure is present only when a genotype column exists."""
    no_geno = clean_frame.drop(columns=["geno"])
    figs = plot_outlier_analysis(no_geno, method="mahalanobis")
    assert "outliers_per_genotype" not in figs
    assert MAHAL_KEYS <= set(figs)


# ---------------------------------------------------------------------------
# which selection
# ---------------------------------------------------------------------------
def test_which_list_narrows(clean_frame):
    """A which list returns exactly those keys."""
    figs = plot_outlier_analysis(
        clean_frame, method="mahalanobis", which=["mahalanobis_outlier_detection"]
    )
    assert set(figs) == {"mahalanobis_outlier_detection"}


def test_which_accepts_single_string(clean_frame):
    """A bare string which returns exactly that one figure (not iterated by char)."""
    figs = plot_outlier_analysis(
        clean_frame, method="mahalanobis", which="mahalanobis_pc_analysis"
    )
    assert set(figs) == {"mahalanobis_pc_analysis"}


def test_which_none_returns_full_set(clean_frame):
    """which=None returns the method's full available set."""
    figs = plot_outlier_analysis(clean_frame, method="mahalanobis", which=None)
    assert MAHAL_KEYS <= set(figs)
    assert "outliers_per_genotype" in figs


def test_which_unavailable_key_raises(clean_frame):
    """An unavailable which key raises ValueError naming the available keys."""
    with pytest.raises(ValueError, match="mahalanobis_outlier_detection"):
        plot_outlier_analysis(clean_frame, method="mahalanobis", which=["not_a_figure"])


def test_which_genotype_key_rejected_without_column(clean_frame):
    """Requesting the per-genotype key without a genotype column raises."""
    no_geno = clean_frame.drop(columns=["geno"])
    with pytest.raises(ValueError):
        plot_outlier_analysis(
            no_geno, method="mahalanobis", which=["outliers_per_genotype"]
        )


# ---------------------------------------------------------------------------
# Deterministic re-detection matching removal
# ---------------------------------------------------------------------------
def test_redetection_matches_removal_mahalanobis(clean_frame):
    """Re-detected set equals what remove_outlier_samples removes (mahalanobis)."""
    _, report = remove_outlier_samples(clean_frame, method="mahalanobis")
    captured = {}

    def _spy(df, mahal_results):
        captured["indices"] = list(mahal_results.get("outlier_indices", []))
        return {"mahalanobis_outlier_detection": plt.figure()}

    # Wrap to capture the detector result the composer feeds the figure function.
    import functools

    real = outlier_visualization.create_mahalanobis_outlier_plots

    @functools.wraps(real)
    def _wrap(df, mahal_results):
        _spy(df, mahal_results)
        return real(df=df, mahal_results=mahal_results)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(outlier_visualization, "create_mahalanobis_outlier_plots", _wrap)
        plot_outlier_analysis(clean_frame, method="mahalanobis")
    assert sorted(captured["indices"]) == sorted(report["outlier_indices"])


def test_redetection_matches_removal_isolation_forest(clean_frame):
    """Re-detected set equals removal's for the seed-load-bearing IF path."""
    _, report = remove_outlier_samples(
        clean_frame, method="isolation_forest", random_state=7
    )
    captured = {}
    import functools

    real = outlier_visualization.create_isolation_forest_plots

    @functools.wraps(real)
    def _wrap(df, iso_results):
        captured["indices"] = list(iso_results.get("outlier_indices", []))
        return real(df=df, iso_results=iso_results)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(outlier_visualization, "create_isolation_forest_plots", _wrap)
        plot_outlier_analysis(clean_frame, method="isolation_forest", random_state=7)
    assert sorted(captured["indices"]) == sorted(report["outlier_indices"])


def test_accepts_random_state_none(clean_frame):
    """random_state=None is accepted without raising."""
    figs = plot_outlier_analysis(clean_frame, random_state=None)
    assert isinstance(figs, dict) and figs


# ---------------------------------------------------------------------------
# Preconditions / misuse
# ---------------------------------------------------------------------------
def test_nan_input_rejected_pointing_to_cleaner(clean_frame):
    """NaN in traits raises ValueError mentioning clean_traits_for_analysis."""
    dirty = clean_frame.copy()
    dirty.loc[0, "trait_0"] = np.nan
    with pytest.raises(ValueError, match="clean_traits_for_analysis"):
        plot_outlier_analysis(dirty)


def test_non_unique_index_rejected(clean_frame):
    """A duplicate index raises ValueError before detection."""
    dup = clean_frame.copy()
    dup.index = [0] * len(dup)
    with pytest.raises(ValueError, match="unique"):
        plot_outlier_analysis(dup)


def test_empty_input_rejected(clean_frame):
    """An empty frame raises ValueError."""
    with pytest.raises(ValueError):
        plot_outlier_analysis(clean_frame.iloc[0:0])


def test_unknown_method_rejected(clean_frame):
    """An unknown method raises ValueError naming the supported methods."""
    with pytest.raises(ValueError, match="mahalanobis"):
        plot_outlier_analysis(clean_frame, method="not_a_method")


def test_unknown_detect_kwarg_rejected(clean_frame):
    """A cross-method detect_kwarg raises before detection."""
    with pytest.raises(ValueError):
        plot_outlier_analysis(clean_frame, method="mahalanobis", contamination=0.2)


def test_missing_trait_cols_rejected(clean_frame):
    """Explicit trait_cols not in the frame raise ValueError (not KeyError)."""
    with pytest.raises(ValueError):
        plot_outlier_analysis(clean_frame, trait_cols=["trait_0", "nope"])


# ---------------------------------------------------------------------------
# Detector-failure surfacing
# ---------------------------------------------------------------------------
def test_detector_error_raised_before_delegation(clean_frame, monkeypatch):
    """A detector error raises before any create_* figure function is called."""
    figure_called = {"hit": False}

    def _boom_detect(*args, **kwargs):
        return {"error": "degenerate PCA"}

    def _mark(*args, **kwargs):
        figure_called["hit"] = True
        return {}

    monkeypatch.setattr(
        outlier_visualization, "detect_outliers_mahalanobis", _boom_detect
    )
    monkeypatch.setattr(
        outlier_visualization, "create_mahalanobis_outlier_plots", _mark
    )
    with pytest.raises(ValueError, match="degenerate PCA"):
        plot_outlier_analysis(clean_frame, method="mahalanobis")
    assert figure_called["hit"] is False


# ---------------------------------------------------------------------------
# IO-freeness
# ---------------------------------------------------------------------------
def test_no_files_written(clean_frame, tmp_path, monkeypatch):
    """The call writes no image files."""
    monkeypatch.chdir(tmp_path)
    plot_outlier_analysis(clean_frame)
    assert not list(tmp_path.rglob("*.png"))


# ---------------------------------------------------------------------------
# Lower-level helper
# ---------------------------------------------------------------------------
def test_select_helper_core_only_without_genotype(clean_frame):
    """The helper returns only core figures when genotype_col is not given."""
    from sleap_roots_analyze.outlier_detection import detect_outliers_mahalanobis

    result = detect_outliers_mahalanobis(clean_frame[[f"trait_{j}" for j in range(5)]])
    figs = _select_outlier_figures(clean_frame, {"mahalanobis": result}, "mahalanobis")
    assert MAHAL_KEYS <= set(figs)
    assert "outliers_per_genotype" not in figs


# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------
def test_importable_and_in_all():
    """Importable from the package root and listed once in __all__."""
    assert sra.plot_outlier_analysis is plot_outlier_analysis
    assert "plot_outlier_analysis" in sra.__all__
    assert sra.__all__.count("plot_outlier_analysis") == 1


def test_type_hints_resolve():
    """get_type_hints resolves (no NameError from **detect_kwargs: Any)."""
    typing.get_type_hints(plot_outlier_analysis)


def test_docstring_has_google_sections():
    """Docstring has Args/Returns/Raises sections (API-docs audit bar)."""
    doc = inspect.getdoc(plot_outlier_analysis) or ""
    assert "Args:" in doc and "Returns:" in doc and "Raises:" in doc
