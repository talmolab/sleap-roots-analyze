"""Regression tests for optional ``columns.replicate`` (issue #142 / PR #143).

Each test pins one previously-uncovered failure mode on the replicate-free code
path. The existing suite only ever passed ``replicate=None`` / ``replicate_col=None``
explicitly, so the paths a real user actually hits (omit replicate in YAML, an
empty-string replicate, NaNs in the replicate column, a missing replicate column,
diagnostics on replicate-free data, a single-genotype dataset) were untested.

Semantics fixed by these tests:
- ``replicate`` defaults to ``"rep"``; the *documented* way to disable it is an
  explicit ``replicate: null`` in YAML (or ``ColumnConfig(replicate=None)``).
  Omitting the key keeps the ``"rep"`` default — it does **not** disable replicate.
- A falsy replicate (``None`` or ``""``) is treated as "unset" consistently across
  the trait-detection and heritability layers.
- Replicate values are never used in the heritability model, so a present-but-NaN
  replicate column yields **identical** H² to ``replicate=None``.
- A replicate column that is named but absent yields a structured ``{"error": ...}``,
  not a raw pandas ``KeyError``.
- Heritability requires >= 2 genotypes; a single genotype yields a structured error
  rather than a silent, meaningless H².
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from sleap_roots_analyze.data_cleanup import get_trait_columns
from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    HeritabilityConfig,
    QCPipeline,
    QCPipelineConfig,
    get_default_qc_config,
)
from sleap_roots_analyze.pipeline.config import load_qc_config
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import FilterHeritabilityStep
from sleap_roots_analyze.statistics import (
    analyze_trait_variance,
    calculate_heritability_estimates,
)


# ---------------------------------------------------------------------------
# 1. Config: omitting replicate vs. the documented `replicate: null` disable path
# ---------------------------------------------------------------------------
def test_omit_replicate_in_yaml_keeps_rep_default(tmp_path):
    """Omitting columns.replicate keeps the "rep" default — it does NOT disable it.

    The documented disable path is an explicit ``replicate: null`` (see next test),
    not omission. This pins B1 with the "rep"-default semantics (issue #142).
    """
    cfg_yaml = tmp_path / "qc.yaml"
    cfg_yaml.write_text(
        "pipeline_name: t\n"
        "columns:\n  genotype: geno\n  barcode: Barcode\n"  # replicate omitted
        "data:\n  csv_path: data.csv\n"
    )
    config = load_qc_config(str(cfg_yaml))
    assert config.columns.replicate == "rep"


def test_replicate_null_in_yaml_disables_replicate(tmp_path):
    """The documented disable path: explicit ``replicate: null`` round-trips to None."""
    cfg_yaml = tmp_path / "qc.yaml"
    cfg_yaml.write_text(
        "pipeline_name: t\n"
        "columns:\n  genotype: geno\n  barcode: Barcode\n  replicate: null\n"
        "data:\n  csv_path: data.csv\n"
    )
    config = load_qc_config(str(cfg_yaml))
    assert config.columns.replicate is None


# ---------------------------------------------------------------------------
# 2. Full QC pipeline end-to-end on a CSV with no replicate column
#    (exercises load_trait_data + CleanupTraitsStep, which the step-level
#     heritability tests skip).
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_qc_pipeline_runs_on_csv_with_no_replicate_column(tmp_path):
    """Load a real replicate-free CSV and run the QC pipeline end to end."""
    rng = np.random.default_rng(0)
    n_per_geno = 12
    genos = ["G1", "G2", "G3"]
    df = pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(n_per_geno * len(genos))],
            "geno": np.repeat(genos, n_per_geno),
            "trait1": rng.normal(50, 8, n_per_geno * len(genos)),
            "trait2": rng.normal(25, 4, n_per_geno * len(genos)),
        }
    )
    # Give trait1 a real genotype signal so heritability is well-defined.
    df.loc[df["geno"] == "G1", "trait1"] += 20
    csv_path = tmp_path / "cylinder_no_rep.csv"
    df.to_csv(csv_path, index=False)

    config = get_default_qc_config()
    config.data.csv_path = str(csv_path)
    config.columns.barcode = "Barcode"
    config.columns.genotype = "geno"
    config.columns.replicate = None  # the documented disable path
    config.outlier_detection.traditional_methods = ["mahalanobis"]
    config.visualization.create_eda_figures = False  # speed
    config.heritability.enabled = True
    config.heritability.threshold = 0.0  # retain traits; we only check H² is real

    pipeline = QCPipeline(config, output_dir=tmp_path / "runs")
    results = pipeline.run()

    # The whole pipeline completes on replicate-free data (no crash in load/cleanup).
    assert pipeline.get_summary().status == "success"
    assert len(results) == 10

    # 08 heritability output contains real H², not error rows.
    h2_csv = pipeline.run_dir / "data" / "08_heritability_results.csv"
    assert h2_csv.exists()
    h2_df = pd.read_csv(h2_csv)
    assert "heritability" in h2_df.columns
    assert h2_df["heritability"].notna().any()
    assert h2_df["heritability"].between(0, 1).all()


# ---------------------------------------------------------------------------
# 3. Empty-string replicate is treated consistently across both layers (I1)
# ---------------------------------------------------------------------------
def test_replicate_empty_string_treated_as_unset():
    """``replicate=""`` behaves like "unset" in both trait detection and heritability."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "geno": np.repeat(["G1", "G2", "G3"], 8),
            "trait1": rng.normal(10, 1, 24),
        }
    )
    df.loc[df["geno"] == "G1", "trait1"] += 3

    # Trait-detection layer: "" is unset, so no "" column is excluded and trait1 is a trait.
    assert get_trait_columns(df, genotype_col="geno", replicate_col="") == ["trait1"]

    # Heritability layer must agree: "" must NOT be treated as a required column.
    result = calculate_heritability_estimates(df, ["trait1"], replicate_col="")
    assert "error" not in result
    assert "error" not in result["trait1"]
    assert 0 <= result["trait1"]["heritability"] <= 1


# ---------------------------------------------------------------------------
# 4. NaNs in the replicate column must not change H² vs. replicate=None (I2)
# ---------------------------------------------------------------------------
def test_heritability_replicate_with_nans_identical_to_none():
    """A present-but-NaN replicate column yields identical H² to replicate=None.

    Replicate values are never used in the model, so NaNs in the replicate column
    must not drop rows from the analysis (which would change n, mean_n_reps, and H²).
    """
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "geno": np.repeat(["G1", "G2", "G3", "G4"], 8),
            "rep": np.tile(range(1, 9), 4).astype(float),
            "trait1": rng.normal(10, 1, 32),
        }
    )
    df.loc[df["geno"] == "G1", "trait1"] += 3
    # NaNs in rep on rows whose trait/genotype are perfectly valid.
    df.loc[[0, 5, 17, 28], "rep"] = np.nan

    with_rep = calculate_heritability_estimates(df, ["trait1"], replicate_col="rep")
    without_rep = calculate_heritability_estimates(df, ["trait1"], replicate_col=None)

    assert "error" not in with_rep["trait1"]
    assert with_rep["trait1"]["heritability"] == pytest.approx(
        without_rep["trait1"]["heritability"]
    )
    assert (
        with_rep["trait1"]["n_observations"] == without_rep["trait1"]["n_observations"]
    )


# ---------------------------------------------------------------------------
# 5. analyze_trait_variance with a named-but-absent replicate column (I3)
# ---------------------------------------------------------------------------
def test_analyze_trait_variance_missing_replicate_returns_structured_error():
    """Should return ``{"error": ...}`` like calculate_heritability_estimates, not raise."""
    df = pd.DataFrame(
        {
            "geno": np.repeat(["G1", "G2"], 5),
            "trait1": np.arange(10.0),  # >=3 rows so the error is the missing column
        }
    )
    result = analyze_trait_variance(df, "trait1", replicate_col="rep")
    assert "error" in result
    assert "Missing required columns" in result["error"]


# ---------------------------------------------------------------------------
# 6. FilterHeritabilityStep diagnostics on replicate-free data (I4)
# ---------------------------------------------------------------------------
def test_filter_heritability_diagnostics_with_no_replicate(tmp_path):
    """generate_diagnostics=True on replicate-free data must actually produce diagnostics.

    Previously FilterHeritabilityStep passed a hardcoded ``replicate_col="Replicate"``
    to ``compare_trait_heritabilities``; on replicate-free data that raised a KeyError
    swallowed by the broad ``except``, silently skipping the diagnostics.
    """
    rng = np.random.default_rng(3)
    n = 30
    df = pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(n)],
            "Genotype": np.repeat(["A", "B", "C"], n // 3),
            "high_h2_trait": rng.normal(50, 10, n),
            "low_h2_trait": rng.normal(15, 3, n),
        }
    )
    # Strong genotype signal for the high-H² trait so the contrast is real.
    df.loc[df["Genotype"] == "A", "high_h2_trait"] += 40

    config = QCPipelineConfig(
        pipeline_name="test_cylinder",
        columns=ColumnConfig(barcode="Barcode", genotype="Genotype", replicate=None),
        data=DataConfig(csv_path="dummy.csv"),
        heritability=HeritabilityConfig(
            enabled=True, threshold=0.3, generate_diagnostics=True
        ),
    )
    prev_result = StepResult(
        data=df,
        metadata={
            "trait_names": ["high_h2_trait", "low_h2_trait"],
            "valid_trait_names": ["high_h2_trait", "low_h2_trait"],
            "heritability_results": {
                "high_h2_trait": {"heritability": 0.8},
                "low_h2_trait": {"heritability": 0.1},  # forced removal
            },
            "samples": n,
        },
        files_generated=[],
    )

    step = FilterHeritabilityStep()
    result = step.execute(df, config, tmp_path, prev_result)

    # Diagnostics were actually generated (not silently skipped).
    diagnostics = result.metadata.get("diagnostic_results")
    assert diagnostics is not None
    assert diagnostics.get("status") != "failed"
    assert "error" not in diagnostics
    assert (tmp_path / "09_heritability_diagnostics.csv").exists()


# ---------------------------------------------------------------------------
# 7. Single-genotype dataset: guarded, not a silent nonsensical H²
# ---------------------------------------------------------------------------
def test_heritability_single_genotype_returns_error():
    """One genotype -> H² is not estimable; return a structured error, not a fake H²."""
    df = pd.DataFrame(
        {
            "geno": ["G1"] * 10,  # a single genotype
            "trait1": np.linspace(
                1.0, 10.0, 10
            ),  # varies, so it's not the no_variance branch
        }
    )
    result = calculate_heritability_estimates(df, ["trait1"], replicate_col=None)
    assert "error" in result["trait1"]
    assert "heritability" not in result["trait1"]


# ---------------------------------------------------------------------------
# 7b. Downstream of the single-genotype guard: heritability filtering must NOT
#     remove traits whose H² was uncomputable (errored). Otherwise a degenerate
#     group (e.g. one genotype) loses every trait, leaving downstream pipeline
#     steps a trait-less DataFrame to crash on.
# ---------------------------------------------------------------------------
def test_filter_heritability_retains_uncomputable_traits(tmp_path):
    """Error (uncomputable) H² traits are kept; only computed-and-low traits drop."""
    df = pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(6)],
            "Genotype": ["A"] * 6,
            "uncomputable_trait": np.linspace(1.0, 6.0, 6),
            "low_h2_trait": np.linspace(1.0, 6.0, 6),
        }
    )
    config = QCPipelineConfig(
        pipeline_name="test",
        columns=ColumnConfig(barcode="Barcode", genotype="Genotype", replicate=None),
        data=DataConfig(csv_path="dummy.csv"),
        heritability=HeritabilityConfig(enabled=True, threshold=0.3),
    )
    prev_result = StepResult(
        data=df,
        metadata={
            "trait_names": ["uncomputable_trait", "low_h2_trait"],
            "valid_trait_names": ["uncomputable_trait", "low_h2_trait"],
            "heritability_results": {
                # No computable H² (single-genotype group reports an error).
                "uncomputable_trait": {"error": "Insufficient genotypes"},
                # A genuinely low, but computed, H² -> still removed.
                "low_h2_trait": {"heritability": 0.1},
            },
            "samples": 6,
        },
        files_generated=[],
    )

    result = FilterHeritabilityStep().execute(df, config, tmp_path, prev_result)
    retained = result.metadata["valid_trait_names"]
    assert "uncomputable_trait" in retained  # kept: can't fail an unmeasured threshold
    assert "low_h2_trait" not in retained  # dropped: computed H² below threshold
