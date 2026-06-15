"""Tests for the input-contract boundary helper (issue #144).

The helper is a non-intrusive, optional side-check: it canonicalizes a *copy* of
the entry frame and runs ``sleap-roots-contracts`` against it, never touching the
frame fed to the pipeline, and degrades to a logged no-op when contracts is absent.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from sleap_roots_analyze.pipeline.config.components import ColumnConfig
from sleap_roots_analyze.validation import input_contract
from sleap_roots_analyze.validation.input_contract import (
    CONTRACTS_AVAILABLE,
    validate_entry_input,
)

# All behavior tests below require the optional dependency to be installed.
contracts = pytest.importorskip("sleap_roots_contracts")
from sleap_roots_contracts.examples import load_analysis_input_example  # noqa: E402

# A config whose role names already match the contract-canonical names, matching
# the packaged example frames (genotype / sample_id / replicate).
CANON_COLUMNS = ColumnConfig(
    genotype="genotype", barcode="sample_id", replicate="replicate"
)


@pytest.fixture
def good_input() -> pd.DataFrame:
    """A well-formed, contract-canonical entry frame."""
    return load_analysis_input_example("turface")


@pytest.fixture
def missing_sample_id(good_input: pd.DataFrame) -> pd.DataFrame:
    """Non-fatal issue: recommended sample_id role absent (warns / strict-raises)."""
    return good_input.drop(columns=["sample_id"])


@pytest.fixture
def missing_genotype(good_input: pd.DataFrame) -> pd.DataFrame:
    """Structural error: required genotype role absent (raises even under warn)."""
    return good_input.drop(columns=["genotype"])


@pytest.fixture
def no_numeric_trait(good_input: pd.DataFrame) -> pd.DataFrame:
    """Structural error: no numeric trait column (raises even under warn)."""
    return good_input[["genotype", "sample_id", "replicate"]].copy()


# --- 3.3: copy isolation, role canonicalization, optional roles, exclusions ---


def test_off_is_noop_even_when_installed(good_input, monkeypatch, caplog):
    """mode='off' never calls the validator and logs nothing, even if installed."""
    calls = []
    monkeypatch.setattr(
        input_contract, "validate_analysis_input", lambda *a, **k: calls.append(1)
    )
    with caplog.at_level(logging.DEBUG):
        validate_entry_input(good_input, columns=CANON_COLUMNS, mode="off")
    assert calls == []
    assert caplog.records == []


def test_validation_runs_on_a_copy(good_input):
    """The frame passed in is never mutated (validation works on a copy)."""
    before = good_input.copy(deep=True)
    validate_entry_input(good_input, columns=CANON_COLUMNS, mode="warn")
    assert_frame_equal(good_input, before)


def test_numeric_replicate_not_leaked_as_trait():
    """A numeric replicate role is canonicalized as a role, not a numeric trait."""
    df = pd.DataFrame(
        {
            "genotype": ["a", "a", "b", "b"],
            "sample_id": ["s1", "s2", "s3", "s4"],
            "replicate": [1, 2, 1, 2],  # numeric on input
            "trait_x": [0.1, 0.2, 0.3, 0.4],
        }
    )
    check = input_contract._build_validation_frame(df, columns=CANON_COLUMNS)
    assert "replicate" in check.columns
    # Roles are cast to string by canonicalize_role_dtypes; a leaked numeric
    # trait would have stayed numeric.
    assert str(check["replicate"].dtype) == "string"


def test_optional_replicate_absent_is_skipped():
    """columns.replicate=None with no replicate column does not error."""
    df = load_analysis_input_example("cylinder_no_replicate")
    cols = ColumnConfig(genotype="genotype", barcode="sample_id", replicate=None)
    check = input_contract._build_validation_frame(df, columns=cols)
    assert "replicate" not in check.columns
    # And the full path runs cleanly.
    validate_entry_input(df, columns=cols, mode="warn")


def test_additional_exclude_dropped_from_validation_copy():
    """additional_exclude columns are dropped from the validation copy."""
    df = pd.DataFrame(
        {
            "genotype": ["a", "a", "b"],
            "sample_id": ["s1", "s2", "s3"],
            "scan_id": [10, 20, 30],  # numeric metadata that must not be a trait
            "trait_x": [0.1, 0.2, 0.3],
        }
    )
    check = input_contract._build_validation_frame(
        df, columns=CANON_COLUMNS, additional_exclude=["scan_id"]
    )
    assert "scan_id" not in check.columns
    assert "trait_x" in check.columns


# --- input guards + actionable error messages (issue #144 review) ---


@pytest.mark.parametrize("mode", ["", "Warn", "foo", "none", "off ", None])
def test_invalid_mode_raises(good_input, mode):
    """A mode outside off|warn|strict is an explicit error, never silent warn semantics."""
    with pytest.raises(ValueError, match=r"off \| warn \| strict"):
        validate_entry_input(good_input, columns=CANON_COLUMNS, mode=mode)


def test_misconfigured_genotype_name_is_actionable():
    """A wrong columns.genotype names configured-vs-available, not a bare contract error."""
    df = pd.DataFrame(
        {
            "geno": ["a", "a", "b"],  # actual genotype column
            "sample_id": ["s1", "s2", "s3"],
            "trait_x": [0.1, 0.2, 0.3],
        }
    )
    cols = ColumnConfig(genotype="WrongName", barcode="sample_id", replicate=None)
    with pytest.raises(ValueError) as exc:
        validate_entry_input(df, columns=cols, mode="warn")
    msg = str(exc.value)
    assert "WrongName" in msg  # the configured (wrong) name
    assert "geno" in msg  # an available column, to point the user at the fix


def test_rename_collision_is_actionable():
    """A frame already carrying a canonical role name yields a clear collision error."""
    df = pd.DataFrame(
        {
            "Genotype": ["a", "a", "b"],  # configured genotype role
            "genotype": ["x", "y", "z"],  # pre-existing canonical-named column
            "sample_id": ["s1", "s2", "s3"],
            "trait_x": [0.1, 0.2, 0.3],
        }
    )
    cols = ColumnConfig(genotype="Genotype", barcode="sample_id", replicate=None)
    with pytest.raises(ValueError, match="already has a column named 'genotype'"):
        validate_entry_input(df, columns=cols, mode="warn")


# --- 3.4: severity behavior + logging ---


def test_good_input_passes_under_warn_and_strict(good_input):
    """Well-formed input raises nothing under either mode."""
    validate_entry_input(good_input, columns=CANON_COLUMNS, mode="warn")
    validate_entry_input(good_input, columns=CANON_COLUMNS, mode="strict")


def test_missing_sample_id_warns_under_warn(missing_sample_id, caplog):
    """A non-fatal issue is logged as a warning and does not raise under warn."""
    with caplog.at_level(logging.WARNING):
        validate_entry_input(missing_sample_id, columns=CANON_COLUMNS, mode="warn")
    assert any("sample_id" in r.message for r in caplog.records)


def test_missing_sample_id_raises_under_strict(missing_sample_id):
    """A non-fatal issue is escalated to an error under strict."""
    with pytest.raises(ValueError, match="sample_id"):
        validate_entry_input(missing_sample_id, columns=CANON_COLUMNS, mode="strict")


def test_missing_genotype_raises_even_under_warn(missing_genotype):
    """A structural error raises even under warn."""
    with pytest.raises(ValueError, match="genotype"):
        validate_entry_input(missing_genotype, columns=CANON_COLUMNS, mode="warn")


def test_no_numeric_trait_raises_even_under_warn(no_numeric_trait):
    """No numeric trait column is a structural error under warn."""
    with pytest.raises(ValueError, match="trait"):
        validate_entry_input(no_numeric_trait, columns=CANON_COLUMNS, mode="warn")


def test_nan_genotype_raises_even_under_warn(good_input):
    """NaN in the genotype role is a structural error under warn."""
    bad = good_input.copy()
    bad.loc[bad.index[0], "genotype"] = np.nan
    with pytest.raises(ValueError, match="genotype"):
        validate_entry_input(bad, columns=CANON_COLUMNS, mode="warn")


# --- 3.5: contracts-absent degradation (logged no-op) ---


def test_contracts_absent_is_logged_noop(good_input, monkeypatch, caplog):
    """With contracts unavailable, the helper degrades to a logged no-op (any mode).

    Monkeypatches the availability flag instead of reloading the module: a reload is
    fragile under pytest-xdist and leaves importers (e.g. ``load_data``) holding a stale
    pre-reload function reference. The genuine ``except ImportError`` fallback is covered
    by module import on environments without the optional dependency.
    """
    monkeypatch.setattr(input_contract, "CONTRACTS_AVAILABLE", False)
    monkeypatch.setattr(input_contract, "validate_analysis_input", None)
    monkeypatch.setattr(input_contract, "canonicalize_role_dtypes", None)
    with caplog.at_level(logging.INFO):
        validate_entry_input(good_input, columns=CANON_COLUMNS, mode="strict")
    assert any("skip" in r.message.lower() for r in caplog.records)


def test_module_exposes_contracts_available_flag():
    """The module advertises whether contracts is importable."""
    assert CONTRACTS_AVAILABLE is True  # installed in the dev group


# --- 5.1 / 5.2: equivalence + contracts-absent proofs on a real golden input ---

from pathlib import Path  # noqa: E402

from sleap_roots_analyze.pipeline import (  # noqa: E402
    ColumnConfig as _ColumnConfig,
    DataConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.steps import LoadDataStep  # noqa: E402

# The committed #120/#146 turface_19 raw EDPIE input (smallest reproduction input).
_TURFACE_19_RAW = (
    Path(__file__).parent
    / "fixtures"
    / "real"
    / "wheat_edpie"
    / "inputs"
    / "raw"
    / "turface_19"
    / "Turface_all_traits_2024_RSR_diameter_angle_traits_removed.csv"
)


def _turface_19_qc_config(mode: str) -> QCPipelineConfig:
    """A QC config over the committed turface_19 raw input with the given mode."""
    return QCPipelineConfig(
        pipeline_name="turface_19_equivalence",
        columns=_ColumnConfig(barcode="Barcode", genotype="geno", replicate="rep"),
        data=DataConfig(csv_path=str(_TURFACE_19_RAW), validate_input=mode),
    )


@pytest.mark.skipif(not _TURFACE_19_RAW.exists(), reason="repro fixture missing")
def test_equivalence_across_modes_on_golden(tmp_path):
    """validate_input off/warn/strict all yield an identical loaded entry frame (#144).

    The validator only runs at LoadDataStep; every downstream QC/PCA stage is a pure
    function of this frame, so identical load output guarantees identical results. The
    clean #120/#146 turface_19 golden also passes strict, so all three modes must agree —
    making "mode never changes r.data" airtight. We assert frame equality directly rather
    than running the heavy (UMAP-flaky) full pipeline three times.
    """
    step = LoadDataStep()
    frames = {}
    for mode in ("off", "warn", "strict"):
        run_dir = tmp_path / mode
        run_dir.mkdir()
        frames[mode] = step.execute(
            data=None,
            config=_turface_19_qc_config(mode),
            run_dir=run_dir,
            prev_result=None,
        ).data

    assert_frame_equal(frames["off"], frames["warn"])
    assert_frame_equal(frames["off"], frames["strict"])


@pytest.mark.skipif(not _TURFACE_19_RAW.exists(), reason="repro fixture missing")
def test_runs_without_contracts_identical_output(tmp_path, monkeypatch, caplog):
    """With contracts unavailable, run-all loads cleanly with identical output (#144)."""
    step = LoadDataStep()
    baseline_dir, absent_dir = tmp_path / "base", tmp_path / "absent"
    baseline_dir.mkdir()
    absent_dir.mkdir()

    # Baseline: validation active (contracts installed).
    r_base = step.execute(
        data=None,
        config=_turface_19_qc_config("warn"),
        run_dir=baseline_dir,
        prev_result=None,
    )

    # Simulate contracts absent: the helper must degrade to a logged no-op.
    monkeypatch.setattr(input_contract, "CONTRACTS_AVAILABLE", False)
    monkeypatch.setattr(input_contract, "validate_analysis_input", None)
    monkeypatch.setattr(input_contract, "canonicalize_role_dtypes", None)
    with caplog.at_level(logging.INFO):
        r_absent = step.execute(
            data=None,
            config=_turface_19_qc_config("warn"),
            run_dir=absent_dir,
            prev_result=None,
        )

    assert_frame_equal(r_base.data, r_absent.data)
    assert any("skip" in r.message.lower() for r in caplog.records)
