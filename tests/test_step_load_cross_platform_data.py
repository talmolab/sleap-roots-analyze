"""Tests for LoadCrossPlatformDataStep."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.config.components import CrossPlatformConfig


@pytest.fixture
def exp1_csv(tmp_path):
    """Create experiment 1 CSV file for testing."""
    data = {
        "Geno": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        "Rep": [1, 2, 3, 1, 2, 3, 1, 2, 3],
        "trait1": [10.5, 11.2, 10.8, 12.3, 11.9, 12.5, 9.8, 10.2, 10.0],
        "trait2": [5.2, 5.5, 5.3, 6.1, 5.9, 6.3, 4.8, 5.0, 4.9],
        "trait3": [15.0, 15.5, 15.2, 16.0, 15.8, 16.2, 14.5, 14.8, 14.6],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "exp1_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def exp2_csv(tmp_path):
    """Create experiment 2 CSV file for testing."""
    data = {
        "geno": ["A", "A", "B", "B", "B", "C", "C"],
        "rep": [1, 2, 1, 2, 3, 1, 2],
        "trait_a": [20.1, 20.5, 22.0, 21.8, 22.2, 19.5, 19.8],
        "trait_b": [10.2, 10.5, 11.0, 10.8, 11.2, 9.8, 10.0],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "exp2_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def cross_platform_config(exp1_csv, exp2_csv):
    """Create a test CrossPlatformConfig."""
    return CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Cylinder",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Turface",
        exp2_genotype_col="geno",
        correlation_method="spearman",
        min_samples_per_genotype=2,
    )


def test_load_cross_platform_data_step_initialization():
    """Test LoadCrossPlatformDataStep initialization."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    assert step.step_name == "LoadCrossPlatformData"
    assert "Load and align" in step.description


def _xp_config(exp1_csv, exp2_csv, mode):
    """Build a CrossPlatformConfig over the test CSVs with the given validate_input."""
    return CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Cylinder",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Turface",
        exp2_genotype_col="geno",
        correlation_method="spearman",
        min_samples_per_genotype=2,
        validate_input=mode,
    )


def test_load_cross_platform_validates_each_frame_once(
    exp1_csv, exp2_csv, tmp_path, monkeypatch
):
    """The boundary helper is called once per experiment frame (exp1, exp2) (#154)."""
    import sleap_roots_analyze.pipeline.steps.load_cross_platform_data as xp_mod

    calls = []
    monkeypatch.setattr(
        xp_mod,
        "validate_cross_platform_experiment",
        lambda df, **kwargs: calls.append(kwargs["mode"]),
    )

    step = xp_mod.LoadCrossPlatformDataStep()
    step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "warn"),
        run_dir=tmp_path,
        prev_result=None,
    )

    assert calls == ["warn", "warn"]  # exp1 and exp2


def test_load_cross_platform_passes_exclude_cols_to_validation(
    exp1_csv, exp2_csv, tmp_path, monkeypatch
):
    """The validator receives the same exclude_cols as the real trait selection (#154).

    Otherwise validation sees a different trait set than the pipeline analyzes — a
    numeric excluded-metadata column would be validated as a trait (or mask a genuine
    "no numeric trait" error).
    """
    import sleap_roots_analyze.pipeline.steps.load_cross_platform_data as xp_mod

    captured = []
    monkeypatch.setattr(
        xp_mod,
        "validate_cross_platform_experiment",
        lambda df, **kwargs: captured.append(kwargs.get("additional_exclude")),
    )

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Cylinder",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Turface",
        exp2_genotype_col="geno",
        min_samples_per_genotype=2,
        exp1_exclude_cols=["Ent", "Sub"],
        exp2_exclude_cols=["scanner"],
    )
    xp_mod.LoadCrossPlatformDataStep().execute(
        data=None, config=config, run_dir=tmp_path, prev_result=None
    )

    assert captured == [["Ent", "Sub"], ["scanner"]]  # exp1 then exp2


def test_load_cross_platform_strict_equivalent_output(exp1_csv, exp2_csv, tmp_path):
    """Strict yields identical loaded output to off (strict is usable, #154)."""
    from pandas.testing import assert_frame_equal

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    off_dir, strict_dir = tmp_path / "off", tmp_path / "strict"
    off_dir.mkdir()
    strict_dir.mkdir()
    step = LoadCrossPlatformDataStep()

    r_off = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "off"),
        run_dir=off_dir,
        prev_result=None,
    )
    r_strict = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "strict"),
        run_dir=strict_dir,
        prev_result=None,
    )

    assert_frame_equal(r_off.data["exp1_df"], r_strict.data["exp1_df"])
    assert_frame_equal(r_off.data["exp2_df"], r_strict.data["exp2_df"])


def test_load_cross_platform_nan_genotype_dropped_equivalence(tmp_path):
    """A blank genotype cell is dropped in alignment; off/warn stay identical (#154).

    Pre-fix the default warn aborted a run that off produced output for, because the
    validator hard-fails on a NaN genotype. Dropping the unusable row in alignment keeps
    the equivalence promise.
    """
    from pandas.testing import assert_frame_equal

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    # Both experiments carry a blank genotype cell among otherwise-shared genotypes.
    exp1 = pd.DataFrame(
        {
            "Geno": ["A", "A", "B", "B", None],
            "Rep": [1, 2, 1, 2, 1],
            "trait1": [10.5, 11.2, 12.3, 11.9, 9.0],
        }
    )
    exp2 = pd.DataFrame(
        {
            "geno": ["A", "A", "B", "B", None],
            "rep": [1, 2, 1, 2, 1],
            "trait_a": [20.1, 20.5, 22.0, 21.8, 5.0],
        }
    )
    exp1_csv = tmp_path / "exp1_nan.csv"
    exp2_csv = tmp_path / "exp2_nan.csv"
    exp1.to_csv(exp1_csv, index=False)
    exp2.to_csv(exp2_csv, index=False)

    off_dir, warn_dir = tmp_path / "off", tmp_path / "warn"
    off_dir.mkdir()
    warn_dir.mkdir()
    step = LoadCrossPlatformDataStep()

    r_off = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "off"),
        run_dir=off_dir,
        prev_result=None,
    )
    # warn must NOT raise (the NaN-genotype row is gone before validation).
    r_warn = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "warn"),
        run_dir=warn_dir,
        prev_result=None,
    )

    assert r_warn.data["exp1_df"]["genotype"].notna().all()
    assert_frame_equal(r_off.data["exp1_df"], r_warn.data["exp1_df"])
    assert_frame_equal(r_off.data["exp2_df"], r_warn.data["exp2_df"])


def test_validation_does_not_preempt_no_common_genotypes(tmp_path):
    """Under default warn, the existing 'No common genotypes' error still surfaces (#154)."""
    exp1 = pd.DataFrame({"Geno": ["A", "B"], "Rep": [1, 1], "trait1": [10.5, 12.3]})
    exp2 = pd.DataFrame({"geno": ["C", "D"], "rep": [1, 1], "trait_a": [20.1, 22.0]})
    exp1_csv = tmp_path / "exp1_nc.csv"
    exp2_csv = tmp_path / "exp2_nc.csv"
    exp1.to_csv(exp1_csv, index=False)
    exp2.to_csv(exp2_csv, index=False)

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    # validate_input defaults to warn here; the genotype-alignment error must win.
    with pytest.raises(ValueError, match="No common genotypes found"):
        LoadCrossPlatformDataStep().execute(
            data=None,
            config=_xp_config(exp1_csv, exp2_csv, "warn"),
            run_dir=tmp_path,
            prev_result=None,
        )


def test_load_cross_platform_validation_does_not_change_output(
    exp1_csv, exp2_csv, tmp_path
):
    """Output with validate_input=warn equals validate_input=off (#154)."""
    from pandas.testing import assert_frame_equal

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    off_dir, warn_dir = tmp_path / "off", tmp_path / "warn"
    off_dir.mkdir()
    warn_dir.mkdir()
    step = LoadCrossPlatformDataStep()

    r_off = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "off"),
        run_dir=off_dir,
        prev_result=None,
    )
    r_warn = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "warn"),
        run_dir=warn_dir,
        prev_result=None,
    )

    assert_frame_equal(r_off.data["exp1_df"], r_warn.data["exp1_df"])
    assert_frame_equal(r_off.data["exp2_df"], r_warn.data["exp2_df"])


def test_load_cross_platform_runs_without_contracts(
    exp1_csv, exp2_csv, tmp_path, monkeypatch, caplog
):
    """With contracts unavailable, the step runs cleanly with identical output (#154)."""
    import logging

    from pandas.testing import assert_frame_equal

    import sleap_roots_analyze.validation.input_contract as ic
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    base_dir, absent_dir = tmp_path / "base", tmp_path / "absent"
    base_dir.mkdir()
    absent_dir.mkdir()
    step = LoadCrossPlatformDataStep()

    r_base = step.execute(
        data=None,
        config=_xp_config(exp1_csv, exp2_csv, "warn"),
        run_dir=base_dir,
        prev_result=None,
    )

    monkeypatch.setattr(ic, "CONTRACTS_AVAILABLE", False)
    monkeypatch.setattr(ic, "validate_analysis_input", None)
    monkeypatch.setattr(ic, "canonicalize_role_dtypes", None)
    with caplog.at_level(logging.INFO):
        r_absent = step.execute(
            data=None,
            config=_xp_config(exp1_csv, exp2_csv, "warn"),
            run_dir=absent_dir,
            prev_result=None,
        )

    assert_frame_equal(r_base.data["exp1_df"], r_absent.data["exp1_df"])
    assert_frame_equal(r_base.data["exp2_df"], r_absent.data["exp2_df"])
    assert any("skip" in r.message.lower() for r in caplog.records)


def test_load_cross_platform_data_step_execute(cross_platform_config, tmp_path):
    """Test LoadCrossPlatformDataStep execution."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that data was loaded
    assert result.data is not None
    assert "exp1_df" in result.data
    assert "exp2_df" in result.data
    assert "common_genotypes" in result.data

    # Check data types
    assert isinstance(result.data["exp1_df"], pd.DataFrame)
    assert isinstance(result.data["exp2_df"], pd.DataFrame)
    assert isinstance(result.data["common_genotypes"], list)

    # Check common genotypes
    assert len(result.data["common_genotypes"]) == 3  # A, B, C
    assert set(result.data["common_genotypes"]) == {"A", "B", "C"}


def test_load_cross_platform_data_step_metadata(cross_platform_config, tmp_path):
    """Test that metadata is correctly populated."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check metadata
    assert "exp1_samples" in result.metadata
    assert "exp2_samples" in result.metadata
    assert "exp1_traits" in result.metadata
    assert "exp2_traits" in result.metadata
    assert "common_genotypes" in result.metadata

    assert result.metadata["exp1_samples"] == 9
    assert result.metadata["exp2_samples"] == 7
    assert result.metadata["exp1_traits"] == 3  # trait1, trait2, trait3
    assert result.metadata["exp2_traits"] == 2  # trait_a, trait_b
    assert result.metadata["common_genotypes"] == 3


def test_load_cross_platform_data_step_trait_identification(
    cross_platform_config, tmp_path
):
    """Test that trait columns are correctly identified."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that trait column names are stored in metadata
    assert "exp1_trait_names" in result.metadata
    assert "exp2_trait_names" in result.metadata

    exp1_traits = result.metadata["exp1_trait_names"]
    exp2_traits = result.metadata["exp2_trait_names"]

    assert "trait1" in exp1_traits
    assert "trait2" in exp1_traits
    assert "trait3" in exp1_traits
    assert "trait_a" in exp2_traits
    assert "trait_b" in exp2_traits

    # Genotype and rep columns should NOT be in traits
    assert "Geno" not in exp1_traits
    assert "Rep" not in exp1_traits
    assert "geno" not in exp2_traits
    assert "rep" not in exp2_traits


def test_load_cross_platform_data_step_min_samples_filter(tmp_path):
    """Test that genotypes are filtered by min_samples_per_genotype."""
    # Create data where genotype C has only 1 sample in exp1
    exp1_data = {
        "Geno": ["A", "A", "A", "B", "B", "B", "C"],
        "Rep": [1, 2, 3, 1, 2, 3, 1],
        "trait1": [10.5, 11.2, 10.8, 12.3, 11.9, 12.5, 9.8],
    }
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_min_samples.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    exp2_data = {
        "geno": ["A", "A", "B", "B", "C", "C"],
        "rep": [1, 2, 1, 2, 1, 2],
        "trait_a": [20.1, 20.5, 22.0, 21.8, 19.5, 19.8],
    }
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_min_samples.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        min_samples_per_genotype=2,  # Require at least 2 samples
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    # C should be excluded because it has only 1 sample in exp1
    # Only A and B should remain
    assert len(result.data["common_genotypes"]) == 2
    assert set(result.data["common_genotypes"]) == {"A", "B"}
    assert "C" not in result.data["common_genotypes"]


def test_load_cross_platform_data_step_no_common_genotypes(tmp_path):
    """Test behavior when there are no common genotypes."""
    exp1_data = {"Geno": ["A", "B"], "Rep": [1, 1], "trait1": [10.5, 12.3]}
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_no_common.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    exp2_data = {"geno": ["C", "D"], "rep": [1, 1], "trait_a": [20.1, 22.0]}
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_no_common.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()

    with pytest.raises(ValueError, match="No common genotypes found"):
        step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)


def test_load_cross_platform_data_step_files_generated(cross_platform_config, tmp_path):
    """Test that appropriate files are generated."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Check that files were generated
    assert len(result.files_generated) > 0

    # Check specific expected files
    expected_files = [
        "cross_platform_exp1_loaded.csv",
        "cross_platform_exp2_loaded.csv",
        "cross_platform_alignment_summary.csv",
    ]

    for expected_file in expected_files:
        file_path = tmp_path / expected_file
        assert file_path.exists(), f"Expected file {expected_file} not found"
        assert file_path in result.files_generated


def test_load_cross_platform_data_step_alignment_summary(
    cross_platform_config, tmp_path
):
    """Test that alignment summary file contains correct information."""
    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    step.execute(
        data=None, config=cross_platform_config, run_dir=tmp_path, prev_result=None
    )

    # Read alignment summary
    summary_path = tmp_path / "cross_platform_alignment_summary.csv"
    summary = pd.read_csv(summary_path)

    # Check columns
    assert "genotype" in summary.columns
    assert "exp1_samples" in summary.columns
    assert "exp2_samples" in summary.columns

    # Check content - should have all 3 common genotypes
    assert len(summary) == 3
    assert set(summary["genotype"]) == {"A", "B", "C"}

    # Check sample counts
    geno_a = summary[summary["genotype"] == "A"].iloc[0]
    assert geno_a["exp1_samples"] == 3
    assert geno_a["exp2_samples"] == 2


def test_cross_platform_config_accepts_exclude_cols():
    """Test that CrossPlatformConfig accepts exp1_exclude_cols and exp2_exclude_cols."""
    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        exp1_exclude_cols=["Ent", "Sub", "Cid"],
        exp2_exclude_cols=["scanner", "region"],
    )

    assert config.exp1_exclude_cols == ["Ent", "Sub", "Cid"]
    assert config.exp2_exclude_cols == ["scanner", "region"]


def test_cross_platform_config_exclude_cols_default_none():
    """Test that exclude_cols parameters default to None."""
    config = CrossPlatformConfig(
        exp1_data_path="dummy1.csv",
        exp1_name="Exp1",
        exp1_genotype_col="geno",
        exp2_data_path="dummy2.csv",
        exp2_name="Exp2",
        exp2_genotype_col="geno",
    )

    assert config.exp1_exclude_cols is None
    assert config.exp2_exclude_cols is None


def test_load_cross_platform_excludes_exp1_columns(tmp_path):
    """Test that LoadCrossPlatformDataStep excludes columns from exp1 traits."""
    # Create exp1 data with metadata columns that should be excluded
    exp1_data = {
        "Geno": ["A", "A", "B", "B"],
        "Rep": [1, 2, 1, 2],
        "Ent": [1, 1, 2, 2],  # Entry number - should be excluded
        "Sub": [1, 1, 1, 1],  # Sub-entry - should be excluded
        "trait1": [10.5, 11.2, 12.3, 11.9],
        "trait2": [5.2, 5.5, 6.1, 5.9],
    }
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_with_metadata.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    # Create exp2 data
    exp2_data = {
        "geno": ["A", "A", "B", "B"],
        "rep": [1, 2, 1, 2],
        "trait_a": [20.1, 20.5, 22.0, 21.8],
    }
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_simple.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        min_samples_per_genotype=2,
        exp1_exclude_cols=["Ent", "Sub"],
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    exp1_traits = result.metadata["exp1_trait_names"]

    # Ent and Sub should NOT be in traits (excluded)
    assert "Ent" not in exp1_traits
    assert "Sub" not in exp1_traits

    # Actual traits should be present
    assert "trait1" in exp1_traits
    assert "trait2" in exp1_traits
    assert len(exp1_traits) == 2


def test_load_cross_platform_excludes_exp2_columns(tmp_path):
    """Test that LoadCrossPlatformDataStep excludes columns from exp2 traits."""
    exp1_data = {
        "Geno": ["A", "A", "B", "B"],
        "Rep": [1, 2, 1, 2],
        "trait1": [10.5, 11.2, 12.3, 11.9],
    }
    exp1_df = pd.DataFrame(exp1_data)
    exp1_csv = tmp_path / "exp1_simple.csv"
    exp1_df.to_csv(exp1_csv, index=False)

    exp2_data = {
        "geno": ["A", "A", "B", "B"],
        "rep": [1, 2, 1, 2],
        "scanner": [1, 1, 2, 2],
        "region": [1, 1, 1, 1],
        "trait_a": [20.1, 20.5, 22.0, 21.8],
        "trait_b": [10.2, 10.5, 11.0, 10.8],
    }
    exp2_df = pd.DataFrame(exp2_data)
    exp2_csv = tmp_path / "exp2_with_metadata.csv"
    exp2_df.to_csv(exp2_csv, index=False)

    config = CrossPlatformConfig(
        exp1_data_path=str(exp1_csv),
        exp1_name="Exp1",
        exp1_genotype_col="Geno",
        exp2_data_path=str(exp2_csv),
        exp2_name="Exp2",
        exp2_genotype_col="geno",
        min_samples_per_genotype=2,
        exp2_exclude_cols=["scanner", "region"],
    )

    from sleap_roots_analyze.pipeline.steps.load_cross_platform_data import (
        LoadCrossPlatformDataStep,
    )

    step = LoadCrossPlatformDataStep()
    result = step.execute(data=None, config=config, run_dir=tmp_path, prev_result=None)

    exp2_traits = result.metadata["exp2_trait_names"]

    assert "scanner" not in exp2_traits
    assert "region" not in exp2_traits
    assert "trait_a" in exp2_traits
    assert "trait_b" in exp2_traits
    assert len(exp2_traits) == 2
