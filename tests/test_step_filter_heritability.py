"""Tests for FilterHeritabilityStep (Step 9)."""

from __future__ import annotations

import json
import logging
import matplotlib
import numpy as np
import pandas as pd
import pytest

# Use non-interactive backend for testing
matplotlib.use("Agg")

from sleap_roots_analyze.pipeline import (
    ColumnConfig,
    DataConfig,
    HeritabilityConfig,
    QCPipelineConfig,
)
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps import FilterHeritabilityStep


@pytest.fixture
def sample_data():
    """Create sample data with standardized column names (as after CleanupTraitsStep)."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"plant{i}" for i in range(20)],
            "Genotype": ["A"] * 10 + ["B"] * 10,
            "Replicate": [1, 2] * 10,
            "high_h2_trait": np.random.randn(20) * 10 + 50,
            "med_h2_trait": np.random.randn(20) * 5 + 25,
            "low_h2_trait": np.random.randn(20) * 3 + 15,
        }
    )


@pytest.fixture
def config():
    """Create config with standardized column names."""
    return QCPipelineConfig(
        pipeline_name="test_qc",
        columns=ColumnConfig(
            barcode="Barcode", genotype="Genotype", replicate="Replicate"
        ),
        data=DataConfig(csv_path="dummy.csv"),
        heritability=HeritabilityConfig(threshold=0.3),
    )


@pytest.fixture
def prev_result(sample_data):
    """Previous result with heritability data."""
    return StepResult(
        data=sample_data,
        metadata={
            "trait_names": [
                "high_h2_trait",
                "med_h2_trait",
                "low_h2_trait",
            ],  # Primary key
            "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
            "heritability_results": {
                "high_h2_trait": {"heritability": 0.8},
                "med_h2_trait": {"heritability": 0.4},
                "low_h2_trait": {"heritability": 0.1},
            },
            "samples": 20,
        },
        files_generated=[],
    )


class TestFilterHeritabilityStepBasic:
    """Test basic functionality."""

    def test_step_initialization(self):
        """Test initialization."""
        step = FilterHeritabilityStep()
        assert step.step_name == "FilterHeritability"
        assert "heritability" in step.description.lower()

    def test_filtering_removes_low_h2_traits(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that low H2 traits are removed."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # low_h2_trait (H2=0.1) should be removed from trait list
        # Note: DataFrame still has all columns, only valid_trait_names changes
        assert "low_h2_trait" not in result.metadata["valid_trait_names"]
        assert "high_h2_trait" in result.metadata["valid_trait_names"]
        assert "med_h2_trait" in result.metadata["valid_trait_names"]

    def test_metadata_tracks_removed_traits(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test metadata tracks removed traits."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Removed traits tracked in metadata
        assert "low_h2_trait" not in result.metadata["valid_trait_names"]

    def test_valid_trait_names_updated(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test valid trait names updated."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        assert "low_h2_trait" not in result.metadata["valid_trait_names"]
        assert "high_h2_trait" in result.metadata["valid_trait_names"]


class TestFilterHeritabilityStepEdgeCases:
    """Test edge cases."""

    def test_all_traits_pass(self, sample_data, config, tmp_path):
        """Test when all traits pass threshold."""
        # Override threshold to 0.0 so all traits pass
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.0),  # All pass
        )

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "heritability_results": {
                    "high_h2_trait": {"heritability": 0.8},
                    "med_h2_trait": {"heritability": 0.4},
                    "low_h2_trait": {"heritability": 0.1},
                },
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # All traits should remain
        assert len(result.metadata["valid_trait_names"]) == 3


class TestFilterHeritabilityStepDiagnostics:
    """Test diagnostic mode functionality."""

    def test_diagnostics_disabled_by_default(
        self, sample_data, config, prev_result, tmp_path
    ):
        """Test that diagnostics are not generated by default."""
        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Diagnostic results should not be in metadata
        assert "diagnostic_results" not in result.metadata

        # Diagnostic files should not exist
        assert not (tmp_path / "09_heritability_diagnostics.csv").exists()
        assert not (tmp_path / "figures" / "09_variance_decomposition.png").exists()
        assert not (tmp_path / "figures" / "09_removed_traits_boxplots.png").exists()

    def test_diagnostics_enabled_generates_files(
        self, sample_data, prev_result, tmp_path
    ):
        """Test that enabling diagnostics generates expected files."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(
                threshold=0.3,
                generate_diagnostics=True,  # Enable diagnostics
            ),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Diagnostic results should be in metadata
        assert "diagnostic_results" in result.metadata
        assert "comparison_df" in result.metadata["diagnostic_results"]

        # Diagnostic files should exist
        assert (tmp_path / "09_heritability_diagnostics.csv").exists()
        assert (tmp_path / "figures" / "09_variance_decomposition.png").exists()
        assert (tmp_path / "figures" / "09_removed_traits_boxplots.png").exists()

        # Check that diagnostic CSV has data
        diag_df = pd.read_csv(tmp_path / "09_heritability_diagnostics.csv")
        assert len(diag_df) == 3  # All 3 traits analyzed
        assert "trait" in diag_df.columns
        assert "heritability" in diag_df.columns
        assert "pct_var_between" in diag_df.columns

    def test_diagnostics_added_to_files_generated(
        self, sample_data, prev_result, tmp_path
    ):
        """Test that diagnostic files are added to files_generated list."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.3, generate_diagnostics=True),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # Check that diagnostic files are in files_generated
        file_names = [f.name for f in result.files_generated]
        assert "09_heritability_diagnostics.csv" in file_names
        assert "09_variance_decomposition.png" in file_names
        assert "09_removed_traits_boxplots.png" in file_names

    def test_diagnostics_not_generated_if_no_removed_traits(
        self, sample_data, tmp_path
    ):
        """Test that diagnostics are not generated if no traits are removed."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(
                threshold=0.0,  # All traits pass
                generate_diagnostics=True,
            ),
        )

        prev_result = StepResult(
            data=sample_data,
            metadata={
                "trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "valid_trait_names": ["high_h2_trait", "med_h2_trait", "low_h2_trait"],
                "heritability_results": {
                    "high_h2_trait": {"heritability": 0.8},
                    "med_h2_trait": {"heritability": 0.4},
                    "low_h2_trait": {"heritability": 0.1},
                },
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        # No traits removed, so diagnostics should not be generated
        assert "diagnostic_results" not in result.metadata
        assert not (tmp_path / "09_heritability_diagnostics.csv").exists()

    def test_diagnostics_metadata_structure(self, sample_data, prev_result, tmp_path):
        """Test that diagnostic results in metadata have expected structure."""
        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.3, generate_diagnostics=True),
        )

        step = FilterHeritabilityStep()
        result = step.execute(sample_data, config, tmp_path, prev_result)

        diag = result.metadata["diagnostic_results"]
        assert "comparison_df" in diag
        assert "diagnostic_csv" in diag
        assert "variance_plot" in diag
        assert "boxplot" in diag
        assert "traits_analyzed" in diag
        assert "traits_removed_plotted" in diag

        # Check values
        assert diag["traits_analyzed"] == 3
        assert diag["traits_removed_plotted"] == 1  # Only low_h2_trait removed

    def test_diagnostics_with_many_removed_traits(self, tmp_path):
        """Test that boxplots are limited to top 10 when many traits are removed."""
        # Create data with 15 traits using standardized column names
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(20)],
                "Genotype": ["A"] * 10 + ["B"] * 10,
                "Replicate": [1, 2] * 10,
            }
        )
        # Add 15 traits, all with low heritability
        for i in range(15):
            data[f"trait_{i}"] = np.random.randn(20) * 3 + 15

        config = QCPipelineConfig(
            pipeline_name="test_qc",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(threshold=0.8, generate_diagnostics=True),
        )

        # All traits have H2 < 0.8
        h2_results = {f"trait_{i}": {"heritability": 0.1} for i in range(15)}
        prev_result = StepResult(
            data=data,
            metadata={
                "trait_names": [f"trait_{i}" for i in range(15)],
                "valid_trait_names": [f"trait_{i}" for i in range(15)],
                "heritability_results": h2_results,
                "samples": 20,
            },
            files_generated=[],
        )

        step = FilterHeritabilityStep()
        result = step.execute(data, config, tmp_path, prev_result)

        # Check that only 10 traits were plotted
        assert result.metadata["diagnostic_results"]["traits_removed_plotted"] == 10


class TestFilterHeritabilityStepDefensiveGuard:
    """Tests for defense-in-depth guard against empty heritability_results.

    When heritability filtering is enabled but heritability_results is empty
    (e.g., because statistics.calculate_heritability=False), the guard should
    prevent silent removal of all traits.
    """

    @pytest.fixture
    def guard_sample_data(self):
        """Sample data for guard tests."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "Barcode": [f"plant{i}" for i in range(20)],
                "Genotype": ["A"] * 10 + ["B"] * 10,
                "Replicate": [1, 2] * 10,
                "trait_1": np.random.randn(20) * 10 + 50,
                "trait_2": np.random.randn(20) * 5 + 25,
                "trait_3": np.random.randn(20) * 3 + 15,
            }
        )

    @pytest.fixture
    def guard_config_enabled(self):
        """Config with heritability filtering enabled."""
        return QCPipelineConfig(
            pipeline_name="test_guard",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(enabled=True, threshold=0.3),
        )

    @pytest.fixture
    def guard_config_disabled(self):
        """Config with heritability filtering disabled."""
        return QCPipelineConfig(
            pipeline_name="test_guard",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(csv_path="dummy.csv"),
            heritability=HeritabilityConfig(enabled=False, threshold=0.3),
        )

    @pytest.fixture
    def prev_result_empty_h2(self, guard_sample_data):
        """Previous result with empty heritability_results."""
        return StepResult(
            data=guard_sample_data,
            metadata={
                "trait_names": ["trait_1", "trait_2", "trait_3"],
                "valid_trait_names": ["trait_1", "trait_2", "trait_3"],
                "heritability_results": {},
                "summary": {"heritability_summary": {"skipped": True}},
                "samples": 20,
            },
            files_generated=[],
        )

    @pytest.fixture
    def prev_result_populated_h2(self, guard_sample_data):
        """Previous result with populated heritability_results."""
        return StepResult(
            data=guard_sample_data,
            metadata={
                "trait_names": ["trait_1", "trait_2", "trait_3"],
                "valid_trait_names": ["trait_1", "trait_2", "trait_3"],
                "heritability_results": {
                    "trait_1": {"heritability": 0.8},
                    "trait_2": {"heritability": 0.4},
                    "trait_3": {"heritability": 0.1},
                },
                "samples": 20,
            },
            files_generated=[],
        )

    def test_guard_prevents_silent_trait_removal(
        self, guard_sample_data, guard_config_enabled, prev_result_empty_h2, tmp_path
    ):
        """Guard should retain all traits when heritability_results is empty."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_enabled, tmp_path, prev_result_empty_h2
        )

        # All 3 trait columns must still be present
        trait_cols = [
            c
            for c in result.data.columns
            if c not in ("Barcode", "Genotype", "Replicate")
        ]
        assert len(trait_cols) == 3
        assert set(trait_cols) == {"trait_1", "trait_2", "trait_3"}

    def test_guard_logs_warning(
        self,
        guard_sample_data,
        guard_config_enabled,
        prev_result_empty_h2,
        tmp_path,
        caplog,
    ):
        """Guard should log a warning when it activates."""
        step = FilterHeritabilityStep()
        with caplog.at_level(logging.WARNING):
            step.execute(
                guard_sample_data, guard_config_enabled, tmp_path, prev_result_empty_h2
            )

        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert any(
            "heritability" in msg.lower() for msg in warning_messages
        ), f"Expected warning about heritability, got: {warning_messages}"

    def test_guard_metadata_includes_flag(
        self, guard_sample_data, guard_config_enabled, prev_result_empty_h2, tmp_path
    ):
        """Guard should set guard_activated and guard_reason in metadata."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_enabled, tmp_path, prev_result_empty_h2
        )

        assert result.metadata.get("guard_activated") is True
        assert isinstance(result.metadata.get("guard_reason"), str)
        assert len(result.metadata["guard_reason"]) > 0

    def test_guard_preserves_metadata(
        self, guard_sample_data, guard_config_enabled, prev_result_empty_h2, tmp_path
    ):
        """Guard should preserve all previous step metadata."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_enabled, tmp_path, prev_result_empty_h2
        )

        assert result.metadata["trait_names"] == ["trait_1", "trait_2", "trait_3"]
        assert result.metadata["valid_trait_names"] == ["trait_1", "trait_2", "trait_3"]
        assert result.metadata["heritability_results"] == {}
        assert result.metadata["samples"] == 20

    def test_guard_generates_correct_files(
        self, guard_sample_data, guard_config_enabled, prev_result_empty_h2, tmp_path
    ):
        """Guard should generate consistent output files."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_enabled, tmp_path, prev_result_empty_h2
        )

        # Summary JSON should show no traits removed and guard activated
        summary = result.metadata["summary"]
        assert summary["traits_removed"] == 0
        assert summary["traits_retained"] == 3
        assert summary["guard_activated"] is True

        # Removed traits JSON should be empty list
        removed_json_path = tmp_path / "09_removed_traits.json"
        assert removed_json_path.exists()
        with open(removed_json_path) as f:
            removed = json.load(f)
        assert removed == []

    def test_guard_does_not_activate_with_populated_results(
        self,
        guard_sample_data,
        guard_config_enabled,
        prev_result_populated_h2,
        tmp_path,
    ):
        """Normal filtering should occur when heritability_results is populated."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_enabled, tmp_path, prev_result_populated_h2
        )

        # Normal filtering: guard should NOT activate
        assert "guard_activated" not in result.metadata
        # With threshold 0.3, trait_3 (h2=0.1) should be removed
        assert "trait_3" not in result.data.columns

    def test_guard_does_not_activate_when_filtering_disabled(
        self, guard_sample_data, guard_config_disabled, prev_result_empty_h2, tmp_path
    ):
        """Disabled path should work normally without guard activation."""
        step = FilterHeritabilityStep()
        result = step.execute(
            guard_sample_data, guard_config_disabled, tmp_path, prev_result_empty_h2
        )

        # Existing disabled path — no guard
        assert "guard_activated" not in result.metadata
        # All traits retained
        assert "trait_1" in result.data.columns
        assert "trait_2" in result.data.columns
        assert "trait_3" in result.data.columns
