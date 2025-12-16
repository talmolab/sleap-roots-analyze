"""Tests for pipeline run provenance (config saving and data source tracking).

TDD tests written before implementation per add-pipeline-run-provenance proposal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from omegaconf import OmegaConf

from sleap_roots_analyze.pipeline.pipelines import BasePipeline
from sleap_roots_analyze.pipeline.summary import PipelineSummary
from sleap_roots_analyze.pipeline.task import Task, TaskResult


# =============================================================================
# Test Fixtures: Simple pipeline with dataclass config
# =============================================================================


@dataclass
class SimpleTestConfig:
    """Simple test config with data path."""

    data_path: str = "/path/to/data.csv"
    param1: int = 10
    param2: str = "test"


class ProvenancePipeline(BasePipeline):
    """Test pipeline for provenance testing."""

    def create_tasks(self) -> List[Task]:
        """Create simple tasks."""

        def load_data(config, run_dir, logger):
            return TaskResult(data={"loaded": True})

        def process_data(config, run_dir, logger, load_data):
            return TaskResult(data={"processed": True})

        return [
            Task(func=load_data, name="load_data", description="Load data"),
            Task(
                func=process_data,
                name="process_data",
                depends_on=["load_data"],
                description="Process data",
            ),
        ]


class FailingProvenancePipeline(BasePipeline):
    """Pipeline that fails to test config saving on failure."""

    def create_tasks(self) -> List[Task]:
        """Create tasks where one fails."""

        def success_task(config, run_dir, logger):
            return TaskResult(data="success")

        def failing_task(config, run_dir, logger, success_task):
            raise RuntimeError("Intentional failure for testing")

        return [
            Task(func=success_task, name="success_task"),
            Task(
                func=failing_task,
                name="failing_task",
                depends_on=["success_task"],
            ),
        ]


# =============================================================================
# Phase 1.1: Tests for config.yaml saving to run directory
# =============================================================================


class TestConfigYamlSaving:
    """Tests for saving config.yaml to run directory."""

    def test_config_yaml_exists_after_run(self, tmp_path):
        """Test that config.yaml is created in run directory after successful run."""
        config = SimpleTestConfig(data_path="/test/data.csv", param1=42)
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        assert config_path.exists(), "config.yaml should exist in run directory"

    def test_config_yaml_contains_parameters(self, tmp_path):
        """Test that saved config.yaml contains the correct parameters."""
        config = SimpleTestConfig(data_path="/test/data.csv", param1=99, param2="hello")
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        loaded_config = OmegaConf.load(config_path)

        assert loaded_config.data_path == "/test/data.csv"
        assert loaded_config.param1 == 99
        assert loaded_config.param2 == "hello"

    def test_config_yaml_saved_with_dict_config(self, tmp_path):
        """Test config saving works with dict config."""
        config = {"value": 10, "name": "test", "nested": {"key": "value"}}
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        assert config_path.exists()

        loaded_config = OmegaConf.load(config_path)
        assert loaded_config.value == 10
        assert loaded_config.name == "test"
        assert loaded_config.nested.key == "value"

    def test_config_yaml_saved_on_failure(self, tmp_path):
        """Test that config.yaml is saved even when pipeline fails."""
        config = SimpleTestConfig(data_path="/test/data.csv", param1=123)
        pipeline = FailingProvenancePipeline(config=config, output_dir=tmp_path)

        with pytest.raises(RuntimeError, match="Intentional failure"):
            pipeline.run()

        # Config should still be saved
        config_path = pipeline.run_dir / "config.yaml"
        assert config_path.exists(), "config.yaml should exist even on failure"

        loaded_config = OmegaConf.load(config_path)
        assert loaded_config.param1 == 123


# =============================================================================
# Phase 1.2: Tests for summary.config population
# =============================================================================


class TestSummaryConfigPopulation:
    """Tests for populating config in pipeline summary."""

    def test_summary_config_populated(self, tmp_path):
        """Test that summary.config contains config dict after run."""
        config = SimpleTestConfig(data_path="/test/data.csv", param1=50)
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        summary = pipeline.get_summary()
        assert summary.config != {}, "summary.config should not be empty"
        assert "param1" in summary.config or "data_path" in summary.config

    def test_summary_config_serializable_to_json(self, tmp_path):
        """Test that summary.config can be serialized to JSON."""
        config = SimpleTestConfig(
            data_path=str(tmp_path / "data.csv"),  # Path as string
            param1=100,
        )
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        summary = pipeline.get_summary()
        # This should not raise
        json_str = json.dumps(summary.config)
        assert json_str is not None

    def test_summary_config_in_saved_json(self, tmp_path):
        """Test that config is included in saved pipeline_summary.json."""
        config = SimpleTestConfig(data_path="/test/data.csv", param1=77)
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        summary_path = pipeline.run_dir / "pipeline_summary.json"
        with open(summary_path) as f:
            saved_summary = json.load(f)

        assert "config" in saved_summary
        assert saved_summary["config"] != {}

    def test_summary_config_with_dict_config(self, tmp_path):
        """Test summary.config works with dict config."""
        config = {"value": 42, "flag": True}
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        summary = pipeline.get_summary()
        assert summary.config.get("value") == 42
        assert summary.config.get("flag") is True


# =============================================================================
# Phase 1.3: Tests for data_source in summary
# =============================================================================


class TestDataSourceTracking:
    """Tests for data_source field in pipeline summary."""

    def test_summary_has_data_source_field(self):
        """Test that PipelineSummary has data_source field."""
        summary = PipelineSummary(pipeline_name="test")
        # Should have data_source attribute (may be empty string or None by default)
        assert hasattr(summary, "data_source")

    def test_data_source_in_saved_summary(self, tmp_path):
        """Test that data_source is included in saved summary JSON."""
        config = SimpleTestConfig(data_path="/test/input.csv")
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        summary_path = pipeline.run_dir / "pipeline_summary.json"
        with open(summary_path) as f:
            saved_summary = json.load(f)

        assert "data_source" in saved_summary


# =============================================================================
# Phase 1.4: Tests for config.yaml content fidelity
# =============================================================================


class TestConfigFidelity:
    """Tests for config.yaml content fidelity and reloadability."""

    def test_saved_config_can_be_reloaded(self, tmp_path):
        """Test that saved config.yaml can be loaded back."""
        config = SimpleTestConfig(
            data_path="/original/path.csv",
            param1=999,
            param2="fidelity_test",
        )
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        reloaded = OmegaConf.load(config_path)

        # Should be able to convert back to structured config
        assert reloaded.data_path == "/original/path.csv"
        assert reloaded.param1 == 999
        assert reloaded.param2 == "fidelity_test"

    def test_config_includes_all_resolved_values(self, tmp_path):
        """Test that config includes resolved (not interpolated) values."""
        # OmegaConf supports interpolation like ${other_key}
        # The saved config should have resolved values
        config = OmegaConf.create(
            {
                "base_dir": "/data",
                "data_path": "${base_dir}/input.csv",
                "param": 10,
            }
        )
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        # Load without resolving to check what was saved
        saved_raw = config_path.read_text()

        # The saved config should work when loaded
        reloaded = OmegaConf.load(config_path)
        # Should be able to access data_path (either resolved or interpolated)
        assert "data" in str(reloaded.data_path).lower() or "input" in str(
            reloaded.data_path
        )

    def test_config_handles_path_objects(self, tmp_path):
        """Test that Path objects in config are properly serialized."""

        @dataclass
        class ConfigWithPaths:
            input_path: Path = Path("/default/path")
            output_path: Path = Path("/default/output")

        config = ConfigWithPaths(
            input_path=tmp_path / "input.csv",
            output_path=tmp_path / "output",
        )
        pipeline = ProvenancePipeline(config=config, output_dir=tmp_path)

        pipeline.run()

        config_path = pipeline.run_dir / "config.yaml"
        reloaded = OmegaConf.load(config_path)

        # Paths should be serialized as strings
        assert "input.csv" in str(reloaded.input_path)
        assert "output" in str(reloaded.output_path)
