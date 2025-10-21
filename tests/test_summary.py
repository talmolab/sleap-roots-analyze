"""Tests for pipeline summary module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary


def test_step_summary_creation():
    """Test creating a StepSummary."""
    step = StepSummary(
        name="test_step",
        description="A test step",
        status="success",
        elapsed_time=1.5,
        files_generated=["file1.csv", "file2.csv"],
        metadata={"param": "value"},
    )

    assert step.name == "test_step"
    assert step.description == "A test step"
    assert step.status == "success"
    assert step.elapsed_time == 1.5
    assert step.files_generated == ["file1.csv", "file2.csv"]
    assert step.metadata == {"param": "value"}
    assert step.error is None


def test_step_summary_defaults():
    """Test StepSummary with default values."""
    step = StepSummary(name="test")

    assert step.name == "test"
    assert step.description == ""
    assert step.status == "pending"
    assert step.elapsed_time == 0.0
    assert step.files_generated == []
    assert step.metadata == {}
    assert step.error is None


def test_pipeline_summary_creation():
    """Test creating a PipelineSummary."""
    step1 = StepSummary(name="step1")
    step2 = StepSummary(name="step2")

    summary = PipelineSummary(
        pipeline_name="test_pipeline",
        version="1.0",
        start_time="2024-01-01T12:00:00",
        end_time="2024-01-01T12:05:00",
        total_elapsed_time=300.0,
        status="success",
        steps=[step1, step2],
        config={"param": "value"},
        environment={"git_commit": "abc123"},
        output_directory="/path/to/output",
    )

    assert summary.pipeline_name == "test_pipeline"
    assert summary.version == "1.0"
    assert summary.start_time == "2024-01-01T12:00:00"
    assert summary.status == "success"
    assert len(summary.steps) == 2


def test_pipeline_summary_defaults():
    """Test PipelineSummary with default values."""
    summary = PipelineSummary(pipeline_name="test")

    assert summary.pipeline_name == "test"
    assert summary.version == "1.0"
    assert summary.start_time == ""
    assert summary.end_time == ""
    assert summary.total_elapsed_time == 0.0
    assert summary.status == "running"
    assert summary.steps == []
    assert summary.config == {}
    assert summary.environment == {}


def test_pipeline_summary_to_dict():
    """Test converting PipelineSummary to dictionary."""
    step = StepSummary(name="step1", description="First step")
    summary = PipelineSummary(pipeline_name="test", version="1.0", steps=[step])

    data = summary.to_dict()

    assert isinstance(data, dict)
    assert data["pipeline_name"] == "test"
    assert data["version"] == "1.0"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["name"] == "step1"


def test_pipeline_summary_to_json():
    """Test converting PipelineSummary to JSON."""
    step = StepSummary(name="step1", files_generated=[Path("file.csv")])
    summary = PipelineSummary(pipeline_name="test", steps=[step])

    json_str = summary.to_json()

    # Should be valid JSON
    data = json.loads(json_str)
    assert data["pipeline_name"] == "test"
    assert len(data["steps"]) == 1

    # Path should be converted to string
    assert isinstance(data["steps"][0]["files_generated"][0], str)


def test_pipeline_summary_save_and_load(tmp_path):
    """Test saving and loading PipelineSummary."""
    step1 = StepSummary(name="step1", description="First step")
    step2 = StepSummary(name="step2", description="Second step")

    original = PipelineSummary(
        pipeline_name="test",
        version="2.0",
        start_time="2024-01-01T12:00:00",
        status="success",
        steps=[step1, step2],
        config={"key": "value"},
    )

    # Save
    save_path = tmp_path / "summary.json"
    original.save(save_path)

    assert save_path.exists()

    # Load
    loaded = PipelineSummary.load(save_path)

    assert loaded.pipeline_name == "test"
    assert loaded.version == "2.0"
    assert loaded.start_time == "2024-01-01T12:00:00"
    assert loaded.status == "success"
    assert len(loaded.steps) == 2
    assert loaded.steps[0].name == "step1"
    assert loaded.steps[1].name == "step2"
    assert loaded.config == {"key": "value"}


def test_pipeline_summary_save_creates_directory(tmp_path):
    """Test that save() creates parent directories."""
    summary = PipelineSummary(pipeline_name="test")

    # Save to a nested path that doesn't exist
    save_path = tmp_path / "nested" / "dir" / "summary.json"
    summary.save(save_path)

    assert save_path.exists()


def test_pipeline_summary_mark_step_success():
    """Test marking a step as successful."""
    step = StepSummary(name="step1", status="pending")
    summary = PipelineSummary(pipeline_name="test", steps=[step])

    summary.mark_step_success(
        step_name="step1",
        elapsed_time=2.5,
        files_generated=["output.csv"],
        metadata={"rows": 100},
    )

    assert summary.steps[0].status == "success"
    assert summary.steps[0].elapsed_time == 2.5
    assert summary.steps[0].files_generated == ["output.csv"]
    assert summary.steps[0].metadata == {"rows": 100}


def test_pipeline_summary_mark_step_failed():
    """Test marking a step as failed."""
    step = StepSummary(name="step1", status="pending")
    summary = PipelineSummary(pipeline_name="test", steps=[step])

    summary.mark_step_failed(step_name="step1", error="Something went wrong")

    assert summary.steps[0].status == "failed"
    assert summary.steps[0].error == "Something went wrong"


def test_pipeline_summary_mark_nonexistent_step():
    """Test marking a nonexistent step (should not raise error)."""
    summary = PipelineSummary(pipeline_name="test", steps=[])

    # Should not raise an error
    summary.mark_step_success(
        step_name="nonexistent",
        elapsed_time=1.0,
        files_generated=[],
        metadata={},
    )


def test_pipeline_summary_finalize():
    """Test finalizing a pipeline summary."""
    summary = PipelineSummary(pipeline_name="test")

    # Set start time
    from datetime import datetime

    summary.start_time = datetime.now().isoformat()

    # Finalize
    summary.finalize(status="success")

    assert summary.status == "success"
    assert summary.end_time != ""
    assert summary.total_elapsed_time >= 0


def test_pipeline_summary_finalize_without_start_time():
    """Test finalizing without a start time."""
    summary = PipelineSummary(pipeline_name="test")

    # Finalize without setting start_time
    summary.finalize(status="success")

    assert summary.status == "success"
    assert summary.end_time != ""
    # total_elapsed_time should still be 0 since start_time wasn't set
    assert summary.total_elapsed_time == 0


def test_pipeline_summary_finalize_calculates_elapsed():
    """Test that finalize calculates elapsed time."""
    from datetime import datetime, timedelta

    summary = PipelineSummary(pipeline_name="test")

    # Set start time to 5 seconds ago
    start = datetime.now() - timedelta(seconds=5)
    summary.start_time = start.isoformat()

    summary.finalize(status="success")

    # Elapsed time should be approximately 5 seconds
    assert summary.total_elapsed_time >= 4.9
    assert summary.total_elapsed_time <= 5.5
