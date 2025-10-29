"""Tests for pipeline base class."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.pipelines import BasePipeline
from sleap_roots_analyze.pipeline.task import Task, TaskResult


class SimplePipeline(BasePipeline):
    """Simple test pipeline."""

    def create_tasks(self):
        """Create a simple two-task pipeline."""

        def load_data(config, run_dir, logger):
            return TaskResult(data={"value": config.get("value", 42)})

        def process_data(config, run_dir, logger, load_data):
            value = load_data.data["value"]
            result_path = run_dir / "result.txt"
            result_path.write_text(f"Processed: {value * 2}")
            return TaskResult(
                data={"processed": value * 2},
                files_generated=[result_path],
            )

        return [
            Task(
                func=load_data,
                name="load_data",
                description="Load data from config",
            ),
            Task(
                func=process_data,
                name="process_data",
                depends_on=["load_data"],
                description="Process loaded data",
            ),
        ]


class FailingPipeline(BasePipeline):
    """Pipeline that fails during execution."""

    def create_tasks(self):
        """Create tasks where one fails."""

        def success_task(config, run_dir, logger):
            return TaskResult(data="success")

        def failing_task(config, run_dir, logger, success_task):
            raise RuntimeError("Task failed!")

        return [
            Task(func=success_task, name="success_task"),
            Task(
                func=failing_task,
                name="failing_task",
                depends_on=["success_task"],
            ),
        ]


def test_pipeline_initialization(tmp_path):
    """Test pipeline initialization."""
    config = {"value": 10}
    pipeline = SimplePipeline(
        config=config,
        output_dir=tmp_path,
        pipeline_name="test_pipeline",
        version="1.0",
    )

    assert pipeline.config == config
    assert pipeline.output_dir == tmp_path
    assert pipeline.pipeline_name == "test_pipeline"
    assert pipeline.version == "1.0"
    assert pipeline.logger is not None
    assert pipeline.run_dir.exists()
    assert pipeline.run_dir.parent == tmp_path


def test_pipeline_creates_run_directory(tmp_path):
    """Test that pipeline creates a timestamped run directory."""
    pipeline = SimplePipeline(config={}, output_dir=tmp_path, pipeline_name="test")

    # Run directory should exist
    assert pipeline.run_dir.exists()

    # Should be inside output_dir
    assert pipeline.run_dir.parent == tmp_path

    # Should start with pipeline name
    assert pipeline.run_dir.name.startswith("test_")


def test_pipeline_run_success(tmp_path):
    """Test successful pipeline execution."""
    config = {"value": 5}
    pipeline = SimplePipeline(config=config, output_dir=tmp_path)

    results = pipeline.run()

    # Check results
    assert "load_data" in results
    assert "process_data" in results
    assert results["load_data"].data == {"value": 5}
    assert results["process_data"].data == {"processed": 10}

    # Check that file was created
    result_file = pipeline.run_dir / "result.txt"
    assert result_file.exists()
    assert result_file.read_text() == "Processed: 10"


def test_pipeline_summary_created(tmp_path):
    """Test that pipeline creates and saves a summary."""
    pipeline = SimplePipeline(config={}, output_dir=tmp_path)

    results = pipeline.run()

    # Summary should exist
    summary_path = pipeline.run_dir / "pipeline_summary.json"
    assert summary_path.exists()

    # Summary should have correct structure
    summary = pipeline.get_summary()
    assert summary.pipeline_name == "pipeline"
    assert summary.status == "success"
    assert len(summary.steps) == 2
    assert summary.steps[0].name == "load_data"
    assert summary.steps[1].name == "process_data"
    assert summary.steps[0].status == "success"
    assert summary.steps[1].status == "success"


def test_pipeline_summary_has_timing(tmp_path):
    """Test that summary includes timing information."""
    pipeline = SimplePipeline(config={}, output_dir=tmp_path)

    results = pipeline.run()

    summary = pipeline.get_summary()

    # Each step should have elapsed time (may be 0 for very fast tasks)
    assert summary.steps[0].elapsed_time >= 0
    assert summary.steps[1].elapsed_time >= 0

    # Overall summary should have times
    assert summary.start_time != ""
    assert summary.end_time != ""
    assert summary.total_elapsed_time >= 0


def test_pipeline_summary_has_files(tmp_path):
    """Test that summary includes generated files."""
    pipeline = SimplePipeline(config={}, output_dir=tmp_path)

    results = pipeline.run()

    summary = pipeline.get_summary()

    # process_data should have generated a file
    assert len(summary.steps[1].files_generated) == 1
    assert "result.txt" in str(summary.steps[1].files_generated[0])


def test_pipeline_failure_handling(tmp_path):
    """Test that pipeline handles failures properly."""
    pipeline = FailingPipeline(config={}, output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="Task failed!"):
        pipeline.run()

    # Summary should still be saved
    summary_path = pipeline.run_dir / "pipeline_summary.json"
    assert summary_path.exists()

    # Summary should show failure
    summary = pipeline.get_summary()
    assert summary.status == "failed"


def test_pipeline_logs_execution_order(tmp_path, caplog):
    """Test that pipeline logs the execution order."""
    import logging

    caplog.set_level(logging.INFO)

    pipeline = SimplePipeline(config={}, output_dir=tmp_path)
    results = pipeline.run()

    # Check that execution order was logged
    assert "Execution order" in caplog.text
    assert "load_data" in caplog.text
    assert "process_data" in caplog.text


def test_pipeline_logs_completion(tmp_path, caplog):
    """Test that pipeline logs completion."""
    import logging

    caplog.set_level(logging.INFO)

    pipeline = SimplePipeline(config={}, output_dir=tmp_path)
    results = pipeline.run()

    # Check completion logging
    assert "Starting pipeline" in caplog.text
    assert "completed successfully" in caplog.text


def test_pipeline_abstract_method():
    """Test that BasePipeline cannot be instantiated without create_tasks."""
    with pytest.raises(TypeError):
        # This should fail because create_tasks is not implemented
        pipeline = BasePipeline(config={}, output_dir=".")


class EmptyPipeline(BasePipeline):
    """Pipeline with no tasks."""

    def create_tasks(self):
        """Return empty task list."""
        return []


def test_pipeline_with_no_tasks(tmp_path):
    """Test pipeline with no tasks."""
    pipeline = EmptyPipeline(config={}, output_dir=tmp_path)

    results = pipeline.run()

    assert results == {}
    assert pipeline.get_summary().status == "success"


class ComplexPipeline(BasePipeline):
    """More complex pipeline with multiple branches."""

    def create_tasks(self):
        """Create tasks with parallel branches."""

        def task1(config, run_dir, logger):
            return TaskResult(data=1)

        def task2(config, run_dir, logger, task1):
            return TaskResult(data=task1.data + 1)

        def task3(config, run_dir, logger, task1):
            return TaskResult(data=task1.data + 2)

        def task4(config, run_dir, logger, task2, task3):
            return TaskResult(data=task2.data + task3.data)

        return [
            Task(func=task1, name="task1", description="First task"),
            Task(
                func=task2,
                name="task2",
                depends_on=["task1"],
                description="Second task",
            ),
            Task(
                func=task3,
                name="task3",
                depends_on=["task1"],
                description="Third task",
            ),
            Task(
                func=task4,
                name="task4",
                depends_on=["task2", "task3"],
                description="Fourth task",
            ),
        ]


def test_complex_pipeline_execution(tmp_path):
    """Test execution of a more complex pipeline."""
    pipeline = ComplexPipeline(config={}, output_dir=tmp_path)

    results = pipeline.run()

    # Check all tasks executed
    assert len(results) == 4

    # Check results: task4 = (1+1) + (1+2) = 5
    assert results["task1"].data == 1
    assert results["task2"].data == 2
    assert results["task3"].data == 3
    assert results["task4"].data == 5

    # Check summary
    summary = pipeline.get_summary()
    assert len(summary.steps) == 4
    assert all(step.status == "success" for step in summary.steps)


def test_pipeline_custom_name_and_version(tmp_path):
    """Test pipeline with custom name and version."""
    pipeline = SimplePipeline(
        config={},
        output_dir=tmp_path,
        pipeline_name="my_analysis",
        version="2.5",
    )

    results = pipeline.run()

    summary = pipeline.get_summary()
    assert summary.pipeline_name == "my_analysis"
    assert summary.version == "2.5"

    # Run directory should use the custom name
    assert "my_analysis" in pipeline.run_dir.name
