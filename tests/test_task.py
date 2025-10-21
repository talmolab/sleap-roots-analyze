"""Tests for pipeline task module."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.task import Task, TaskResult


def test_task_result_creation():
    """Test creating a TaskResult."""
    result = TaskResult(
        data={"key": "value"},
        metadata={"elapsed": 1.5},
        files_generated=["output.csv"],
    )

    assert result.data == {"key": "value"}
    assert result.metadata == {"elapsed": 1.5}
    assert result.files_generated == ["output.csv"]


def test_task_result_default_fields():
    """Test TaskResult with default fields."""
    result = TaskResult(data="some data")

    assert result.data == "some data"
    assert result.metadata == {}
    assert result.files_generated == []


def test_task_creation():
    """Test creating a Task."""

    def dummy_func():
        return "result"

    task = Task(
        func=dummy_func,
        name="test_task",
        depends_on=["dep1", "dep2"],
        output_prefix="test_",
        description="A test task",
    )

    assert task.func == dummy_func
    assert task.name == "test_task"
    assert task.depends_on == ["dep1", "dep2"]
    assert task.output_prefix == "test_"
    assert task.description == "A test task"


def test_task_default_dependencies():
    """Test Task with default (empty) dependencies."""

    def dummy_func():
        return "result"

    task = Task(func=dummy_func, name="test_task")

    assert task.depends_on == []
    assert task.output_prefix == ""
    assert task.description == ""


def test_task_repr():
    """Test Task string representation."""

    def dummy_func():
        return "result"

    task = Task(func=dummy_func, name="my_task", depends_on=["dep1", "dep2"])

    repr_str = repr(task)
    assert "my_task" in repr_str
    assert "dep1" in repr_str
    assert "dep2" in repr_str


def test_task_repr_no_dependencies():
    """Test Task repr with no dependencies."""

    def dummy_func():
        return "result"

    task = Task(func=dummy_func, name="my_task")

    repr_str = repr(task)
    assert "my_task" in repr_str
    assert "none" in repr_str


def test_task_execute_simple(tmp_path):
    """Test executing a simple task."""

    def simple_func(config, run_dir, logger):
        return "success"

    task = Task(func=simple_func, name="simple")

    logger = logging.getLogger("test")
    result = task.execute(config={"param": "value"}, run_dir=tmp_path, logger=logger)

    assert isinstance(result, TaskResult)
    assert result.data == "success"
    assert "elapsed_time" in result.metadata
    assert "task_name" in result.metadata
    assert result.metadata["task_name"] == "simple"


def test_task_execute_returns_task_result(tmp_path):
    """Test task that returns TaskResult directly."""

    def func_with_result(config, run_dir, logger):
        return TaskResult(
            data={"result": "data"},
            metadata={"custom": "metadata"},
            files_generated=["file1.csv", "file2.csv"],
        )

    task = Task(func=func_with_result, name="custom_result")

    logger = logging.getLogger("test")
    result = task.execute(config={}, run_dir=tmp_path, logger=logger)

    assert result.data == {"result": "data"}
    assert result.metadata["custom"] == "metadata"
    assert "elapsed_time" in result.metadata  # Should be added
    assert "task_name" in result.metadata
    assert len(result.files_generated) == 2


def test_task_execute_with_dependencies(tmp_path):
    """Test executing a task with dependency results."""

    def load_func(config, run_dir, logger):
        return TaskResult(data={"loaded": "data"})

    def process_func(config, run_dir, logger, load_data):
        # Access the dependency's data
        input_data = load_data.data
        return TaskResult(data={"processed": input_data})

    load_task = Task(func=load_func, name="load_data")
    process_task = Task(
        func=process_func, name="process_data", depends_on=["load_data"]
    )

    logger = logging.getLogger("test")

    # Execute load task
    load_result = load_task.execute(config={}, run_dir=tmp_path, logger=logger)

    # Execute process task with dependency result
    process_result = process_task.execute(
        config={},
        run_dir=tmp_path,
        logger=logger,
        dependency_results={"load_data": load_result},
    )

    assert process_result.data == {"processed": {"loaded": "data"}}


def test_task_execute_error_propagation(tmp_path):
    """Test that exceptions in task functions are propagated."""

    def failing_func(config, run_dir, logger):
        raise ValueError("Task failed!")

    task = Task(func=failing_func, name="failing")

    logger = logging.getLogger("test")

    with pytest.raises(ValueError, match="Task failed!"):
        task.execute(config={}, run_dir=tmp_path, logger=logger)


def test_task_execute_multiple_dependencies(tmp_path):
    """Test task with multiple dependencies."""

    def dep1_func(config, run_dir, logger):
        return TaskResult(data={"from": "dep1"})

    def dep2_func(config, run_dir, logger):
        return TaskResult(data={"from": "dep2"})

    def main_func(config, run_dir, logger, dep1, dep2):
        combined = {"dep1": dep1.data, "dep2": dep2.data}
        return TaskResult(data=combined)

    dep1 = Task(func=dep1_func, name="dep1")
    dep2 = Task(func=dep2_func, name="dep2")
    main = Task(func=main_func, name="main", depends_on=["dep1", "dep2"])

    logger = logging.getLogger("test")

    # Execute dependencies
    dep1_result = dep1.execute(config={}, run_dir=tmp_path, logger=logger)
    dep2_result = dep2.execute(config={}, run_dir=tmp_path, logger=logger)

    # Execute main task
    main_result = main.execute(
        config={},
        run_dir=tmp_path,
        logger=logger,
        dependency_results={"dep1": dep1_result, "dep2": dep2_result},
    )

    assert main_result.data == {
        "dep1": {"from": "dep1"},
        "dep2": {"from": "dep2"},
    }


def test_task_execute_logs_timing(tmp_path, caplog):
    """Test that task execution logs timing information."""

    def slow_func(config, run_dir, logger):
        import time

        time.sleep(0.1)
        return "done"

    task = Task(func=slow_func, name="slow")

    logger = logging.getLogger("test")
    caplog.set_level(logging.INFO)

    result = task.execute(config={}, run_dir=tmp_path, logger=logger)

    # Check that elapsed time was recorded
    assert result.metadata["elapsed_time"] >= 0.1

    # Check that logging occurred
    assert "Executing task: slow" in caplog.text
    assert "completed" in caplog.text


def test_task_execute_logs_error(tmp_path, caplog):
    """Test that task execution logs errors."""

    def failing_func(config, run_dir, logger):
        raise RuntimeError("Something went wrong")

    task = Task(func=failing_func, name="failing")

    logger = logging.getLogger("test")
    caplog.set_level(logging.ERROR)

    with pytest.raises(RuntimeError):
        task.execute(config={}, run_dir=tmp_path, logger=logger)

    # Check that error was logged
    assert "failed" in caplog.text
    assert "Something went wrong" in caplog.text
