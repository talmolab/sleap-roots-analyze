"""Tests for pipeline DAG executor."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.dag import DAGExecutor, DAGValidationError
from sleap_roots_analyze.pipeline.task import Task, TaskResult


def test_dag_executor_creation():
    """Test creating a DAG executor."""

    def func1(config, run_dir, logger):
        return "result1"

    def func2(config, run_dir, logger, task1):
        return "result2"

    task1 = Task(func=func1, name="task1")
    task2 = Task(func=func2, name="task2", depends_on=["task1"])

    executor = DAGExecutor([task1, task2])

    assert len(executor.tasks) == 2
    assert "task1" in executor.tasks
    assert "task2" in executor.tasks


def test_dag_executor_duplicate_names():
    """Test that duplicate task names raise an error."""

    def func():
        pass

    task1 = Task(func=func, name="duplicate")
    task2 = Task(func=func, name="duplicate")

    with pytest.raises(DAGValidationError, match="Duplicate task names"):
        DAGExecutor([task1, task2])


def test_dag_validate_missing_dependency():
    """Test validation fails for missing dependencies."""

    def func(config, run_dir, logger):
        pass

    task = Task(func=func, name="task1", depends_on=["nonexistent"])

    executor = DAGExecutor([task])

    with pytest.raises(DAGValidationError, match="non-existent"):
        executor.validate()


def test_dag_validate_cycle_detection():
    """Test that cycles in the DAG are detected."""

    def func(config, run_dir, logger):
        pass

    # Create a cycle: task1 -> task2 -> task3 -> task1
    task1 = Task(func=func, name="task1", depends_on=["task3"])
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task2"])

    executor = DAGExecutor([task1, task2, task3])

    with pytest.raises(DAGValidationError, match="Cycle detected"):
        executor.validate()


def test_dag_validate_self_dependency():
    """Test that self-dependency creates a cycle."""

    def func(config, run_dir, logger):
        pass

    task = Task(func=func, name="task1", depends_on=["task1"])

    executor = DAGExecutor([task])

    with pytest.raises(DAGValidationError, match="Cycle detected"):
        executor.validate()


def test_dag_validate_success():
    """Test validation succeeds for valid DAG."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1"])
    task4 = Task(func=func, name="task4", depends_on=["task2", "task3"])

    executor = DAGExecutor([task1, task2, task3, task4])

    # Should not raise
    executor.validate()


def test_dag_get_execution_order_linear():
    """Test execution order for linear dependency chain."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task2"])

    executor = DAGExecutor([task1, task2, task3])

    order = executor.get_execution_order()

    assert order == ["task1", "task2", "task3"]


def test_dag_get_execution_order_parallel():
    """Test execution order with parallel branches."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1"])
    task4 = Task(func=func, name="task4", depends_on=["task2", "task3"])

    executor = DAGExecutor([task1, task2, task3, task4])

    order = executor.get_execution_order()

    # task1 must be first
    assert order[0] == "task1"
    # task4 must be last
    assert order[-1] == "task4"
    # task2 and task3 can be in any order, but both before task4
    assert set(order[1:3]) == {"task2", "task3"}


def test_dag_get_execution_order_complex():
    """Test execution order for complex DAG."""

    def func(config, run_dir, logger):
        pass

    # Create a more complex DAG
    #     task1
    #    /     \
    # task2   task3
    #    \     / \
    #    task4  task5
    #       \   /
    #       task6

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1"])
    task4 = Task(func=func, name="task4", depends_on=["task2", "task3"])
    task5 = Task(func=func, name="task5", depends_on=["task3"])
    task6 = Task(func=func, name="task6", depends_on=["task4", "task5"])

    executor = DAGExecutor([task1, task2, task3, task4, task5, task6])

    order = executor.get_execution_order()

    # Verify topological constraints
    assert order.index("task1") < order.index("task2")
    assert order.index("task1") < order.index("task3")
    assert order.index("task2") < order.index("task4")
    assert order.index("task3") < order.index("task4")
    assert order.index("task3") < order.index("task5")
    assert order.index("task4") < order.index("task6")
    assert order.index("task5") < order.index("task6")


def test_dag_get_execution_order_with_cycle():
    """Test that execution order raises error for cyclic DAG."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1", depends_on=["task2"])
    task2 = Task(func=func, name="task2", depends_on=["task1"])

    executor = DAGExecutor([task1, task2])

    with pytest.raises(DAGValidationError, match="cycle"):
        executor.get_execution_order()


def test_dag_execute_simple(tmp_path):
    """Test executing a simple DAG."""
    results = []

    def func1(config, run_dir, logger):
        results.append("task1")
        return TaskResult(data="result1")

    def func2(config, run_dir, logger, task1):
        results.append("task2")
        return TaskResult(data=f"result2_after_{task1.data}")

    task1 = Task(func=func1, name="task1")
    task2 = Task(func=func2, name="task2", depends_on=["task1"])

    executor = DAGExecutor([task1, task2])

    logger = logging.getLogger("test")
    task_results = executor.execute(config={}, run_dir=tmp_path, logger=logger)

    # Check execution order
    assert results == ["task1", "task2"]

    # Check results
    assert "task1" in task_results
    assert "task2" in task_results
    assert task_results["task1"].data == "result1"
    assert task_results["task2"].data == "result2_after_result1"


def test_dag_execute_parallel_branches(tmp_path):
    """Test executing DAG with parallel branches."""
    execution_log = []

    def func1(config, run_dir, logger):
        execution_log.append("task1")
        return TaskResult(data=1)

    def func2(config, run_dir, logger, task1):
        execution_log.append("task2")
        return TaskResult(data=task1.data + 1)

    def func3(config, run_dir, logger, task1):
        execution_log.append("task3")
        return TaskResult(data=task1.data + 2)

    def func4(config, run_dir, logger, task2, task3):
        execution_log.append("task4")
        return TaskResult(data=task2.data + task3.data)

    task1 = Task(func=func1, name="task1")
    task2 = Task(func=func2, name="task2", depends_on=["task1"])
    task3 = Task(func=func3, name="task3", depends_on=["task1"])
    task4 = Task(func=func4, name="task4", depends_on=["task2", "task3"])

    executor = DAGExecutor([task1, task2, task3, task4])

    logger = logging.getLogger("test")
    results = executor.execute(config={}, run_dir=tmp_path, logger=logger)

    # Check that task1 executed first
    assert execution_log[0] == "task1"
    # Check that task4 executed last
    assert execution_log[-1] == "task4"
    # Check final result: (1+1) + (1+2) = 5
    assert results["task4"].data == 5


def test_dag_execute_propagates_errors(tmp_path):
    """Test that errors in tasks are propagated."""

    def func1(config, run_dir, logger):
        return TaskResult(data="ok")

    def func2(config, run_dir, logger, task1):
        raise RuntimeError("Task2 failed!")

    task1 = Task(func=func1, name="task1")
    task2 = Task(func=func2, name="task2", depends_on=["task1"])

    executor = DAGExecutor([task1, task2])

    logger = logging.getLogger("test")

    with pytest.raises(RuntimeError, match="Task2 failed!"):
        executor.execute(config={}, run_dir=tmp_path, logger=logger)


def test_dag_execute_validates_before_execution(tmp_path):
    """Test that DAG is validated before execution."""

    def func(config, run_dir, logger):
        pass

    task = Task(func=func, name="task1", depends_on=["nonexistent"])

    executor = DAGExecutor([task])

    logger = logging.getLogger("test")

    with pytest.raises(DAGValidationError, match="non-existent"):
        executor.execute(config={}, run_dir=tmp_path, logger=logger)


def test_dag_get_task_graph():
    """Test getting the task dependency graph."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1", "task2"])

    executor = DAGExecutor([task1, task2, task3])

    graph = executor.get_task_graph()

    assert graph == {
        "task1": [],
        "task2": ["task1"],
        "task3": ["task1", "task2"],
    }


def test_dag_get_reverse_graph():
    """Test getting the reverse dependency graph."""

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1"])
    task4 = Task(func=func, name="task4", depends_on=["task2", "task3"])

    executor = DAGExecutor([task1, task2, task3, task4])

    reverse_graph = executor.get_reverse_graph()

    assert reverse_graph == {
        "task1": ["task2", "task3"],
        "task2": ["task4"],
        "task3": ["task4"],
    }


def test_dag_no_tasks():
    """Test DAG executor with no tasks."""
    executor = DAGExecutor([])

    assert len(executor.tasks) == 0

    # Should validate successfully (empty DAG is valid)
    executor.validate()

    # Execution order should be empty
    order = executor.get_execution_order()
    assert order == []


def test_dag_single_task(tmp_path):
    """Test DAG executor with a single task."""

    def func(config, run_dir, logger):
        return TaskResult(data="single")

    task = Task(func=func, name="single")

    executor = DAGExecutor([task])
    executor.validate()

    order = executor.get_execution_order()
    assert order == ["single"]

    logger = logging.getLogger("test")
    results = executor.execute(config={}, run_dir=tmp_path, logger=logger)

    assert len(results) == 1
    assert results["single"].data == "single"


def test_dag_visualize(tmp_path):
    """Test DAG visualization."""
    pytest.importorskip("matplotlib")

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])
    task3 = Task(func=func, name="task3", depends_on=["task1"])

    executor = DAGExecutor([task1, task2, task3])

    output_path = tmp_path / "dag.png"
    executor.visualize(output_path)

    assert output_path.exists()
    # Check file is not empty
    assert output_path.stat().st_size > 0


def test_dag_visualize_complex(tmp_path):
    """Test visualization of complex DAG."""
    pytest.importorskip("matplotlib")

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="load")
    task2 = Task(func=func, name="clean", depends_on=["load"])
    task3 = Task(func=func, name="analyze", depends_on=["clean"])
    task4 = Task(func=func, name="plot", depends_on=["analyze"])

    executor = DAGExecutor([task1, task2, task3, task4])

    output_path = tmp_path / "complex_dag.png"
    executor.visualize(output_path, figsize=(10, 6))

    assert output_path.exists()


def test_dag_visualize_custom_style(tmp_path):
    """Test DAG visualization with custom styling."""
    pytest.importorskip("matplotlib")

    def func(config, run_dir, logger):
        pass

    task1 = Task(func=func, name="task1")
    task2 = Task(func=func, name="task2", depends_on=["task1"])

    executor = DAGExecutor([task1, task2])

    output_path = tmp_path / "styled_dag.png"
    executor.visualize(
        output_path,
        node_color="lightgreen",
        node_size=2000,
        font_size=12,
    )

    assert output_path.exists()


def test_dag_visualize_no_matplotlib(tmp_path):
    """Test that visualization fails gracefully without matplotlib."""

    def func(config, run_dir, logger):
        pass

    task = Task(func=func, name="task1")
    executor = DAGExecutor([task])

    # Mock matplotlib import failure
    import sys

    matplotlib_backup = sys.modules.get("matplotlib")
    pyplot_backup = sys.modules.get("matplotlib.pyplot")

    try:
        # Simulate matplotlib not being installed
        sys.modules["matplotlib"] = None
        sys.modules["matplotlib.pyplot"] = None

        # Force reimport to trigger the ImportError
        import importlib

        import sleap_roots_analyze.pipeline.dag as dag_module

        importlib.reload(dag_module)

        with pytest.raises(ImportError, match="matplotlib is required"):
            executor.visualize(tmp_path / "output.png")

    finally:
        # Restore matplotlib modules
        if matplotlib_backup is not None:
            sys.modules["matplotlib"] = matplotlib_backup
        elif "matplotlib" in sys.modules:
            del sys.modules["matplotlib"]

        if pyplot_backup is not None:
            sys.modules["matplotlib.pyplot"] = pyplot_backup
        elif "matplotlib.pyplot" in sys.modules:
            del sys.modules["matplotlib.pyplot"]

        # Reload the module to restore original state
        import sleap_roots_analyze.pipeline.dag as dag_module

        importlib.reload(dag_module)
