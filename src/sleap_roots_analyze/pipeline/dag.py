"""DAG executor for pipeline tasks using NetworkX.

This module provides a DAG (Directed Acyclic Graph) executor that validates
task dependencies and executes tasks in topological order. Uses NetworkX for
graph management, aligning with SLEAP Roots' trait pipeline approach.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx

from sleap_roots_analyze.pipeline.task import Task, TaskResult


class DAGValidationError(Exception):
    """Exception raised when DAG validation fails."""

    pass


class DAGExecutor:
    """DAG executor for pipeline tasks using NetworkX.

    This executor validates the DAG structure (no cycles, all dependencies exist),
    determines the execution order using topological sort, and executes tasks
    sequentially in dependency order.

    Uses NetworkX for graph management, consistent with SLEAP Roots' approach.

    Args:
        tasks: List of Task objects to execute.

    Example:
        >>> load_task = Task(func=load_data, name="load", depends_on=[])
        >>> clean_task = Task(func=clean_data, name="clean", depends_on=["load"])
        >>> executor = DAGExecutor([load_task, clean_task])
        >>> executor.validate()
        >>> results = executor.execute(config, run_dir, logger)
    """

    def __init__(self, tasks: List[Task]):
        """Initialize the DAG executor.

        Args:
            tasks: List of Task objects to manage and execute.
        """
        self.tasks = {task.name: task for task in tasks}
        self._validate_unique_names(tasks)

        # Build NetworkX directed graph
        self.graph = nx.DiGraph()
        for task in tasks:
            self.graph.add_node(task.name)
            for dep in task.depends_on:
                # Add edge from dependency to task (dep -> task)
                self.graph.add_edge(dep, task.name)

    def _validate_unique_names(self, tasks: List[Task]) -> None:
        """Validate that all task names are unique.

        Args:
            tasks: List of tasks to validate.

        Raises:
            DAGValidationError: If duplicate task names are found.
        """
        names = [task.name for task in tasks]
        duplicates = [name for name in names if names.count(name) > 1]
        if duplicates:
            raise DAGValidationError(f"Duplicate task names found: {set(duplicates)}")

    def validate(self) -> None:
        """Validate the DAG structure.

        Checks:
        1. All dependencies reference existing tasks
        2. No cycles exist in the dependency graph

        Raises:
            DAGValidationError: If validation fails.
        """
        # Check that all dependencies exist
        for task_name, task in self.tasks.items():
            for dep in task.depends_on:
                if dep not in self.tasks:
                    raise DAGValidationError(
                        f"Task '{task_name}' depends on non-existent task '{dep}'"
                    )

        # Check for cycles using NetworkX
        if not nx.is_directed_acyclic_graph(self.graph):
            raise DAGValidationError(
                "Cycle detected in task dependencies involving "
                f"{list(nx.simple_cycles(self.graph))[0]}"
            )

    def get_execution_order(self) -> List[str]:
        """Get task names in topological order.

        Uses NetworkX's topological sort algorithm.

        Returns:
            List of task names in the order they should be executed.

        Raises:
            DAGValidationError: If the DAG contains cycles.
        """
        try:
            return list(nx.topological_sort(self.graph))
        except (nx.NetworkXError, nx.NetworkXUnfeasible):
            raise DAGValidationError(
                "Cannot determine execution order - cycle detected in DAG"
            )

    def execute(self, config: Any, run_dir: Path, logger: Any) -> Dict[str, TaskResult]:
        """Execute all tasks in dependency order.

        Args:
            config: Configuration object to pass to each task.
            run_dir: Directory for pipeline run outputs.
            logger: Logger instance for recording execution.

        Returns:
            Dictionary mapping task names to their TaskResults.

        Raises:
            DAGValidationError: If the DAG is invalid.
            Exception: Any exception raised by task execution is propagated.
        """
        # Validate DAG structure
        self.validate()

        # Get execution order
        execution_order = self.get_execution_order()

        logger.info(
            f"Executing {len(execution_order)} tasks in order: "
            f"{', '.join(execution_order)}"
        )

        # Execute tasks in order
        results: Dict[str, TaskResult] = {}

        for task_name in execution_order:
            task = self.tasks[task_name]

            # Collect results from dependencies
            dependency_results = {
                dep: results[dep] for dep in task.depends_on if dep in results
            }

            # Execute the task
            result = task.execute(config, run_dir, logger, dependency_results)
            results[task_name] = result

        logger.info(f"Successfully executed all {len(execution_order)} tasks")
        return results

    def get_task_graph(self) -> Dict[str, List[str]]:
        """Get the dependency graph as an adjacency list.

        Returns:
            Dictionary mapping task names to lists of tasks they depend on.
        """
        return {name: task.depends_on for name, task in self.tasks.items()}

    def get_reverse_graph(self) -> Dict[str, List[str]]:
        """Get the reverse dependency graph (tasks that depend on each task).

        Returns:
            Dictionary mapping task names to lists of tasks that depend on them.
        """
        reverse_graph: Dict[str, List[str]] = defaultdict(list)
        for task_name, task in self.tasks.items():
            for dep in task.depends_on:
                reverse_graph[dep].append(task_name)
        return dict(reverse_graph)

    def visualize(
        self,
        output_path: str | Path,
        figsize: tuple[int, int] = (12, 8),
        **kwargs,
    ) -> None:
        """Visualize the DAG structure.

        Requires matplotlib to be installed.

        Args:
            output_path: Path to save the visualization (PNG, PDF, SVG, etc.).
            figsize: Figure size as (width, height) in inches.
            **kwargs: Additional arguments passed to networkx.draw().

        Raises:
            ImportError: If matplotlib is not installed.

        Example:
            >>> executor = DAGExecutor(tasks)
            >>> executor.visualize("pipeline_dag.png")
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        output_path = Path(output_path)

        plt.figure(figsize=figsize)

        # Use hierarchical layout for DAGs
        # Try to use graphviz layout if available, otherwise spring layout
        try:
            pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
        except (ImportError, AttributeError):
            # Fallback to spring layout
            pos = nx.spring_layout(self.graph, k=2, iterations=50)

        # Draw the graph
        nx.draw(
            self.graph,
            pos,
            with_labels=True,
            node_color=kwargs.pop("node_color", "lightblue"),
            node_size=kwargs.pop("node_size", 3000),
            font_size=kwargs.pop("font_size", 10),
            font_weight=kwargs.pop("font_weight", "bold"),
            arrows=True,
            arrowsize=kwargs.pop("arrowsize", 20),
            edge_color=kwargs.pop("edge_color", "gray"),
            **kwargs,
        )

        plt.title("Pipeline DAG Structure", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save the figure
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
