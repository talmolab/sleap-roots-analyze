"""Base pipeline class using DAG execution.

This module provides the abstract BasePipeline class that uses the DAG executor
for running analysis pipelines.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sleap_roots_analyze.pipeline.dag import DAGExecutor
from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary
from sleap_roots_analyze.pipeline.task import Task, TaskResult


class BasePipeline(ABC):
    """Abstract base class for DAG-based analysis pipelines.

    Subclasses must implement the `create_tasks()` method to define the
    pipeline's tasks and their dependencies.

    Args:
        config: Configuration object for the pipeline.
        output_dir: Directory for pipeline outputs.
        pipeline_name: Name of the pipeline.
        version: Pipeline version.

    Example:
        >>> class MyPipeline(BasePipeline):
        ...     def create_tasks(self) -> List[Task]:
        ...         load_task = Task(
        ...             func=self.load_data,
        ...             name="load_data",
        ...             description="Load input data"
        ...         )
        ...         process_task = Task(
        ...             func=self.process_data,
        ...             name="process_data",
        ...             depends_on=["load_data"],
        ...             description="Process loaded data"
        ...         )
        ...         return [load_task, process_task]
        ...
        ...     def load_data(self, config, run_dir, logger):
        ...         return TaskResult(data=pd.read_csv(config.data_path))
        ...
        ...     def process_data(self, config, run_dir, logger, load_data):
        ...         df = load_data.data
        ...         return TaskResult(data=df.dropna())
        >>>
        >>> pipeline = MyPipeline(config, output_dir="./outputs")
        >>> results = pipeline.run()
    """

    def __init__(
        self,
        config: Any,
        output_dir: str | Path,
        pipeline_name: str = "pipeline",
        version: str = "1.0",
    ):
        """Initialize the pipeline.

        Args:
            config: Configuration object.
            output_dir: Directory for pipeline outputs.
            pipeline_name: Name of the pipeline.
            version: Pipeline version.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.pipeline_name = pipeline_name
        self.version = version

        # Set up logging
        self.logger = self._setup_logger()

        # Create run directory with timestamp
        self.run_dir = self._create_run_directory()

        # Initialize summary
        self.summary = PipelineSummary(
            pipeline_name=pipeline_name,
            version=version,
            output_directory=str(self.run_dir),
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the pipeline.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(f"{self.pipeline_name}")
        logger.setLevel(logging.INFO)

        # Only add handler if logger doesn't already have one
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _create_run_directory(self) -> Path:
        """Create a timestamped run directory.

        Returns:
            Path to the created run directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / f"{self.pipeline_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @abstractmethod
    def create_tasks(self) -> List[Task]:
        """Create and return the list of tasks for this pipeline.

        This method must be implemented by subclasses to define the pipeline's
        tasks and their dependencies.

        Returns:
            List of Task objects with their dependencies specified.
        """
        pass

    def run(self) -> Dict[str, TaskResult]:
        """Execute the pipeline DAG.

        Returns:
            Dictionary mapping task names to their TaskResults.

        Raises:
            Exception: Any exception raised during task execution is propagated
                after being logged and recorded in the summary.
        """
        self.logger.info(f"Starting pipeline: {self.pipeline_name}")
        self.summary.start_time = datetime.now().isoformat()

        try:
            # Create tasks
            tasks = self.create_tasks()
            self.logger.info(f"Created {len(tasks)} tasks")

            # Initialize step summaries
            self.summary.steps = [
                StepSummary(name=task.name, description=task.description)
                for task in tasks
            ]

            # Create and execute DAG
            executor = DAGExecutor(tasks)

            # Log the execution order
            execution_order = executor.get_execution_order()
            self.logger.info(f"Execution order: {', '.join(execution_order)}")

            # Execute
            results = executor.execute(
                config=self.config,
                run_dir=self.run_dir,
                logger=self.logger,
            )

            # Update summary with results
            for task_name, task_result in results.items():
                self.summary.mark_step_success(
                    step_name=task_name,
                    elapsed_time=task_result.metadata.get("elapsed_time", 0.0),
                    files_generated=task_result.files_generated,
                    metadata=task_result.metadata,
                )

            # Finalize summary
            self.summary.finalize(status="success")
            self.logger.info("Pipeline completed successfully")

            # Save summary
            summary_path = self.run_dir / "pipeline_summary.json"
            self.summary.save(summary_path)
            self.logger.info(f"Summary saved to {summary_path}")

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.summary.finalize(status="failed")

            # Save summary even on failure
            summary_path = self.run_dir / "pipeline_summary.json"
            self.summary.save(summary_path)

            raise

    def get_summary(self) -> PipelineSummary:
        """Get the current pipeline summary.

        Returns:
            The PipelineSummary object for this run.
        """
        return self.summary
