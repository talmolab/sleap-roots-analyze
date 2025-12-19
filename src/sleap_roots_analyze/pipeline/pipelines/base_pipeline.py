"""Base pipeline class using DAG execution.

This module provides the abstract BasePipeline class that uses the DAG executor
for running analysis pipelines.
"""

from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from omegaconf import DictConfig, OmegaConf

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

        # Create run directory first (needed for file logging)
        self.run_dir = self._create_run_directory()

        # Set up logging (after run_dir exists for file handler)
        self.logger = self._setup_logger()

        # Initialize summary
        self.summary = PipelineSummary(
            pipeline_name=pipeline_name,
            version=version,
            output_directory=str(self.run_dir),
        )

    def _setup_logger(self) -> logging.Logger:
        """Set up logger for the pipeline.

        Creates both a StreamHandler for console output and a FileHandler
        for per-run logging to {run_dir}/pipeline.log.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(f"{self.pipeline_name}")
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Add StreamHandler for console output if not already present
        # Note: FileHandler inherits from StreamHandler, so we explicitly exclude it
        # to avoid counting file handlers as console handlers. This check prevents
        # duplicate console output in interactive sessions (notebooks, REPL).
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        )
        if not has_stream_handler:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

        # Add FileHandler for per-run log file
        # Each run gets its own log file in run_dir
        log_file_path = self.run_dir / "pipeline.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Store reference to close handler later if needed
        self._file_handler = file_handler

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

    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to a JSON-serializable dictionary.

        Handles various config types: dict, dataclass, OmegaConf DictConfig.

        Returns:
            Dictionary representation of the config.
        """
        config = self.config

        # Handle OmegaConf DictConfig
        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)

        # Handle dataclass
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            return self._dataclass_to_dict(config)

        # Handle plain dict
        if isinstance(config, dict):
            return self._serialize_dict(config)

        # Try to convert using OmegaConf (works for structured configs)
        try:
            omega_conf = OmegaConf.structured(config)
            return OmegaConf.to_container(omega_conf, resolve=True)
        except Exception:
            # Last resort: try to get __dict__
            if hasattr(config, "__dict__"):
                return self._serialize_dict(vars(config))
            return {"_config_type": str(type(config))}

    def _dataclass_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Recursively convert a dataclass to a dict with Path serialization.

        Args:
            obj: Dataclass object to convert.

        Returns:
            Dictionary with all values serialized.
        """
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = self._serialize_value(value)
        return result

    def _serialize_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize a dictionary, converting Path objects to strings.

        Args:
            d: Dictionary to serialize.

        Returns:
            Serialized dictionary.
        """
        return {k: self._serialize_value(v) for k, v in d.items()}

    def _serialize_value(self, value: Any) -> Any:
        """Serialize a single value, handling Path and nested structures.

        Args:
            value: Value to serialize.

        Returns:
            Serialized value.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return self._serialize_dict(value)
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return self._dataclass_to_dict(value)
        return value

    def _save_config(self) -> Path:
        """Save the resolved configuration to the run directory.

        Saves the config as config.yaml in the run directory for reproducibility.

        Returns:
            Path to the saved config file.
        """
        config_path = self.run_dir / "config.yaml"

        # Convert config to OmegaConf for saving
        config_dict = self._config_to_dict()
        omega_conf = OmegaConf.create(config_dict)
        OmegaConf.save(omega_conf, config_path)

        self.logger.info(f"Config saved to {config_path}")
        return config_path

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

        # Save config for provenance (do this early so it's saved even on failure)
        self._save_config()

        # Populate summary.config for JSON summary
        self.summary.config = self._config_to_dict()

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

            # Close file handler to release file lock
            self._close_file_handler()

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self.summary.finalize(status="failed")

            # Save summary even on failure
            summary_path = self.run_dir / "pipeline_summary.json"
            self.summary.save(summary_path)

            # Close file handler to release file lock
            self._close_file_handler()

            raise

    def _close_file_handler(self) -> None:
        """Close the per-run file handler to release file lock.

        This is called after pipeline completion to allow file cleanup
        (especially important on Windows where open files can't be deleted).
        """
        if hasattr(self, "_file_handler") and self._file_handler:
            self._file_handler.close()
            self.logger.removeHandler(self._file_handler)
            self._file_handler = None

    def get_summary(self) -> PipelineSummary:
        """Get the current pipeline summary.

        Returns:
            The PipelineSummary object for this run.
        """
        return self.summary
