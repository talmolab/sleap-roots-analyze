"""Core abstractions for pipeline steps.

This module provides the base classes and data structures for building
modular, testable pipeline steps. These are higher-level abstractions
built on top of the general DAG Task/TaskResult framework.

All pipeline steps (both QC and Viz) extend BaseStep and return StepResult.

Relationship with Task/TaskResult:
    - Task/TaskResult: General DAG framework (in task.py) - wraps any callable
    - BaseStep/StepResult: Pipeline-specific abstractions - domain logic for data processing
    - All pipeline steps (QC and Viz) extend BaseStep and return StepResult
    - These steps are wrapped in Tasks when building the pipeline DAG

Example:
    >>> class LoadDataStep(BaseStep):
    ...     def execute(self, data, config, run_dir, prev_result=None):
    ...         df = pd.read_csv(config.data.csv_path)
    ...         return StepResult(data=df, metadata={"rows": len(df)})
    >>>
    >>> # Wrap in a Task for DAG execution:
    >>> load_task = Task(
    ...     func=load_step.execute,
    ...     name="load_data",
    ...     description="Load CSV data"
    ... )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class StepResult:
    """Result from executing a pipeline step.

    Attributes:
        data: The primary output data (usually a DataFrame).
        metadata: Additional information about the step execution.
        files_generated: List of files created by this step.
    """

    data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    files_generated: List[Path] = field(default_factory=list)


class BaseStep(ABC):
    """Abstract base class for all pipeline steps.

    Each step should:
    1. Execute a specific data processing operation
    2. Save outputs to the run directory
    3. Return a StepResult with data and metadata
    """

    def __init__(self, step_name: str = "", description: str = ""):
        """Initialize the step.

        Args:
            step_name: Name of the step (auto-generated from class name if empty).
            description: Human-readable description of what this step does.
        """
        self.step_name = step_name or self.__class__.__name__
        self.description = description

    @abstractmethod
    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the step.

        Args:
            data: Input data (typically a DataFrame from previous step).
            config: Pipeline configuration object.
            run_dir: Directory to save output files.
            prev_result: Result from the previous step (contains metadata).

        Returns:
            StepResult with processed data, metadata, and files generated.
        """
        pass

    def save_dataframe(self, df: pd.DataFrame, filename: str, run_dir: Path) -> Path:
        """Save a DataFrame to CSV in the run directory.

        Args:
            df: DataFrame to save.
            filename: Name of the file (should include .csv extension).
            run_dir: Directory to save to.

        Returns:
            Path to the saved file.
        """
        output_path = run_dir / filename
        df.to_csv(output_path, index=False)
        return output_path

    def save_json(self, data: dict, filename: str, run_dir: Path) -> Path:
        """Save a dictionary to JSON in the run directory.

        Args:
            data: Dictionary to save.
            filename: Name of the file (should include .json extension).
            run_dir: Directory to save to.

        Returns:
            Path to the saved file.
        """
        import json

        from sleap_roots_analyze.data_utils import path_to_posix

        output_path = run_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            # path_to_posix normalizes any PurePath to forward-slash on every OS so
            # producers can store bare Path; everything else keeps the prior str()
            # fallback. Shared with convert_to_json_serializable so both serializer
            # sinks use one path predicate (#157).
            json.dump(data, f, indent=2, default=path_to_posix)
        return output_path

    def reorder_dataframe_columns(
        self, df: pd.DataFrame, trait_cols: list[str]
    ) -> pd.DataFrame:
        """Reorder DataFrame columns: metadata first, then traits (sorted).

        This ensures consistent column ordering across all pipeline outputs:
        - Metadata columns (Plot, Rep, geno, Barcode, etc.) come first
        - Trait columns (measurements) come after, sorted alphabetically

        Args:
            df: DataFrame to reorder.
            trait_cols: List of trait column names.

        Returns:
            DataFrame with reordered columns.
        """
        metadata_cols = [col for col in df.columns if col not in trait_cols]
        return df[metadata_cols + sorted(trait_cols)]
