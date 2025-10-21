"""Pipeline infrastructure for sleap-roots-analyze.

This module provides a lightweight DAG-based pipeline framework for executing
data analysis workflows with explicit dependency management.
"""

from sleap_roots_analyze.pipeline.config import (
    ClusteringConfig,
    DataConfig,
    LoggingConfig,
    OutlierDetectionConfig,
    PCAConfig,
    PipelineConfig,
    VisualizationConfig,
    get_default_config,
    load_config,
    merge_configs,
    save_config,
    validate_config,
)
from sleap_roots_analyze.pipeline.dag import DAGExecutor, DAGValidationError
from sleap_roots_analyze.pipeline.pipeline import BasePipeline
from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary
from sleap_roots_analyze.pipeline.task import Task, TaskResult
from sleap_roots_analyze.pipeline.utils import (
    create_run_directory,
    get_environment_info,
    get_git_branch,
    get_git_commit_hash,
    get_package_version,
    get_package_versions,
)

__all__ = [
    # Core DAG components
    "Task",
    "TaskResult",
    "DAGExecutor",
    "DAGValidationError",
    # Pipeline and summary
    "BasePipeline",
    "PipelineSummary",
    "StepSummary",
    # Configuration
    "PipelineConfig",
    "DataConfig",
    "OutlierDetectionConfig",
    "PCAConfig",
    "ClusteringConfig",
    "VisualizationConfig",
    "LoggingConfig",
    "load_config",
    "save_config",
    "get_default_config",
    "merge_configs",
    "validate_config",
    # Utilities
    "create_run_directory",
    "get_git_commit_hash",
    "get_git_branch",
    "get_package_version",
    "get_package_versions",
    "get_environment_info",
]
