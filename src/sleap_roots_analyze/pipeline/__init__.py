"""Pipeline infrastructure for sleap-roots-analyze.

This module provides a lightweight DAG-based pipeline framework for executing
data analysis workflows with explicit dependency management.
"""

from sleap_roots_analyze.pipeline.config import (
    CleanupConfig,
    ClusteringConfig,
    ColumnConfig,
    CoreQCConfig,
    CrossPlatformConfig,
    DataConfig,
    HeritabilityConfig,
    LoggingConfig,
    MergeTraitsConfig,
    OutlierDetectionConfig,
    PCAConfig,
    QCPipelineConfig,
    RootCoreConfig,
    RootCoreSourceConfig,
    VisualizationConfig,
    VizPipelineConfig,
    get_default_qc_config,
    get_default_viz_config,
    load_cross_platform_config,
    load_qc_config,
    load_viz_config,
    merge_qc_configs,
    save_qc_config,
    save_viz_config,
    validate_qc_config,
    validate_viz_config,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.pipeline.dag import DAGExecutor, DAGValidationError
from sleap_roots_analyze.pipeline.pipelines import (
    BasePipeline,
    CrossPlatformPipeline,
    QCPipeline,
    VizPipeline,
)
from sleap_roots_analyze.pipeline.summary import PipelineSummary, StepSummary
from sleap_roots_analyze.pipeline.task import Task, TaskResult
from sleap_roots_analyze.pipeline.utils import (
    create_code_archive,
    create_run_directory,
    get_code_snapshot,
    get_environment_info,
    get_git_branch,
    get_git_commit_hash,
    get_git_remote_url,
    get_package_version,
    get_package_versions,
    is_git_dirty,
)

__all__ = [
    # Core DAG components
    "Task",
    "TaskResult",
    "DAGExecutor",
    "DAGValidationError",
    # Pipeline and summary
    "BasePipeline",
    "CrossPlatformPipeline",
    "QCPipeline",
    "VizPipeline",
    "PipelineSummary",
    "StepSummary",
    # Core step abstractions
    "BaseStep",
    "StepResult",
    # Configuration Components
    "ColumnConfig",
    "DataConfig",
    "CleanupConfig",
    "OutlierDetectionConfig",
    "HeritabilityConfig",
    "PCAConfig",
    "ClusteringConfig",
    "VisualizationConfig",
    "LoggingConfig",
    # Root Core Configuration
    "RootCoreSourceConfig",
    "CoreQCConfig",
    "MergeTraitsConfig",
    "RootCoreConfig",
    # QC Pipeline Configuration
    "QCPipelineConfig",
    "load_qc_config",
    "save_qc_config",
    "get_default_qc_config",
    "validate_qc_config",
    "merge_qc_configs",
    # Viz Pipeline Configuration
    "VizPipelineConfig",
    "load_viz_config",
    "save_viz_config",
    "get_default_viz_config",
    "validate_viz_config",
    # Cross-Platform Configuration
    "CrossPlatformConfig",
    "load_cross_platform_config",
    # Utilities
    "create_run_directory",
    "get_git_commit_hash",
    "get_git_branch",
    "get_git_remote_url",
    "is_git_dirty",
    "get_package_version",
    "get_package_versions",
    "get_environment_info",
    "create_code_archive",
    "get_code_snapshot",
]
