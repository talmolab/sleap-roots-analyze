"""Visualization pipeline for comprehensive trait visualization.

This module provides a 12-step visualization pipeline for generating
publication-quality plots, interactive visualizations, and dashboards from
trait data.
"""

from __future__ import annotations

from sleap_roots_analyze.viz_pipeline.config import (
    VizPipelineConfig,
    load_viz_config,
    save_viz_config,
    validate_viz_config,
)

# VizPipeline will be added in Phase 2
# from sleap_roots_analyze.viz_pipeline.viz_pipeline import VizPipeline

__all__ = [
    # "VizPipeline",  # Phase 2
    "VizPipelineConfig",
    "load_viz_config",
    "save_viz_config",
    "validate_viz_config",
]
