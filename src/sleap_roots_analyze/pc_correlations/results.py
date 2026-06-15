"""Typed result objects for the cross-platform PC-correlation workflow.

Mirrors the typed-result pattern used by :class:`EnrichmentResult` (and the
broader serializable-result-types effort, issues #127-#130): the orchestrator
returns a structured object rather than a bare ``dict`` so consumers — notably
the bloom-mcp reproduction tool — get discoverable, typed fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CrossPlatformPCResult:
    """Bundled result of :func:`cross_platform_pc_correlations`.

    Attributes:
        correlations: Tidy table with one row per cross-platform PC test
            (``platform1``/``platform2``/``platform1_pc``/``platform2_pc``,
            ``spearman_r``/``spearman_p``, CI/power columns, and the
            ``p_adj_*``/``significant_*`` FDR columns for each requested scope).
        summary: Headline numbers (``n_tests``, ``n_genotypes``,
            ``tests_per_pair``, ``significant_counts``,
            ``n_significant_combined``, ``n_significant_per_pair``,
            ``min_detectable_r_80_power``, ``mean_achieved_power``).
        genotype_means: Mapping of platform name to its genotype-indexed,
            common-panel-aligned mean PC scores.
        common_genotypes: Genotypes present in every platform.
        output_paths: Mapping of artifact name to the path written on disk.
    """

    correlations: pd.DataFrame
    summary: dict[str, Any]
    genotype_means: dict[str, pd.DataFrame]
    common_genotypes: list[str]
    output_paths: dict[str, str] = field(default_factory=dict)
