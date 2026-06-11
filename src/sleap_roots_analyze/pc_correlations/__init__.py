"""Cross-platform PC-correlation and trait-enrichment workflows (issue #119).

This subpackage ports the wheat EDPIE paper's two cross-platform analyses into
the public API as reusable, DAG-structured workflows:

- :func:`cross_platform_pc_correlations` — PC-level cross-platform correlation
  workflow (load per-platform PCA outputs + QC genotype labels, aggregate
  sample PC scores to genotype means, align genotypes, correlate every PC
  against every PC across platform pairs, FDR-correct at combined and per-pair
  scopes, and export tidy CSVs, figures, and metadata).
- :func:`trait_correlation_enrichment` — trait-level binomial enrichment
  workflow over existing ``cross_platform_correlations.csv`` files. This is an
  *independent* analysis; it is not downstream of the PC workflow.

The two orchestrators compose the DAG nodes in :mod:`aggregate`,
:mod:`correlate`, :mod:`enrichment`, and :mod:`visualize`, reusing the shared
confidence-interval and power helpers from
:mod:`sleap_roots_analyze.cross_experiment_analysis`.
"""

from __future__ import annotations

from sleap_roots_analyze.pc_correlations.enrichment import EnrichmentResult
from sleap_roots_analyze.pc_correlations.results import CrossPlatformPCResult
from sleap_roots_analyze.pc_correlations.workflow import (
    cross_platform_pc_correlations,
    trait_correlation_enrichment,
)

__all__ = [
    "cross_platform_pc_correlations",
    "trait_correlation_enrichment",
    "CrossPlatformPCResult",
    "EnrichmentResult",
]
