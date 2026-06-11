#!/usr/bin/env python
"""Reproduce the wheat EDPIE PC-level cross-platform correlation analysis.

Thin CLI wrapper around the public
:func:`sleap_roots_analyze.cross_platform_pc_correlations`; all logic lives in
the library. Paths are supplied at the command line — nothing is hard-coded.

Usage:
    uv run scripts/run_pc_correlations.py \
        --pipeline-run /path/to/pipeline_run \
        --output-dir ./outputs/pc_correlations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless figure rendering

from sleap_roots_analyze import cross_platform_pc_correlations
from sleap_roots_analyze.pc_correlations.aggregate import WHEAT_EDPIE_PLATFORMS


def main() -> None:
    """Parse arguments and run the PC-correlation workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-run", type=Path, required=True, help="Pipeline-run directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory."
    )
    parser.add_argument("--primary-fdr-method", default="fdr_by")
    parser.add_argument("--fdr-scope", default="both")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    result = cross_platform_pc_correlations(
        pipeline_run=args.pipeline_run,
        platform_config=WHEAT_EDPIE_PLATFORMS,
        output_dir=args.output_dir,
        primary_fdr_method=args.primary_fdr_method,
        alpha=args.alpha,
        fdr_scope=args.fdr_scope,
        make_figures=not args.no_figures,
    )
    summary = result["summary"]
    print(
        f"PC tests: {summary['n_tests']} | genotypes: {summary['n_genotypes']} | "
        f"combined-{args.primary_fdr_method} significant: "
        f"{summary['n_significant_combined']}"
    )
    print(f"Artifacts written under: {args.output_dir}")


if __name__ == "__main__":
    main()
