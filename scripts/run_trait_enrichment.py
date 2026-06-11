#!/usr/bin/env python
"""Reproduce the wheat EDPIE trait-level cross-platform enrichment analysis.

Thin CLI wrapper around the public
:func:`sleap_roots_analyze.trait_correlation_enrichment`; all logic lives in the
library. Correlation files are supplied at the command line — nothing is
hard-coded.

Usage:
    uv run scripts/run_trait_enrichment.py \
        --pair "Turface vs Cylinder=/path/to/cross_platform_correlations.csv" \
        --pair "Field vs Cylinder=/path/to/cross_platform_correlations.csv" \
        --output-dir ./outputs/trait_enrichment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless figure rendering

from sleap_roots_analyze import trait_correlation_enrichment


def _parse_pair(spec: str) -> tuple[str, Path]:
    """Parse a ``"Label=path"`` argument into ``(label, Path)``."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--pair must be 'Label=path', got {spec!r}")
    label, path = spec.split("=", 1)
    return label.strip(), Path(path.strip())


def main() -> None:
    """Parse arguments and run the trait-enrichment workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A platform-pair label and its cross_platform_correlations.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory."
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--p-value-column", default="spearman_p")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    correlation_files = dict(args.pair)
    result = trait_correlation_enrichment(
        correlation_files=correlation_files,
        output_dir=args.output_dir,
        alpha=args.alpha,
        p_value_column=args.p_value_column,
        make_figure=not args.no_figure,
    )
    combined = result["results"][0]
    print(
        f"Combined: {combined.n_significant}/{combined.n_tests} significant "
        f"({combined.fold_enrichment:.2f}x, {combined.interpretation})"
    )
    print(f"Artifacts written under: {args.output_dir}")


if __name__ == "__main__":
    main()
