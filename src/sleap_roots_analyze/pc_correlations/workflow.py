"""Public orchestrators for the two cross-platform analysis workflows.

- :func:`cross_platform_pc_correlations` runs the PC-level DAG from a pipeline
  run directory and writes correlations, figures, and metadata.
- :func:`trait_correlation_enrichment` runs the independent trait-level binomial
  enrichment DAG over existing ``cross_platform_correlations.csv`` files.

Both return an in-memory ``dict`` (alongside the written artifacts) so they are
testable without reading files back.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sleap_roots_analyze.pc_correlations.aggregate import (
    align_genotypes_across_platforms,
    load_platform_data,
)
from sleap_roots_analyze.pc_correlations.correlate import (
    DEFAULT_FDR_METHODS,
    calculate_all_platform_correlations,
    summarize_correlations,
)
from sleap_roots_analyze.pc_correlations.enrichment import (
    calculate_all_enrichment_tests,
    create_enrichment_figure,
    results_to_dataframe,
)
from sleap_roots_analyze.pc_correlations.visualize import (
    create_correlation_heatmap,
    create_correlation_summary_figure,
    create_sensitivity_analysis_figure,
    save_figure,
)


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON-serialisable types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    return obj


def _significant_columns(fdr_scope: str, primary_method: str) -> dict[str, str]:
    """Return the significance column names for each requested scope."""
    if fdr_scope == "both":
        return {
            "combined": f"significant_combined_{primary_method}",
            "per_pair": f"significant_per_pair_{primary_method}",
        }
    return {fdr_scope: f"significant_{primary_method}"}


def cross_platform_pc_correlations(
    pipeline_run: Path,
    platform_config: dict[str, dict[str, Any]],
    output_dir: Path,
    fdr_methods: list[str] | None = None,
    primary_fdr_method: str = "fdr_by",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    fdr_scope: str = "both",
    make_figures: bool = True,
) -> dict[str, Any]:
    """Run the PC-level cross-platform correlation workflow in one call.

    Loads per-platform sample PC scores and QC genotype labels from a pipeline
    run, aggregates sample PC scores to genotype means, aligns platforms to their
    common genotypes, correlates every PC against every PC across platform pairs,
    applies FDR correction at the requested scope(s), and writes tidy CSVs,
    figures, and a ``metadata.json``.

    Args:
        pipeline_run: Path to the pipeline-run directory.
        platform_config: Mapping of platform name to ``{pca_dir, final_data,
            genotype_col, n_pcs}``.
        output_dir: Directory for the written artifacts (created if missing).
        fdr_methods: Multiple-testing methods. Defaults to BY/BH/Bonferroni.
        primary_fdr_method: Method used for the headline significance counts and
            the ``significant_*`` export files.
        alpha: Significance level for FDR, power, and CIs.
        confidence_level: Confidence level for the correlation intervals.
        fdr_scope: ``"combined"``, ``"per_pair"``, or ``"both"`` (default).
        make_figures: Whether to render correlation and sensitivity figures.

    Returns:
        Dict with keys ``correlations`` (DataFrame), ``summary`` (dict),
        ``genotype_means`` (mapping of platform to aligned genotype-mean PCs),
        ``common_genotypes`` (list), and ``output_paths`` (mapping of artifact
        name to path).
    """
    if fdr_methods is None:
        fdr_methods = list(DEFAULT_FDR_METHODS)
    output_dir = Path(output_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1-2. Load per-platform PCA outputs and aggregate to genotype means.
    platform_data = load_platform_data(pipeline_run, platform_config)
    common_genotypes, aligned_data = align_genotypes_across_platforms(platform_data)

    # 3-4. Correlate PCs across platform pairs with the requested FDR scope(s).
    platform_n_pcs = {
        name: int(cfg["n_pcs"])
        for name, cfg in platform_config.items()
        if "n_pcs" in cfg
    }
    correlations = calculate_all_platform_correlations(
        aligned_data=aligned_data,
        platform_n_pcs=platform_n_pcs,
        fdr_methods=fdr_methods,
        alpha=alpha,
        confidence_level=confidence_level,
        fdr_scope=fdr_scope,
    )
    summary = summarize_correlations(
        correlations, primary_method=primary_fdr_method, fdr_scope=fdr_scope
    )

    # 5. Export tidy CSVs.
    output_paths: dict[str, str] = {}
    corr_path = data_dir / "correlations.csv"
    correlations.to_csv(corr_path, index=False)
    output_paths["correlations"] = str(corr_path)

    for scope, sig_col in _significant_columns(fdr_scope, primary_fdr_method).items():
        sig_path = data_dir / f"significant_{scope}.csv"
        subset = (
            correlations[correlations[sig_col]]
            if sig_col in correlations.columns
            else correlations.iloc[0:0]
        )
        subset.to_csv(sig_path, index=False)
        output_paths[f"significant_{scope}"] = str(sig_path)

    for name, means in aligned_data.items():
        safe = name.lower().replace(" ", "_")
        means_path = data_dir / f"genotype_means_{safe}.csv"
        means.to_csv(means_path)
        output_paths[f"genotype_means_{safe}"] = str(means_path)

    # 6. Figures (optional).
    if make_figures and not correlations.empty:
        figures_dir = output_dir / "figures"
        scopes = (
            [
                ("combined", f"combined_{primary_fdr_method}"),
                ("per_pair", f"per_pair_{primary_fdr_method}"),
            ]
            if fdr_scope == "both"
            else [(fdr_scope, primary_fdr_method)]
        )
        for scope, method_col in scopes:
            if f"significant_{method_col}" not in correlations.columns:
                continue
            scope_dir = figures_dir / scope
            fig = create_correlation_summary_figure(correlations, fdr_method=method_col)
            save_figure(fig, scope_dir, "correlation_summary")
            fig.clf()
            for p1, p2 in combinations(platform_config.keys(), 2):
                fig = create_correlation_heatmap(
                    correlations, p1, p2, fdr_method=method_col
                )
                save_figure(fig, scope_dir, f"heatmap_{p1.lower()}_{p2.lower()}")
                fig.clf()
        fig = create_sensitivity_analysis_figure(correlations, methods=fdr_methods)
        save_figure(fig, figures_dir, "sensitivity_analysis")
        fig.clf()
        output_paths["figures_dir"] = str(figures_dir)

    # 7. Metadata.
    metadata = {
        "analysis_type": "pc_level_cross_platform_correlation",
        "pipeline_run": str(pipeline_run),
        "parameters": {
            "fdr_methods": fdr_methods,
            "primary_fdr_method": primary_fdr_method,
            "fdr_scope": fdr_scope,
            "alpha": alpha,
            "confidence_level": confidence_level,
        },
        "platforms": {
            name: {"n_pcs_used": platform_n_pcs.get(name)} for name in platform_config
        },
        "results": _to_native(summary),
        "common_genotypes": list(common_genotypes),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    output_paths["metadata"] = str(metadata_path)

    return {
        "correlations": correlations,
        "summary": summary,
        "genotype_means": aligned_data,
        "common_genotypes": common_genotypes,
        "output_paths": output_paths,
    }


def trait_correlation_enrichment(
    correlation_files: dict[str, Path],
    output_dir: Path,
    alpha: float = 0.05,
    p_value_column: str = "spearman_p",
    confidence_level: float = 0.95,
    make_figure: bool = True,
) -> dict[str, Any]:
    """Run the trait-level binomial enrichment workflow in one call.

    Counts nominally significant (``p < alpha``) trait correlations in each
    supplied ``cross_platform_correlations.csv`` and tests whether the count
    deviates from chance, per pair and pooled. This workflow is independent of
    :func:`cross_platform_pc_correlations`.

    Args:
        correlation_files: Mapping of platform-pair label to correlation CSV path.
        output_dir: Directory for the written artifacts (created if missing).
        alpha: Nominal significance threshold (null proportion).
        p_value_column: Column holding p-values in the correlation CSVs.
        confidence_level: Confidence level for the proportion intervals.
        make_figure: Whether to render the enrichment summary figure.

    Returns:
        Dict with keys ``results`` (list of :class:`EnrichmentResult`),
        ``results_df`` (DataFrame), and ``output_paths`` (mapping of artifact
        name to path).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = calculate_all_enrichment_tests(
        correlation_files=correlation_files,
        alpha=alpha,
        p_value_column=p_value_column,
        confidence_level=confidence_level,
    )
    results_df = results_to_dataframe(results)

    output_paths: dict[str, str] = {}
    csv_path = output_dir / "enrichment_results.csv"
    results_df.to_csv(csv_path, index=False)
    output_paths["enrichment_results"] = str(csv_path)

    if make_figure:
        fig = create_enrichment_figure(results)
        paths = save_figure(fig, output_dir, "enrichment_summary")
        fig.clf()
        output_paths["enrichment_summary"] = str(paths[0])

    combined = results[0]
    metadata = {
        "analysis_type": "trait_level_enrichment_test",
        "parameters": {
            "alpha": alpha,
            "p_value_column": p_value_column,
            "confidence_level": confidence_level,
            "test_method": "binomial (exact)",
            "ci_method": "Clopper-Pearson (exact)",
        },
        "input_files": {name: str(path) for name, path in correlation_files.items()},
        "results_summary": _to_native(
            {
                "combined": {
                    "n_tests": combined.n_tests,
                    "n_significant": combined.n_significant,
                    "fold_enrichment": combined.fold_enrichment,
                    "interpretation": combined.interpretation,
                },
                "per_pair": {
                    r.platform_pair: {
                        "n_tests": r.n_tests,
                        "n_significant": r.n_significant,
                        "fold_enrichment": r.fold_enrichment,
                        "interpretation": r.interpretation,
                    }
                    for r in results[1:]
                },
            }
        ),
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    output_paths["metadata"] = str(metadata_path)

    return {
        "results": results,
        "results_df": results_df,
        "output_paths": output_paths,
    }
