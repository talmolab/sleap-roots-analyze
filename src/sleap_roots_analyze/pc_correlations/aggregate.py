"""Load and aggregate per-platform PC scores by genotype.

These functions form the first nodes of the PC-level cross-platform correlation
DAG: they load the pipeline's sample-level PCA output and QC genotype labels,
aggregate the sample PC scores to genotype-level means, and align platforms to a
shared panel of common genotypes.

The aggregation happens *after* PCA (the pipeline computes sample-level PCA
upstream and writes ``pc_scores.csv``), preserving the required
sample-PCA -> genotype-mean ordering — never average-then-PCA.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_pc_scores(
    pca_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load PC scores, loadings, and explained variance from PCA output.

    Args:
        pca_dir: Path to the ``pca/`` directory containing ``pc_scores.csv``,
            ``loadings.csv``, and ``explained_variance.csv``.

    Returns:
        Tuple of ``(pc_scores, loadings, explained_variance)`` DataFrames.
    """
    pca_dir = Path(pca_dir)
    pc_scores = pd.read_csv(pca_dir / "pc_scores.csv", index_col=0)
    loadings = pd.read_csv(pca_dir / "loadings.csv", index_col=0)
    explained_variance = pd.read_csv(pca_dir / "explained_variance.csv", index_col=0)

    return pc_scores, loadings, explained_variance


def load_genotype_mapping(
    final_data_path: Path, genotype_col: str = "Genotype"
) -> pd.Series:
    """Load genotype labels from a QC final-data file.

    Args:
        final_data_path: Path to the QC ``10_final_data.csv`` (rows aligned with
            the PCA sample rows).
        genotype_col: Name of the genotype column.

    Returns:
        Series of genotype labels aligned by position with the PC score rows.
    """
    df = pd.read_csv(final_data_path)
    return df[genotype_col]


def aggregate_pc_scores_by_genotype(
    pc_scores: pd.DataFrame,
    genotype_series: pd.Series,
    pc_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate sample-level PC scores to genotype means.

    Args:
        pc_scores: Sample-level PC scores (rows = samples, columns include
            ``PC1``, ``PC2``, ...).
        genotype_series: Genotype labels aligned by position with ``pc_scores``.
        pc_columns: PC columns to aggregate. Defaults to every column of
            ``pc_scores``.

    Returns:
        DataFrame of genotype means (index = genotype, columns = the requested
        PCs plus an ``n_samples`` count).
    """
    if pc_columns is None:
        pc_columns = list(pc_scores.columns)

    # PC scores and genotype labels come from two separately loaded files
    # (pc_scores.csv and the QC final_data) that must be row-aligned; guard with
    # a clear message rather than pandas' generic length-mismatch error.
    if len(genotype_series) != len(pc_scores):
        raise ValueError(
            f"PC scores have {len(pc_scores)} rows but genotype labels have "
            f"{len(genotype_series)}; pc_scores.csv and the QC final_data must "
            "be row-aligned (same samples, same order)."
        )

    df = pc_scores[pc_columns].copy()
    df["genotype"] = genotype_series.to_numpy()

    agg_funcs: dict[str, str] = {col: "mean" for col in pc_columns}
    agg_funcs["genotype"] = "count"  # count samples per genotype

    result = df.groupby("genotype").agg(agg_funcs)
    result = result.rename(columns={"genotype": "n_samples"})

    return result


def load_platform_data(
    pipeline_run: Path,
    platform_config: dict[str, dict[str, object]],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Load PC scores and genotype data for every platform.

    Args:
        pipeline_run: Path to the pipeline-run directory.
        platform_config: Mapping of platform name to its configuration, e.g.::

            {
                "Turface": {
                    "pca_dir": "viz/.../data/pca",
                    "final_data": "qc/.../10_final_data.csv",
                    "genotype_col": "Genotype",
                    "n_pcs": 3,
                },
                ...
            }

    Returns:
        Mapping of platform name to a dict with keys ``pc_scores``, ``loadings``,
        ``explained_variance``, ``genotypes`` (labels), and ``genotype_means``.
    """
    pipeline_run = Path(pipeline_run)
    result: dict[str, dict[str, pd.DataFrame]] = {}

    for platform_name, config in platform_config.items():
        pca_dir = pipeline_run / str(config["pca_dir"])
        final_data_path = pipeline_run / str(config["final_data"])
        genotype_col = str(config.get("genotype_col", "Genotype"))

        pc_scores, loadings, explained_variance = load_pc_scores(pca_dir)
        genotypes = load_genotype_mapping(final_data_path, genotype_col)
        genotype_means = aggregate_pc_scores_by_genotype(pc_scores, genotypes)

        result[platform_name] = {
            "pc_scores": pc_scores,
            "loadings": loadings,
            "explained_variance": explained_variance,
            "genotypes": genotypes,
            "genotype_means": genotype_means,
        }

    return result


def align_genotypes_across_platforms(
    platform_data: dict[str, dict[str, pd.DataFrame]],
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    """Find genotypes common to all platforms and return aligned PC means.

    Args:
        platform_data: Output of :func:`load_platform_data`.

    Returns:
        Tuple of ``(common_genotypes, aligned_data)`` where ``common_genotypes``
        is the sorted intersection and ``aligned_data`` maps each platform name
        to its ``genotype_means`` filtered to those genotypes.

    Note:
        For three or more platforms this is the intersection across **all**
        platforms, not the true pairwise intersection. Every pair is therefore
        correlated on the same shared panel (a uniform ``n``), which can
        understate the genotypes actually available to a given pair — and hence
        makes the headline power / minimum-detectable-correlation slightly
        conservative. This is intentional (one comparable panel across all
        pairs).
    """
    genotype_sets = [
        set(data["genotype_means"].index) for data in platform_data.values()
    ]
    common_genotypes = sorted(set.intersection(*genotype_sets)) if genotype_sets else []

    aligned_data: dict[str, pd.DataFrame] = {}
    for platform_name, data in platform_data.items():
        aligned_data[platform_name] = data["genotype_means"].loc[common_genotypes]

    return common_genotypes, aligned_data


# Example platform configuration for the wheat EDPIE analysis. Provided as a
# template only; public functions never hard-code these paths.
WHEAT_EDPIE_PLATFORMS: dict[str, dict[str, object]] = {
    "Turface": {
        "pca_dir": "viz/viz_turface_19genotypes_20260212_192803/data/pca",
        "final_data": "qc/turface_19genotypes_qc_20260212_191941/10_final_data.csv",
        "genotype_col": "Genotype",
        "n_pcs": 3,  # PC1-PC3 explain 96% variance
    },
    "Cylinder": {
        "pca_dir": "viz/viz_cylinder_edpie_20260212_192902/data/pca",
        "final_data": "qc/cylinder_edpie_qc_20260212_192028/10_final_data.csv",
        "genotype_col": "Genotype",
        "n_pcs": 4,  # PC1-PC4 explain 75.5% variance
    },
    "Field": {
        "pca_dir": "viz/viz_root_coring_20260212_194044/data/pca",
        "final_data": "qc/edpie_root_core_qc_20260212_192315/10_final_data.csv",
        "genotype_col": "Genotype",
        "n_pcs": 5,  # PC1-PC5 explain 76% variance
    },
}
