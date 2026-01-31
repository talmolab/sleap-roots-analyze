"""Step: Reduce trait redundancy before cross-platform correlation analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import numpy as np

from sleap_roots_analyze.cross_experiment_analysis import (
    cluster_correlated_traits,
    select_cluster_representatives,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class ReduceTraitRedundancyStep(BaseStep):
    """Reduce trait redundancy before cross-platform correlation analysis.

    This step optionally clusters highly correlated traits within exp2 and selects
    one representative per cluster. This reduces the multiple testing burden in
    downstream correlation analysis.

    When trait_reduction_method is "none": Pass through data unchanged.
    When trait_reduction_method is "clustering": Cluster exp2 traits and select representatives.

    Outputs:
        - trait_clusters.csv: Cluster membership file (only when clustering enabled)
          Columns: trait, cluster_id, is_representative, variance
    """

    def __init__(self):
        """Initialize ReduceTraitRedundancyStep."""
        super().__init__(
            step_name="ReduceTraitRedundancy",
            description="Reduce trait redundancy via hierarchical clustering",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the trait redundancy reduction step.

        Args:
            data: Data dict from LoadCrossPlatformDataStep with exp1_df, exp2_df.
            config: CrossPlatformConfig with trait reduction parameters.
            run_dir: Directory to save outputs.
            prev_result: Previous step result with metadata including trait names.

        Returns:
            StepResult with potentially reduced trait DataFrames and metadata.
        """
        # Get data from previous step
        exp1_df = data["exp1_df"].copy()
        exp2_df = data["exp2_df"].copy()

        # Get trait names from previous step metadata
        exp1_trait_names = prev_result.metadata["exp1_trait_names"]
        exp2_trait_names = prev_result.metadata["exp2_trait_names"]

        # Initialize output metadata from previous step
        metadata = dict(prev_result.metadata)
        files_generated = []

        if config.trait_reduction_method == "none":
            # No reduction - pass through unchanged
            logger.info(
                "Trait reduction method is 'none', passing data through unchanged"
            )

            metadata["trait_reduction_method"] = "none"
            metadata["reduction_ratio"] = 0.0
            metadata["original_exp2_traits"] = len(exp2_trait_names)
            metadata["reduced_exp2_traits"] = len(exp2_trait_names)

            result_data = {
                "exp1_df": exp1_df,
                "exp2_df": exp2_df,
                "common_genotypes": data.get("common_genotypes", []),
            }

            return StepResult(
                data=result_data,
                metadata=metadata,
                files_generated=files_generated,
            )

        # Clustering-based reduction
        logger.info(
            "Applying clustering-based trait reduction with threshold=%.2f, linkage=%s",
            config.trait_clustering_threshold,
            config.trait_clustering_linkage,
        )

        # Extract trait data from exp2 (rows are genotypes/samples, columns are traits)
        # Need to aggregate by genotype first for trait correlations
        exp2_trait_data = exp2_df.groupby("genotype")[exp2_trait_names].mean()

        # Cluster traits
        clusters = cluster_correlated_traits(
            exp2_trait_data,
            threshold=config.trait_clustering_threshold,
            linkage=config.trait_clustering_linkage,
        )

        # Select representatives
        representatives = select_cluster_representatives(exp2_trait_data, clusters)

        # Calculate variances for cluster membership file
        variances = {trait: exp2_trait_data[trait].var() for trait in exp2_trait_names}

        # Build cluster membership dataframe
        cluster_records = []
        for cluster_id, traits in clusters.items():
            for trait in traits:
                cluster_records.append(
                    {
                        "trait": trait,
                        "cluster_id": cluster_id,
                        "is_representative": trait in representatives,
                        "variance": variances.get(trait, np.nan),
                    }
                )

        cluster_df = pd.DataFrame(cluster_records)

        # Save cluster membership file
        cluster_file = run_dir / "trait_clusters.csv"
        cluster_df.to_csv(cluster_file, index=False)
        files_generated.append(str(cluster_file))

        logger.info(
            "Saved cluster membership to %s (%d traits -> %d clusters)",
            cluster_file,
            len(exp2_trait_names),
            len(clusters),
        )

        # Filter exp2_df to only include representative columns (plus genotype)
        keep_cols = ["genotype"] + representatives
        exp2_df_reduced = exp2_df[keep_cols].copy()

        # Calculate reduction statistics
        original_count = len(exp2_trait_names)
        reduced_count = len(representatives)
        reduction_ratio = (
            (original_count - reduced_count) / original_count
            if original_count > 0
            else 0.0
        )

        logger.info(
            "Trait reduction: %d -> %d traits (%.1f%% reduction)",
            original_count,
            reduced_count,
            reduction_ratio * 100,
        )

        # Update metadata
        metadata["trait_reduction_method"] = "clustering"
        metadata["trait_clustering_threshold"] = config.trait_clustering_threshold
        metadata["trait_clustering_linkage"] = config.trait_clustering_linkage
        metadata["original_exp2_traits"] = original_count
        metadata["reduced_exp2_traits"] = reduced_count
        metadata["n_clusters"] = len(clusters)
        metadata["reduction_ratio"] = reduction_ratio
        metadata["exp2_trait_names"] = representatives  # Update to reduced list

        result_data = {
            "exp1_df": exp1_df,
            "exp2_df": exp2_df_reduced,
            "common_genotypes": data.get("common_genotypes", []),
        }

        return StepResult(
            data=result_data,
            metadata=metadata,
            files_generated=files_generated,
        )
