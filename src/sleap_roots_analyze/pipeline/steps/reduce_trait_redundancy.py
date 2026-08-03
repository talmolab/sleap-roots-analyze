"""Step: Reduce trait redundancy before cross-platform correlation analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage as hierarchical_linkage
from scipy.spatial.distance import squareform

from sleap_roots_analyze.cross_experiment_analysis import (
    cluster_correlated_traits,
    select_cluster_representatives,
)
from sleap_roots_analyze.data_utils import sanitize_single_trait_name
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class ReduceTraitRedundancyStep(BaseStep):
    """Reduce trait redundancy before cross-platform correlation analysis.

    This step clusters highly correlated traits and selects one representative
    per cluster, reducing the multiple testing burden in downstream correlation
    analysis.

    Configuration options:
        - trait_reduction_method: "none" or "clustering"
        - trait_reduction_target: "exp1", "exp2", or "both"
        - trait_clustering_threshold: Correlation threshold for clustering (0-1)
        - trait_clustering_linkage: "complete", "average", or "single"

    Outputs (when clustering enabled):
        - exp1_trait_clusters.csv: Cluster membership for exp1 (if exp1 clustered)
        - exp2_trait_clusters.csv: Cluster membership for exp2 (if exp2 clustered)
        - exp1_trait_clustering_dendrogram.png: Dendrogram for exp1 (if exp1 clustered)
        - exp2_trait_clustering_dendrogram.png: Dendrogram for exp2 (if exp2 clustered)
        - exp1_trait_cluster_heatmap.png: Correlation heatmap for exp1 (if exp1 clustered)
        - exp2_trait_cluster_heatmap.png: Correlation heatmap for exp2 (if exp2 clustered)
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
        exp1_trait_names = list(prev_result.metadata["exp1_trait_names"])
        exp2_trait_names = list(prev_result.metadata["exp2_trait_names"])

        # Initialize output metadata from previous step
        metadata = dict(prev_result.metadata)
        files_generated: List[Path] = []

        if config.trait_reduction_method == "none":
            # No reduction - pass through unchanged
            logger.info(
                "Trait reduction method is 'none', passing data through unchanged"
            )

            metadata["trait_reduction_method"] = "none"
            metadata["trait_reduction_target"] = None
            metadata["exp1_original_traits"] = len(exp1_trait_names)
            metadata["exp1_reduced_traits"] = len(exp1_trait_names)
            metadata["exp2_original_traits"] = len(exp2_trait_names)
            metadata["exp2_reduced_traits"] = len(exp2_trait_names)

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
        target = config.trait_reduction_target
        logger.info(
            "Applying clustering-based trait reduction (target=%s, threshold=%.2f, linkage=%s)",
            target,
            config.trait_clustering_threshold,
            config.trait_clustering_linkage,
        )

        # Update metadata with clustering parameters
        metadata["trait_reduction_method"] = "clustering"
        metadata["trait_reduction_target"] = target
        metadata["trait_clustering_threshold"] = config.trait_clustering_threshold
        metadata["trait_clustering_linkage"] = config.trait_clustering_linkage

        # Process exp1 if target includes it
        if target in ["exp1", "both"]:
            exp1_df, exp1_trait_names, exp1_files, exp1_n_clusters = (
                self._cluster_experiment(
                    exp1_df,
                    exp1_trait_names,
                    config,
                    run_dir,
                    exp_name="exp1",
                    exp_display_name=config.exp1_name,
                )
            )
            files_generated.extend(exp1_files)
            metadata["exp1_original_traits"] = len(
                prev_result.metadata["exp1_trait_names"]
            )
            metadata["exp1_reduced_traits"] = len(exp1_trait_names)
            metadata["exp1_n_clusters"] = exp1_n_clusters
            metadata["exp1_trait_names"] = exp1_trait_names
        else:
            metadata["exp1_original_traits"] = len(exp1_trait_names)
            metadata["exp1_reduced_traits"] = len(exp1_trait_names)

        # Process exp2 if target includes it
        if target in ["exp2", "both"]:
            exp2_df, exp2_trait_names, exp2_files, exp2_n_clusters = (
                self._cluster_experiment(
                    exp2_df,
                    exp2_trait_names,
                    config,
                    run_dir,
                    exp_name="exp2",
                    exp_display_name=config.exp2_name,
                )
            )
            files_generated.extend(exp2_files)
            metadata["exp2_original_traits"] = len(
                prev_result.metadata["exp2_trait_names"]
            )
            metadata["exp2_reduced_traits"] = len(exp2_trait_names)
            metadata["exp2_n_clusters"] = exp2_n_clusters
            metadata["exp2_trait_names"] = exp2_trait_names
        else:
            metadata["exp2_original_traits"] = len(exp2_trait_names)
            metadata["exp2_reduced_traits"] = len(exp2_trait_names)

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

    def _cluster_experiment(
        self,
        df: pd.DataFrame,
        trait_names: List[str],
        config: Any,
        run_dir: Path,
        exp_name: str,
        exp_display_name: str,
    ) -> Tuple[pd.DataFrame, List[str], List[Path], int]:
        """Cluster traits for a single experiment.

        Args:
            df: DataFrame with genotype column and trait columns.
            trait_names: List of trait column names.
            config: CrossPlatformConfig with clustering parameters.
            run_dir: Directory to save outputs.
            exp_name: Short name for files (e.g., "exp1", "exp2").
            exp_display_name: Display name for plots (e.g., "Turface 19 Genotypes").

        Returns:
            Tuple of (reduced_df, representative_trait_names, files_generated, n_clusters).
        """
        files_generated: List[Path] = []

        # Aggregate by genotype for trait correlations
        trait_data = df.groupby("genotype")[trait_names].mean()

        # Compute correlation matrix
        corr_matrix = trait_data.corr(method="spearman")

        # Cluster traits
        clusters = cluster_correlated_traits(
            trait_data,
            threshold=config.trait_clustering_threshold,
            linkage=config.trait_clustering_linkage,
        )

        # Select representatives
        representatives = select_cluster_representatives(trait_data, clusters)

        # Calculate variances for cluster membership file
        variances = {trait: trait_data[trait].var() for trait in trait_names}

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
        cluster_file = run_dir / f"{exp_name}_trait_clusters.csv"
        cluster_df.to_csv(cluster_file, index=False)
        files_generated.append(cluster_file)

        logger.info(
            "Saved %s cluster membership to %s (%d traits -> %d clusters -> %d representatives)",
            exp_name,
            cluster_file,
            len(trait_names),
            len(clusters),
            len(representatives),
        )

        # Generate dendrogram visualization
        dendrogram_file = self._create_dendrogram(
            corr_matrix,
            trait_names,
            clusters,
            config.trait_clustering_threshold,
            config.trait_clustering_linkage,
            run_dir,
            exp_name,
            exp_display_name,
        )
        files_generated.append(dendrogram_file)

        # Generate cluster heatmap visualization
        heatmap_file = self._create_cluster_heatmap(
            corr_matrix,
            clusters,
            representatives,
            run_dir,
            exp_name,
            exp_display_name,
        )
        files_generated.append(heatmap_file)

        # Filter df to only include representative columns (plus genotype)
        keep_cols = ["genotype"] + representatives
        df_reduced = df[keep_cols].copy()

        n_clusters = len(clusters)
        return df_reduced, representatives, files_generated, n_clusters

    def _create_dendrogram(
        self,
        corr_matrix: pd.DataFrame,
        trait_names: List[str],
        clusters: Dict[int, List[str]],
        threshold: float,
        linkage_method: str,
        run_dir: Path,
        exp_name: str,
        exp_display_name: str,
    ) -> Path:
        """Create and save a dendrogram visualization.

        Args:
            corr_matrix: Correlation matrix of traits.
            trait_names: List of trait names.
            clusters: Dictionary mapping cluster_id to list of traits.
            threshold: Clustering threshold.
            linkage_method: Linkage method used.
            run_dir: Directory to save output.
            exp_name: Short name for file.
            exp_display_name: Display name for title.

        Returns:
            Path to saved dendrogram image.
        """
        # Convert correlation to distance (1 - |r|)
        dist_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(dist_matrix, 0)

        # Compute linkage
        condensed_dist = squareform(dist_matrix)
        Z = hierarchical_linkage(condensed_dist, method=linkage_method)

        # Create figure
        n_traits = len(trait_names)
        fig_height = max(8, n_traits * 0.15)
        fig, ax = plt.subplots(figsize=(12, fig_height))

        # Plot dendrogram
        dendrogram(
            Z,
            labels=trait_names,
            orientation="right",
            leaf_font_size=8,
            ax=ax,
            color_threshold=1 - threshold,  # Convert threshold to distance
        )

        # Add threshold line
        ax.axvline(
            x=1 - threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Threshold (|r| = {threshold})",
        )

        ax.set_xlabel("Distance (1 - |r|)")
        ax.set_title(
            f"Trait Clustering Dendrogram: {exp_display_name}\n"
            f"({len(clusters)} clusters from {n_traits} traits)"
        )
        ax.legend(loc="upper right")

        # Save figure
        output_path = run_dir / f"{exp_name}_trait_clustering_dendrogram.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved %s dendrogram to %s", exp_name, output_path)
        return output_path

    def _create_cluster_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        clusters: Dict[int, List[str]],
        representatives: List[str],
        run_dir: Path,
        exp_name: str,
        exp_display_name: str,
    ) -> Path:
        """Create and save a correlation heatmap with cluster structure.

        Args:
            corr_matrix: Correlation matrix of traits.
            clusters: Dictionary mapping cluster_id to list of traits.
            representatives: List of representative trait names.
            run_dir: Directory to save output.
            exp_name: Short name for file.
            exp_display_name: Display name for title.

        Returns:
            Path to saved heatmap image.
        """
        # Order traits by cluster membership
        ordered_traits = []
        cluster_boundaries = []
        current_pos = 0

        for cluster_id in sorted(clusters.keys()):
            traits_in_cluster = clusters[cluster_id]
            ordered_traits.extend(traits_in_cluster)
            current_pos += len(traits_in_cluster)
            cluster_boundaries.append(current_pos)

        # Reorder correlation matrix
        ordered_corr = corr_matrix.loc[ordered_traits, ordered_traits]

        # Create figure
        n_traits = len(ordered_traits)
        fig_size = max(10, n_traits * 0.12)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))

        # Plot heatmap
        sns.heatmap(
            ordered_corr,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            ax=ax,
            xticklabels=True,
            yticklabels=True,
            cbar_kws={"label": "Spearman r", "shrink": 0.5},
        )

        # Highlight cluster boundaries
        for boundary in cluster_boundaries[:-1]:
            ax.axhline(y=boundary, color="black", linewidth=2)
            ax.axvline(x=boundary, color="black", linewidth=2)

        # Sanitize tick labels for better readability
        sanitized_labels = [
            sanitize_single_trait_name(trait, abbreviate=False)
            for trait in ordered_traits
        ]
        ax.set_xticklabels(sanitized_labels)
        ax.set_yticklabels(sanitized_labels)

        # Create mapping from sanitized back to original for representative highlighting
        sanitized_to_original = dict(zip(sanitized_labels, ordered_traits))

        # Mark representative traits
        for i, trait in enumerate(ordered_traits):
            if trait in representatives:
                ax.get_yticklabels()[i].set_weight("bold")
                ax.get_yticklabels()[i].set_color("green")
                ax.get_xticklabels()[i].set_weight("bold")
                ax.get_xticklabels()[i].set_color("green")

        ax.set_title(
            f"Trait Correlation Heatmap: {exp_display_name}\n"
            f"({len(clusters)} clusters, {len(representatives)} representatives in bold green)"
        )

        # Rotate labels
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(fontsize=6)

        # Save figure
        output_path = run_dir / f"{exp_name}_trait_cluster_heatmap.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info("Saved %s cluster heatmap to %s", exp_name, output_path)
        return output_path
