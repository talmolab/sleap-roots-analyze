"""Step 5: Detect outliers using configured methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from sleap_roots_analyze.data_cleanup import get_numeric_traits_only
from sleap_roots_analyze.outlier_detection import (
    detect_outliers_gmm,
    detect_outliers_hierarchical,
    detect_outliers_isolation_forest,
    detect_outliers_kmeans,
    detect_outliers_mahalanobis,
    detect_outliers_pca,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class DetectOutliersStep(BaseStep):
    """Detect outliers using configured detection methods.

    This step runs all configured outlier detection methods but does NOT
    remove any outliers. It just stores the detection results for later
    visualization and removal steps.

    Supports 6 methods:
    - Traditional: PCA, Isolation Forest, Mahalanobis
    - Clustering: K-Means, GMM, Hierarchical

    Outputs:
        - None (results stored in metadata for next steps)
    """

    def __init__(self):
        """Initialize DetectOutliersStep."""
        super().__init__(
            step_name="DetectOutliers",
            description="Detect outliers using configured methods (no removal)",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute outlier detection.

        Args:
            data: DataFrame from previous step (ExploratoryAnalysisStep).
            config: Pipeline configuration with:
                - outlier_detection.traditional_methods: List of traditional methods
                - outlier_detection.clustering_methods: List of clustering methods
                - outlier_detection.pca.*: PCA method parameters
                - outlier_detection.isolation_forest.*: IF parameters
                - outlier_detection.mahalanobis.*: Mahalanobis parameters
                - outlier_detection.kmeans.*: K-Means parameters
                - outlier_detection.gmm.*: GMM parameters
                - outlier_detection.hierarchical.*: Hierarchical parameters
                - columns.*: Column names for metadata preservation
            run_dir: Directory to save outputs.
            prev_result: Result from ExploratoryAnalysisStep (contains trait names).

        Returns:
            StepResult with outlier detection results in metadata.
        """
        df = data
        trait_cols = prev_result.metadata["valid_trait_names"]

        # Get numeric traits only (for detection)
        numeric_df = get_numeric_traits_only(
            df,
            barcode_col=config.columns.barcode,
            genotype_col=config.columns.genotype,
            replicate_col=config.columns.replicate,
        )

        # Store all outlier detection results
        outlier_results = {}

        # Get configured methods
        traditional_methods = config.outlier_detection.traditional_methods
        clustering_methods = config.outlier_detection.clustering_methods

        # Run traditional methods
        if "pca" in traditional_methods:
            outlier_results["pca"] = detect_outliers_pca(
                numeric_df,
                explained_variance=config.outlier_detection.pca.explained_variance,
                threshold=config.outlier_detection.pca.threshold,
            )

        if "isolation_forest" in traditional_methods:
            outlier_results["isolation_forest"] = detect_outliers_isolation_forest(
                numeric_df,
                contamination=config.outlier_detection.isolation_forest.contamination,
            )

        if "mahalanobis" in traditional_methods:
            outlier_results["mahalanobis"] = detect_outliers_mahalanobis(
                numeric_df,
                variance_threshold=config.outlier_detection.mahalanobis.variance_threshold,
                use_chi_squared=config.outlier_detection.mahalanobis.use_chi_squared,
                chi2_percentile=config.outlier_detection.mahalanobis.chi2_percentile,
            )

        # Run clustering methods
        if "kmeans" in clustering_methods:
            outlier_results["kmeans"] = detect_outliers_kmeans(
                numeric_df,
                n_clusters=config.outlier_detection.kmeans.n_clusters,
                max_clusters=config.outlier_detection.kmeans.max_clusters,
                distance_threshold=config.outlier_detection.kmeans.distance_threshold,
            )

        if "gmm" in clustering_methods:
            outlier_results["gmm"] = detect_outliers_gmm(
                numeric_df,
                n_components=config.outlier_detection.gmm.n_components,
                max_components=config.outlier_detection.gmm.max_components,
                percentile_threshold=config.outlier_detection.gmm.percentile_threshold,
            )

        if "hierarchical" in clustering_methods:
            outlier_results["hierarchical"] = detect_outliers_hierarchical(
                numeric_df,
                n_clusters=config.outlier_detection.hierarchical.n_clusters,
                linkage_method=config.outlier_detection.hierarchical.linkage_method,
                distance_threshold=config.outlier_detection.hierarchical.distance_threshold,
            )

        # Count outliers per method
        outlier_counts = {}
        for method_name, result in outlier_results.items():
            outlier_counts[method_name] = len(result.get("outlier_indices", []))

        # Create metadata
        metadata = {
            "samples": len(df),
            "methods_run": list(outlier_results.keys()),
            "outlier_results": outlier_results,
            "outlier_counts": outlier_counts,
            "total_methods": len(outlier_results),
        }

        return StepResult(data=df, metadata=metadata, files_generated=[])
