"""User exposed API."""

from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    remove_nan_samples,
    remove_low_heritability_traits,
    link_rhizovision_images_to_samples,
)

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    build_feature_metrics_df,
)

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_pca,
    detect_outliers_isolation_forest,
    calculate_outlier_threshold,
    identify_outliers_from_distances,
    remove_outliers_from_data,
)

__all__ = [
    # Data cleanup functions
    "load_trait_data",
    "get_trait_columns",
    "remove_nan_samples",
    "remove_low_heritability_traits",
    "link_rhizovision_images_to_samples",
    # PCA functions
    "perform_pca_analysis",
    "calculate_mahalanobis_distances",
    "calculate_pca_metrics",
    "build_feature_metrics_df",
    # Outlier detection functions
    "detect_outliers_mahalanobis",
    "detect_outliers_pca",
    "detect_outliers_isolation_forest",
    "calculate_outlier_threshold",
    "identify_outliers_from_distances",
    "remove_outliers_from_data",
]
