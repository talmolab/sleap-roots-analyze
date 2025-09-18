"""User exposed API."""

from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    get_numeric_traits_only,
    remove_nan_samples,
    remove_low_heritability_traits,
    inspect_nan_samples,
)

from sleap_roots_analyze.data_utils import (
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
    combine_outlier_methods,
)

from sleap_roots_analyze.visualization import (
    create_trait_histograms,
    create_trait_boxplots_by_genotype,
    create_correlation_heatmap,
    save_figure_with_unique_name,
    create_exploratory_summary_plots,
    create_trait_eda_plots,
)

from sleap_roots_analyze.outlier_visualization import (
    create_isolation_forest_plots,
    create_outlier_overlap_heatmap,
    create_outliers_per_genotype_plot,
    create_mahalanobis_outlier_plots,
    create_pca_outlier_plot,
    create_comprehensive_outlier_comparison,
)

__all__ = [
    # Data cleanup functions
    "load_trait_data",
    "get_trait_columns",
    "get_numeric_traits_only",
    "remove_nan_samples",
    "remove_low_heritability_traits",
    "link_rhizovision_images_to_samples",
    "inspect_nan_samples",
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
    "combine_outlier_methods",
    # Visualization functions
    "create_trait_histograms",
    "create_trait_boxplots_by_genotype",
    "create_correlation_heatmap",
    "save_figure_with_unique_name",
    "create_exploratory_summary_plots",
    "create_trait_eda_plots",
    # Outlier visualization functions
    "create_isolation_forest_plots",
    "create_outlier_overlap_heatmap",
    "create_outliers_per_genotype_plot",
    "create_mahalanobis_outlier_plots",
    "create_pca_outlier_plot",
    "create_comprehensive_outlier_comparison",
]
