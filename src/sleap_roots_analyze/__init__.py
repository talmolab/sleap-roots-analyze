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
    run_pca_and_export_artifacts,
)

from sleap_roots_analyze.umap import (
    perform_umap_analysis,
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
    create_heritability_plot,
    create_heritability_threshold_plot,
    identify_extreme_samples_in_pc_space,
    create_pca_scree_plot,
    create_feature_contribution_plot,
    create_pca_biplot,
    create_umap_colored_by_top_traits,
    identify_extreme_genotypes_by_pc,
    create_pc_genotype_boxplots,
    create_feature_contribution_heatmap,
    create_publication_figure,
    identify_extreme_phenotypes,
    create_phenotype_variation_plot,
)

from sleap_roots_analyze.outlier_visualization import (
    create_isolation_forest_plots,
    create_outlier_overlap_heatmap,
    create_outliers_per_genotype_plot,
    create_mahalanobis_outlier_plots,
    create_pca_outlier_plot,
    create_comprehensive_outlier_comparison,
)

from sleap_roots_analyze.cross_experiment_analysis import (
    load_and_align_experiments,
    calculate_genotype_means,
    calculate_genotype_statistics,
    calculate_cross_experiment_correlations,
    calculate_cross_experiment_correlations_extended,
    summarize_statistic_combinations,
    create_statistic_combination_heatmap,
    create_cross_experiment_heatmap,
    create_top_correlations_plot,
    create_scatter_plot_grid,
    calculate_per_trait_correlations,
    create_joint_plot,
    identify_significant_correlations,
    calculate_correlation_confidence_intervals,
    summarize_correlation_results,
    create_genotype_boxplots,
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
    "run_pca_and_export_artifacts",
    # UMAP functions
    "perform_umap_analysis",
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
    "create_heritability_plot",
    "create_heritability_threshold_plot",
    "identify_extreme_samples_in_pc_space",
    "create_pca_scree_plot",
    "create_feature_contribution_plot",
    "create_pca_biplot",
    "create_umap_colored_by_top_traits",
    "identify_extreme_genotypes_by_pc",
    "create_pc_genotype_boxplots",
    "create_feature_contribution_heatmap",
    "create_publication_figure",
    "identify_extreme_phenotypes",
    "create_phenotype_variation_plot",
    # Outlier visualization functions
    "create_isolation_forest_plots",
    "create_outlier_overlap_heatmap",
    "create_outliers_per_genotype_plot",
    "create_mahalanobis_outlier_plots",
    "create_pca_outlier_plot",
    "create_comprehensive_outlier_comparison",
    # Cross-experiment analysis functions
    "load_and_align_experiments",
    "calculate_genotype_means",
    "calculate_genotype_statistics",
    "calculate_cross_experiment_correlations",
    "calculate_cross_experiment_correlations_extended",
    "summarize_statistic_combinations",
    "create_statistic_combination_heatmap",
    "create_cross_experiment_heatmap",
    "create_top_correlations_plot",
    "create_scatter_plot_grid",
    "calculate_per_trait_correlations",
    "create_joint_plot",
    "identify_significant_correlations",
    "calculate_correlation_confidence_intervals",
    "summarize_correlation_results",
    "create_genotype_boxplots",
]
