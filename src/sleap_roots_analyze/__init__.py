"""User exposed API."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sleap-roots-analyze")
except PackageNotFoundError:
    __version__ = "unknown"

from sleap_roots_analyze.cli import main

from sleap_roots_analyze.statistics import (
    calculate_trait_statistics,
    perform_anova_by_genotype,
    calculate_heritability_estimates,
    identify_high_heritability_traits,
    analyze_heritability_thresholds,
    analyze_trait_variance,
    diagnose_heritability_issues,
    compare_trait_heritabilities,
)
from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    get_numeric_traits_only,
    remove_nan_samples,
    remove_low_heritability_traits,
    inspect_nan_samples,
    apply_data_cleanup_filters,
    build_clean_validation_report,
    validate_clean_traits,
    clean_traits_for_analysis,
)

from sleap_roots_analyze.data_utils import (
    link_rhizovision_images_to_samples,
    link_cylinder_images_from_scan_path,
)

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    build_feature_metrics_df,
    run_pca_and_export_artifacts,
)

from sleap_roots_analyze.result_types import (
    FeatureContribution,
    PCAResult,
    HeritabilityResult,
    TraitHeritability,
    ClusterResult,
    KMeansResult,
    GMMResult,
    HierarchicalResult,
    UMAPResult,
)

from sleap_roots_analyze.umap import (
    perform_umap_analysis,
)

from sleap_roots_analyze.clustering import (
    perform_kmeans_clustering,
    perform_gmm_clustering,
    perform_hierarchical_clustering,
    cut_dendrogram,
    calculate_optimal_clusters_hierarchical,
    calculate_cluster_quality_metrics,
    hierarchical_cluster_labels,
)

from sleap_roots_analyze.outlier_detection import (
    detect_outliers_mahalanobis,
    detect_outliers_pca,
    detect_outliers_isolation_forest,
    detect_outliers_kmeans,
    detect_outliers_gmm,
    detect_outliers_hierarchical,
    calculate_outlier_threshold,
    identify_outliers_from_distances,
    remove_outliers_from_data,
    combine_outlier_methods,
)

from sleap_roots_analyze.outlier_removal import (
    remove_outlier_samples,
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
    create_variance_decomposition_plot,
    create_trait_by_genotype_boxplots,
    create_heritability_diagnostic_dashboard,
    identify_extreme_samples_in_pc_space,
    create_pca_scree_plot,
    create_feature_contribution_plot,
    create_pca_biplot,
    create_umap_colored_by_top_traits,
    create_umap_single_trait,
    identify_extreme_genotypes_by_pc,
    create_pc_genotype_boxplots,
    create_feature_contribution_heatmap,
    create_publication_figure,
    identify_extreme_phenotypes,
    create_phenotype_variation_plot,
    create_genotype_image_grid,
    create_regression_plot,
)

from sleap_roots_analyze.cluster_visualization import (
    create_cluster_scatter_pca,
    create_distance_distribution_plot,
    create_cluster_size_barplot,
    create_bic_aic_comparison_plot,
    create_silhouette_plot,
    create_dendrogram,
)

from sleap_roots_analyze.outlier_visualization import (
    create_isolation_forest_plots,
    create_outlier_overlap_heatmap,
    create_outliers_per_genotype_plot,
    create_mahalanobis_outlier_plots,
    create_pca_outlier_plot,
    create_comprehensive_outlier_comparison,
    create_kmeans_outlier_plots,
    create_gmm_outlier_plots,
    create_hierarchical_outlier_plots,
    plot_outlier_analysis,
    select_outlier_figures,
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

from sleap_roots_analyze.pipeline import (
    VizPipeline,
    VizPipelineConfig,
    load_viz_config,
    save_viz_config,
    validate_viz_config,
)

from sleap_roots_analyze.viz_utils import (
    calculate_grid_dimensions,
    calculate_figure_size,
    calculate_subplot_grid_size,
    calculate_correlation_matrix_size,
    calculate_barplot_size,
    suggest_layout_params,
)

from sleap_roots_analyze.interactive_visualization import (
    encode_image_to_base64,
    create_interactive_scatter_with_images,
    create_interactive_pca_with_images,
    create_interactive_umap_with_images,
    create_interactive_umap_with_hover_highlight,
    create_trait_explorer_dashboard,
    create_interactive_scatter_with_preview,
    create_html_with_image_viewer,
    create_interactive_image_gallery,
    create_interactive_scatter_plot,
    create_interactive_pca_plot,
)

from sleap_roots_analyze.root_core_analysis import (
    create_sample_identifier,
    validate_unique_identifiers,
    melt_depth_data,
    aggregate_by_replicate,
)

from sleap_roots_analyze.depth_profile_plots import (
    plot_depth_profile_faceted,
    plot_depth_profile_replicates,
)

from sleap_roots_analyze.pc_correlations import (
    cross_platform_pc_correlations,
    trait_correlation_enrichment,
    CrossPlatformPCResult,
    EnrichmentResult,
)

__all__ = [
    # CLI
    "main",
    # Pipeline
    "VizPipeline",
    "VizPipelineConfig",
    "load_viz_config",
    "save_viz_config",
    "validate_viz_config",
    # Adaptive sizing utilities
    "calculate_grid_dimensions",
    "calculate_figure_size",
    "calculate_subplot_grid_size",
    "calculate_correlation_matrix_size",
    "calculate_barplot_size",
    "suggest_layout_params",
    # Data cleanup functions
    "load_trait_data",
    "get_trait_columns",
    "get_numeric_traits_only",
    "remove_nan_samples",
    "remove_low_heritability_traits",
    "link_rhizovision_images_to_samples",
    "link_cylinder_images_from_scan_path",
    "inspect_nan_samples",
    "apply_data_cleanup_filters",
    "build_clean_validation_report",
    "validate_clean_traits",
    "clean_traits_for_analysis",
    # PCA functions
    "perform_pca_analysis",
    "calculate_mahalanobis_distances",
    "calculate_pca_metrics",
    "build_feature_metrics_df",
    "run_pca_and_export_artifacts",
    # Serializable result types (#130)
    "PCAResult",
    "FeatureContribution",
    "HeritabilityResult",
    "TraitHeritability",
    "ClusterResult",
    "KMeansResult",
    "GMMResult",
    "HierarchicalResult",
    "UMAPResult",
    # UMAP functions
    "perform_umap_analysis",
    # Clustering functions
    "perform_kmeans_clustering",
    "perform_gmm_clustering",
    "perform_hierarchical_clustering",
    "cut_dendrogram",
    "calculate_optimal_clusters_hierarchical",
    "calculate_cluster_quality_metrics",
    "hierarchical_cluster_labels",
    # Outlier detection functions
    "detect_outliers_mahalanobis",
    "detect_outliers_pca",
    "detect_outliers_isolation_forest",
    "detect_outliers_kmeans",
    "detect_outliers_gmm",
    "detect_outliers_hierarchical",
    "calculate_outlier_threshold",
    "identify_outliers_from_distances",
    "remove_outliers_from_data",
    "combine_outlier_methods",
    # Outlier-removal entry point (#165)
    "remove_outlier_samples",
    # Visualization functions
    "create_trait_histograms",
    "create_trait_boxplots_by_genotype",
    "create_correlation_heatmap",
    "save_figure_with_unique_name",
    "create_exploratory_summary_plots",
    "create_trait_eda_plots",
    "create_heritability_plot",
    "create_heritability_threshold_plot",
    "create_variance_decomposition_plot",
    "create_trait_by_genotype_boxplots",
    "create_heritability_diagnostic_dashboard",
    "identify_extreme_samples_in_pc_space",
    "create_pca_scree_plot",
    "create_feature_contribution_plot",
    "create_pca_biplot",
    "create_umap_colored_by_top_traits",
    "create_umap_single_trait",
    "identify_extreme_genotypes_by_pc",
    "create_pc_genotype_boxplots",
    "create_feature_contribution_heatmap",
    "create_publication_figure",
    "identify_extreme_phenotypes",
    "create_phenotype_variation_plot",
    "create_genotype_image_grid",
    "create_regression_plot",
    # Cluster visualization functions
    "create_cluster_scatter_pca",
    "create_distance_distribution_plot",
    "create_cluster_size_barplot",
    "create_bic_aic_comparison_plot",
    "create_silhouette_plot",
    "create_dendrogram",
    # Outlier visualization functions
    "create_isolation_forest_plots",
    "create_outlier_overlap_heatmap",
    "create_outliers_per_genotype_plot",
    "create_mahalanobis_outlier_plots",
    "create_pca_outlier_plot",
    "create_comprehensive_outlier_comparison",
    "create_kmeans_outlier_plots",
    "create_gmm_outlier_plots",
    "create_hierarchical_outlier_plots",
    "plot_outlier_analysis",
    "select_outlier_figures",
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
    # Root core analysis functions
    "create_sample_identifier",
    "validate_unique_identifiers",
    "melt_depth_data",
    "aggregate_by_replicate",
    # Depth profile visualization functions
    "plot_depth_profile_faceted",
    "plot_depth_profile_replicates",
    # Statistics / heritability functions
    "calculate_trait_statistics",
    "perform_anova_by_genotype",
    "calculate_heritability_estimates",
    "identify_high_heritability_traits",
    "analyze_heritability_thresholds",
    "analyze_trait_variance",
    "diagnose_heritability_issues",
    "compare_trait_heritabilities",
    # PC-correlation & trait-enrichment workflows
    "cross_platform_pc_correlations",
    "trait_correlation_enrichment",
    "CrossPlatformPCResult",
    "EnrichmentResult",
]
