"""Reusable configuration components for all pipelines.

This module contains all configuration dataclasses that can be composed by different
pipelines. These are building blocks that pipelines use to create their configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class AdaptiveSizingConfig:
    """Adaptive figure sizing configuration.

    Attributes:
        enabled: Whether to use adaptive sizing.
        base_width: Base width for single-column figures.
        base_height: Base height for single-row figures.
        width_per_item: Width increment per additional column/item.
        height_per_item: Height increment per additional row/item.
        min_width: Minimum figure width.
        max_width: Maximum figure width.
        min_height: Minimum figure height.
        max_height: Maximum figure height.
    """

    enabled: bool = True
    base_width: float = 8.0
    base_height: float = 6.0
    width_per_item: float = 2.0
    height_per_item: float = 2.0
    min_width: float = 6.0
    max_width: float = 20.0
    min_height: float = 4.0
    max_height: float = 16.0


@dataclass
class CleanupConfig:
    """Data cleanup filters configuration.

    Attributes:
        max_nan_fraction: Maximum fraction of NaN values allowed per sample.
        max_zeros_per_trait: Maximum fraction of zero values allowed per trait.
        max_nans_per_trait: Maximum fraction of NaN values allowed per trait.
        min_samples_per_trait: Minimum number of valid samples required per trait.
    """

    max_nan_fraction: float = 0.0
    max_zeros_per_trait: float = 0.5
    max_nans_per_trait: float = 0.2
    min_samples_per_trait: int = 10


@dataclass
class ClusteringConfig:
    """Clustering analysis configuration.

    Attributes:
        enabled: Whether to perform clustering.
        methods: List of clustering methods (kmeans, gmm, hierarchical).
        n_clusters: Number of clusters (None = auto-optimize).
        auto_optimize: Whether to automatically optimize number of clusters.
        min_clusters: Minimum clusters for auto-optimization.
        max_clusters: Maximum clusters for auto-optimization.
    """

    enabled: bool = False
    methods: List[str] = field(default_factory=lambda: ["kmeans"])
    n_clusters: Optional[int] = None
    auto_optimize: bool = True
    min_clusters: int = 2
    max_clusters: int = 10


@dataclass
class ColumnConfig:
    """Column name mappings configuration.

    Attributes:
        barcode: Name of the barcode/plant ID column.
        genotype: Name of the genotype column.
        replicate: Name of the replicate column (None if not present).
        image_path: Name of the image path column (optional).
    """

    barcode: str = "Barcode"
    genotype: str = "geno"
    replicate: Optional[str] = "rep"
    image_path: Optional[str] = "image_path"


@dataclass
class DashboardConfig:
    """Dashboard generation configuration.

    Attributes:
        enabled: Whether to generate dashboards.
        create_summary_dashboard: Whether to create summary dashboard.
        create_trait_dashboard: Whether to create per-trait dashboards.
    """

    enabled: bool = False
    create_summary_dashboard: bool = True
    create_trait_dashboard: bool = False


@dataclass
class DataConfig:
    """Data loading and processing configuration.

    Attributes:
        csv_path: Path to trait CSV file. Can be None for root core mode (auto-filled).
        image_dir: Directory containing images (optional).
        output_dir: Directory for output files.
        additional_exclude_cols: Additional columns to exclude from analysis.
        traits_to_include: List of trait names to include. If None, includes all.
        traits_to_exclude: List of trait names to exclude.
    """

    csv_path: str | None = MISSING
    image_dir: Optional[str] = None
    output_dir: str = "./outputs"
    additional_exclude_cols: Optional[List[str]] = None
    traits_to_include: Optional[List[str]] = None
    traits_to_exclude: List[str] = field(default_factory=list)


@dataclass
class GMMOutlierConfig:
    """GMM clustering outlier detection configuration.

    Attributes:
        n_components: Number of components (None = auto-select via BIC).
        max_components: Maximum components for auto-selection.
        percentile_threshold: Percentile threshold for outlier detection.
    """

    n_components: Optional[int] = None
    max_components: int = 5
    percentile_threshold: float = 99.0


@dataclass
class HeritabilityConfig:
    """Heritability analysis and filtering configuration.

    Attributes:
        enabled: Whether to filter traits by heritability.
        threshold: Minimum heritability (H²) threshold.
        generate_diagnostics: Whether to generate diagnostic plots and comparison CSV
            for removed traits. Only takes effect when enabled=True. Outputs include:
            - Comparison CSV with variance components for all traits
            - Variance decomposition plot showing H², variance components, and metrics
            - Boxplots of removed traits by genotype (limited to top 10 if >10 removed)
    """

    enabled: bool = True
    threshold: float = 0.60
    generate_diagnostics: bool = False


@dataclass
class HierarchicalOutlierConfig:
    """Hierarchical clustering outlier detection configuration.

    Attributes:
        n_clusters: Number of clusters (None = auto-optimize).
        linkage_method: Linkage method for hierarchical clustering.
        distance_threshold: Distance threshold for outlier detection.
    """

    n_clusters: Optional[int] = None
    linkage_method: str = "ward"
    distance_threshold: float = 2.0


@dataclass
class InteractiveVisualizationConfig:
    """Interactive visualization generation configuration.

    Attributes:
        enabled: Whether to generate interactive plots.
        create_pca_plots: Whether to create interactive PCA plots.
        create_umap_plots: Whether to create interactive UMAP plots.
        create_cluster_plots: Whether to create interactive clustering plots.
        show_images_on_hover: Whether to show images on hover.
    """

    enabled: bool = True
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    show_images_on_hover: bool = True


@dataclass
class InterestingGenotypesConfig:
    """Interesting genotypes identification configuration.

    Attributes:
        enabled: Whether to identify interesting genotypes.
        methods: List of methods (pc_extreme, trait_extreme, heritable_extreme).
        pc_threshold: Z-score threshold for PC-based extremes.
        trait_percentile: Percentile threshold for trait-based extremes.
        min_heritability: Minimum heritability for heritable extremes.
        max_genotypes: Maximum number of genotypes per method.
        generate_image_grids: Whether to generate image grids.
        images_per_genotype: Number of images per genotype.
    """

    enabled: bool = True
    methods: List[str] = field(
        default_factory=lambda: ["pc_extreme", "trait_extreme", "heritable_extreme"]
    )
    pc_threshold: float = 2.0
    trait_percentile: float = 95.0
    min_heritability: float = 0.60
    max_genotypes: int = 10
    generate_image_grids: bool = True
    images_per_genotype: int = 9


@dataclass
class IsolationForestConfig:
    """Isolation Forest outlier detection configuration.

    Attributes:
        contamination: Expected proportion of outliers.
    """

    contamination: float = 0.1


@dataclass
class KMeansOutlierConfig:
    """K-Means clustering outlier detection configuration.

    Attributes:
        n_clusters: Number of clusters (None = auto-optimize).
        max_clusters: Maximum clusters for auto-optimization.
        distance_threshold: Distance threshold for outlier detection.
    """

    n_clusters: Optional[int] = None
    max_clusters: int = 10
    distance_threshold: float = 2.0


@dataclass
class LoggingConfig:
    """Logging configuration.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to log to a file.
        log_file: Path to log file.
    """

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "pipeline.log"


@dataclass
class MahalanobisConfig:
    """Mahalanobis distance outlier detection configuration.

    Attributes:
        variance_threshold: Variance threshold for PCA.
        use_chi_squared: Whether to use chi-squared distribution threshold.
        chi2_percentile: Chi-squared percentile for threshold.
    """

    variance_threshold: float = 0.95
    use_chi_squared: bool = True
    chi2_percentile: float = 99.0


@dataclass
class OutlierDetectionConfig:
    """Outlier detection configuration.

    Attributes:
        traditional_methods: Traditional methods (pca, isolation_forest, mahalanobis).
        clustering_methods: Clustering methods (kmeans, gmm, hierarchical).
        pca: PCA method parameters.
        isolation_forest: Isolation Forest parameters.
        mahalanobis: Mahalanobis distance parameters.
        kmeans: K-Means clustering parameters.
        gmm: GMM clustering parameters.
        hierarchical: Hierarchical clustering parameters.
    """

    traditional_methods: List[str] = field(default_factory=list)
    clustering_methods: List[str] = field(default_factory=list)
    pca: "PCAOutlierConfig" = field(default_factory=lambda: PCAOutlierConfig())
    isolation_forest: IsolationForestConfig = field(
        default_factory=IsolationForestConfig
    )
    mahalanobis: MahalanobisConfig = field(default_factory=MahalanobisConfig)
    kmeans: KMeansOutlierConfig = field(default_factory=KMeansOutlierConfig)
    gmm: GMMOutlierConfig = field(default_factory=GMMOutlierConfig)
    hierarchical: HierarchicalOutlierConfig = field(
        default_factory=HierarchicalOutlierConfig
    )


@dataclass
class OutlierRemovalConfig:
    """Outlier removal configuration.

    Attributes:
        strategy: Removal strategy (single, consensus, subset).
        method: Method name (required for "single" strategy).
        min_methods: Minimum methods required (for "subset" strategy).
    """

    strategy: str = "single"
    method: str = "mahalanobis"
    min_methods: int = 2


@dataclass
class PCAConfig:
    """PCA analysis configuration.

    Attributes:
        n_components: Number of components (or variance ratio if < 1).
        standardize: Whether to standardize data before PCA.
        feature_selection_strategy: Strategy for selecting top features.
        n_top_features: Number of top features to select per component.
    """

    n_components: float = 0.95
    standardize: bool = True
    feature_selection_strategy: str = "top_variance"
    n_top_features: int = 10


@dataclass
class PCAOutlierConfig:
    """PCA-based outlier detection configuration.

    Attributes:
        explained_variance: Variance threshold for PCA.
        threshold: Distance threshold for outlier detection.
    """

    explained_variance: float = 0.95
    threshold: float = 2.5


@dataclass
class StaticVisualizationConfig:
    """Static visualization generation configuration.

    Attributes:
        enabled: Whether to generate static plots.
        formats: List of output formats (png, pdf, svg).
        dpi: DPI for raster formats.
        create_pca_plots: Whether to create PCA plots.
        create_umap_plots: Whether to create UMAP plots.
        create_cluster_plots: Whether to create clustering plots.
        create_trait_distributions: Whether to create trait histograms.
        create_trait_correlations: Whether to create correlation plots.
        create_heritability_plots: Whether to create heritability plots.
        create_genotype_comparisons: Whether to create genotype comparison plots.
        pca_biplot_top_features: Number of top features to show in PCA biplot (default: 10).
        pca_heatmap_features: Number of features to show in PCA contribution heatmap (default: 20).
        pca_n_components: Number of principal components to show in PC boxplots (default: 3).
        histogram_batch_size: Number of traits per histogram figure (default: 9).
        boxplot_batch_size: Number of traits per boxplot figure (default: 6).
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        legend_fontsize: Font size for legend text.
        bbox_inches: Bounding box mode for savefig ("tight" or None).
        transparent: Whether to save with transparent background.
    """

    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    create_trait_distributions: bool = True
    create_trait_correlations: bool = True
    create_heritability_plots: bool = True
    create_genotype_comparisons: bool = True
    # Visualization parameters
    pca_biplot_top_features: int = 10
    pca_heatmap_features: int = 20
    pca_n_components: int = 3
    histogram_batch_size: int = 9
    boxplot_batch_size: int = 6
    # Font sizes
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    # Savefig parameters
    bbox_inches: Optional[str] = "tight"
    transparent: bool = False


@dataclass
class StatisticsConfig:
    """Statistical analysis configuration.

    Attributes:
        calculate_anova: Whether to calculate ANOVA.
        calculate_heritability: Whether to calculate heritability.
        alpha: Significance level for tests.
    """

    calculate_anova: bool = True
    calculate_heritability: bool = True
    alpha: float = 0.05


@dataclass
class SummaryConfig:
    """Summary report generation configuration.

    Attributes:
        enabled: Whether to generate summary report.
        formats: List of output formats (markdown, html, json).
    """

    enabled: bool = True
    formats: List[str] = field(default_factory=lambda: ["markdown", "json"])


@dataclass
class UMAPConfig:
    """UMAP analysis configuration.

    Attributes:
        enabled: Whether to perform UMAP analysis.
        n_neighbors: Number of neighbors for UMAP.
        min_dist: Minimum distance for UMAP.
        metric: Distance metric for UMAP.
        random_state: Random state for reproducibility.
    """

    enabled: bool = False
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42


@dataclass
class VisualizationConfig:
    """Visualization generation configuration (for QC pipeline).

    Attributes:
        create_pca_plots: Whether to create PCA plots.
        create_umap_plots: Whether to create UMAP plots.
        create_cluster_plots: Whether to create clustering plots.
        create_outlier_plots: Whether to create outlier detection plots.
        interactive: Whether to create interactive plots.
        dpi: DPI for static plots.
        figsize: Figure size for static plots (width, height).
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        legend_fontsize: Font size for legend text.
        figure_format: Output format for figures (png, pdf, svg, eps).
        bbox_inches: Bounding box mode for savefig ("tight" or None).
        facecolor: Figure face color (None = default).
        edgecolor: Figure edge color (None = default).
        transparent: Whether to save with transparent background.
    """

    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    create_outlier_plots: bool = True
    interactive: bool = False
    dpi: int = 300
    figsize: tuple[int, int] = (10, 8)

    # Font sizes
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10

    # Figure format
    figure_format: str = "png"

    # Savefig parameters
    bbox_inches: Optional[str] = "tight"
    facecolor: Optional[str] = None
    edgecolor: Optional[str] = None
    transparent: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate font sizes are positive
        for name in [
            "title_fontsize",
            "label_fontsize",
            "tick_fontsize",
            "legend_fontsize",
        ]:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # Validate figure format
        valid_formats = {"png", "pdf", "svg", "eps"}
        if self.figure_format not in valid_formats:
            raise ValueError(
                f"figure_format must be one of {valid_formats}, got {self.figure_format}"
            )

        # Validate bbox_inches
        if self.bbox_inches not in [None, "tight"]:
            raise ValueError(
                f"bbox_inches must be 'tight' or None, got {self.bbox_inches}"
            )

        # Validate DPI is positive
        if self.dpi <= 0:
            raise ValueError(f"dpi must be positive, got {self.dpi}")


@dataclass
class RootCoreSourceConfig:
    """Configuration for a single root core data source.

    Attributes:
        csv_path: Path to root core CSV file.
        data_type: Type of data ("biomass" or "counting").
        depth_column_prefix: Prefix for wide-format column names (e.g., "RootDW_", "RootCount_").
        value_column_name: Name of value column in long format (default: "Value").
        aggregation_method: Method for aggregating cores ("mean", "median", or callable).
        depth_mapping: Manual depth mapping for biomass data {column_name: depth_cm}.
            Required for data_type="biomass", optional for data_type="counting" (auto-parsed).
        genotype_column: Name of genotype column in CSV (default: "geno"). If the CSV uses
            a different column name (e.g., "salk_geno"), specify it here and it will be
            renamed to "geno" for standardization across all sources.
    """

    csv_path: str = MISSING
    data_type: str = MISSING  # "biomass" or "counting"
    depth_column_prefix: str = MISSING
    value_column_name: str = "Value"
    aggregation_method: str = "mean"
    depth_mapping: Optional[dict] = None
    genotype_column: str = "geno"  # Column name for genotype, will be renamed to "geno"


@dataclass
class CoreQCConfig:
    """Configuration for core-level quality control.

    Attributes:
        enabled: Whether to perform core-level QC.
        outlier_method: Method for outlier detection ("mahalanobis").
        contamination: Expected proportion of outliers (0.0-0.5).
        max_missing_proportion: Maximum proportion of missing depths allowed per core.
        remove_outliers: Whether to remove flagged outliers before aggregation.
    """

    enabled: bool = True
    outlier_method: str = "mahalanobis"
    contamination: float = 0.1
    max_missing_proportion: float = 0.5
    remove_outliers: bool = True


@dataclass
class MergeTraitsConfig:
    """Configuration for merging above-ground traits with root data.

    Attributes:
        above_ground_csv: Path to above-ground trait CSV file.
        join_keys: Column names to use for merging (default: ["Plot", "Rep", "geno"]).
        join_type: Type of join ("inner", "left", "right", "outer").
        duplicate_strategy: How to handle duplicate columns ("fail", "skip", "suffix").
        output_path: Path for merged output CSV file.
    """

    above_ground_csv: str = MISSING
    join_keys: List[str] = field(default_factory=lambda: ["Plot", "Rep", "geno"])
    join_type: str = "inner"
    duplicate_strategy: str = "fail"  # "fail", "skip", or "suffix"
    output_path: str = "merged_traits.csv"


@dataclass
class RootCoreConfig:
    """Root core data processing configuration.

    Attributes:
        sources: List of root core data sources (biomass, counting, or both).
        core_qc: Configuration for core-level quality control.
        merge_traits: Configuration for merging with above-ground traits (optional).
    """

    sources: List[RootCoreSourceConfig] = field(default_factory=list)
    core_qc: CoreQCConfig = field(default_factory=CoreQCConfig)
    merge_traits: Optional[MergeTraitsConfig] = None
