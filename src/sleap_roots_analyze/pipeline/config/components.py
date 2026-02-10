"""Reusable configuration components for all pipelines.

This module contains all configuration dataclasses that can be composed by different
pipelines. These are building blocks that pipelines use to create their configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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

    IMPORTANT: All threshold parameters should be explicitly set in your config
    to avoid silent data removal. Default values are provided for convenience but
    may not match your analysis requirements.

    Attributes:
        max_nan_fraction: Max fraction of NaN values per sample (0.0-1.0).
            Samples exceeding this will be removed. Recommended: 0.25
        max_zeros_per_trait: Max fraction of zero values per trait (0.0-1.0).
            Traits exceeding this will be removed. Recommended: 0.5
        max_nans_per_trait: Max fraction of NaN values per trait (0.0-1.0).
            Traits exceeding this will be removed. Recommended: 0.2-0.3
        min_samples_per_trait: Minimum number of valid samples required per trait.
            Traits with fewer samples will be removed. Recommended: 10
        custom_replacements: Optional dict mapping old terms to new terms for
            domain-specific trait name terminology (e.g., {"crown": "seminal"} to
            replace "crown" with "seminal" in wheat trait names). Applied during
            trait name sanitization. Keys are matched case-insensitively.
    """

    max_nan_fraction: float = 0.0
    max_zeros_per_trait: float = 0.5
    max_nans_per_trait: float = 0.2
    min_samples_per_trait: int = 10
    custom_replacements: Optional[Dict[str, str]] = None


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

    IMPORTANT: If filtering is enabled, threshold should be explicitly set in your
    config to match your scientific objectives.

    Attributes:
        enabled: Whether to filter traits by heritability.
        threshold: Minimum heritability (H²) for trait retention (0.0-1.0).
            Typical range: 0.3 (permissive) to 0.6 (stringent).
            Must be explicitly set if enabled=True. Default: 0.60
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
        create_scatter_with_images: Whether to create scatter plot with image tooltips.
        create_image_viewer: Whether to create HTML image viewer with PCA overlay.
        create_image_gallery: Whether to create interactive image gallery.
    """

    enabled: bool = True
    create_pca_plots: bool = True
    create_umap_plots: bool = False
    create_cluster_plots: bool = False
    show_images_on_hover: bool = True
    create_scatter_with_images: bool = True
    create_image_viewer: bool = True
    create_image_gallery: bool = True


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
        create_phenotype_variation_plots: Whether to create phenotype variation plots
            showing box plots with jittered points for top heritable traits.
        phenotype_variation_top_n: Number of top heritable traits to generate
            phenotype variation plots for (default: 10).
        regression_trait_pairs: List of trait pairs [[x, y], ...] for regression plots.
            Each pair specifies the x and y trait names for a regression analysis.
            Empty list (default) means no regression plots are generated.
        create_genotype_image_grids: Whether to create genotype image grids for
            extreme genotypes (requires image paths in metadata).
        feature_contribution_variance_threshold: Variance threshold for determining
            number of PCs to show in feature contribution plot. When None (default),
            inherits from pca.n_components (if < 1) or uses 0.95.
        feature_contribution_top_n: Number of top features to show in the feature
            contribution bar chart (default: 20).
        pca_biplot_top_features: Number of top features to show in PCA biplot (default: 10).
        pca_heatmap_features: Number of features to show in PCA contribution heatmap (default: 20).
        pca_n_components: Number of principal components to show in PC boxplots (default: 3).
        histogram_batch_size: Number of traits per histogram figure (default: 9).
        boxplot_batch_size: Number of traits per boxplot figure (default: 6).
        genotypes_to_color: Optional list of genotype names to color distinctly in PCA plots.
            If provided, only these genotypes will be colored with distinct colors in PCA biplots
            and PC boxplots, while all other genotypes will be gray. If None (default), all
            genotypes are colored.
        highlight_genotypes: Optional list of genotype names to highlight with special formatting
            (larger points, edge colors in PCA biplot; bold labels, gold color in PC boxplots).
            Works independently of genotypes_to_color.
        title_fontsize: Font size for plot titles.
        label_fontsize: Font size for axis labels.
        tick_fontsize: Font size for tick labels.
        legend_fontsize: Font size for legend text.
        bbox_inches: Bounding box mode for savefig ("tight" or None).
        transparent: Whether to save with transparent background.
        save_pdf: Whether to generate PDF files alongside other formats (default: True).
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
    create_phenotype_variation_plots: bool = True
    phenotype_variation_top_n: int = 10
    regression_trait_pairs: List[List[str]] = field(default_factory=list)
    create_genotype_image_grids: bool = True
    # Feature contribution plot parameters
    feature_contribution_variance_threshold: Optional[float] = None
    feature_contribution_top_n: int = 20
    # Visualization parameters
    pca_biplot_top_features: int = 10
    pca_heatmap_features: int = 20
    pca_n_components: int = 3
    histogram_batch_size: int = 9
    boxplot_batch_size: int = 6
    # Genotype highlighting
    genotypes_to_color: Optional[List[str]] = None
    highlight_genotypes: Optional[List[str]] = None
    # Font sizes
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    legend_fontsize: int = 10
    # Savefig parameters
    bbox_inches: Optional[str] = "tight"
    transparent: bool = False
    # File output options
    save_pdf: bool = True  # Whether to generate PDF files alongside other formats


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
        enable_batched_plots: Whether to create batched plots for many traits.
        batched_plot_threshold: Trait count threshold for creating batches.
        batch_size: Number of traits per batch figure.
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

    # Batched plot configuration
    enable_batched_plots: bool = True
    batched_plot_threshold: int = 16  # Create batches when > this many traits
    batch_size: int = 16  # Traits per batch figure

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

    IMPORTANT: aggregation_method should be explicitly set in your config to match
    your statistical requirements, though a sensible default is provided.

    Attributes:
        csv_path: Path to root core CSV file.
        data_type: Type of data ("biomass" or "counting").
        depth_column_prefix: Prefix for wide-format column names (e.g., "RootDW_", "RootCount_").
        value_column_name: Name of value column in long format (default: "Value").
        aggregation_method: Method for aggregating cores ("mean" or "median").
            Recommended: "median" (robust to outliers, typos, measurement errors).
            Use "mean" only if you're confident data has no outliers.
            Default: "median"
        depth_mapping: Manual depth mapping for biomass data {column_name: depth_cm}.
            Required for data_type="biomass", optional for data_type="counting" (auto-parsed).
            Maps depth range labels to midpoint values (e.g., {"0-30": 15.0, "30-60": 45.0}).
        depth_range_mapping: Optional mapping from depth midpoints to range labels for display.
            Used to show actual measurement ranges in outputs instead of midpoint values.
            Example: {15.0: "0-30", 45.0: "30-60"} will display "Root Biomass DW (g) 0-30cm"
            instead of "Rootdw 15Cm". Improves scientific clarity in visualizations.
            Default: None (midpoint notation used)
        genotype_column: Name of genotype column in CSV (default: "geno"). If the CSV uses
            a different column name (e.g., "salk_geno"), specify it here and it will be
            renamed to "geno" for standardization across all sources.
    """

    csv_path: str = MISSING
    data_type: str = MISSING  # "biomass" or "counting"
    depth_column_prefix: str = MISSING
    value_column_name: str = "Value"
    aggregation_method: str = "median"  # Robust to outliers and measurement errors
    depth_mapping: Optional[dict] = None
    depth_range_mapping: Optional[dict] = None  # Maps midpoints to ranges for display
    genotype_column: str = "geno"  # Column name for genotype, will be renamed to "geno"


@dataclass
class CoreQCConfig:
    """Configuration for core-level quality control.

    This step performs measurement error detection (quality control), NOT statistical
    hypothesis testing. It uses per-group analysis to detect gross errors (damaged cores,
    typos, sampling failures) before aggregation.

    **Statistical Framework:**
    - QC paradigm: Fixed threshold (30% deviation) based on domain knowledge
    - Per-group detection: Respects nested structure (cores within plots)
    - Sample size: Works with N=3 (non-parametric, no variance estimation)
    - Independence: Analyzes each plot independently (satisfies i.i.d. assumption)

    **Why not population-level methods?**
    Cores within plots are NOT independent (share genotype, soil, spatial location).
    Population-level outlier detection (Mahalanobis, Isolation Forest) would violate
    independence assumptions and confound plot effects with measurement errors.

    **Missing Data Detection:**
    Flags cores with excessive missing depths (>50% NaN). This catches incomplete
    or damaged samples before aggregation.

    **Value Outlier Detection (Optional, Opt-In):**
    Flags cores with anomalous values using percent deviation from within-group median.
    Formula: |value - median| / median > threshold (e.g., 0.30 = 30%).

    Example: GH_7371 Rep 1 had cores [0.76g, 0.71g, 0.31g]. Core 2 (56% deviation)
    was a measurement error that destroyed heritability (H²: 0.27→0.50 after removal).

    **Publication-Quality Methods Section:**
    "Quality control of core-level data was performed within each plot-depth group
    (n=3 cores). Cores with biomass values deviating >30% from the within-group
    median were flagged as potential measurement errors and excluded prior to
    aggregation. This threshold was chosen conservatively to remove gross errors
    while preserving natural biological variation (typically <20% within plots)."

    Attributes:
        enabled: Whether to perform core-level QC.
        max_missing_proportion: Maximum proportion of missing depths per core (0-1).
                                Cores exceeding this are flagged.
        remove_outliers: Whether to remove flagged cores before aggregation.
        detect_value_outliers: Enable value-based outlier detection (opt-in).
                               Default: False (disabled for backward compatibility).
        max_deviation_from_median: Threshold for percent deviation (0-1).
                                   Cores exceeding this are flagged as value outliers.
                                   Default: 0.30 (30% - conservative, minimal false positives).
        min_cores_after_qc: Minimum cores to keep per group (safety mechanism).
                            Prevents removing all cores if all are flagged.
                            Default: 1 (always keep at least one core).
    """

    enabled: bool = False  # Default to disabled
    max_missing_proportion: float = 0.5  # Flag cores with >50% missing depths
    remove_outliers: bool = True  # Remove flagged cores if enabled
    detect_value_outliers: bool = False  # Opt-in: value-based outlier detection
    max_deviation_from_median: float = 0.30  # 30% threshold (conservative)
    min_cores_after_qc: int = (
        1  # Safety: keep at least 1 core per group  # Remove flagged cores if enabled
    )


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
        generate_depth_profiles: Whether to generate depth profile visualizations.
    """

    sources: List[RootCoreSourceConfig] = field(default_factory=list)
    core_qc: CoreQCConfig = field(default_factory=CoreQCConfig)
    merge_traits: Optional[MergeTraitsConfig] = None
    generate_depth_profiles: bool = True


@dataclass(frozen=True)
class CrossPlatformConfig:
    """Cross-platform analysis configuration.

    Configuration for comparing trait data across different experimental platforms
    (e.g., cylinder vs turface, field vs growth chamber).

    Attributes:
        exp1_data_path: Path to experiment 1 cleaned traits CSV.
        exp1_name: Display name for experiment 1 (e.g., "Cylinder").
        exp1_genotype_col: Column name containing genotype identifiers in experiment 1.
        exp2_data_path: Path to experiment 2 cleaned traits CSV.
        exp2_name: Display name for experiment 2 (e.g., "Turface").
        exp2_genotype_col: Column name containing genotype identifiers in experiment 2.
        correlation_method: Statistical correlation method ("spearman", "pearson", "kendall").
        min_samples_per_genotype: Minimum samples required per genotype for analysis.
        significance_level: P-value threshold for significance testing.
        fdr_correction_method: Method for multiple testing correction.
            - "fdr_bh": Benjamini-Hochberg (assumes test independence)
            - "fdr_by": Benjamini-Yekutieli (valid under arbitrary dependence) - DEFAULT
            - "none": No correction (use raw p-values)
        confidence_level: Confidence level for correlation coefficient intervals.
            Must be in (0, 1) exclusive range. Default 0.95 for 95% CI.
        min_genotypes_for_correlation: Minimum number of genotypes required for a valid
            correlation calculation. Trait pairs with fewer valid genotypes (after NaN
            removal) will be excluded from results. Default 10, which is recommended
            for accurate Fisher z-transformation approximation. Must be >= 3.
        power_analysis_alpha: Significance level for power analysis calculations.
            Must be in (0, 1) exclusive range. Default 0.05.
        power_analysis_power: Target statistical power for minimum detectable effect
            size calculation. Must be in (0, 1) exclusive range. Default 0.80.
        top_n_correlations: Number of top correlations to display in summary.
        top_n_joint_plots: Number of joint plots to generate for top correlations.
        top_n_boxplots: Number of boxplots to generate for top correlations.
        figsize_summary: Figure size for summary visualization (width, height).
        figsize_joint: Figure size for joint plots (width, height).
        figsize_boxplot: Figure size for boxplots (width, height).
        exp1_exclude_cols: List of column names to exclude from experiment 1 trait analysis.
        exp2_exclude_cols: List of column names to exclude from experiment 2 trait analysis.
        trait_reduction_method: Method for reducing trait redundancy before correlation.
            - "none": No reduction, use all traits (default)
            - "clustering": Hierarchical clustering of correlated traits
        trait_clustering_threshold: Minimum |r| for traits to be considered redundant.
            Must be in (0, 1] range. Default 0.8. Higher values = more stringent = more
            clusters = fewer traits removed.
        trait_clustering_linkage: Linkage method for hierarchical clustering.
            - "complete": Maximum distance between all pairs (default, most conservative)
            - "average": Average distance between all pairs
            - "single": Minimum distance between any pair (most aggressive)
    """

    # Required parameters
    exp1_data_path: str
    exp1_name: str
    exp1_genotype_col: str
    exp2_data_path: str
    exp2_name: str
    exp2_genotype_col: str

    # Optional parameters with defaults
    correlation_method: str = "spearman"
    min_samples_per_genotype: int = 3
    significance_level: float = 0.05
    fdr_correction_method: str = "fdr_by"
    confidence_level: float = 0.95
    min_genotypes_for_correlation: int = 10
    power_analysis_alpha: float = 0.05
    power_analysis_power: float = 0.80
    top_n_correlations: int = 20
    top_n_joint_plots: int = 6
    top_n_boxplots: int = 6
    figsize_summary: tuple = (14, 12)
    figsize_joint: tuple = (10, 10)
    figsize_boxplot: tuple = (14, 6)
    exp1_exclude_cols: Optional[List[str]] = None
    exp2_exclude_cols: Optional[List[str]] = None

    # Trait redundancy reduction parameters
    trait_reduction_method: str = "none"
    trait_reduction_target: Optional[str] = None  # "exp1", "exp2", or "both"
    trait_clustering_threshold: float = 0.8
    trait_clustering_linkage: str = "complete"

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate correlation method
        valid_methods = ["spearman", "pearson", "kendall"]
        if self.correlation_method not in valid_methods:
            raise ValueError(
                f"correlation_method must be one of {valid_methods}, "
                f"got '{self.correlation_method}'"
            )

        # Validate FDR correction method
        valid_fdr_methods = ["fdr_bh", "fdr_by", "none"]
        if self.fdr_correction_method not in valid_fdr_methods:
            raise ValueError(
                f"fdr_correction_method must be one of {valid_fdr_methods}, "
                f"got '{self.fdr_correction_method}'"
            )

        # Validate confidence level
        if not (0 < self.confidence_level < 1):
            raise ValueError(
                f"confidence_level must be in (0, 1) exclusive range, "
                f"got {self.confidence_level}"
            )

        # Validate min_genotypes_for_correlation (minimum 3 for valid correlation)
        if self.min_genotypes_for_correlation < 3:
            raise ValueError(
                f"min_genotypes_for_correlation must be >= 3, "
                f"got {self.min_genotypes_for_correlation}"
            )

        # Validate power_analysis_alpha
        if not (0 < self.power_analysis_alpha < 1):
            raise ValueError(
                f"power_analysis_alpha must be in (0, 1) exclusive range, "
                f"got {self.power_analysis_alpha}"
            )

        # Validate power_analysis_power
        if not (0 < self.power_analysis_power < 1):
            raise ValueError(
                f"power_analysis_power must be in (0, 1) exclusive range, "
                f"got {self.power_analysis_power}"
            )

        # Validate trait reduction method
        valid_reduction_methods = ["none", "clustering"]
        if self.trait_reduction_method not in valid_reduction_methods:
            raise ValueError(
                f"trait_reduction_method must be one of {valid_reduction_methods}, "
                f"got '{self.trait_reduction_method}'"
            )

        # Validate trait_reduction_target is required when clustering is enabled
        if self.trait_reduction_method == "clustering":
            if self.trait_reduction_target is None:
                raise ValueError(
                    "trait_reduction_target must be specified when "
                    "trait_reduction_method is 'clustering'"
                )
            valid_targets = ["exp1", "exp2", "both"]
            if self.trait_reduction_target not in valid_targets:
                raise ValueError(
                    f"trait_reduction_target must be one of {valid_targets}, "
                    f"got '{self.trait_reduction_target}'"
                )

        # Validate trait clustering threshold (must be in (0, 1] range)
        if not (0 < self.trait_clustering_threshold <= 1):
            raise ValueError(
                f"trait_clustering_threshold must be in (0, 1] range, "
                f"got {self.trait_clustering_threshold}"
            )

        # Validate trait clustering linkage
        valid_linkage_methods = ["complete", "average", "single"]
        if self.trait_clustering_linkage not in valid_linkage_methods:
            raise ValueError(
                f"trait_clustering_linkage must be one of {valid_linkage_methods}, "
                f"got '{self.trait_clustering_linkage}'"
            )
