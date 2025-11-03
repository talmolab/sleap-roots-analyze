"""Test fixtures for visualization pipeline steps (Steps 9-11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline.core import StepResult


# ============================================================================
# MOCK DATA FIXTURES - Simulated analysis results
# ============================================================================


@pytest.fixture
def mock_pca_results():
    """Create mock PCA results for visualization testing.

    Returns a dictionary mimicking the structure returned by perform_pca_analysis.
    """
    n_samples = 50
    n_features = 6
    n_components = 3

    # Create mock PC scores
    pc_scores = pd.DataFrame(
        np.random.randn(n_samples, n_components),
        columns=[f"PC{i+1}" for i in range(n_components)],
    )

    # Create mock loadings
    loadings = np.random.randn(n_features, n_components)

    # Create transformed data (same as pc_scores for compatibility)
    transformed_data = pc_scores.to_numpy()

    return {
        "pca": None,  # sklearn PCA object (not needed for visualization)
        "pc_scores": pc_scores,
        "transformed_data": transformed_data,  # For biplot compatibility
        "loadings": loadings,
        "explained_variance": np.array([3.5, 1.8, 0.9]),
        "explained_variance_ratio": np.array([0.45, 0.30, 0.15]),
        "cumulative_variance_ratio": np.array([0.45, 0.75, 0.90]),
        "eigenvalues": np.array([3.5, 1.8, 0.9]),  # Same as explained_variance
        "feature_names": [f"trait{i+1}" for i in range(n_features)],
        "n_components": n_components,
        "total_variance_explained": 0.90,
    }


@pytest.fixture
def mock_heritability_results():
    """Create mock heritability results for visualization testing.

    Returns a dictionary with H² estimates and standard errors.
    """
    return {
        "trait1": {"H2": 0.75, "SE": 0.05, "p_value": 0.001},
        "trait2": {"H2": 0.62, "SE": 0.08, "p_value": 0.01},
        "trait3": {"H2": 0.48, "SE": 0.10, "p_value": 0.05},
        "trait4": {"H2": 0.82, "SE": 0.04, "p_value": 0.0001},
        "trait5": {"H2": 0.35, "SE": 0.12, "p_value": 0.15},
    }


@pytest.fixture
def mock_umap_results():
    """Create mock UMAP results for visualization testing."""
    n_samples = 50

    return {
        "umap_embedding": np.random.randn(n_samples, 2),
        "n_neighbors": 15,
        "min_dist": 0.1,
        "metric": "euclidean",
    }


@pytest.fixture
def sample_trait_data():
    """Create sample trait data for visualization testing.

    Returns a DataFrame with metadata columns and trait columns.
    """
    np.random.seed(42)
    n_samples = 50
    n_genotypes = 5

    return pd.DataFrame(
        {
            "Barcode": [f"sample_{i}" for i in range(n_samples)],
            "geno": [f"geno_{i%n_genotypes}" for i in range(n_samples)],
            "rep": [i % 3 + 1 for i in range(n_samples)],
            "trait1": np.random.randn(n_samples) * 10 + 50,
            "trait2": np.random.randn(n_samples) * 5 + 20,
            "trait3": np.random.randn(n_samples) * 2 + 100,
            "trait4": np.random.randn(n_samples) * 15 + 75,
            "trait5": np.random.randn(n_samples) * 8 + 30,
            "trait6": np.random.randn(n_samples) * 3 + 60,
        }
    )


# ============================================================================
# CONFIGURATION FIXTURES - Test configs for visualization steps
# ============================================================================


@pytest.fixture
def static_viz_config_enabled():
    """QCPipelineConfig with static visualization enabled."""
    from sleap_roots_analyze.pipeline.config import (
        QCPipelineConfig,
        StaticVisualizationConfig,
    )

    config = QCPipelineConfig(pipeline_name="test_static_viz")
    config.data.csv_path = "dummy.csv"
    config.static_viz = StaticVisualizationConfig(
        enabled=True,
        formats=["png"],
        dpi=100,
        create_pca_plots=True,
        create_umap_plots=False,
        create_cluster_plots=False,
        create_trait_distributions=True,
        create_trait_correlations=True,
        create_heritability_plots=True,
        create_genotype_comparisons=True,
    )
    return config


@pytest.fixture
def static_viz_config_disabled():
    """QCPipelineConfig with static visualization disabled."""
    from sleap_roots_analyze.pipeline.config import (
        QCPipelineConfig,
        StaticVisualizationConfig,
    )

    config = QCPipelineConfig(pipeline_name="test_static_viz_disabled")
    config.data.csv_path = "dummy.csv"
    config.static_viz = StaticVisualizationConfig(enabled=False)
    return config


@pytest.fixture
def static_viz_config_multiformat():
    """QCPipelineConfig with multiple output formats."""
    from sleap_roots_analyze.pipeline.config import (
        QCPipelineConfig,
        StaticVisualizationConfig,
    )

    config = QCPipelineConfig(pipeline_name="test_static_viz_multiformat")
    config.data.csv_path = "dummy.csv"
    config.static_viz = StaticVisualizationConfig(
        enabled=True,
        formats=["png", "pdf", "svg"],
        dpi=300,
        create_pca_plots=True,
        create_trait_distributions=True,
        create_trait_correlations=True,
    )
    return config


@pytest.fixture
def interactive_viz_config_enabled():
    """QCPipelineConfig with interactive visualization enabled."""
    from sleap_roots_analyze.pipeline.config import (
        QCPipelineConfig,
        InteractiveVisualizationConfig,
    )

    config = QCPipelineConfig(pipeline_name="test_interactive_viz")
    config.data.csv_path = "dummy.csv"
    config.interactive_viz = InteractiveVisualizationConfig(
        enabled=True,
        create_pca_plots=True,
        create_umap_plots=True,
        show_images_on_hover=False,
    )
    return config


@pytest.fixture
def interactive_viz_config_disabled():
    """QCPipelineConfig with interactive visualization disabled."""
    from sleap_roots_analyze.pipeline.config import QCPipelineConfig

    config = QCPipelineConfig(pipeline_name="test_interactive_viz_disabled")
    config.data.csv_path = "dummy.csv"
    config.interactive_viz.enabled = False
    return config


@pytest.fixture
def dashboard_config_enabled():
    """QCPipelineConfig with dashboard generation enabled."""
    from sleap_roots_analyze.pipeline.config import (
        QCPipelineConfig,
        DashboardConfig,
    )

    config = QCPipelineConfig(pipeline_name="test_dashboard")
    config.data.csv_path = "dummy.csv"
    config.dashboard = DashboardConfig(
        enabled=True,
        include_summary=True,
        include_figures=True,
        include_file_list=True,
    )
    return config


@pytest.fixture
def dashboard_config_disabled():
    """QCPipelineConfig with dashboard generation disabled."""
    from sleap_roots_analyze.pipeline.config import QCPipelineConfig

    config = QCPipelineConfig(pipeline_name="test_dashboard_disabled")
    config.data.csv_path = "dummy.csv"
    config.dashboard.enabled = False
    return config


# ============================================================================
# STEP RESULT FIXTURES - Pre-configured results for testing
# ============================================================================


@pytest.fixture
def prev_result_with_pca(sample_trait_data, mock_pca_results):
    """StepResult with PCA results for testing visualization steps."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]

    return StepResult(
        data=sample_trait_data,
        metadata={
            "trait_names": trait_cols,
            "valid_trait_names": trait_cols,
            "trait_cols": trait_cols,
            "metadata_cols": ["Barcode", "geno", "rep"],
            "pca_results": mock_pca_results,
        },
    )


@pytest.fixture
def prev_result_with_heritability(sample_trait_data, mock_heritability_results):
    """StepResult with heritability results for testing visualization steps."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5"]

    return StepResult(
        data=sample_trait_data,
        metadata={
            "trait_names": trait_cols,
            "valid_trait_names": trait_cols,
            "heritability_results": mock_heritability_results,
        },
    )


@pytest.fixture
def prev_result_with_all_viz_data(
    sample_trait_data, mock_pca_results, mock_heritability_results, mock_umap_results
):
    """StepResult with all visualization data (PCA, heritability, UMAP)."""
    trait_cols = ["trait1", "trait2", "trait3", "trait4", "trait5", "trait6"]

    return StepResult(
        data=sample_trait_data,
        metadata={
            "trait_names": trait_cols,
            "valid_trait_names": trait_cols,
            "trait_cols": trait_cols,
            "metadata_cols": ["Barcode", "geno", "rep"],
            "pca_results": mock_pca_results,
            "heritability_results": mock_heritability_results,
            "umap_results": mock_umap_results,
        },
    )


@pytest.fixture
def prev_result_minimal(sample_trait_data):
    """Minimal StepResult without PCA/heritability for testing basic visualization."""
    trait_cols = ["trait1", "trait2", "trait3"]

    return StepResult(
        data=sample_trait_data,
        metadata={
            "trait_names": trait_cols,
            "valid_trait_names": trait_cols,
        },
    )


@pytest.fixture
def prev_result_with_generated_figures(sample_trait_data, tmp_path):
    """StepResult with mock generated figure paths for dashboard testing."""
    # Create mock figure files
    static_dir = tmp_path / "static_figures"
    static_dir.mkdir()
    interactive_dir = tmp_path / "interactive_figures"
    interactive_dir.mkdir()

    # Create dummy static files
    static_files = [
        static_dir / "pca_scree_plot.png",
        static_dir / "pca_biplot.png",
        static_dir / "trait_correlations.png",
        static_dir / "trait_histograms_batch1.png",
    ]
    for f in static_files:
        f.write_text("mock figure")

    # Create dummy interactive files
    interactive_files = [
        interactive_dir / "interactive_pca.html",
        interactive_dir / "interactive_umap.html",
    ]
    for f in interactive_files:
        f.write_text("<html><body>mock plotly figure</body></html>")

    return StepResult(
        data=sample_trait_data,
        metadata={
            "trait_names": ["trait1", "trait2", "trait3"],
            "static_figures": static_files,
            "interactive_figures": interactive_files,
        },
    )


# ============================================================================
# HELPER FUNCTIONS - Utilities for testing
# ============================================================================


def count_files_by_extension(directory: Path, ext: str) -> int:
    """Count files with given extension in directory.

    Args:
        directory: Directory to search.
        ext: File extension (without dot, e.g., 'png').

    Returns:
        Number of files with the extension.
    """
    return len(list(directory.glob(f"**/*.{ext}")))


def verify_html_structure(html_path: Path) -> bool:
    """Basic HTML validation.

    Args:
        html_path: Path to HTML file.

    Returns:
        True if file contains basic HTML structure.
    """
    content = html_path.read_text()
    return (
        "<html>" in content.lower()
        and "</html>" in content.lower()
        and "<body>" in content.lower()
    )


def setup_matplotlib_backend():
    """Set matplotlib to non-interactive backend for testing."""
    import matplotlib

    matplotlib.use("Agg")
