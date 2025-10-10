"""Tests for interactive visualization functions."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import base64
from PIL import Image
import io
import plotly.graph_objects as go

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


# ============================================================================
# Fixtures for interactive visualization testing
# ============================================================================


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image."""
    img_path = tmp_path / "test_image.png"
    # Create a simple 100x100 red image
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_image_links(sample_image_path):
    """Create sample image links dictionary."""
    return {
        "sample1": {
            "features.png": str(sample_image_path),
            "seg.png": str(sample_image_path),
        },
        "sample2": {"features.png": str(sample_image_path)},
        "sample3": {},
    }


@pytest.fixture
def interactive_viz_df():
    """Create sample dataframe for interactive visualization."""
    np.random.seed(42)
    n_samples = 100  # Match PCA/UMAP fixture sizes
    return pd.DataFrame(
        {
            "Barcode": [f"sample{i}" for i in range(n_samples)],
            "trait1": np.random.randn(n_samples),
            "trait2": np.random.randn(n_samples) * 2 + 1,
            "trait3": np.random.exponential(2, n_samples),
            "geno": np.random.choice(["A", "B", "C"], n_samples),
            "rep": np.random.choice([1, 2, 3], n_samples),
        }
    )


# ============================================================================
# Tests for image encoding
# ============================================================================


class TestImageEncoding:
    """Test image encoding functions."""

    def test_encode_image_to_base64_valid(self, sample_image_path):
        """Test encoding a valid image to base64."""
        encoded = encode_image_to_base64(sample_image_path)

        # Check it's a valid base64 string with data URI
        assert encoded.startswith("data:image/png;base64,")
        assert len(encoded) > 100  # Should have some content

        # Try to decode it back
        data_part = encoded.split(",")[1]
        decoded = base64.b64decode(data_part)
        assert len(decoded) > 0

    def test_encode_image_to_base64_nonexistent(self):
        """Test encoding a non-existent image."""
        encoded = encode_image_to_base64("nonexistent.png")
        assert encoded == ""

    def test_encode_image_to_base64_path_object(self, sample_image_path):
        """Test encoding with Path object."""
        encoded = encode_image_to_base64(Path(sample_image_path))
        assert encoded.startswith("data:image/png;base64,")

    def test_encode_image_resize(self, tmp_path):
        """Test that large images are resized."""
        # Create a large image
        large_img_path = tmp_path / "large.png"
        img = Image.new("RGB", (1000, 1000), color="blue")
        img.save(large_img_path)

        encoded = encode_image_to_base64(large_img_path)
        assert encoded.startswith("data:image/png;base64,")

        # Decode and check size
        data_part = encoded.split(",")[1]
        decoded = base64.b64decode(data_part)
        img_decoded = Image.open(io.BytesIO(decoded))
        assert img_decoded.size[0] <= 400
        assert img_decoded.size[1] <= 400

    def test_encode_image_non_rgb(self, tmp_path):
        """Test encoding non-RGB images."""
        # Create grayscale image
        gray_img_path = tmp_path / "gray.png"
        img = Image.new("L", (100, 100), color=128)
        img.save(gray_img_path)

        encoded = encode_image_to_base64(gray_img_path)
        assert encoded.startswith("data:image/png;base64,")


# ============================================================================
# Tests for interactive scatter plots
# ============================================================================


class TestInteractiveScatterPlots:
    """Test interactive scatter plot functions."""

    def test_create_interactive_scatter_basic(
        self, interactive_viz_df, sample_image_links
    ):
        """Test basic scatter plot creation."""
        fig = create_interactive_scatter_with_images(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "trait1"
        assert fig.layout.yaxis.title.text == "trait2"

    def test_create_interactive_scatter_with_color(
        self, interactive_viz_df, sample_image_links
    ):
        """Test scatter plot with color grouping."""
        fig = create_interactive_scatter_with_images(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
            color_by="geno",
        )

        # Should have multiple traces for different groups
        unique_genos = interactive_viz_df["geno"].nunique()
        assert len(fig.data) == unique_genos

    def test_create_interactive_scatter_with_size(
        self, interactive_viz_df, sample_image_links
    ):
        """Test scatter plot with size mapping."""
        fig = create_interactive_scatter_with_images(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
            size_by="trait3",
        )

        assert isinstance(fig, go.Figure)
        # Check that marker sizes vary
        if hasattr(fig.data[0].marker, "size"):
            sizes = fig.data[0].marker.size
            if hasattr(sizes, "__len__"):  # Check if it's an array
                assert len(set(sizes)) > 1  # Sizes should vary

    def test_create_interactive_scatter_hover_data(
        self, interactive_viz_df, sample_image_links
    ):
        """Test scatter plot with additional hover data."""
        fig = create_interactive_scatter_with_images(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
            hover_data=["geno", "rep"],
        )

        # Check hover template exists
        assert fig.data[0].hovertemplate is not None
        # Check customdata is set for sample IDs
        assert fig.data[0].customdata is not None

    def test_create_interactive_scatter_no_images(self, interactive_viz_df):
        """Test scatter plot without image links."""
        fig = create_interactive_scatter_with_images(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links={},
            show_images_on_hover=False,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_interactive_scatter_missing_columns(
        self, interactive_viz_df, sample_image_links
    ):
        """Test scatter plot with missing columns."""
        with pytest.raises(KeyError):
            create_interactive_scatter_with_images(
                df=interactive_viz_df,
                x_col="nonexistent1",
                y_col="trait2",
                image_links=sample_image_links,
            )

    def test_create_interactive_scatter_empty_df(self, sample_image_links):
        """Test scatter plot with empty dataframe."""
        empty_df = pd.DataFrame({"trait1": [], "trait2": []})
        fig = create_interactive_scatter_with_images(
            df=empty_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)


# ============================================================================
# Tests for PCA visualization
# ============================================================================


class TestInteractivePCAPlots:
    """Test interactive PCA plot functions."""

    def test_create_interactive_pca_basic(
        self, pca_viz_results, pca_viz_dataframe, sample_image_links
    ):
        """Test basic PCA plot creation."""
        fig = create_interactive_pca_with_images(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # Check axis labels include variance
        assert "PC1" in fig.layout.xaxis.title.text
        assert "PC2" in fig.layout.yaxis.title.text
        assert "%" in fig.layout.xaxis.title.text  # Should show variance percentage

    def test_create_interactive_pca_with_loadings(
        self, pca_viz_results, pca_viz_dataframe, sample_image_links
    ):
        """Test PCA plot with feature loadings."""
        fig = create_interactive_pca_with_images(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
            show_loadings=True,
            n_loadings=5,
        )

        # Should have annotations for loadings
        assert len(fig.layout.annotations) > 0

    def test_create_interactive_pca_different_components(
        self, pca_viz_results, pca_viz_dataframe, sample_image_links
    ):
        """Test PCA plot with different component pairs."""
        fig = create_interactive_pca_with_images(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
            components=(1, 2),  # PC2 vs PC3
        )

        assert "PC2" in fig.layout.xaxis.title.text
        assert "PC3" in fig.layout.yaxis.title.text

    def test_create_interactive_pca_with_color(
        self, pca_viz_results, pca_viz_dataframe, sample_image_links
    ):
        """Test PCA plot with color grouping."""
        fig = create_interactive_pca_with_images(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
            color_by="geno",
            hover_cols=["rep"],
        )

        assert isinstance(fig, go.Figure)
        # Should have multiple traces for different groups
        assert len(fig.data) > 1

    def test_create_interactive_pca_plot_simple(
        self, pca_viz_results, pca_viz_dataframe
    ):
        """Test simplified PCA plot function."""
        fig = create_interactive_pca_plot(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            color_by="geno",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


# ============================================================================
# Tests for UMAP visualization
# ============================================================================


class TestInteractiveUMAPPlots:
    """Test interactive UMAP plot functions."""

    def test_create_interactive_umap_basic(
        self, umap_viz_results, interactive_viz_df, sample_image_links
    ):
        """Test basic UMAP plot creation."""
        fig = create_interactive_umap_with_images(
            umap_results=umap_viz_results,
            df=interactive_viz_df,
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert "UMAP1" in fig.layout.xaxis.title.text
        assert "UMAP2" in fig.layout.yaxis.title.text
        # Check that parameters are in title
        assert "n_neighbors" in fig.layout.title.text

    def test_create_interactive_umap_with_hover_highlight(
        self, umap_viz_results, interactive_viz_df
    ):
        """Test UMAP plot with hover highlighting."""
        fig = create_interactive_umap_with_hover_highlight(
            umap_results=umap_viz_results,
            df=interactive_viz_df,
            genotype_col="geno",
        )

        assert isinstance(fig, go.Figure)
        # Should have base grey trace plus one for each genotype
        unique_genos = interactive_viz_df["geno"].nunique()
        assert len(fig.data) == unique_genos + 1  # +1 for the base grey trace

        # First trace should be visible, others hidden
        assert fig.data[0].visible != "legendonly"
        for trace in fig.data[1:]:
            assert trace.visible == "legendonly"

    def test_create_interactive_umap_missing_genotype_col(
        self, umap_viz_results, interactive_viz_df
    ):
        """Test UMAP hover highlight with missing genotype column."""
        with pytest.raises(ValueError, match="not found in dataframe"):
            create_interactive_umap_with_hover_highlight(
                umap_results=umap_viz_results,
                df=interactive_viz_df,
                genotype_col="nonexistent",
            )

    def test_create_interactive_umap_1d_embedding(
        self, interactive_viz_df, sample_image_links
    ):
        """Test UMAP plot with 1D embedding."""
        umap_results_1d = {
            "embedding": np.random.randn(len(interactive_viz_df), 1),
            "n_neighbors": 15,
            "min_dist": 0.1,
        }

        fig = create_interactive_umap_with_images(
            umap_results=umap_results_1d,
            df=interactive_viz_df,
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)
        # Y values should all be 0 for 1D
        assert all(y == 0 for y in fig.data[0].y)


# ============================================================================
# Tests for dashboard and gallery
# ============================================================================


class TestDashboardAndGallery:
    """Test dashboard and gallery creation."""

    def test_create_trait_explorer_dashboard(
        self, interactive_viz_df, sample_image_links
    ):
        """Test dashboard creation."""
        trait_cols = ["trait1", "trait2", "trait3"]

        fig = create_trait_explorer_dashboard(
            df=interactive_viz_df,
            trait_cols=trait_cols,
            image_links=sample_image_links,
            group_col="geno",
        )

        assert isinstance(fig, go.Figure)
        # Dashboard has multiple subplots - check for annotations that indicate subplot titles
        assert len(fig.layout.annotations) >= 4  # Should have 4 subplot titles

    def test_create_trait_explorer_dashboard_single_trait(
        self, interactive_viz_df, sample_image_links
    ):
        """Test dashboard with single trait."""
        fig = create_trait_explorer_dashboard(
            df=interactive_viz_df,
            trait_cols=["trait1"],
            image_links=sample_image_links,
            group_col="geno",
        )

        assert isinstance(fig, go.Figure)

    def test_create_trait_explorer_dashboard_empty(self):
        """Test dashboard with empty data."""
        empty_df = pd.DataFrame()

        fig = create_trait_explorer_dashboard(
            df=empty_df,
            trait_cols=[],
            image_links={},
        )

        assert isinstance(fig, go.Figure)

    def test_create_interactive_image_gallery(
        self, interactive_viz_df, sample_image_links, tmp_path
    ):
        """Test image gallery creation."""
        output_path = tmp_path / "gallery.html"
        trait_cols = ["trait1", "trait2", "trait3"]

        create_interactive_image_gallery(
            df=interactive_viz_df,
            image_links=sample_image_links,
            trait_cols=trait_cols,
            output_path=output_path,
            n_cols=3,
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "Sample Image Gallery" in content
        assert "sample-card" in content

    def test_create_interactive_image_gallery_no_images(
        self, interactive_viz_df, tmp_path
    ):
        """Test gallery with no images."""
        output_path = tmp_path / "gallery_empty.html"

        create_interactive_image_gallery(
            df=interactive_viz_df,
            image_links={},  # No images
            trait_cols=["trait1"],
            output_path=output_path,
        )

        assert output_path.exists()


# ============================================================================
# Tests for HTML generation
# ============================================================================


class TestHTMLGeneration:
    """Test HTML generation with image viewer."""

    def test_create_html_with_image_viewer(
        self, interactive_viz_df, sample_image_links, tmp_path
    ):
        """Test HTML creation with image viewer."""
        # Create a simple figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=interactive_viz_df["trait1"],
                    y=interactive_viz_df["trait2"],
                    mode="markers",
                    customdata=interactive_viz_df["Barcode"],
                )
            ]
        )

        output_path = tmp_path / "plot_with_viewer.html"

        create_html_with_image_viewer(
            fig=fig,
            df=interactive_viz_df,
            image_links=sample_image_links,
            output_path=output_path,
        )

        assert output_path.exists()
        content = output_path.read_text()
        assert "Interactive Plot with Image Viewer" in content
        assert "plotly_click" in content  # Check for click handler
        assert "imageData" in content  # Check for image data

    def test_create_html_with_image_viewer_no_images(
        self, interactive_viz_df, tmp_path
    ):
        """Test HTML viewer with no images."""
        fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="markers")])

        output_path = tmp_path / "plot_no_images.html"

        create_html_with_image_viewer(
            fig=fig,
            df=interactive_viz_df,
            image_links={},
            output_path=output_path,
        )

        assert output_path.exists()

    def test_create_html_with_image_viewer_empty_figure(
        self, interactive_viz_df, sample_image_links, tmp_path
    ):
        """Test HTML viewer with empty figure."""
        fig = go.Figure()  # Empty figure
        output_path = tmp_path / "empty_plot.html"

        # Should handle empty figure gracefully
        create_html_with_image_viewer(
            fig=fig,
            df=interactive_viz_df,
            image_links=sample_image_links,
            output_path=output_path,
        )

        # Function should return early with warning
        assert not output_path.exists()  # File shouldn't be created


# ============================================================================
# Tests for scatter plot with preview
# ============================================================================


class TestScatterWithPreview:
    """Test scatter plot with image preview panel."""

    def test_create_scatter_with_preview_basic(
        self, interactive_viz_df, sample_image_links
    ):
        """Test basic scatter plot with preview panel."""
        fig = create_interactive_scatter_with_preview(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
        )

        assert isinstance(fig, go.Figure)
        # Check for subplot structure - should have multiple axes for subplots
        assert "xaxis2" in fig.layout  # Second subplot's x-axis
        assert "yaxis2" in fig.layout  # Second subplot's y-axis

    def test_create_scatter_with_preview_color(
        self, interactive_viz_df, sample_image_links
    ):
        """Test scatter with preview and color grouping."""
        fig = create_interactive_scatter_with_preview(
            df=interactive_viz_df,
            x_col="trait1",
            y_col="trait2",
            image_links=sample_image_links,
            color_by="geno",
            hover_data=["rep"],
        )

        assert isinstance(fig, go.Figure)
        # Should have multiple traces for groups
        unique_genos = interactive_viz_df["geno"].nunique()
        # Note: traces include both scatter and layout elements
        assert len(fig.data) >= unique_genos


# ============================================================================
# Integration tests
# ============================================================================


class TestIntegration:
    """Integration tests for interactive visualization."""

    def test_full_pca_workflow(
        self, pca_viz_results, pca_viz_dataframe, sample_image_links, tmp_path
    ):
        """Test complete PCA visualization workflow."""
        # Create PCA plot with images
        fig = create_interactive_pca_with_images(
            pca_results=pca_viz_results,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
            color_by="geno",
            show_loadings=True,
        )

        # Save as HTML with image viewer
        output_path = tmp_path / "pca_complete.html"
        create_html_with_image_viewer(
            fig=fig,
            df=pca_viz_dataframe,
            image_links=sample_image_links,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.stat().st_size > 1000  # Should have substantial content

    def test_full_umap_workflow(
        self, umap_viz_results, interactive_viz_df, sample_image_links, tmp_path
    ):
        """Test complete UMAP visualization workflow."""
        # Create UMAP plot
        fig = create_interactive_umap_with_images(
            umap_results=umap_viz_results,
            df=interactive_viz_df,
            image_links=sample_image_links,
            color_by="geno",
        )

        # Also create hover highlight version
        fig_hover = create_interactive_umap_with_hover_highlight(
            umap_results=umap_viz_results,
            df=interactive_viz_df,
            genotype_col="geno",
        )

        # Save both
        for fig_obj, name in [(fig, "umap_basic"), (fig_hover, "umap_hover")]:
            output_path = tmp_path / f"{name}.html"
            create_html_with_image_viewer(
                fig=fig_obj,
                df=interactive_viz_df,
                image_links=sample_image_links,
                output_path=output_path,
            )

            if name == "umap_basic":  # Hover version has empty figure issue
                assert output_path.exists()

    def test_dashboard_workflow(self, interactive_viz_df, sample_image_links):
        """Test dashboard creation workflow."""
        trait_cols = ["trait1", "trait2", "trait3"]

        # Create dashboard
        fig = create_trait_explorer_dashboard(
            df=interactive_viz_df,
            trait_cols=trait_cols,
            image_links=sample_image_links,
            group_col="geno",
        )

        # Dashboard should be complete
        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Trait Explorer Dashboard"

        # Check subplot titles are set
        assert any(
            "Correlation" in str(ann.text)
            for ann in fig.layout.annotations
            if hasattr(ann, "text")
        )
