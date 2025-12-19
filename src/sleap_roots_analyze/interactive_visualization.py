"""Interactive visualization functions with image hover capabilities.

This module provides functions for creating interactive plots that allow
users to hover over data points and view associated images.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import io


def encode_image_to_base64(image_path: Union[str, Path]) -> str:
    """Encode an image file to base64 string for embedding in HTML.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 encoded image string.
    """
    image_path = Path(image_path)

    if not image_path.exists():
        return ""

    # Open and resize image if needed
    with Image.open(image_path) as img:
        # Resize large images to reduce data size
        max_size = (400, 400)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        # Encode to base64
        encoded = base64.b64encode(buffer.read()).decode()

    return f"data:image/png;base64,{encoded}"


def create_interactive_scatter_with_images(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    image_links: Dict[str, Dict[str, str]],
    color_by: Optional[str] = None,
    size_by: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: str = "Interactive Scatter Plot with Images",
    width: int = 1000,
    height: int = 800,
    marker_size: int = 10,
    show_images_on_hover: bool = True,
    id_col: str = "Barcode",
) -> go.Figure:
    """Create an interactive scatter plot with image hover capabilities.

    Args:
        df: DataFrame containing the data.
        x_col: Column name for x-axis.
        y_col: Column name for y-axis.
        image_links: Dictionary mapping sample IDs to image paths.
            Format: {barcode: {'features.png': path, 'seg.png': path}}.
        color_by: Column name to color points by.
        size_by: Column name to size points by.
        hover_data: Additional columns to show on hover.
        title: Plot title.
        width: Plot width.
        height: Plot height.
        marker_size: Base marker size.
        show_images_on_hover: Whether to show images on hover.
        id_col: Column containing sample IDs for image linking (default: "Barcode").

    Returns:
        Plotly figure object with interactive hover.
    """
    # Prepare hover data
    if hover_data is None:
        hover_data = []

    # Ensure we have the ID column for image linking
    # Use provided id_col parameter, fallback to index if not in df
    if id_col not in df.columns:
        id_col = df.index.name if df.index.name else "index"
    if id_col not in hover_data and id_col in df.columns:
        hover_data = [id_col] + hover_data

    # Create figure
    fig = go.Figure()

    # Handle coloring
    if color_by and color_by in df.columns:
        unique_groups = df[color_by].unique()
        colors = px.colors.qualitative.Plotly
        color_map = {
            group: colors[i % len(colors)] for i, group in enumerate(unique_groups)
        }

        for group in unique_groups:
            mask = df[color_by] == group
            group_df = df[mask]

            # Prepare hover text with images
            hover_texts = []
            for idx, row in group_df.iterrows():
                hover_text_parts = [f"<b>{color_by}: {group}</b>"]
                hover_text_parts.append(f"{x_col}: {row[x_col]:.3f}")
                hover_text_parts.append(f"{y_col}: {row[y_col]:.3f}")

                # Add additional hover data
                for col in hover_data:
                    if col in row:
                        hover_text_parts.append(f"{col}: {row[col]}")

                # Add image if available
                if show_images_on_hover and id_col in row:
                    sample_id = row[id_col]
                    if sample_id in image_links:
                        # Use features image if available
                        if "features.png" in image_links[sample_id]:
                            hover_text_parts.append("<b>📷 Image available</b>")

                hover_text = "<br>".join(hover_text_parts)
                hover_texts.append(hover_text)

            # Handle sizing
            if size_by and size_by in df.columns:
                sizes = group_df[size_by].values
                # Normalize sizes
                sizes = (
                    ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * 20 + 5)
                    if sizes.max() > sizes.min()
                    else [marker_size] * len(sizes)
                )
            else:
                sizes = marker_size

            # Get sample IDs for this group
            sample_ids = []
            for idx, row in group_df.iterrows():
                if id_col in row:
                    sample_ids.append(row[id_col])
                else:
                    sample_ids.append(f"Sample_{idx}")

            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=group_df[x_col],
                    y=group_df[y_col],
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        color=color_map[group],
                        size=sizes,
                        opacity=0.7,
                        line=dict(width=1, color="white"),
                    ),
                    customdata=sample_ids,  # Use sample IDs for customdata
                    text=hover_texts,  # Use hover texts for text
                    hovertemplate="%{text}<extra></extra>",  # Show text on hover
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                )
            )
    else:
        # Single color plot
        hover_texts = []
        for idx, row in df.iterrows():
            hover_text_parts = []
            hover_text_parts.append(f"{x_col}: {row[x_col]:.3f}")
            hover_text_parts.append(f"{y_col}: {row[y_col]:.3f}")

            # Add additional hover data
            for col in hover_data:
                if col in row:
                    hover_text_parts.append(f"{col}: {row[col]}")

            # Add image if available
            if show_images_on_hover and id_col in row:
                sample_id = row[id_col]
                if sample_id in image_links:
                    if "features.png" in image_links[sample_id]:
                        hover_text_parts.append("<b>📷 Image available</b>")

            hover_text = "<br>".join(hover_text_parts)
            hover_texts.append(hover_text)

        # Handle sizing
        if size_by and size_by in df.columns:
            sizes = df[size_by].values
            sizes = (
                ((sizes - sizes.min()) / (sizes.max() - sizes.min()) * 20 + 5)
                if sizes.max() > sizes.min()
                else [marker_size] * len(sizes)
            )
        else:
            sizes = marker_size

        # Get sample IDs
        sample_ids = []
        for idx, row in df.iterrows():
            if id_col in row:
                sample_ids.append(row[id_col])
            else:
                sample_ids.append(f"Sample_{idx}")

        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(
                    color="blue",
                    size=sizes,
                    opacity=0.7,
                    line=dict(width=1, color="white"),
                ),
                customdata=sample_ids,  # Use sample IDs for customdata
                text=hover_texts,  # Use hover texts for text
                hovertemplate="%{text}<extra></extra>",  # Show text on hover
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
            )
        )

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=width,
        height=height,
        hovermode="closest",
        template="plotly_white",
        showlegend=bool(color_by),
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

    return fig


def create_interactive_pca_with_images(
    pca_results: Dict,
    df: pd.DataFrame,
    image_links: Dict[str, Dict[str, str]],
    color_by: Optional[str] = None,
    components: Tuple[int, int] = (0, 1),
    hover_cols: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 800,
    show_loadings: bool = False,
    n_loadings: int = 10,
    id_col: str = "Barcode",
) -> go.Figure:
    """Create an interactive PCA plot with image hover.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with metadata.
        image_links: Dictionary mapping sample IDs to image paths.
        color_by: Column name to color points by.
        components: Which components to plot (0-indexed).
        hover_cols: Additional columns to show on hover.
        title: Plot title.
        width: Plot width.
        height: Plot height.
        show_loadings: Whether to show feature loadings as arrows.
        n_loadings: Number of top loadings to show.
        id_col: Column containing sample IDs (default: "Barcode").

    Returns:
        Interactive PCA plot with images.
    """
    # Get PC scores
    pc_x, pc_y = components
    X_pca = pca_results["transformed_data"]

    # Create dataframe for plotting
    plot_df = df.copy()
    plot_df[f"PC{pc_x + 1}"] = X_pca[:, pc_x]
    plot_df[f"PC{pc_y + 1}"] = X_pca[:, pc_y]

    # Default title
    if title is None:
        var_x = pca_results["explained_variance_ratio"][pc_x] * 100
        var_y = pca_results["explained_variance_ratio"][pc_y] * 100
        title = f"PCA Plot - PC{pc_x + 1} vs PC{pc_y + 1} ({var_x:.1f}% vs {var_y:.1f}% variance)"

    # Create scatter plot with images
    fig = create_interactive_scatter_with_images(
        plot_df,
        x_col=f"PC{pc_x + 1}",
        y_col=f"PC{pc_y + 1}",
        image_links=image_links,
        color_by=color_by,
        hover_data=hover_cols,
        title=title,
        width=width,
        height=height,
        id_col=id_col,
    )

    # Add loadings if requested
    if show_loadings and "loadings" in pca_results:
        loadings = pca_results["loadings"]
        feature_contributions = pca_results["feature_contributions"]

        # Get top contributing features
        top_features = feature_contributions.nlargest(
            n_loadings, "total_contribution"
        ).index

        # Scale loadings for visualization
        max_score = max(
            abs(X_pca[:, pc_x].max()),
            abs(X_pca[:, pc_x].min()),
            abs(X_pca[:, pc_y].max()),
            abs(X_pca[:, pc_y].min()),
        )
        scale = max_score * 0.7

        # Add loading arrows and labels with smart positioning
        import numpy as np

        # Pattern of offsets for label positioning to avoid overlaps
        offsets = [-0.3, 0, 0.3, -0.2, 0.2, -0.4, 0.1, 0.4, -0.1, 0.35]

        for i, feature in enumerate(top_features):
            feature_idx = list(pca_results["feature_names"]).index(feature)

            # Calculate arrow end position
            x_end = loadings[feature_idx, pc_x] * scale
            y_end = loadings[feature_idx, pc_y] * scale

            # Add arrow
            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",  # No text on arrow itself
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                opacity=0.6,
            )

            # Calculate label position with jitter
            angle = np.arctan2(y_end, x_end)
            angle_offset = offsets[i % len(offsets)]

            # Vary radius to create layers and avoid overlaps
            radius_mult = 1.2 + (i % 4) * 0.1  # 1.2, 1.3, 1.4, 1.5
            radius = np.sqrt(x_end**2 + y_end**2) * radius_mult

            # Calculate jittered label position
            label_x = radius * np.cos(angle + angle_offset)
            label_y = radius * np.sin(angle + angle_offset)

            # Determine text alignment based on position
            xanchor = "left" if label_x > 0 else "right"
            yanchor = "bottom" if label_y > 0 else "top"

            # Add label at jittered position
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=feature,
                showarrow=False,
                xref="x",
                yref="y",
                xanchor=xanchor,
                yanchor=yanchor,
                font=dict(size=10, color="darkred"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="darkred",
                borderwidth=1,
            )

    # Update axis labels with variance
    fig.update_xaxes(
        title=f"PC{pc_x + 1} ({pca_results['explained_variance_ratio'][pc_x]:.1%} variance)"
    )
    fig.update_yaxes(
        title=f"PC{pc_y + 1} ({pca_results['explained_variance_ratio'][pc_y]:.1%} variance)"
    )

    return fig


def create_interactive_umap_with_images(
    umap_results: Dict,
    df: pd.DataFrame,
    image_links: Dict[str, Dict[str, str]],
    color_by: Optional[str] = None,
    hover_cols: Optional[List[str]] = None,
    title: str = "UMAP Visualization",
    width: int = 1000,
    height: int = 800,
    id_col: str = "Barcode",
) -> go.Figure:
    """Create an interactive UMAP plot with image hover.

    Args:
        umap_results: Results from perform_umap_analysis.
        df: Original dataframe with metadata.
        image_links: Dictionary mapping sample IDs to image paths.
        color_by: Column name to color points by.
        hover_cols: Additional columns to show on hover.
        title: Plot title.
        width: Plot width.
        height: Plot height.
        id_col: Column containing sample IDs (default: "Barcode").

    Returns:
        Interactive UMAP plot with images.
    """
    # Get UMAP embedding
    embedding = umap_results["embedding"]

    # Create dataframe for plotting
    plot_df = df.copy()
    plot_df["UMAP1"] = embedding[:, 0]
    plot_df["UMAP2"] = embedding[:, 1] if embedding.shape[1] > 1 else 0

    # Add UMAP parameters to title
    title += f" (n_neighbors={umap_results['n_neighbors']}, min_dist={umap_results['min_dist']})"

    # Create scatter plot with images
    fig = create_interactive_scatter_with_images(
        plot_df,
        x_col="UMAP1",
        y_col="UMAP2",
        image_links=image_links,
        color_by=color_by,
        hover_data=hover_cols,
        title=title,
        width=width,
        height=height,
        id_col=id_col,
    )

    return fig


def create_interactive_umap_with_hover_highlight(
    umap_results: Dict,
    df: pd.DataFrame,
    genotype_col: str = "geno",
    title: str = "UMAP Visualization with Genotype Hover Highlighting",
    width: int = 1000,
    height: int = 800,
    grey_color: str = "lightgrey",
    highlight_color: str = "red",
    marker_size: int = 8,
    highlight_size: int = 12,
) -> go.Figure:
    """Create an interactive UMAP plot where hovering over genotype names highlights samples.

    All samples are shown in grey by default. When hovering over a genotype name in the
    legend, all samples for that genotype are highlighted in red. This helps explore how
    individual genotypes cluster in UMAP space without the clutter of many colored categories.

    Args:
        umap_results: Results from perform_umap_analysis containing 'embedding' array.
        df: Original dataframe with metadata including genotype column.
        genotype_col: Column name containing genotype information.
        title: Plot title.
        width: Plot width in pixels.
        height: Plot height in pixels.
        grey_color: Color for non-highlighted samples.
        highlight_color: Color for highlighted samples on hover.
        marker_size: Size of markers when not highlighted.
        highlight_size: Size of markers when highlighted.

    Returns:
        Interactive UMAP plot with hover-based genotype highlighting.
    """
    # Get UMAP embedding
    embedding = umap_results["embedding"]

    # Check if genotype column exists
    if genotype_col not in df.columns:
        raise ValueError(f"Genotype column '{genotype_col}' not found in dataframe")

    # Get unique genotypes
    unique_genotypes = sorted(df[genotype_col].unique())

    # Create figure
    fig = go.Figure()

    # First, add all points as a single grey trace
    fig.add_trace(
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(len(embedding)),
            mode="markers",
            name="All samples",
            marker=dict(
                size=marker_size,
                color=grey_color,
                line=dict(width=0.5, color="darkgrey"),
            ),
            hovertemplate="<b>%{text}</b><br>"
            + "UMAP1: %{x:.2f}<br>"
            + "UMAP2: %{y:.2f}<br>"
            + "<extra></extra>",
            text=df[genotype_col].values,
            showlegend=False,
        )
    )

    # Then add a trace for each genotype (initially hidden)
    for i, genotype in enumerate(unique_genotypes):
        # Get mask for this genotype
        mask = df[genotype_col] == genotype

        # Add scatter trace for this genotype
        fig.add_trace(
            go.Scatter(
                x=embedding[mask, 0],
                y=(
                    embedding[mask, 1]
                    if embedding.shape[1] > 1
                    else np.zeros(mask.sum())
                ),
                mode="markers",
                name=genotype,
                marker=dict(
                    size=highlight_size,
                    color=highlight_color,
                    line=dict(width=1, color="darkred"),
                ),
                hovertemplate=f"<b>{genotype}</b><br>"
                + "UMAP1: %{x:.2f}<br>"
                + "UMAP2: %{y:.2f}<br>"
                + "<extra></extra>",
                visible="legendonly",  # Hidden by default, shown on legend click
                legendgroup=genotype,
                showlegend=True,
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="UMAP 1"),
        yaxis=dict(title="UMAP 2"),
        width=width,
        height=height,
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            title=dict(
                text=f"{genotype_col.capitalize()}<br><sup>(click to highlight)</sup>",
                font=dict(size=14),
            ),
            itemsizing="constant",
            font=dict(size=12),
            bordercolor="Black",
            borderwidth=1,
            itemclick="toggle",  # Allow clicking to toggle visibility
            itemdoubleclick="toggleothers",  # Double-click to isolate
        ),
        annotations=[
            dict(
                text="<i>Click genotype names to highlight samples<br>"
                + "Double-click to show only that genotype</i>",
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor="left",
                yanchor="top",
                font=dict(size=11, color="grey"),
                showarrow=False,
            )
        ],
    )

    # Add UMAP parameters to title if available
    if "n_neighbors" in umap_results and "min_dist" in umap_results:
        fig.update_layout(
            title=dict(
                text=title
                + f"<br><sup>n_neighbors={umap_results['n_neighbors']}, "
                + f"min_dist={umap_results['min_dist']}</sup>"
            )
        )

    return fig


def create_trait_explorer_dashboard(
    df: pd.DataFrame,
    trait_cols: List[str],
    image_links: Dict[str, Dict[str, str]],
    group_col: str = "geno",
    width: int = 1400,
    height: int = 900,
) -> go.Figure:
    """Create an interactive dashboard for exploring traits.

    Args:
        df: DataFrame containing trait data.
        trait_cols: List of trait columns.
        image_links: Dictionary mapping sample IDs to image paths.
        group_col: Column for grouping (e.g., 'geno').
        width: Dashboard width.
        height: Dashboard height.

    Returns:
        Interactive dashboard.
    """
    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Trait Correlation Matrix",
            "Trait Distribution",
            "Group Comparison",
            "Sample Explorer",
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "box"}],
            [{"type": "bar"}, {"type": "scatter"}],
        ],
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5],
    )

    # 1. Correlation heatmap
    corr_matrix = df[trait_cols].corr()
    fig.add_trace(
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 8},
            hovertemplate="%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # 2. Trait distribution (box plot for first trait)
    if trait_cols:
        trait = trait_cols[0]
        for group in df[group_col].unique():
            group_data = df[df[group_col] == group][trait]
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=str(group),
                    boxpoints="outliers",
                    jitter=0.3,
                    pointpos=-1.8,
                ),
                row=1,
                col=2,
            )

    # 3. Group means (bar plot for first trait)
    if trait_cols:
        trait = trait_cols[0]
        group_means = df.groupby(group_col)[trait].agg(["mean", "std"])
        fig.add_trace(
            go.Bar(
                x=group_means.index,
                y=group_means["mean"],
                error_y=dict(type="data", array=group_means["std"], visible=True),
                name="Mean ± SD",
            ),
            row=2,
            col=1,
        )

    # 4. Sample scatter (first two traits)
    if len(trait_cols) >= 2:
        # This would be replaced with the interactive scatter with images
        # For now, just a placeholder
        fig.add_trace(
            go.Scatter(
                x=df[trait_cols[0]],
                y=df[trait_cols[1]],
                mode="markers",
                marker=dict(size=8, opacity=0.7),
                text=df["Barcode"] if "Barcode" in df.columns else None,
                hovertemplate="%{text}<br>%{x}<br>%{y}<extra></extra>",
            ),
            row=2,
            col=2,
        )

    # Update layout
    fig.update_layout(
        title_text="Trait Explorer Dashboard",
        showlegend=True,
        width=width,
        height=height,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(title_text="Traits", row=1, col=1)
    fig.update_yaxes(title_text="Traits", row=1, col=1)
    fig.update_yaxes(title_text=trait_cols[0] if trait_cols else "Value", row=1, col=2)
    fig.update_xaxes(title_text=group_col, row=2, col=1)
    fig.update_yaxes(title_text=trait_cols[0] if trait_cols else "Value", row=2, col=1)
    fig.update_xaxes(
        title_text=trait_cols[0] if len(trait_cols) >= 1 else "Trait 1", row=2, col=2
    )
    fig.update_yaxes(
        title_text=trait_cols[1] if len(trait_cols) >= 2 else "Trait 2", row=2, col=2
    )

    return fig


def create_interactive_scatter_with_preview(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    image_links: Dict[str, Dict[str, Path]],
    color_by: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    title: Optional[str] = None,
    width: int = 1000,
    height: int = 800,
    id_col: str = "Barcode",
) -> go.Figure:
    """Create interactive scatter plot with image preview panel.

    This creates a subplot with the scatter plot on the left and an image
    preview area on the right. Clicking on a point shows the image.

    Args:
        df: DataFrame with data
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        image_links: Dictionary mapping sample IDs to image paths
        color_by: Column to use for coloring points
        hover_data: Additional columns to show on hover
        title: Plot title
        width: Total figure width
        height: Figure height
        id_col: Column containing sample IDs

    Returns:
        Plotly figure with scatter plot and image preview
    """
    # Create subplot with 1 row, 2 columns
    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=("Scatter Plot", "Click a point to view image"),
        horizontal_spacing=0.05,
    )

    # Prepare hover data
    if hover_data is None:
        hover_data = []

    # Create hover text without images
    hover_texts = []
    sample_ids = []

    for idx, row in df.iterrows():
        hover_parts = [
            f"<b>Sample: {row.get(id_col, 'Unknown')}</b>",
            f"{x_col}: {row[x_col]:.3f}",
            f"{y_col}: {row[y_col]:.3f}",
        ]

        # Add additional hover data
        for col in hover_data:
            if col in row:
                hover_parts.append(f"{col}: {row[col]}")

        # Check if image available
        sample_id = row.get(id_col, None)
        if sample_id and sample_id in image_links:
            if "features.png" in image_links[sample_id]:
                hover_parts.append("📷 Image available (click to view)")

        hover_texts.append("<br>".join(hover_parts))
        sample_ids.append(sample_id)

    # Add scatter plot
    if color_by and color_by in df.columns:
        # Color by groups
        groups = df[color_by].unique()
        colors = px.colors.qualitative.Set1[: len(groups)]

        for i, group in enumerate(groups):
            mask = df[color_by] == group
            fig.add_trace(
                go.Scatter(
                    x=df.loc[mask, x_col],
                    y=df.loc[mask, y_col],
                    mode="markers",
                    name=str(group),
                    marker=dict(color=colors[i % len(colors)], size=8, opacity=0.7),
                    text=[hover_texts[j] for j, m in enumerate(mask) if m],
                    hoverinfo="text",
                    customdata=[sample_ids[j] for j, m in enumerate(mask) if m],
                ),
                row=1,
                col=1,
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(color="blue", size=8, opacity=0.7),
                text=hover_texts,
                hoverinfo="text",
                customdata=sample_ids,
            ),
            row=1,
            col=1,
        )

    # Add placeholder for image
    fig.add_layout_image(
        dict(
            source="",
            xref="x2",
            yref="y2",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            xanchor="left",
            yanchor="top",
        ),
        row=1,
        col=2,
    )

    # Update axes
    fig.update_xaxes(title_text=x_col, row=1, col=1)
    fig.update_yaxes(title_text=y_col, row=1, col=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=title or f"{y_col} vs {x_col}",
        width=width,
        height=height,
        template="plotly_white",
        showlegend=True,
        hovermode="closest",
    )

    # Add JavaScript for image display on click
    # Note: This requires saving as HTML and including image data

    return fig


def create_html_with_image_viewer(
    fig: go.Figure,
    df: pd.DataFrame,
    image_links: Dict[str, Dict[str, Path]],
    output_path: Union[str, Path],
    id_col: str = "Barcode",
) -> None:
    """Save interactive plot as HTML with custom image viewer.

    Creates an HTML file with the plot and JavaScript code to display
    images when points are clicked.

    Args:
        fig: Plotly figure
        df: DataFrame with data
        image_links: Dictionary mapping sample IDs to image paths
        output_path: Path to save HTML file
        id_col: Column containing sample IDs
    """
    import json
    import re

    output_path = Path(output_path)

    # Debug: Check if figure has data
    if not fig.data:
        print(f"Warning: Figure has no data traces!")
        return

    # Prepare image data as base64
    image_data = {}
    for sample_id, links in image_links.items():
        if "features.png" in links and links["features.png"]:
            img_path = Path(links["features.png"])
            if img_path.exists():
                with open(img_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                    image_data[sample_id] = f"data:image/png;base64,{encoded}"

    # Use fig.write_html to generate proper HTML with all data preserved
    # Then inject our custom JavaScript
    temp_html = fig.to_html(include_plotlyjs="cdn", div_id="plot")

    # Extract the plot div and script from the generated HTML
    # Find the div
    div_match = re.search(r'<div id="plot".*?</div>', temp_html, re.DOTALL)
    plot_div = div_match.group(0) if div_match else '<div id="plot"></div>'

    # Find the Plotly script
    script_match = re.search(
        r'<script type="text/javascript">(.*?Plotly\.newPlot.*?)</script>',
        temp_html,
        re.DOTALL,
    )
    plotly_script = script_match.group(1) if script_match else ""

    # Create custom HTML with image viewer
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Interactive Plot with Image Viewer</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            #container {{ display: flex; gap: 20px; }}
            #plot-container {{ flex: 1; }}
            #image-panel {{
                width: 400px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background: #f9f9f9;
            }}
            #sample-image {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ccc;
            }}
            #image-info {{
                margin-top: 10px;
                font-size: 14px;
            }}
            .no-image {{
                color: #666;
                font-style: italic;
                text-align: center;
                padding: 50px 0;
            }}
        </style>
    </head>
    <body>
        <h2>{fig.layout.title.text or "Interactive Plot"}</h2>
        <div id="container">
            <div id="plot-container">
                {plot_div}
            </div>
            <div id="image-panel">
                <h3>Sample Image</h3>
                <div id="image-container">
                    <p class="no-image">Click on a data point to view its image</p>
                </div>
                <div id="image-info"></div>
            </div>
        </div>
        
        <script>
            var imageData = {json.dumps(image_data)};
            
            // Plotly plot creation
            {plotly_script}
            
            // Add click handler
            var plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data) {{
                var point = data.points[0];
                var sampleId = point.customdata;
                
                // Handle different customdata formats
                if (Array.isArray(sampleId)) {{
                    sampleId = sampleId[0];
                }}
                
                // Debug logging
                console.log('Clicked point:', point);
                console.log('Sample ID:', sampleId);
                console.log('Available images:', Object.keys(imageData));
                
                var imageContainer = document.getElementById('image-container');
                var imageInfo = document.getElementById('image-info');
                
                if (sampleId && imageData[sampleId]) {{
                    imageContainer.innerHTML = '<img id="sample-image" src="' + imageData[sampleId] + '" alt="Sample ' + sampleId + '">';
                    
                    var infoHtml = '<strong>Sample ID:</strong> ' + sampleId;
                    if (point.x !== undefined) infoHtml += '<br><strong>X:</strong> ' + point.x.toFixed(3);
                    if (point.y !== undefined) infoHtml += '<br><strong>Y:</strong> ' + point.y.toFixed(3);
                    
                    imageInfo.innerHTML = infoHtml;
                }} else {{
                    imageContainer.innerHTML = '<p class="no-image">No image available for sample: ' + (sampleId || 'Unknown') + '</p>';
                    imageInfo.innerHTML = '';
                }}
            }});
        </script>
    </body>
    </html>
    """

    # Save HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Saved interactive plot with image viewer to: {output_path.as_posix()}")


def create_interactive_image_gallery(
    df: pd.DataFrame,
    image_links: Dict[str, Dict[str, Path]],
    trait_cols: List[str],
    output_path: Union[str, Path],
    id_col: str = "Barcode",
    n_cols: int = 4,
    image_width: int = 200,
) -> None:
    """Create an interactive image gallery with trait data.

    Creates an HTML page showing all sample images in a grid with
    trait values displayed on hover.

    Args:
        df: DataFrame with trait data
        image_links: Dictionary mapping sample IDs to image paths
        trait_cols: List of trait columns to display
        output_path: Path to save HTML file
        id_col: Column containing sample IDs
        n_cols: Number of columns in image grid
        image_width: Width of each image in pixels
    """
    output_path = Path(output_path)

    # Create HTML content
    html_parts = [
        f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Sample Image Gallery</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #333; }}
            .gallery {{
                display: grid;
                grid-template-columns: repeat({n_cols}, 1fr);
                gap: 20px;
                margin-top: 20px;
            }}
            .sample-card {{
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                background: #f9f9f9;
                position: relative;
            }}
            .sample-image {{
                width: 100%;
                height: auto;
                border: 1px solid #ccc;
                cursor: pointer;
            }}
            .sample-info {{
                margin-top: 5px;
                font-size: 12px;
            }}
            .trait-tooltip {{
                position: absolute;
                background: rgba(0,0,0,0.9);
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 12px;
                z-index: 1000;
                display: none;
                max-width: 300px;
            }}
            .sample-card:hover .trait-tooltip {{
                display: block;
                top: 100%;
                left: 0;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <h1>Sample Image Gallery</h1>
        <p>Hover over images to see trait values</p>
        <div class="gallery">
    """
    ]

    # Add sample cards
    for idx, row in df.iterrows():
        sample_id = row.get(id_col, f"Sample_{idx}")

        # Get image if available
        img_src = ""
        if sample_id in image_links and "features.png" in image_links[sample_id]:
            img_path = image_links[sample_id]["features.png"]
            if img_path is not None:  # Check if path exists (not None)
                img_path = Path(img_path)
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()
                        img_src = f"data:image/png;base64,{encoded}"

        if img_src:
            # Create trait tooltip content
            tooltip_parts = [f"<strong>{sample_id}</strong><br>"]
            for trait in trait_cols[:10]:  # Limit to first 10 traits
                if trait in row:
                    value = row[trait]
                    if pd.notna(value):
                        tooltip_parts.append(f"{trait}: {value:.3f}")

            html_parts.append(
                f"""
            <div class="sample-card">
                <img class="sample-image" src="{img_src}" alt="{sample_id}">
                <div class="sample-info">
                    <strong>{sample_id}</strong>
                    {"<br>Genotype: " + str(row.get("geno", "")) if "geno" in row else ""}
                </div>
                <div class="trait-tooltip">
                    {"<br>".join(tooltip_parts)}
                </div>
            </div>
            """
            )

    html_parts.append(
        """
        </div>
    </body>
    </html>
    """
    )

    # Save HTML file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(html_parts))

    print(f"Saved image gallery to: {output_path.as_posix()}")


def create_interactive_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: Optional[str] = None,
    hover_data: Optional[List[str]] = None,
    image_links: Optional[Dict] = None,
    title: str = "Interactive Scatter Plot",
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Create an interactive scatter plot with hover information and optional images.

    Args:
        df: DataFrame containing the data.
        x_col: Column name for x-axis.
        y_col: Column name for y-axis.
        color_by: Column name to color points by.
        hover_data: Additional columns to show on hover.
        image_links: Dictionary mapping sample identifiers to image paths.
        title: Plot title.
        width: Plot width.
        height: Plot height.

    Returns:
        Plotly figure object.
    """
    # Prepare hover data
    if hover_data is None:
        hover_data = []

    # Create base scatter plot
    if color_by:
        fig = px.scatter(
            df, x=x_col, y=y_col, color=color_by, hover_data=hover_data, title=title
        )
    else:
        fig = px.scatter(df, x=x_col, y=y_col, hover_data=hover_data, title=title)

    # Update layout
    fig.update_layout(
        width=width, height=height, hovermode="closest", template="plotly_white"
    )

    # Add hover template with images if available
    if image_links:
        # This is a simplified version - full image hover would require
        # a more complex setup with a custom hover template
        hover_template = "<b>Sample: %{customdata[0]}</b><br>"
        hover_template += f"{x_col}: %{{x}}<br>"
        hover_template += f"{y_col}: %{{y}}<br>"

        for i, col in enumerate(hover_data[1:], 1):
            hover_template += f"{col}: %{{customdata[{i}]}}<br>"

        hover_template += "<extra></extra>"

        fig.update_traces(hovertemplate=hover_template)

    fig.update_xaxes(title=x_col)
    fig.update_yaxes(title=y_col)

    return fig


def create_interactive_pca_plot(
    pca_results: Dict,
    df: pd.DataFrame,
    color_by: Optional[str] = None,
    hover_cols: Optional[List[str]] = None,
    components: Tuple[int, int] = (0, 1),
    image_links: Optional[Dict] = None,
    title: str = "Interactive PCA Plot",
    width: int = 800,
    height: int = 600,
    show_loadings: bool = False,
    n_loadings: int = 10,
) -> go.Figure:
    """Create an interactive PCA plot using Plotly.

    Args:
        pca_results: Results from perform_pca_analysis.
        df: Original dataframe with metadata.
        color_by: Column name to color points by.
        hover_cols: Columns to show on hover.
        components: Which components to plot (0-indexed).
        image_links: Dictionary mapping sample identifiers to image paths.
        title: Plot title.
        width: Plot width.
        height: Plot height.
        show_loadings: Whether to show feature loadings as arrows.
        n_loadings: Number of top loadings to show.

    Returns:
        Plotly figure object.
    """
    # Get PC scores
    pc_x, pc_y = components
    X_pca = pca_results["transformed_data"]

    # Create dataframe for plotting
    plot_df = df.copy()
    plot_df[f"PC{pc_x + 1}"] = X_pca[:, pc_x]
    plot_df[f"PC{pc_y + 1}"] = X_pca[:, pc_y]

    # Create hover data
    if hover_cols is None:
        hover_cols = []
    if "Barcode" in df.columns and "Barcode" not in hover_cols:
        hover_cols = ["Barcode"] + hover_cols

    # Create figure
    fig = create_interactive_scatter_plot(
        plot_df,
        x_col=f"PC{pc_x + 1}",
        y_col=f"PC{pc_y + 1}",
        color_by=color_by,
        hover_data=hover_cols,
        image_links=image_links,
        title=title,
        width=width,
        height=height,
    )

    # Add loadings if requested
    if show_loadings and "loadings" in pca_results:
        loadings = pca_results["loadings"]
        feature_contributions = pca_results["feature_contributions"]

        # Get top contributing features
        top_features = feature_contributions.nlargest(
            n_loadings, "total_contribution"
        ).index

        # Scale loadings for visualization
        max_score = max(
            abs(X_pca[:, pc_x].max()),
            abs(X_pca[:, pc_x].min()),
            abs(X_pca[:, pc_y].max()),
            abs(X_pca[:, pc_y].min()),
        )
        scale = max_score * 0.7

        # Add loading arrows and labels with smart positioning
        import numpy as np

        # Pattern of offsets for label positioning to avoid overlaps
        offsets = [-0.3, 0, 0.3, -0.2, 0.2, -0.4, 0.1, 0.4, -0.1, 0.35]

        for i, feature in enumerate(top_features):
            feature_idx = list(pca_results["feature_names"]).index(feature)

            # Calculate arrow end position
            x_end = loadings[feature_idx, pc_x] * scale
            y_end = loadings[feature_idx, pc_y] * scale

            # Add arrow
            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=0,
                ay=0,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",  # No text on arrow itself
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                opacity=0.6,
            )

            # Calculate label position with jitter
            angle = np.arctan2(y_end, x_end)
            angle_offset = offsets[i % len(offsets)]

            # Vary radius to create layers and avoid overlaps
            radius_mult = 1.2 + (i % 4) * 0.1  # 1.2, 1.3, 1.4, 1.5
            radius = np.sqrt(x_end**2 + y_end**2) * radius_mult

            # Calculate jittered label position
            label_x = radius * np.cos(angle + angle_offset)
            label_y = radius * np.sin(angle + angle_offset)

            # Determine text alignment based on position
            xanchor = "left" if label_x > 0 else "right"
            yanchor = "bottom" if label_y > 0 else "top"

            # Add label at jittered position
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=feature,
                showarrow=False,
                xref="x",
                yref="y",
                xanchor=xanchor,
                yanchor=yanchor,
                font=dict(size=10, color="darkred"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="darkred",
                borderwidth=1,
            )

    # Update axis labels with explained variance
    fig.update_xaxes(
        title=f"PC{pc_x + 1} ({pca_results['explained_variance_ratio'][pc_x]:.1%} variance)"
    )
    fig.update_yaxes(
        title=f"PC{pc_y + 1} ({pca_results['explained_variance_ratio'][pc_y]:.1%} variance)"
    )

    return fig
