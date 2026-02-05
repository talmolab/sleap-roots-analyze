"""Cross-platform analysis summary generator.

Generates detailed markdown summaries from cross-platform correlation
pipeline outputs, including trait reduction statistics, correlation
analysis results, and embedded visualizations.
"""

from __future__ import annotations

import base64
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ============================================================================
# STATISTICAL CONSTANTS
# ============================================================================

# Standard power threshold: 80% power is the conventional minimum for
# adequate statistical power in the behavioral/biological sciences.
# This threshold is used for:
# - Calculating minimum detectable effect sizes
# - Determining whether a study is "underpowered" (>50% of tests below this)
# - Sample size recommendations
TARGET_POWER = 0.80

# Default significance level (alpha) when not specified in config.
# This is only used as a fallback - the preferred source is config.significance_level.
DEFAULT_ALPHA = 0.05

# ============================================================================
# MEMORY CONSTANTS
# ============================================================================

# Default threshold for embedding images as base64 (10 MB).
# Files exceeding this threshold will use file paths instead.
# This limit is based on practical browser/editor rendering constraints:
# - VS Code markdown preview: blocks data URIs for security
# - Browser DOM rendering: degrades above 10-20 MB
DEFAULT_EMBED_THRESHOLD_BYTES = 10 * 1024 * 1024  # 10 MB


# ============================================================================
# SIZE CALCULATION HELPERS
# ============================================================================


def _calculate_total_image_size(paths: List[Path]) -> int:
    """Calculate total size of image files in bytes.

    Args:
        paths: List of paths to image files.

    Returns:
        Total size in bytes. Missing files are skipped.
    """
    total = 0
    for path in paths:
        if path and path.exists():
            try:
                total += path.stat().st_size
            except (OSError, IOError):
                logger.warning("Could not read size of %s", path)
    return total


# ============================================================================
# POWER CALCULATION HELPERS
# ============================================================================


def _calculate_minimum_detectable_r(
    n: int, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Calculate minimum detectable correlation coefficient.

    Uses Fisher's z transformation to determine the smallest |r| that can be
    detected with given sample size, alpha, and power.

    Args:
        n: Sample size.
        alpha: Significance level (default 0.05).
        power: Statistical power (default 0.8).

    Returns:
        Minimum detectable |r| value.
    """
    if n < 4:
        return 1.0  # Cannot detect any correlation with too few samples

    # Critical z-values
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = scipy_stats.norm.ppf(power)

    # Standard error of Fisher's z
    se = 1 / math.sqrt(n - 3)

    # Minimum detectable Fisher's z
    min_z = (z_alpha + z_beta) * se

    # Convert back to r using inverse Fisher transformation
    min_r = math.tanh(min_z)

    return min(min_r, 1.0)


def _calculate_required_n(r: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Calculate required sample size to detect a given correlation.

    Uses Fisher's z transformation to determine required n for detecting
    a correlation of magnitude |r| at given alpha and power.

    Args:
        r: Target correlation coefficient magnitude.
        alpha: Significance level (default 0.05).
        power: Statistical power (default 0.8).

    Returns:
        Required sample size (n).
    """
    if abs(r) >= 1.0:
        return 4  # Minimum meaningful sample size
    if abs(r) < 0.01:
        return 10000  # Cap at reasonable maximum

    # Critical z-values
    z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = scipy_stats.norm.ppf(power)

    # Fisher's z transformation of target r
    z_r = math.atanh(abs(r))

    # Required sample size
    n = ((z_alpha + z_beta) / z_r) ** 2 + 3

    return max(4, int(math.ceil(n)))


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class TraitReductionStats:
    """Statistics about trait redundancy reduction."""

    original_traits: int
    n_clusters: int
    representative_traits: int

    @property
    def reduction_pct(self) -> float:
        """Calculate reduction percentage."""
        if self.original_traits == 0:
            return 0.0
        return (1 - self.representative_traits / self.original_traits) * 100


@dataclass
class TopCorrelation:
    """A single top correlation result."""

    exp1_trait: str
    exp2_trait: str
    r_value: float
    p_value: float
    p_adjusted: float
    power: float
    n_genotypes: int
    significant_fdr: bool
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


@dataclass
class CorrelationStats:
    """Statistics about cross-platform correlations."""

    total_correlations: int
    nominal_significant: int
    fdr_significant: int
    top_correlations: List[TopCorrelation] = field(default_factory=list)


@dataclass
class PowerStats:
    """Statistical power analysis results."""

    min_power: float
    max_power: float
    median_power: float
    pct_above_80: float
    alpha: float = 0.05
    n_genotypes_modal: int = 0
    minimum_detectable_r: float = 0.0
    recommended_n_for_r40: int = 0


@dataclass
class ConfigInfo:
    """Configuration parameters for the analysis."""

    correlation_method: str
    trait_reduction_method: str
    trait_reduction_target: Optional[str]
    trait_clustering_threshold: Optional[float]
    fdr_correction_method: Optional[str] = None
    significance_level: Optional[float] = None


@dataclass
class ValidationResult:
    """Result of validation guardrails."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CrossPlatformRunSummary:
    """Summary for a single cross-platform comparison."""

    run_dir: Path
    exp1_name: str
    exp2_name: str
    config: ConfigInfo
    correlation_stats: CorrelationStats
    power_stats: Optional[PowerStats] = None
    exp1_trait_reduction: Optional[TraitReductionStats] = None
    exp2_trait_reduction: Optional[TraitReductionStats] = None
    exp1_dendrogram_path: Optional[Path] = None
    exp2_dendrogram_path: Optional[Path] = None
    exp1_heatmap_path: Optional[Path] = None
    exp2_heatmap_path: Optional[Path] = None
    representative_heatmap_path: Optional[Path] = None
    correlation_summary_path: Optional[Path] = None
    joint_plot_paths: List[Path] = field(default_factory=list)


@dataclass
class CrossPlatformSummary:
    """Aggregated summary from one or more cross-platform comparisons."""

    run_summaries: List[CrossPlatformRunSummary]
    validation_result: ValidationResult
    source_dir: Path

    def _embed_image_base64(self, image_path: Path) -> str:
        """Convert image file to base64 data URI.

        Args:
            image_path: Path to image file.

        Returns:
            Base64 data URI string or empty string if file not found.
        """
        if not image_path or not image_path.exists():
            return ""

        try:
            with open(image_path, "rb") as f:
                image_data = f.read()
            b64_data = base64.b64encode(image_data).decode("utf-8")
            return f"data:image/png;base64,{b64_data}"
        except Exception as e:
            logger.warning("Failed to embed image %s: %s", image_path, e)
            return ""

    def _format_image_reference(
        self, image_path: Optional[Path], alt_text: str, embed_images: bool
    ) -> str:
        """Format image reference as markdown.

        Args:
            image_path: Path to image file.
            alt_text: Alt text for image.
            embed_images: If True, embed as base64; otherwise use relative path.

        Returns:
            Markdown image reference or empty string if path is None.
        """
        if not image_path:
            return ""

        if embed_images:
            data_uri = self._embed_image_base64(image_path)
            if data_uri:
                return f"![{alt_text}]({data_uri})"
            return ""
        else:
            return f"![{alt_text}]({image_path.name})"

    def _format_fdr_interpretation(
        self, stats: CorrelationStats, power_stats: Optional[PowerStats]
    ) -> List[str]:
        """Generate interpretation section when no FDR-significant correlations found.

        Args:
            stats: Correlation statistics.
            power_stats: Power analysis statistics.

        Returns:
            List of markdown lines for interpretation section.
        """
        if stats.fdr_significant > 0:
            return []

        lines = []
        lines.append("\n#### Interpretation: No FDR-Significant Correlations\n")

        # Warning about FDR correction
        lines.append(
            "**Note:** No correlations survived FDR correction. This is common when:\n"
            "- Sample sizes are small\n"
            "- Effect sizes are modest\n"
            "- Testing many correlations simultaneously\n"
        )

        # Show nominal significant count
        if stats.nominal_significant > 0:
            lines.append(
                f"**Nominal Significant (p < 0.05):** {stats.nominal_significant} correlations "
                f"reached nominal significance before FDR correction. These may warrant "
                "further investigation with larger sample sizes.\n"
            )

        # Sample size recommendations
        if power_stats and power_stats.n_genotypes_modal > 0:
            current_n = power_stats.n_genotypes_modal
            lines.append("**Sample Size Recommendations:**\n")
            lines.append(f"- Current modal sample size: n = {current_n}")

            # Calculate required n for detecting moderate effect (r=0.4) at target power
            if power_stats.recommended_n_for_r40 > 0:
                lines.append(
                    f"- To detect |r| = 0.40 at {TARGET_POWER*100:.0f}% power: n ≈ {power_stats.recommended_n_for_r40}"
                )

            # Calculate required n for detecting small effect (r=0.3)
            required_n_r30 = _calculate_required_n(0.3, power_stats.alpha, TARGET_POWER)
            lines.append(
                f"- To detect |r| = 0.30 at {TARGET_POWER*100:.0f}% power: n ≈ {required_n_r30}\n"
            )

        return lines

    def _collect_all_image_paths(self) -> List[Path]:
        """Collect all image paths from run summaries.

        Returns:
            List of all image paths referenced in the summary.
        """
        paths = []
        for run in self.run_summaries:
            if run.exp1_dendrogram_path:
                paths.append(run.exp1_dendrogram_path)
            if run.exp2_dendrogram_path:
                paths.append(run.exp2_dendrogram_path)
            if run.exp1_heatmap_path:
                paths.append(run.exp1_heatmap_path)
            if run.exp2_heatmap_path:
                paths.append(run.exp2_heatmap_path)
            if run.representative_heatmap_path:
                paths.append(run.representative_heatmap_path)
            if run.correlation_summary_path:
                paths.append(run.correlation_summary_path)
            paths.extend(run.joint_plot_paths)
        return paths

    def to_markdown(
        self,
        embed_images: bool = False,
        image_mode: Optional[str] = None,
        embed_threshold_bytes: int = DEFAULT_EMBED_THRESHOLD_BYTES,
    ) -> str:
        """Render summary as markdown.

        Args:
            embed_images: Deprecated. Use image_mode instead.
                If True, embed images as base64 data URIs.
            image_mode: Image handling mode. One of:
                - "file_path": Always use relative file paths (default)
                - "embed": Embed as base64 if under threshold, else fallback
                - "auto": Same as embed (smart selection based on size)
                - None: Uses embed_images parameter for backward compatibility
            embed_threshold_bytes: Maximum total image size for embedding.
                Default is 10 MB. If exceeded in embed/auto mode, falls back
                to file paths with a warning.

        Returns:
            Markdown-formatted summary string.
        """
        # Determine effective embed mode
        if image_mode is None:
            # Backward compatibility: use embed_images parameter
            should_embed = embed_images
        elif image_mode == "file_path":
            should_embed = False
        elif image_mode in ("embed", "auto"):
            # Check total size against threshold
            all_images = self._collect_all_image_paths()
            total_size = _calculate_total_image_size(all_images)

            if total_size > embed_threshold_bytes:
                logger.warning(
                    "Total image size (%d bytes) exceeds threshold (%d bytes). "
                    "Using file paths instead of embedding.",
                    total_size,
                    embed_threshold_bytes,
                )
                should_embed = False
            else:
                should_embed = True
        else:
            logger.warning("Unknown image_mode '%s', using file_path", image_mode)
            should_embed = False
        lines = []
        lines.append("# Cross-Platform Correlation Analysis Summary\n")

        for run_summary in self.run_summaries:
            lines.append(f"## {run_summary.exp1_name} vs {run_summary.exp2_name}\n")

            # Configuration section
            lines.append("### Configuration\n")
            lines.append(
                f"- **Correlation Method**: {run_summary.config.correlation_method}"
            )
            lines.append(
                f"- **Trait Reduction**: {run_summary.config.trait_reduction_method}"
            )
            if run_summary.config.trait_reduction_target:
                lines.append(
                    f"- **Reduction Target**: {run_summary.config.trait_reduction_target}"
                )
            if run_summary.config.trait_clustering_threshold:
                lines.append(
                    f"- **Clustering Threshold**: {run_summary.config.trait_clustering_threshold}"
                )
            lines.append("")

            # Trait reduction section (if clustering enabled)
            if run_summary.exp1_trait_reduction or run_summary.exp2_trait_reduction:
                lines.append("### Trait Reduction\n")

                if run_summary.exp1_trait_reduction:
                    red = run_summary.exp1_trait_reduction
                    lines.append(f"**{run_summary.exp1_name}**:")
                    lines.append(
                        f"- {red.original_traits} original traits → {red.n_clusters} clusters → "
                        f"{red.representative_traits} representatives ({red.reduction_pct:.1f}% reduction)"
                    )
                    img_ref = self._format_image_reference(
                        run_summary.exp1_dendrogram_path,
                        "Exp1 Dendrogram",
                        should_embed,
                    )
                    if img_ref:
                        lines.append(f"\n{img_ref}\n")
                    img_ref = self._format_image_reference(
                        run_summary.exp1_heatmap_path, "Exp1 Heatmap", should_embed
                    )
                    if img_ref:
                        lines.append(f"{img_ref}\n")
                    lines.append("")

                if run_summary.exp2_trait_reduction:
                    red = run_summary.exp2_trait_reduction
                    lines.append(f"**{run_summary.exp2_name}**:")
                    lines.append(
                        f"- {red.original_traits} original traits → {red.n_clusters} clusters → "
                        f"{red.representative_traits} representatives ({red.reduction_pct:.1f}% reduction)"
                    )
                    img_ref = self._format_image_reference(
                        run_summary.exp2_dendrogram_path,
                        "Exp2 Dendrogram",
                        should_embed,
                    )
                    if img_ref:
                        lines.append(f"\n{img_ref}\n")
                    img_ref = self._format_image_reference(
                        run_summary.exp2_heatmap_path, "Exp2 Heatmap", should_embed
                    )
                    if img_ref:
                        lines.append(f"{img_ref}\n")
                    lines.append("")

                # Cross-platform representative heatmap (when clustering is enabled)
                if run_summary.representative_heatmap_path:
                    lines.append("**Cross-Platform Representative Heatmap:**\n")
                    img_ref = self._format_image_reference(
                        run_summary.representative_heatmap_path,
                        "Cross-Platform Representative Heatmap",
                        should_embed,
                    )
                    if img_ref:
                        lines.append(f"{img_ref}\n")
                    lines.append("")

            # Correlation statistics section
            lines.append("### Correlation Statistics\n")
            stats = run_summary.correlation_stats
            lines.append(
                "| Metric | Value |\n"
                "| --- | --- |\n"
                f"| Total Correlations | {stats.total_correlations} |\n"
                f"| Nominal Significant (p < 0.05) | {stats.nominal_significant} |\n"
                f"| FDR Significant | {stats.fdr_significant} |\n"
            )

            # Top correlations table with inline definitions
            if stats.top_correlations:
                lines.append("\n#### Top Correlations\n")
                # Check if we have CI data
                has_ci = any(
                    corr.ci_low is not None and corr.ci_high is not None
                    for corr in stats.top_correlations
                )

                if has_ci:
                    lines.append(
                        "| Exp1 Trait | Exp2 Trait | ρ (Spearman) | 95% CI | p | q (FDR) | Power | n | Sig? |\n"
                        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
                    )
                else:
                    lines.append(
                        "| Exp1 Trait | Exp2 Trait | ρ (Spearman) | p | q (FDR) | Power | n | Sig? |\n"
                        "| --- | --- | --- | --- | --- | --- | --- | --- |"
                    )

                for corr in stats.top_correlations[:10]:  # Limit to top 10
                    sig_marker = "✓" if corr.significant_fdr else ""

                    if has_ci and corr.ci_low is not None and corr.ci_high is not None:
                        ci_str = f"[{corr.ci_low:.2f}, {corr.ci_high:.2f}]"
                        lines.append(
                            f"| {corr.exp1_trait} | {corr.exp2_trait} | "
                            f"{corr.r_value:.3f} | {ci_str} | {corr.p_value:.4f} | "
                            f"{corr.p_adjusted:.4f} | {corr.power:.2f} | "
                            f"{corr.n_genotypes} | {sig_marker} |"
                        )
                    else:
                        lines.append(
                            f"| {corr.exp1_trait} | {corr.exp2_trait} | "
                            f"{corr.r_value:.3f} | {corr.p_value:.4f} | "
                            f"{corr.p_adjusted:.4f} | {corr.power:.2f} | "
                            f"{corr.n_genotypes} | {sig_marker} |"
                        )

                # Add legend
                lines.append("")
                lines.append(
                    "*Legend: ρ = Spearman correlation coefficient, p = nominal p-value, "
                    "q = FDR-adjusted p-value, Power = achieved statistical power, "
                    "n = sample size (genotypes), Sig? = FDR-significant (q < 0.05)*\n"
                )

            # FDR=0 interpretation section
            fdr_lines = self._format_fdr_interpretation(stats, run_summary.power_stats)
            lines.extend(fdr_lines)

            # Power statistics section
            if run_summary.power_stats:
                lines.append("### Power Analysis\n")
                power = run_summary.power_stats

                # Power parameters table
                target_pct = TARGET_POWER * 100
                lines.append("**Analysis Parameters:**\n")
                lines.append(
                    f"- **Significance level (α):** {power.alpha}\n"
                    f"- **Modal sample size (n):** {power.n_genotypes_modal}\n"
                    f"- **Minimum detectable |r| at {target_pct:.0f}% power:** {power.minimum_detectable_r:.2f}\n"
                )
                if power.recommended_n_for_r40 > 0:
                    lines.append(
                        f"- **Required n for |r|=0.40 at {target_pct:.0f}% power:** {power.recommended_n_for_r40}\n"
                    )
                lines.append("")

                # Power distribution
                lines.append("**Power Distribution:**\n")
                lines.append(
                    f"- **Min Power:** {power.min_power:.2f}\n"
                    f"- **Median Power:** {power.median_power:.2f}\n"
                    f"- **Max Power:** {power.max_power:.2f}\n"
                    f"- **% Above {target_pct:.0f}%:** {power.pct_above_80:.1f}%\n"
                )

                # Power warning if underpowered
                if power.pct_above_80 < 50:
                    lines.append(
                        "\n**⚠️ Warning: Study may be underpowered.** "
                        f"Only {power.pct_above_80:.1f}% of correlations have ≥{target_pct:.0f}% power. "
                        "Consider increasing sample size for future studies.\n"
                    )

            # Joint plots section
            if run_summary.joint_plot_paths:
                lines.append("### Top Correlation Joint Plots\n")
                for i, path in enumerate(run_summary.joint_plot_paths[:3]):
                    img_ref = self._format_image_reference(
                        path, f"Joint Plot {i+1}", should_embed
                    )
                    if img_ref:
                        lines.append(f"{img_ref}\n")
                lines.append("")

            # Correlation summary plot
            if run_summary.correlation_summary_path:
                lines.append("### Correlation Summary\n")
                img_ref = self._format_image_reference(
                    run_summary.correlation_summary_path,
                    "Correlation Summary",
                    should_embed,
                )
                if img_ref:
                    lines.append(f"{img_ref}\n")

            lines.append("---\n")

        return "\n".join(lines)

    def to_html(
        self,
        image_mode: str = "file_path",
        embed_threshold_bytes: int = DEFAULT_EMBED_THRESHOLD_BYTES,
    ) -> str:
        """Render summary as HTML for browser viewing.

        Args:
            image_mode: Image handling mode (file_path, embed, or auto).
            embed_threshold_bytes: Maximum size for embedding images.

        Returns:
            HTML-formatted summary string.
        """
        import html as html_module

        # Get markdown content first
        markdown_content = self.to_markdown(
            image_mode=image_mode,
            embed_threshold_bytes=embed_threshold_bytes,
        )

        # Convert markdown to HTML manually (basic conversion)
        html_body = self._markdown_to_html(markdown_content)

        # Build complete HTML document with styling
        html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Cross-Platform Analysis Summary</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #24292e;
        }}
        h1, h2, h3, h4 {{
            border-bottom: 1px solid #eaecef;
            padding-bottom: 0.3em;
            margin-top: 1.5em;
        }}
        h1 {{ font-size: 2em; }}
        h2 {{ font-size: 1.5em; }}
        h3 {{ font-size: 1.25em; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        th, td {{
            border: 1px solid #dfe2e5;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f6f8fa;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 20px auto;
            border: 1px solid #dfe2e5;
            border-radius: 4px;
        }}
        pre {{
            background: #f6f8fa;
            padding: 16px;
            overflow-x: auto;
            border-radius: 4px;
        }}
        code {{
            background: #f6f8fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }}
        hr {{
            border: none;
            border-top: 1px solid #eaecef;
            margin: 24px 0;
        }}
        ul, ol {{
            padding-left: 2em;
        }}
        strong {{
            font-weight: 600;
        }}
        .warning {{
            background-color: #fffbdd;
            border: 1px solid #f9c513;
            border-radius: 4px;
            padding: 12px;
            margin: 16px 0;
        }}
    </style>
</head>
<body>
{body}
</body>
</html>"""

        return html_template.format(body=html_body)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to HTML (basic conversion).

        Args:
            markdown: Markdown-formatted string.

        Returns:
            HTML-formatted string.
        """
        import html as html_module
        import re

        lines = markdown.split("\n")
        html_lines = []
        in_table = False
        in_list = False
        table_rows = []

        for line in lines:
            # Headers
            if line.startswith("# "):
                html_lines.append(f"<h1>{html_module.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                html_lines.append(f"<h2>{html_module.escape(line[3:])}</h2>")
            elif line.startswith("### "):
                html_lines.append(f"<h3>{html_module.escape(line[4:])}</h3>")
            elif line.startswith("#### "):
                html_lines.append(f"<h4>{html_module.escape(line[5:])}</h4>")
            # Horizontal rule
            elif line.strip() == "---":
                html_lines.append("<hr>")
            # Table rows
            elif line.startswith("|"):
                if not in_table:
                    in_table = True
                    table_rows = []
                table_rows.append(line)
            # End of table
            elif in_table and not line.startswith("|"):
                html_lines.append(self._table_to_html(table_rows))
                in_table = False
                table_rows = []
                # Process current line
                html_lines.append(self._process_inline_markdown(line))
            # Images
            elif line.strip().startswith("!["):
                match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
                if match:
                    alt, src = match.groups()
                    html_lines.append(
                        f'<img src="{html_module.escape(src)}" alt="{html_module.escape(alt)}">'
                    )
                else:
                    html_lines.append(self._process_inline_markdown(line))
            # List items
            elif line.strip().startswith("- "):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                content = self._process_inline_markdown(line.strip()[2:])
                html_lines.append(f"<li>{content}</li>")
            # End of list
            elif in_list and not line.strip().startswith("- "):
                html_lines.append("</ul>")
                in_list = False
                html_lines.append(self._process_inline_markdown(line))
            # Regular paragraph
            else:
                html_lines.append(self._process_inline_markdown(line))

        # Close any open elements
        if in_table:
            html_lines.append(self._table_to_html(table_rows))
        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def _table_to_html(self, rows: List[str]) -> str:
        """Convert markdown table rows to HTML table.

        Args:
            rows: List of markdown table row strings.

        Returns:
            HTML table string.
        """
        import html as html_module

        if len(rows) < 2:
            return ""

        html_parts = ["<table>"]

        # Header row
        header_cells = [c.strip() for c in rows[0].split("|")[1:-1]]
        html_parts.append("<thead><tr>")
        for cell in header_cells:
            html_parts.append(f"<th>{self._process_inline_markdown(cell)}</th>")
        html_parts.append("</tr></thead>")

        # Body rows (skip separator row)
        html_parts.append("<tbody>")
        for row in rows[2:]:  # Skip header and separator
            cells = [c.strip() for c in row.split("|")[1:-1]]
            html_parts.append("<tr>")
            for cell in cells:
                html_parts.append(f"<td>{self._process_inline_markdown(cell)}</td>")
            html_parts.append("</tr>")
        html_parts.append("</tbody></table>")

        return "".join(html_parts)

    def _process_inline_markdown(self, text: str) -> str:
        """Process inline markdown (bold, italic, code).

        Args:
            text: Text with potential inline markdown.

        Returns:
            HTML with inline elements converted.
        """
        import html as html_module
        import re

        if not text.strip():
            return "<p></p>"

        # Escape HTML first
        result = html_module.escape(text)

        # Bold: **text** -> <strong>text</strong>
        result = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", result)

        # Italic: *text* -> <em>text</em>
        result = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", result)

        # Inline code: `text` -> <code>text</code>
        result = re.sub(r"`([^`]+)`", r"<code>\1</code>", result)

        # Convert special characters back that were over-escaped
        result = result.replace("&lt;strong&gt;", "<strong>")
        result = result.replace("&lt;/strong&gt;", "</strong>")
        result = result.replace("&lt;em&gt;", "<em>")
        result = result.replace("&lt;/em&gt;", "</em>")
        result = result.replace("&lt;code&gt;", "<code>")
        result = result.replace("&lt;/code&gt;", "</code>")

        return f"<p>{result}</p>" if result.strip() else ""


# ============================================================================
# SUMMARY GENERATOR
# ============================================================================


class CrossPlatformSummaryGenerator:
    """Generator for cross-platform analysis summaries.

    Parses pipeline output directories and generates comprehensive
    markdown summaries with embedded visualizations.
    """

    def __init__(self, run_dir: Path):
        """Initialize generator with pipeline run directory.

        Args:
            run_dir: Path to a cross-platform run directory or parent directory
                    containing multiple cross-platform comparisons.
        """
        self.run_dir = Path(run_dir)

    def generate(self) -> CrossPlatformSummary:
        """Generate summary from pipeline outputs.

        Returns:
            CrossPlatformSummary containing all parsed statistics.
        """
        validation_result = ValidationResult(passed=True)
        run_summaries = []

        # Find cross-platform run directories
        run_dirs = self._find_cross_platform_runs()

        if not run_dirs:
            validation_result.warnings.append(
                f"No cross-platform runs found in {self.run_dir}"
            )
            return CrossPlatformSummary(
                run_summaries=[],
                validation_result=validation_result,
                source_dir=self.run_dir,
            )

        for run_path in run_dirs:
            try:
                run_summary = self._parse_run_directory(run_path)
                run_summaries.append(run_summary)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", run_path, e)
                validation_result.warnings.append(
                    f"Failed to parse {run_path.name}: {e}"
                )

        return CrossPlatformSummary(
            run_summaries=run_summaries,
            validation_result=validation_result,
            source_dir=self.run_dir,
        )

    def _find_cross_platform_runs(self) -> List[Path]:
        """Find cross-platform output directories.

        Returns:
            List of paths to cross-platform run directories.
        """
        run_dirs = []

        # Check if run_dir itself is a cross-platform run
        if (self.run_dir / "cross_platform_correlations.csv").exists():
            return [self.run_dir]

        # Check for cross_platform subdirectory
        cross_platform_dir = self.run_dir / "cross_platform"
        if cross_platform_dir.exists():
            for subdir in cross_platform_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith("cross_platform_"):
                    if (subdir / "cross_platform_correlations.csv").exists():
                        run_dirs.append(subdir)
                    elif (subdir / "pipeline_summary.json").exists():
                        run_dirs.append(subdir)

        # Check immediate subdirectories for cross-platform runs
        for subdir in self.run_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("cross_platform_"):
                if (subdir / "cross_platform_correlations.csv").exists():
                    if subdir not in run_dirs:
                        run_dirs.append(subdir)

        return sorted(run_dirs)

    def _parse_run_directory(self, run_dir: Path) -> CrossPlatformRunSummary:
        """Parse a single cross-platform run directory.

        Args:
            run_dir: Path to cross-platform run directory.

        Returns:
            CrossPlatformRunSummary with parsed statistics.
        """
        # Read pipeline summary
        metadata = self._read_metadata(run_dir)
        config = self._extract_config(metadata)

        # Extract experiment names
        exp1_name = self._get_exp_name(metadata, "exp1_name", "Experiment 1")
        exp2_name = self._get_exp_name(metadata, "exp2_name", "Experiment 2")

        # Read correlation data
        correlation_stats = self._read_correlations(run_dir)

        # Calculate power statistics (pass config for alpha)
        power_stats = self._calculate_power_stats(run_dir, config)

        # Read trait reduction statistics
        exp1_trait_reduction = None
        exp2_trait_reduction = None

        if config.trait_reduction_method == "clustering":
            target = config.trait_reduction_target or "both"
            if target in ["exp1", "both"]:
                exp1_trait_reduction = self._read_trait_clusters(
                    run_dir, "exp1", metadata
                )
            if target in ["exp2", "both"]:
                exp2_trait_reduction = self._read_trait_clusters(
                    run_dir, "exp2", metadata
                )

        # Find visualization files
        exp1_dendrogram = self._find_file(
            run_dir, "exp1_trait_clustering_dendrogram.png"
        )
        exp2_dendrogram = self._find_file(
            run_dir, "exp2_trait_clustering_dendrogram.png"
        )
        exp1_heatmap = self._find_file(run_dir, "exp1_trait_cluster_heatmap.png")
        exp2_heatmap = self._find_file(run_dir, "exp2_trait_cluster_heatmap.png")
        representative_heatmap = self._find_file(
            run_dir, "cross_platform_representative_heatmap.png"
        )
        correlation_summary = self._find_file(run_dir, "correlation_summary.png")
        joint_plots = self._find_joint_plots(run_dir)

        return CrossPlatformRunSummary(
            run_dir=run_dir,
            exp1_name=exp1_name,
            exp2_name=exp2_name,
            config=config,
            correlation_stats=correlation_stats,
            power_stats=power_stats,
            exp1_trait_reduction=exp1_trait_reduction,
            exp2_trait_reduction=exp2_trait_reduction,
            exp1_dendrogram_path=exp1_dendrogram,
            exp2_dendrogram_path=exp2_dendrogram,
            exp1_heatmap_path=exp1_heatmap,
            exp2_heatmap_path=exp2_heatmap,
            representative_heatmap_path=representative_heatmap,
            correlation_summary_path=correlation_summary,
            joint_plot_paths=joint_plots,
        )

    def _read_metadata(self, run_dir: Path) -> Dict[str, Any]:
        """Read pipeline_summary.json metadata.

        Args:
            run_dir: Path to run directory.

        Returns:
            Dictionary with pipeline metadata.
        """
        summary_file = run_dir / "pipeline_summary.json"
        if not summary_file.exists():
            return {}

        with open(summary_file) as f:
            return json.load(f)

    def _extract_config(self, metadata: Dict[str, Any]) -> ConfigInfo:
        """Extract configuration info from metadata.

        Args:
            metadata: Pipeline metadata dictionary.

        Returns:
            ConfigInfo with extracted parameters.
        """
        config = metadata.get("config", {})

        return ConfigInfo(
            correlation_method=config.get("correlation_method", "spearman"),
            trait_reduction_method=config.get("trait_reduction_method", "none"),
            trait_reduction_target=config.get("trait_reduction_target"),
            trait_clustering_threshold=config.get("trait_clustering_threshold"),
            fdr_correction_method=config.get("fdr_correction_method"),
            significance_level=config.get("significance_level"),
        )

    def _get_exp_name(self, metadata: Dict[str, Any], key: str, default: str) -> str:
        """Extract experiment name from metadata.

        Args:
            metadata: Pipeline metadata.
            key: Key to look for (e.g., "exp1_name").
            default: Default value if not found.

        Returns:
            Experiment name string.
        """
        # Try config first
        config = metadata.get("config", {})
        if key in config:
            return config[key]

        # Try steps metadata
        for step in metadata.get("steps", []):
            step_meta = step.get("metadata", {})
            if key in step_meta:
                return step_meta[key]

        return default

    def _read_correlations(self, run_dir: Path) -> CorrelationStats:
        """Read and parse correlation CSV.

        Args:
            run_dir: Path to run directory.

        Returns:
            CorrelationStats with parsed data.
        """
        corr_file = run_dir / "cross_platform_correlations.csv"
        if not corr_file.exists():
            return CorrelationStats(
                total_correlations=0,
                nominal_significant=0,
                fdr_significant=0,
                top_correlations=[],
            )

        df = pd.read_csv(corr_file)

        if df.empty:
            return CorrelationStats(
                total_correlations=0,
                nominal_significant=0,
                fdr_significant=0,
                top_correlations=[],
            )

        # Count correlations
        total = len(df)
        nominal_sig = (
            (df["spearman_p"] < 0.05).sum() if "spearman_p" in df.columns else 0
        )
        fdr_sig = df["significant_fdr"].sum() if "significant_fdr" in df.columns else 0

        # Get top correlations by |r|
        df_sorted = df.reindex(
            df["spearman_r"].abs().sort_values(ascending=False).index
        )

        top_correlations = []
        for _, row in df_sorted.head(20).iterrows():
            # Extract CI values if present
            ci_low = None
            ci_high = None
            if "spearman_ci_low" in df.columns and pd.notna(row.get("spearman_ci_low")):
                ci_low = float(row["spearman_ci_low"])
            if "spearman_ci_high" in df.columns and pd.notna(
                row.get("spearman_ci_high")
            ):
                ci_high = float(row["spearman_ci_high"])

            top_correlations.append(
                TopCorrelation(
                    exp1_trait=row["exp1_trait"],
                    exp2_trait=row["exp2_trait"],
                    r_value=row["spearman_r"],
                    p_value=row.get("spearman_p", 0),
                    p_adjusted=row.get("spearman_p_adjusted", 0),
                    power=row.get("achieved_power", 0),
                    n_genotypes=int(row.get("n_genotypes", 0)),
                    significant_fdr=bool(row.get("significant_fdr", False)),
                    ci_low=ci_low,
                    ci_high=ci_high,
                )
            )

        return CorrelationStats(
            total_correlations=total,
            nominal_significant=int(nominal_sig),
            fdr_significant=int(fdr_sig),
            top_correlations=top_correlations,
        )

    def _calculate_power_stats(
        self, run_dir: Path, config: Optional[ConfigInfo] = None
    ) -> Optional[PowerStats]:
        """Calculate power statistics from correlation CSV.

        Args:
            run_dir: Path to run directory.
            config: Configuration info for reading significance_level.

        Returns:
            PowerStats or None if no power data available.
        """
        corr_file = run_dir / "cross_platform_correlations.csv"
        if not corr_file.exists():
            return None

        df = pd.read_csv(corr_file)

        if df.empty or "achieved_power" not in df.columns:
            return None

        power_values = df["achieved_power"].dropna()
        if power_values.empty:
            return None

        # Get alpha from config, falling back to default
        alpha = DEFAULT_ALPHA
        if config and config.significance_level is not None:
            alpha = config.significance_level

        # Calculate modal sample size (most common n_genotypes)
        n_genotypes_modal = 0
        if "n_genotypes" in df.columns:
            n_values = df["n_genotypes"].dropna()
            if not n_values.empty:
                mode_result = n_values.mode()
                if len(mode_result) > 0:
                    n_genotypes_modal = int(mode_result.iloc[0])
                else:
                    n_genotypes_modal = int(n_values.median())

        # Calculate minimum detectable effect size at target power
        minimum_detectable_r = 0.0
        if n_genotypes_modal >= 4:
            minimum_detectable_r = _calculate_minimum_detectable_r(
                n_genotypes_modal, alpha, TARGET_POWER
            )

        # Calculate required n for detecting r=0.40 at target power
        recommended_n_for_r40 = _calculate_required_n(0.4, alpha, TARGET_POWER)

        return PowerStats(
            min_power=float(power_values.min()),
            max_power=float(power_values.max()),
            median_power=float(power_values.median()),
            pct_above_80=float(
                (power_values >= TARGET_POWER).sum() / len(power_values) * 100
            ),
            alpha=alpha,
            n_genotypes_modal=n_genotypes_modal,
            minimum_detectable_r=minimum_detectable_r,
            recommended_n_for_r40=recommended_n_for_r40,
        )

    def _read_trait_clusters(
        self, run_dir: Path, exp_name: str, metadata: Dict[str, Any]
    ) -> Optional[TraitReductionStats]:
        """Read trait cluster membership file.

        Args:
            run_dir: Path to run directory.
            exp_name: "exp1" or "exp2".
            metadata: Pipeline metadata for fallback stats.

        Returns:
            TraitReductionStats or None if not available.
        """
        cluster_file = run_dir / f"{exp_name}_trait_clusters.csv"

        # Try to get stats from cluster file
        if cluster_file.exists():
            df = pd.read_csv(cluster_file)
            original_traits = len(df)
            n_clusters = df["cluster_id"].nunique()
            representative_traits = df["is_representative"].sum()

            return TraitReductionStats(
                original_traits=original_traits,
                n_clusters=n_clusters,
                representative_traits=int(representative_traits),
            )

        # Fall back to metadata
        for step in metadata.get("steps", []):
            step_meta = step.get("metadata", {})
            original_key = f"{exp_name}_original_traits"
            reduced_key = f"{exp_name}_reduced_traits"
            clusters_key = f"{exp_name}_n_clusters"

            if original_key in step_meta:
                return TraitReductionStats(
                    original_traits=step_meta[original_key],
                    n_clusters=step_meta.get(
                        clusters_key, step_meta.get(reduced_key, 0)
                    ),
                    representative_traits=step_meta.get(reduced_key, 0),
                )

        return None

    def _find_file(self, run_dir: Path, filename: str) -> Optional[Path]:
        """Find a file in the run directory.

        Args:
            run_dir: Path to run directory.
            filename: Name of file to find.

        Returns:
            Path to file or None if not found.
        """
        file_path = run_dir / filename
        return file_path if file_path.exists() else None

    def _find_joint_plots(self, run_dir: Path) -> List[Path]:
        """Find joint plot files in run directory.

        Args:
            run_dir: Path to run directory.

        Returns:
            List of paths to joint plot PNGs.
        """
        joint_plots = []
        for f in run_dir.glob("joint_plot_*.png"):
            joint_plots.append(f)
        return sorted(joint_plots)
