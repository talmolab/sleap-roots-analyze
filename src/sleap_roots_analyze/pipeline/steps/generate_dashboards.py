"""Step 11: Generate HTML dashboards combining all visualizations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class GenerateDashboardsStep(BaseStep):
    """Generate HTML dashboards combining all visualizations.

    Creates an HTML dashboard that links to all generated static
    and interactive figures, providing a central navigation point
    for all pipeline outputs.

    Features:
    - Summary statistics cards (sample counts, figure counts)
    - Figure gallery with thumbnails/links
    - File listing with download links
    - Responsive CSS styling with gradient theme
    - Automatic detection of all generated figures

    Configuration:
        Control dashboard generation via config.dashboard:
        - enabled: Enable/disable step
        - create_summary_dashboard: Create main summary dashboard
        - create_trait_dashboard: Create per-trait dashboards (future)

    Outputs:
        - dashboard.html: Main dashboard HTML file
        - 11_dashboard_manifest.json: Dashboard generation metadata

    Example:
        ```python
        config.dashboard.enabled = True
        config.dashboard.create_summary_dashboard = True

        step = GenerateDashboardsStep()
        result = step.execute(data, config, run_dir, prev_result)

        # Open in browser: open dashboard.html
        ```
    """

    def __init__(self):
        """Initialize GenerateDashboardsStep."""
        super().__init__(
            step_name="GenerateDashboards",
            description="Generate HTML dashboards with all visualizations",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute dashboard generation.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration with dashboard settings.
            run_dir: Directory to save outputs.
            prev_result: Result from previous step (contains all figure paths).

        Returns:
            StepResult with DataFrame and dashboard file path.
        """
        if not config.dashboard.enabled:
            logger.info("Dashboard generation disabled, skipping")
            return StepResult(
                data=data,
                metadata=prev_result.metadata if prev_result else {},
            )

        logger.info("Generating HTML dashboard...")
        df = data.copy()

        # Get metadata from previous steps
        metadata = prev_result.metadata if prev_result else {}
        static_figures = metadata.get("static_figures", [])
        interactive_figures = metadata.get("interactive_figures", [])
        summary_files = metadata.get("summary_files", [])

        # Generate dashboard HTML
        dashboard_path = run_dir / "dashboard.html"
        self._create_dashboard_html(
            dashboard_path,
            run_dir,
            static_figures,
            interactive_figures,
            summary_files,
            df,
            config,
        )

        # Save manifest
        manifest = {
            "dashboard_path": str(dashboard_path.relative_to(run_dir)),
            "total_static_figures": len(static_figures),
            "total_interactive_figures": len(interactive_figures),
            "run_directory": str(run_dir),
        }
        manifest_file = self.save_json(manifest, "11_dashboard_manifest.json", run_dir)

        logger.info(f"Dashboard created: {dashboard_path}")

        # Update metadata
        new_metadata = {
            **metadata,
            "dashboard_path": dashboard_path,
            "dashboard_manifest": manifest,
        }

        return StepResult(
            data=df,
            metadata=new_metadata,
            files_generated=[dashboard_path, manifest_file],
        )

    def _create_dashboard_html(
        self,
        output_path: Path,
        run_dir: Path,
        static_figures: list,
        interactive_figures: list,
        summary_files: list,
        df: pd.DataFrame,
        config: Any,
    ):
        """Create HTML dashboard."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization Pipeline Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .stat-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .section {{
            margin: 30px 0;
        }}
        .figure-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .figure-card {{
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .figure-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        }}
        .figure-link {{
            color: #4CAF50;
            text-decoration: none;
            font-weight: 500;
            display: block;
            margin: 10px 0;
        }}
        .figure-link:hover {{
            text-decoration: underline;
        }}
        .file-list {{
            list-style-type: none;
            padding: 0;
        }}
        .file-list li {{
            padding: 8px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-left: 3px solid #4CAF50;
            border-radius: 4px;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
            font-style: italic;
        }}
        .nav-buttons {{
            margin: 20px 0;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        .btn {{
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            display: inline-block;
            transition: background-color 0.3s;
        }}
        .btn:hover {{
            background-color: #45a049;
        }}
        .btn-secondary {{
            background-color: #008CBA;
        }}
        .btn-secondary:hover {{
            background-color: #007399;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Visualization Pipeline Dashboard</h1>
        <p class="timestamp">Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{len(df)}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(static_figures)}</div>
                <div class="stat-label">Static Figures</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(interactive_figures)}</div>
                <div class="stat-label">Interactive Plots</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{config.pipeline_name}</div>
                <div class="stat-label">Pipeline</div>
            </div>
        </div>

        <div class="nav-buttons">
            <a href="#static" class="btn">📊 Static Figures</a>
            <a href="#interactive" class="btn btn-secondary">🎨 Interactive Plots</a>
            <a href="#files" class="btn">📁 All Files</a>
        </div>

        <div class="section" id="static">
            <h2>📊 Static Figures</h2>
            <div class="figure-grid">
                {self._generate_figure_cards(static_figures, run_dir, "static")}
            </div>
        </div>

        <div class="section" id="interactive">
            <h2>🎨 Interactive Visualizations</h2>
            <div class="figure-grid">
                {self._generate_figure_cards(interactive_figures, run_dir, "interactive")}
            </div>
        </div>

        <div class="section" id="files">
            <h2>📁 Output Files</h2>
            <ul class="file-list">
                {self._generate_file_list(run_dir)}
            </ul>
        </div>
    </div>
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")

    def _generate_figure_cards(
        self, figures: list, run_dir: Path, category: str
    ) -> str:
        """Generate HTML for figure cards."""
        if not figures:
            return "<p>No figures generated in this category.</p>"

        cards = []
        for fig_path in figures:
            if isinstance(fig_path, (str, Path)):
                fig_path = Path(fig_path)
                rel_path = fig_path.relative_to(run_dir)
                name = fig_path.stem.replace("_", " ").title()
                cards.append(
                    f"""
                <div class="figure-card">
                    <strong>{name}</strong>
                    <a href="{rel_path}" class="figure-link" target="_blank">
                        Open {fig_path.suffix.upper()[1:]}
                    </a>
                </div>
                """
                )
        return "\n".join(cards)

    def _generate_file_list(self, run_dir: Path) -> str:
        """Generate HTML list of all output files."""
        files = []
        for file_path in sorted(run_dir.rglob("*")):
            if file_path.is_file() and file_path.name != "dashboard.html":
                rel_path = file_path.relative_to(run_dir)
                files.append(
                    f"""
                <li>
                    <a href="{rel_path}" target="_blank">{rel_path}</a>
                </li>
                """
                )
        return "\n".join(files) if files else "<li>No files found</li>"
