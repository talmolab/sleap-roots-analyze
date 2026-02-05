"""Convert cross-platform summary to HTML for browser viewing.

Usage:
    uv run python scripts/convert_summary_to_html.py [run_dir]
    uv run python scripts/convert_summary_to_html.py pipeline_runs/2026-02-04_120723
    uv run python scripts/convert_summary_to_html.py --latest

If run_dir not provided, uses the latest pipeline run.

This script generates an HTML file that can be opened directly in a browser.
The HTML version works around markdown rendering limitations in Chrome and
VS Code's data URI blocking.
"""

import sys
from pathlib import Path


def find_latest_run() -> Path | None:
    """Find the most recent pipeline run directory."""
    pipeline_runs = Path("pipeline_runs")
    if not pipeline_runs.exists():
        return None

    # Find directories with cross_platform subdirectories
    candidates = []
    for run_dir in pipeline_runs.iterdir():
        if run_dir.is_dir() and (run_dir / "cross_platform").exists():
            candidates.append(run_dir)

    if not candidates:
        return None

    # Return most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    """Main entry point."""
    # Determine run directory
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg == "--latest":
            run_dir = find_latest_run()
            if not run_dir:
                print("Error: No pipeline runs found")
                sys.exit(1)
            print(f"Using latest run: {run_dir}")
        else:
            run_dir = Path(arg)
    else:
        run_dir = find_latest_run()
        if not run_dir:
            print("Error: No pipeline runs found. Specify a run directory.")
            print("Usage: uv run python scripts/convert_summary_to_html.py [run_dir]")
            sys.exit(1)
        print(f"Using latest run: {run_dir}")

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    # Check for SUMMARY.md
    summary_md = run_dir / "SUMMARY.md"
    if not summary_md.exists():
        print(f"Error: No SUMMARY.md found in {run_dir}")
        sys.exit(1)

    # Use the CrossPlatformSummaryGenerator to create HTML
    try:
        from sleap_roots_analyze.summary.cross_platform_summary import (
            CrossPlatformSummaryGenerator,
        )

        generator = CrossPlatformSummaryGenerator(run_dir)
        summary = generator.generate()

        if not summary.run_summaries:
            print("Warning: No cross-platform runs found, generating from markdown")
            # Fall back to markdown conversion
            html_content = convert_markdown_to_html(summary_md)
        else:
            # Use the native to_html method
            html_content = summary.to_html(image_mode="file_path")

    except ImportError:
        print("Warning: Could not import summary module, using markdown conversion")
        html_content = convert_markdown_to_html(summary_md)

    # Write HTML file
    html_path = run_dir / "SUMMARY.html"
    html_path.write_text(html_content, encoding="utf-8")

    size_kb = html_path.stat().st_size / 1024
    print(f"Created: {html_path}")
    print(f"Size: {size_kb:.1f} KB")
    print(f"\nOpen in browser: start {html_path}")


def convert_markdown_to_html(md_path: Path) -> str:
    """Convert markdown file to HTML using client-side rendering.

    This is a fallback for when the summary module can't be imported.
    """
    import json

    md_content = md_path.read_text(encoding="utf-8")
    md_escaped = json.dumps(md_content)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Cross-Platform Analysis Summary</title>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; display: block; margin: 20px auto; }}
        pre {{ background: #f6f8fa; padding: 16px; overflow-x: auto; }}
        code {{ background: #f6f8fa; padding: 2px 6px; border-radius: 3px; }}
        h1, h2, h3 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div id="content">Loading...</div>
    <script>
        const md = {md_escaped};
        document.getElementById('content').innerHTML = marked.parse(md);
    </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
