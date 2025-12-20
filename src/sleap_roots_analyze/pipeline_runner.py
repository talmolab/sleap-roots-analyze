"""Pipeline runner for orchestrating multi-pipeline execution.

This module provides the PipelineRunner class that orchestrates execution of
QC, Viz, and Cross-Platform pipelines with automatic path updates between
dependent pipelines and comprehensive summary generation.

Usage:
    runner = PipelineRunner(manifest_path, output_dir)
    runner.run_all()  # Run all pipelines
    runner.run_all(qc_only=True)  # Run only QC pipelines
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class PipelineRunner:
    """Orchestrates multi-pipeline execution with dependency management."""

    def __init__(
        self,
        manifest_path: Path | str,
        output_dir: Path | str = "pipeline_runs",
        verbose: bool = False,
    ):
        """Initialize the pipeline runner.

        Args:
            manifest_path: Path to the run manifest YAML file
            output_dir: Base directory for pipeline outputs
            verbose: Enable verbose output
        """
        self.manifest_path = Path(manifest_path)
        self.output_dir = Path(output_dir)
        self.verbose = verbose
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_timestamp

        # Load and validate manifest
        self.manifest = self._load_manifest()
        self._validate_manifest()

        # Track run results
        self.qc_outputs: dict[str, Path] = {}
        self.viz_outputs: dict[str, Path] = {}
        self.cross_platform_outputs: dict[str, Path] = {}
        self.run_results: dict[str, dict[str, Any]] = {
            "qc": {},
            "viz": {},
            "cross_platform": {},
        }

    def _load_manifest(self) -> dict[str, Any]:
        """Load the run manifest YAML file."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        with open(self.manifest_path) as f:
            return yaml.safe_load(f)

    def _validate_manifest(self) -> list[str]:
        """Validate all config files in the manifest exist.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        base_dir = self.manifest_path.parent

        # Check QC configs
        for config in self.manifest.get("qc_configs", []):
            config_path = base_dir / config
            if not config_path.exists():
                errors.append(f"QC config not found: {config_path}")

        # Check Viz configs
        for config in self.manifest.get("viz_configs", []):
            config_path = base_dir / config
            if not config_path.exists():
                errors.append(f"Viz config not found: {config_path}")

        # Check Cross-Platform configs
        for config in self.manifest.get("cross_platform_configs", []):
            config_path = base_dir / config
            if not config_path.exists():
                errors.append(f"Cross-platform config not found: {config_path}")

        return errors

    def dry_run(self) -> dict[str, Any]:
        """Validate manifest and show execution plan without running.

        Returns:
            Dictionary with validation results and execution plan
        """
        errors = self._validate_manifest()
        plan = {
            "manifest": str(self.manifest_path),
            "run_name": self.manifest.get("run_name", "Unnamed"),
            "output_dir": str(self.run_dir),
            "timestamp": self.run_timestamp,
            "validation_errors": errors,
            "valid": len(errors) == 0,
            "execution_plan": {
                "qc_configs": self.manifest.get("qc_configs", []),
                "viz_configs": self.manifest.get("viz_configs", []),
                "cross_platform_configs": self.manifest.get(
                    "cross_platform_configs", []
                ),
            },
        }
        return plan

    def run_all(
        self,
        qc_only: bool = False,
        viz_only: bool = False,
        cross_only: bool = False,
        no_summary: bool = False,
    ) -> dict[str, Any]:
        """Execute pipelines in dependency order.

        Args:
            qc_only: Run only QC pipelines
            viz_only: Run only Viz pipelines (requires existing QC outputs)
            cross_only: Run only Cross-Platform pipelines (requires existing QC outputs)
            no_summary: Skip summary generation

        Returns:
            Dictionary with run results
        """
        # Create run directory structure
        self._create_run_directories()

        # Determine which pipelines to run
        run_qc = not viz_only and not cross_only
        run_viz = not qc_only and not cross_only
        run_cross = not qc_only and not viz_only

        # Run QC pipelines first (if requested)
        if run_qc:
            self._run_qc_pipelines()

        # Run Viz pipelines (if requested)
        if run_viz and self.manifest.get("viz_configs"):
            self._run_viz_pipelines()

        # Run Cross-Platform pipelines (if requested)
        if run_cross and self.manifest.get("cross_platform_configs"):
            self._run_cross_platform_pipelines()

        # Generate summary
        if not no_summary:
            self.generate_summary()

        # Create latest symlink
        self._create_latest_symlink()

        return self.run_results

    def _create_run_directories(self) -> None:
        """Create the timestamped run directory structure."""
        (self.run_dir / "qc").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "viz").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "cross_platform").mkdir(parents=True, exist_ok=True)

    def _run_qc_pipelines(self) -> None:
        """Run all QC pipelines."""
        base_dir = self.manifest_path.parent
        qc_output_dir = self.run_dir / "qc"

        for config_rel in self.manifest.get("qc_configs", []):
            config_path = base_dir / config_rel
            print(f"\n{'=' * 60}")
            print(f"Running QC: {config_rel}")
            print(f"{'=' * 60}")

            result = self._run_pipeline_command("qc", config_path, qc_output_dir)

            # Track output path
            if result.get("success"):
                self.qc_outputs[config_rel] = Path(result.get("output_path", ""))

            self.run_results["qc"][config_rel] = result

    def _run_viz_pipelines(self) -> None:
        """Run all Viz pipelines with updated paths."""
        base_dir = self.manifest_path.parent
        viz_output_dir = self.run_dir / "viz"
        qc_mapping = self.manifest.get("qc_mapping", {})

        for config_rel in self.manifest.get("viz_configs", []):
            config_path = base_dir / config_rel
            print(f"\n{'=' * 60}")
            print(f"Running Viz: {config_rel}")
            print(f"{'=' * 60}")

            # Update config with new QC output path if mapping exists
            updated_config = self._update_viz_config(
                config_path, config_rel, qc_mapping
            )

            result = self._run_pipeline_command(
                "viz", updated_config or config_path, viz_output_dir
            )

            if result.get("success"):
                self.viz_outputs[config_rel] = Path(result.get("output_path", ""))

            self.run_results["viz"][config_rel] = result

    def _run_cross_platform_pipelines(self) -> None:
        """Run all Cross-Platform pipelines with updated paths."""
        base_dir = self.manifest_path.parent
        cross_output_dir = self.run_dir / "cross_platform"
        qc_mapping = self.manifest.get("qc_mapping", {})

        for config_rel in self.manifest.get("cross_platform_configs", []):
            config_path = base_dir / config_rel
            print(f"\n{'=' * 60}")
            print(f"Running Cross-Platform: {config_rel}")
            print(f"{'=' * 60}")

            # Update config with new QC output paths if mapping exists
            updated_config = self._update_cross_platform_config(
                config_path, config_rel, qc_mapping
            )

            result = self._run_pipeline_command(
                "cross-platform", updated_config or config_path, cross_output_dir
            )

            if result.get("success"):
                self.cross_platform_outputs[config_rel] = Path(
                    result.get("output_path", "")
                )

            self.run_results["cross_platform"][config_rel] = result

    @staticmethod
    def _extract_filename(path: str) -> str:
        """Extract the filename from a path string.

        Args:
            path: A file path (can use forward or back slashes)

        Returns:
            The filename portion of the path
        """
        # Normalize path separators and extract filename
        normalized = path.replace("\\", "/")
        return Path(normalized).name

    @staticmethod
    def _update_yaml_path_preserving_structure(
        content: str,
        key_pattern: str,
        new_dir: Path,
        original_filename: str,
    ) -> str:
        """Update a path value in YAML content while preserving structure.

        Uses regex substitution to update only the path value, preserving
        comments, key ordering, and formatting.

        Args:
            content: The full YAML file content as a string
            key_pattern: Regex pattern to match the key (e.g., 'exp1_data_path')
            new_dir: The new directory to use
            original_filename: The filename to preserve from the original path

        Returns:
            Updated YAML content with the new path
        """
        # Build the new path with forward slashes for consistency
        new_path = (new_dir / original_filename).as_posix()

        # Pattern matches: key: "value" or key: 'value' or key: value
        # Captures: (key: )(optional quote)(path)(optional quote)
        pattern = rf'({key_pattern}:\s*)(["\']?)([^"\'\n]+)(["\']?)'

        def replacer(match: re.Match) -> str:
            prefix = match.group(1)  # "key: " part
            open_quote = match.group(2)  # opening quote if any
            close_quote = match.group(4)  # closing quote if any
            return f"{prefix}{open_quote}{new_path}{close_quote}"

        return re.sub(pattern, replacer, content)

    def _update_viz_config(
        self,
        config_path: Path,
        config_rel: str,
        qc_mapping: dict[str, Any],
    ) -> Path | None:
        """Update viz config with new QC output path.

        Preserves the original filename choice and YAML structure (comments,
        key ordering, formatting) by using regex substitution instead of
        yaml.dump().

        Returns updated config path or None if no update needed.
        """
        if config_rel not in qc_mapping:
            return None

        qc_config_rel = qc_mapping[config_rel]
        if qc_config_rel not in self.qc_outputs:
            print(f"  Warning: QC output not found for {qc_config_rel}")
            return None

        qc_output_path = self.qc_outputs[qc_config_rel]

        # Read original config to extract the filename choice
        content = config_path.read_text(encoding="utf-8")

        # Parse YAML to get original csv_path and extract filename
        with open(config_path) as f:
            config = yaml.safe_load(f)

        original_path = config.get("data", {}).get("csv_path", "")
        if not original_path:
            print("  Warning: No data.csv_path found in config")
            return None

        # Extract the filename the user chose (e.g., 07_data_outliers_removed.csv)
        original_filename = self._extract_filename(original_path)

        # Verify the file exists in the new QC output
        new_data_path = qc_output_path / original_filename
        if not new_data_path.exists():
            print(f"  Warning: QC data not found: {new_data_path}")
            # Fall back to 10_final_data.csv if original filename doesn't exist
            new_data_path = qc_output_path / "10_final_data.csv"
            if not new_data_path.exists():
                print(f"  Warning: Fallback QC data not found: {new_data_path}")
                return None
            original_filename = "10_final_data.csv"
            print(f"  Using fallback: {original_filename}")

        # Update the path in YAML content while preserving structure
        # Pattern for nested YAML: csv_path: "value" (with possible indentation)
        updated_content = self._update_yaml_path_preserving_structure(
            content, r"csv_path", qc_output_path, original_filename
        )

        # Write updated config
        updated_path = self.run_dir / "viz" / f"_updated_{config_path.name}"
        updated_path.write_text(updated_content, encoding="utf-8")

        print(f"  Updated data.csv_path -> {new_data_path}")
        return updated_path

    def _update_cross_platform_config(
        self,
        config_path: Path,
        config_rel: str,
        qc_mapping: dict[str, Any],
    ) -> Path | None:
        """Update cross-platform config with new QC output paths.

        Preserves the original filename choice for each experiment and YAML
        structure (comments, key ordering, formatting) by using regex
        substitution instead of yaml.dump().

        Returns updated config path or None if no update needed.
        """
        if config_rel not in qc_mapping:
            return None

        mapping = qc_mapping[config_rel]
        if not isinstance(mapping, dict):
            return None

        # Read original config content
        content = config_path.read_text(encoding="utf-8")

        # Parse YAML to get original paths
        with open(config_path) as f:
            config = yaml.safe_load(f)

        updated = False
        updated_content = content

        # Update exp1 path
        exp1_qc = mapping.get("exp1")
        if exp1_qc and exp1_qc in self.qc_outputs:
            qc_output = self.qc_outputs[exp1_qc]
            original_path = config.get("exp1_data_path", "")

            if original_path:
                # Extract the filename the user chose
                original_filename = self._extract_filename(original_path)

                # Verify the file exists in the new QC output
                new_data_path = qc_output / original_filename
                if not new_data_path.exists():
                    # Fall back to 10_final_data.csv
                    new_data_path = qc_output / "10_final_data.csv"
                    if new_data_path.exists():
                        original_filename = "10_final_data.csv"
                        print(f"  exp1: Using fallback {original_filename}")
                    else:
                        print(f"  Warning: exp1 QC data not found: {new_data_path}")
                        original_filename = None

                if original_filename:
                    updated_content = self._update_yaml_path_preserving_structure(
                        updated_content, r"exp1_data_path", qc_output, original_filename
                    )
                    print(
                        f"  Updated exp1_data_path -> {qc_output / original_filename}"
                    )
                    updated = True

        # Update exp2 path
        exp2_qc = mapping.get("exp2")
        if exp2_qc and exp2_qc in self.qc_outputs:
            qc_output = self.qc_outputs[exp2_qc]
            original_path = config.get("exp2_data_path", "")

            if original_path:
                # Extract the filename the user chose
                original_filename = self._extract_filename(original_path)

                # Verify the file exists in the new QC output
                new_data_path = qc_output / original_filename
                if not new_data_path.exists():
                    # Fall back to 10_final_data.csv
                    new_data_path = qc_output / "10_final_data.csv"
                    if new_data_path.exists():
                        original_filename = "10_final_data.csv"
                        print(f"  exp2: Using fallback {original_filename}")
                    else:
                        print(f"  Warning: exp2 QC data not found: {new_data_path}")
                        original_filename = None

                if original_filename:
                    updated_content = self._update_yaml_path_preserving_structure(
                        updated_content, r"exp2_data_path", qc_output, original_filename
                    )
                    print(
                        f"  Updated exp2_data_path -> {qc_output / original_filename}"
                    )
                    updated = True

        if not updated:
            return None

        # Write updated config
        updated_path = self.run_dir / "cross_platform" / f"_updated_{config_path.name}"
        updated_path.write_text(updated_content, encoding="utf-8")

        return updated_path

    def _run_pipeline_command(
        self,
        pipeline_type: str,
        config_path: Path,
        output_dir: Path,
    ) -> dict[str, Any]:
        """Run a single pipeline command.

        Args:
            pipeline_type: One of 'qc', 'viz', 'cross-platform'
            config_path: Path to config file
            output_dir: Output directory

        Returns:
            Dictionary with run results
        """
        start_time = datetime.now()

        cmd = [
            sys.executable,
            "-m",
            "sleap_roots_analyze",
            pipeline_type,
            str(config_path),
            "-o",
            str(output_dir),
        ]

        if self.verbose:
            print(f"  Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(Path.cwd()),
            )

            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                print(f"  Completed in {elapsed:.1f}s")
                # Try to find the output directory
                output_path = self._find_pipeline_output(output_dir, config_path)
                return {
                    "success": True,
                    "elapsed_seconds": elapsed,
                    "output_path": str(output_path) if output_path else None,
                }
            else:
                print(f"  FAILED (exit code {result.returncode})")
                if result.stderr:
                    print(f"  Error: {result.stderr[:500]}")
                return {
                    "success": False,
                    "elapsed_seconds": elapsed,
                    "error": result.stderr,
                    "returncode": result.returncode,
                }

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            print(f"  FAILED: {e}")
            return {
                "success": False,
                "elapsed_seconds": elapsed,
                "error": str(e),
            }

    def _find_pipeline_output(self, output_dir: Path, config_path: Path) -> Path | None:
        """Find the most recent pipeline output directory."""
        # Look for directories created after we started
        if not output_dir.exists():
            return None

        # Find most recently modified subdirectory
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        if not subdirs:
            return None

        # Return most recent
        return max(subdirs, key=lambda d: d.stat().st_mtime)

    def _create_latest_symlink(self) -> None:
        """Create or update the 'latest' symlink."""
        latest_path = self.output_dir / "latest"

        try:
            # Remove existing symlink
            if latest_path.exists() or latest_path.is_symlink():
                latest_path.unlink()

            # Create new symlink (use relative path)
            latest_path.symlink_to(self.run_timestamp, target_is_directory=True)
            print(f"\nUpdated symlink: latest -> {self.run_timestamp}")
        except OSError as e:
            # Symlinks may not work on some Windows configurations
            print(f"\nWarning: Could not create symlink: {e}")

    def generate_summary(self) -> Path:
        """Generate comprehensive markdown summary.

        Returns:
            Path to the generated summary file
        """
        summary_path = self.run_dir / "SUMMARY.md"

        # Get git info
        git_info = self._get_git_info()

        lines = [
            f"# {self.manifest.get('run_name', 'Pipeline Run')} Summary",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Run Directory:** `{self.run_dir}`",
            f"**Git Commit:** `{git_info.get('commit', 'unknown')}`",
            f"**Git Branch:** `{git_info.get('branch', 'unknown')}`",
            f"**Manifest:** `{self.manifest_path}`",
            "",
            "---",
            "",
        ]

        # QC Results
        if self.run_results["qc"]:
            lines.extend(self._format_qc_summary())

        # Viz Results
        if self.run_results["viz"]:
            lines.extend(self._format_viz_summary())

        # Cross-Platform Results
        if self.run_results["cross_platform"]:
            lines.extend(self._format_cross_platform_summary())

        # Configuration Comparison Section
        lines.extend(self._format_config_comparison())

        # Methods Section
        lines.extend(self._format_methods_section())

        # Write summary with UTF-8 encoding for proper Unicode character display
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\nSummary written to: {summary_path}")
        return summary_path

    def _get_git_info(self) -> dict[str, str]:
        """Get current git information."""
        info = {}
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["commit"] = result.stdout.strip()[:12]

            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
        except Exception:
            pass

        return info

    def _format_qc_summary(self) -> list[str]:
        """Format QC results for summary with scientific metrics.

        Reads pipeline summary JSON from each run to extract sample counts,
        trait counts, genotype counts, heritability threshold, and mean H².
        Also reads heritability CSV to get H² values for removed traits.
        """
        lines = [
            "## QC Pipeline Results",
            "",
            "| Config | Status | Samples | Traits | Genotypes | H² Threshold | Mean H² | Time | Output Path |",
            "|--------|--------|---------|--------|-----------|--------------|---------|------|-------------|",
        ]

        removed_traits_data = []

        for config, result in self.run_results["qc"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")

            # Default values for failed runs
            n_samples = "N/A"
            n_traits = "N/A"
            n_genotypes = "N/A"
            h2_threshold = "N/A"
            mean_h2 = "N/A"

            if result.get("success") and output != "N/A":
                output_path = Path(output)
                # Read pipeline summary for metrics
                summary = self._read_pipeline_summary(output)

                if summary:
                    final_data = summary.get("final_data", {})
                    n_samples = str(final_data.get("n_samples", "N/A"))
                    n_traits = str(final_data.get("n_traits", "N/A"))
                    n_genotypes = str(final_data.get("n_genotypes", "N/A"))

                    # Get heritability info
                    config_data = summary.get("configuration", {})
                    h2_config = config_data.get("heritability", {})

                    if h2_config.get("enabled", True):
                        h2_threshold = str(h2_config.get("threshold", "N/A"))

                        # Get mean heritability from step summaries
                        step_summaries = summary.get("step_summaries", {})
                        h2_filter = step_summaries.get("heritability_filter", {})
                        mean_val = h2_filter.get("mean_heritability_retained")
                        if mean_val is not None:
                            mean_h2 = f"{mean_val:.2f}"

                        # Collect removed traits with their H² values
                        removed_names = h2_filter.get("removed_trait_names", [])
                        if removed_names:
                            # Try to read heritability values from CSV
                            h2_values = self._read_heritability_values(output_path)
                            removed_with_h2 = []
                            for trait in removed_names:
                                h2_val = h2_values.get(trait)
                                if h2_val is not None:
                                    removed_with_h2.append((trait, h2_val))
                                else:
                                    removed_with_h2.append((trait, None))
                            removed_traits_data.append((config, removed_with_h2))
                    else:
                        h2_threshold = "Disabled"
                        mean_h2 = "N/A"

            lines.append(
                f"| {config} | {status} | {n_samples} | {n_traits} | {n_genotypes} | "
                f"{h2_threshold} | {mean_h2} | {time_str} | `{output}` |"
            )

        lines.extend(["", ""])

        # Add removed traits section
        lines.extend(self._format_removed_traits_section(removed_traits_data))

        return lines

    def _format_removed_traits_section(
        self, removed_traits_data: list[tuple[str, list[tuple[str, float | None]]]]
    ) -> list[str]:
        """Format the removed traits subsection with heritability values.

        Args:
            removed_traits_data: List of (config_name, [(trait_name, h2_value), ...]) tuples

        Returns:
            List of markdown lines
        """
        lines = ["### Removed Traits by Dataset", ""]

        if not removed_traits_data:
            lines.append("No traits were removed by heritability filtering.")
            lines.extend(["", ""])
            return lines

        for config, removed_traits in removed_traits_data:
            if removed_traits:
                lines.append(f"**{config}** ({len(removed_traits)} traits removed):")
                for trait, h2_val in removed_traits:
                    if h2_val is not None:
                        lines.append(f"- {trait} (H²={h2_val:.2f})")
                    else:
                        lines.append(f"- {trait}")
                lines.append("")
            else:
                lines.append(f"**{config}**: No traits removed")
                lines.append("")

        lines.append("")
        return lines

    def _format_viz_summary(self) -> list[str]:
        """Format Viz results for summary with figure counts.

        Counts PNG figures from static_figures directory and HTML interactive files
        from interactive_figures, pca, and umap directories.
        """
        lines = [
            "## Visualization Pipeline Results",
            "",
            "| Config | Status | Figures | Interactive | Time | Output Path |",
            "|--------|--------|---------|-------------|------|-------------|",
        ]

        for config, result in self.run_results["viz"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")

            # Default counts
            figure_count = "N/A"
            interactive_count = "N/A"

            if result.get("success") and output != "N/A":
                output_path = Path(output)

                # Count PNG files in static_figures directory (primary location)
                static_figures_dir = output_path / "static_figures"
                png_count = self._count_files(static_figures_dir, "*.png")
                # Also check figures/ for backwards compatibility
                figures_dir = output_path / "figures"
                png_count += self._count_files(figures_dir, "*.png")
                # Count any PNG in root as well
                png_count += self._count_files(output_path, "*.png")
                figure_count = str(png_count) if png_count > 0 else "0"

                # Count interactive HTML files from multiple locations
                interactive_figures_dir = output_path / "interactive_figures"
                html_count = self._count_files(interactive_figures_dir, "*.html")
                # Also check pca/ and umap/ directories
                pca_dir = output_path / "pca"
                html_count += self._count_files(pca_dir, "*.html")
                umap_dir = output_path / "umap"
                html_count += self._count_files(umap_dir, "*.html")
                interactive_count = str(html_count) if html_count > 0 else "0"

            lines.append(
                f"| {config} | {status} | {figure_count} | {interactive_count} | "
                f"{time_str} | `{output}` |"
            )

        lines.extend(["", ""])
        return lines

    def _format_cross_platform_summary(self) -> list[str]:
        """Format Cross-Platform results for summary with alignment metrics.

        Reads alignment CSV and correlations CSV to extract common genotypes,
        sample/trait counts, and top correlation value.
        """
        lines = [
            "## Cross-Platform Analysis Results",
            "",
            "| Config | Status | Common Genotypes | Exp1 Samples | Exp2 Samples | Top Correlation | Time | Output Path |",
            "|--------|--------|------------------|--------------|--------------|-----------------|------|-------------|",
        ]

        for config, result in self.run_results["cross_platform"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")

            # Default values
            common_genos = "N/A"
            exp1_samples = "N/A"
            exp2_samples = "N/A"
            top_corr = "N/A"

            if result.get("success") and output != "N/A":
                output_path = Path(output)

                # Try to read alignment summary CSV
                alignment_path = output_path / "cross_platform_alignment_summary.csv"
                if alignment_path.exists():
                    try:
                        import pandas as pd

                        align_df = pd.read_csv(alignment_path)
                        # Handle the actual CSV format: genotype, exp1_samples, exp2_samples
                        # where each row is one genotype
                        if "genotype" in align_df.columns:
                            common_genos = str(len(align_df))
                            if "exp1_samples" in align_df.columns:
                                exp1_samples = str(int(align_df["exp1_samples"].sum()))
                            if "exp2_samples" in align_df.columns:
                                exp2_samples = str(int(align_df["exp2_samples"].sum()))
                        # Also handle metric/value format for backwards compatibility
                        elif (
                            "metric" in align_df.columns and "value" in align_df.columns
                        ):
                            metrics = dict(zip(align_df["metric"], align_df["value"]))
                            common_genos = str(int(metrics.get("common_genotypes", 0)))
                            exp1_samples = str(int(metrics.get("exp1_samples", 0)))
                            exp2_samples = str(int(metrics.get("exp2_samples", 0)))
                        elif "common_genotypes" in align_df.columns:
                            common_genos = str(align_df["common_genotypes"].iloc[0])
                    except Exception:
                        pass

                # Try to read correlations CSV for top correlation
                corr_path = output_path / "cross_platform_correlations.csv"
                if corr_path.exists():
                    try:
                        import pandas as pd

                        corr_df = pd.read_csv(corr_path)
                        if "correlation" in corr_df.columns and len(corr_df) > 0:
                            # Get top absolute correlation
                            top_val = corr_df["correlation"].abs().max()
                            top_corr = f"{top_val:.2f}"
                    except Exception:
                        pass

            lines.append(
                f"| {config} | {status} | {common_genos} | {exp1_samples} | {exp2_samples} | "
                f"{top_corr} | {time_str} | `{output}` |"
            )

        lines.extend(["", ""])
        return lines

    @staticmethod
    def _read_pipeline_summary(run_dir: Path | str) -> dict[str, Any]:
        """Read and parse a pipeline summary JSON file.

        Safely reads the pipeline summary from a run directory, handling
        missing files and malformed JSON gracefully.

        Args:
            run_dir: Path to the pipeline run directory

        Returns:
            Dictionary with parsed summary contents, or empty dict if not found/invalid
        """
        run_dir = Path(run_dir)

        # Try QC summary first (10_pipeline_summary.json), then generic
        summary_paths = [
            run_dir / "10_pipeline_summary.json",
            run_dir / "pipeline_summary.json",
        ]

        for summary_path in summary_paths:
            if summary_path.exists():
                try:
                    with open(summary_path) as f:
                        return json.load(f)
                except json.JSONDecodeError:
                    print(f"  Warning: Malformed JSON in {summary_path}")
                    return {}
                except Exception as e:
                    print(f"  Warning: Error reading {summary_path}: {e}")
                    return {}

        return {}

    @staticmethod
    def _read_heritability_values(run_dir: Path) -> dict[str, float]:
        """Read heritability values from the heritability results CSV.

        Args:
            run_dir: Path to the pipeline run directory

        Returns:
            Dictionary mapping trait names to their H² values
        """
        h2_values: dict[str, float] = {}

        # Try different possible heritability CSV locations
        h2_paths = [
            run_dir / "08_heritability_results.csv",
            run_dir / "09_heritability_full.csv",
            run_dir / "heritability_results.csv",
        ]

        for h2_path in h2_paths:
            if h2_path.exists():
                try:
                    import pandas as pd

                    h2_df = pd.read_csv(h2_path)
                    # Handle different column naming conventions
                    trait_col = None
                    h2_col = None

                    for col in ["trait", "Trait", "trait_name"]:
                        if col in h2_df.columns:
                            trait_col = col
                            break

                    for col in ["heritability", "Heritability", "H2", "h2"]:
                        if col in h2_df.columns:
                            h2_col = col
                            break

                    if trait_col and h2_col:
                        for _, row in h2_df.iterrows():
                            trait = row[trait_col]
                            h2_val = row[h2_col]
                            if pd.notna(h2_val):
                                h2_values[trait] = float(h2_val)
                        break  # Found valid CSV, stop searching

                except Exception:
                    pass

        return h2_values

    @staticmethod
    def _count_files(directory: Path, pattern: str) -> int:
        """Count files matching a pattern in a directory.

        Args:
            directory: Directory to search
            pattern: Glob pattern (e.g., '*.png')

        Returns:
            Count of matching files
        """
        if not directory.exists():
            return 0
        return len(list(directory.glob(pattern)))

    @staticmethod
    def _read_csv_safe(csv_path: Path) -> dict[str, Any]:
        """Read a CSV file safely and return as dict.

        Args:
            csv_path: Path to CSV file

        Returns:
            Dictionary with CSV data or empty dict if not found
        """
        if not csv_path.exists():
            return {}
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            return df.to_dict(orient="list")
        except Exception:
            return {}

    def _format_methods_section(self) -> list[str]:
        """Generate publication-ready methods section template.

        Extracts actual config values from QC runs to fill in placeholders.

        Returns:
            List of markdown lines for methods section
        """
        # Collect config values from QC runs
        config_values = self._collect_qc_config_values()

        # Format values, handling cases where configs differ
        max_nan = self._format_config_value(config_values.get("max_nan_fraction", []))
        max_zeros = self._format_config_value(
            config_values.get("max_zeros_per_trait", [])
        )
        max_nans = self._format_config_value(
            config_values.get("max_nans_per_trait", [])
        )
        h2_threshold = self._format_config_value(config_values.get("h2_threshold", []))
        chi2_percentile = self._format_config_value(
            config_values.get("chi2_percentile", []), default="97.5"
        )

        lines = [
            "## Methods",
            "",
            "### Data Quality Control",
            "",
            "Phenotypic trait data underwent quality control using the sleap-roots-analyze "
            "pipeline. The QC process included:",
            "",
            f"1. **Data Cleanup**: Samples with excessive missing values (>{max_nan}% NaN) "
            f"were removed. Traits with high zero frequency (>{max_zeros}%) or "
            f"excessive missing data (>{max_nans}%) were excluded.",
            "",
            f"2. **Outlier Detection**: Multivariate outliers were identified using Mahalanobis "
            f"distance on PCA-transformed data, with outliers defined as samples exceeding "
            f"the {chi2_percentile}th percentile of the chi-squared distribution.",
            "",
            f"3. **Heritability Filtering**: Broad-sense heritability (H²) was calculated for each "
            f"trait using mixed-effects models. Traits with H² < {h2_threshold} were excluded "
            "from downstream analysis to focus on genetically heritable phenotypes.",
            "",
            "### Visualization",
            "",
            "Publication-quality visualizations were generated including trait distributions, "
            "correlation heatmaps, PCA biplots, and genotype comparison plots. Interactive "
            "HTML visualizations were created for exploratory analysis.",
            "",
            "### Cross-Platform Analysis",
            "",
            "Cross-platform trait correlations were calculated using Spearman rank correlation "
            "on genotype means. Significance was assessed using permutation testing with "
            "Bonferroni correction for multiple comparisons.",
            "",
            "---",
            "",
        ]
        return lines

    def _collect_qc_config_values(self) -> dict[str, list[Any]]:
        """Collect config values from all QC run summaries.

        Returns:
            Dictionary mapping config keys to lists of values from each run
        """
        values: dict[str, list[Any]] = {
            "max_nan_fraction": [],
            "max_zeros_per_trait": [],
            "max_nans_per_trait": [],
            "h2_threshold": [],
            "chi2_percentile": [],
        }

        for _config, result in self.run_results.get("qc", {}).items():
            if not result.get("success"):
                continue
            output = result.get("output_path")
            if not output:
                continue

            summary = self._read_pipeline_summary(output)
            config_data = summary.get("configuration", {})

            # Extract cleanup config
            cleanup = config_data.get("cleanup", {})
            if "max_nan_fraction" in cleanup:
                values["max_nan_fraction"].append(cleanup["max_nan_fraction"])
            if "max_zeros_per_trait" in cleanup:
                values["max_zeros_per_trait"].append(cleanup["max_zeros_per_trait"])
            if "max_nans_per_trait" in cleanup:
                values["max_nans_per_trait"].append(cleanup["max_nans_per_trait"])

            # Extract heritability config
            h2_config = config_data.get("heritability", {})
            if h2_config.get("enabled", True) and "threshold" in h2_config:
                values["h2_threshold"].append(h2_config["threshold"])

            # Extract outlier detection chi2 percentile if available
            outlier_config = config_data.get("outlier_detection", {})
            if "chi2_percentile" in outlier_config:
                values["chi2_percentile"].append(outlier_config["chi2_percentile"])

        return values

    @staticmethod
    def _flatten_config_dict(
        config: dict[str, Any],
        prefix: str = "",
        exclude_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Flatten a nested dictionary to dot-notation keys.

        Args:
            config: The nested dictionary to flatten
            prefix: Current key prefix for recursion
            exclude_keys: Top-level keys to exclude from flattening

        Returns:
            Flattened dictionary with dot-notation keys

        Example:
            >>> _flatten_config_dict({"cleanup": {"max_nan": 0.0}})
            {"cleanup.max_nan": 0.0}
        """
        exclude_keys = exclude_keys or []
        result: dict[str, Any] = {}

        for key, value in config.items():
            # Skip excluded top-level keys
            if not prefix and key in exclude_keys:
                continue

            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                # Recursively flatten nested dicts
                result.update(
                    PipelineRunner._flatten_config_dict(value, full_key, exclude_keys)
                )
            else:
                # Leaf value - add to result
                result[full_key] = value

        return result

    @staticmethod
    def _extract_all_config_params(config_path: Path | str) -> dict[str, Any]:
        """Extract ALL parameters from a config file.

        Loads a YAML config file and flattens it to dot-notation keys,
        excluding environment-specific paths.

        Args:
            config_path: Path to the YAML config file

        Returns:
            Flattened dictionary of all config parameters
        """
        config_path = Path(config_path)

        if not config_path.exists():
            return {}

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            return {}

        # Exclude environment-specific keys that aren't useful for comparison
        exclude_keys = [
            "data",  # Contains file paths
            "columns",  # Column names vary by dataset
        ]

        return PipelineRunner._flatten_config_dict(config, exclude_keys=exclude_keys)

    @staticmethod
    def _format_value_for_table(value: Any) -> str:
        """Format a config value for display in a markdown table.

        Args:
            value: The value to format

        Returns:
            String representation suitable for markdown table
        """
        if value is None:
            return "N/A"
        elif isinstance(value, list):
            if len(value) == 0:
                return "(none)"
            return ", ".join(str(v) for v in value)
        elif isinstance(value, bool):
            return str(value).lower()
        elif isinstance(value, float):
            # Format floats nicely
            if value == int(value):
                return str(int(value))
            return f"{value:.4g}"
        else:
            return str(value)

    @staticmethod
    def _format_comparison_table(configs: dict[str, dict[str, Any]]) -> list[str]:
        """Format config parameters as a markdown comparison table.

        Args:
            configs: Dictionary mapping config names to their flattened parameters

        Returns:
            List of markdown lines for the table
        """
        if not configs:
            return ["No configurations to compare.", ""]

        # Get all unique parameters across all configs
        all_params: set[str] = set()
        for params in configs.values():
            all_params.update(params.keys())

        # Sort parameters for consistent ordering
        sorted_params = sorted(all_params)
        config_names = list(configs.keys())

        # Build table header
        header = "| Parameter | " + " | ".join(config_names) + " |"
        separator = "|" + "|".join(["---"] * (len(config_names) + 1)) + "|"

        lines = [header, separator]

        # Build table rows
        for param in sorted_params:
            row_values = []
            for config_name in config_names:
                value = configs[config_name].get(param)
                if value is None and param not in configs[config_name]:
                    row_values.append("N/A")
                else:
                    row_values.append(PipelineRunner._format_value_for_table(value))

            row = f"| {param} | " + " | ".join(row_values) + " |"
            lines.append(row)

        lines.append("")
        return lines

    def _format_config_comparison(self) -> list[str]:
        """Generate the configuration comparison section for the summary.

        Reads all config files from the manifest and generates comparison tables
        for QC, Viz, and Cross-Platform pipelines.

        Returns:
            List of markdown lines for the configuration comparison section
        """
        lines = ["## Configuration Comparison", ""]
        base_dir = self.manifest_path.parent

        # QC Pipeline Configuration
        qc_configs = self.manifest.get("qc_configs", [])
        if qc_configs:
            lines.extend(["### QC Pipeline Configuration", ""])
            config_params = {}
            for config_rel in qc_configs:
                config_path = base_dir / config_rel
                params = self._extract_all_config_params(config_path)
                if params:
                    # Use a short name for the column header
                    config_name = Path(config_rel).stem
                    config_params[config_name] = params
            if config_params:
                lines.extend(self._format_comparison_table(config_params))

        # Visualization Pipeline Configuration
        viz_configs = self.manifest.get("viz_configs", [])
        if viz_configs:
            lines.extend(["### Visualization Pipeline Configuration", ""])
            config_params = {}
            for config_rel in viz_configs:
                config_path = base_dir / config_rel
                params = self._extract_all_config_params(config_path)
                if params:
                    config_name = Path(config_rel).stem
                    config_params[config_name] = params
            if config_params:
                lines.extend(self._format_comparison_table(config_params))

        # Cross-Platform Pipeline Configuration
        cross_configs = self.manifest.get("cross_platform_configs", [])
        if cross_configs:
            lines.extend(["### Cross-Platform Pipeline Configuration", ""])
            config_params = {}
            for config_rel in cross_configs:
                config_path = base_dir / config_rel
                params = self._extract_all_config_params(config_path)
                if params:
                    config_name = Path(config_rel).stem
                    config_params[config_name] = params
            if config_params:
                lines.extend(self._format_comparison_table(config_params))

        return lines

    def _format_config_value(self, values: list[Any], default: str = "N/A") -> str:
        """Format config values, handling multiple different values.

        Args:
            values: List of values from different configs
            default: Default value if list is empty

        Returns:
            Formatted string - single value if all same, or "varied" notation
        """
        if not values:
            return default

        # Convert to strings and get unique values
        str_values = [str(v) for v in values]
        unique = set(str_values)

        if len(unique) == 1:
            # All values are the same
            val = values[0]
            if isinstance(val, float):
                # Format percentage values nicely
                if val < 1:
                    return f"{val * 100:.0f}"
                return f"{val:.2f}".rstrip("0").rstrip(".")
            return str(val)
        else:
            # Values differ across configs
            return f"varied ({', '.join(sorted(unique))})"


def run_all_pipelines(
    manifest_path: str | Path = "configs/active/run_manifest.yaml",
    output_dir: str | Path = "pipeline_runs",
    dry_run: bool = False,
    qc_only: bool = False,
    viz_only: bool = False,
    cross_only: bool = False,
    no_summary: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Convenience function to run all pipelines.

    Args:
        manifest_path: Path to run manifest file
        output_dir: Output directory for pipeline runs
        dry_run: Validate and show plan without running
        qc_only: Run only QC pipelines
        viz_only: Run only Viz pipelines
        cross_only: Run only Cross-Platform pipelines
        no_summary: Skip summary generation
        verbose: Enable verbose output

    Returns:
        Dictionary with run results or dry-run plan
    """
    runner = PipelineRunner(
        manifest_path=manifest_path,
        output_dir=output_dir,
        verbose=verbose,
    )

    if dry_run:
        return runner.dry_run()

    return runner.run_all(
        qc_only=qc_only,
        viz_only=viz_only,
        cross_only=cross_only,
        no_summary=no_summary,
    )
