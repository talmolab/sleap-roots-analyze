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
            print(f"\n{'='*60}")
            print(f"Running QC: {config_rel}")
            print(f"{'='*60}")

            result = self._run_pipeline_command(
                "qc", config_path, qc_output_dir
            )

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
            print(f"\n{'='*60}")
            print(f"Running Viz: {config_rel}")
            print(f"{'='*60}")

            # Update config with new QC output path if mapping exists
            updated_config = self._update_viz_config(config_path, config_rel, qc_mapping)

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
            print(f"\n{'='*60}")
            print(f"Running Cross-Platform: {config_rel}")
            print(f"{'='*60}")

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

    def _update_viz_config(
        self,
        config_path: Path,
        config_rel: str,
        qc_mapping: dict[str, Any],
    ) -> Path | None:
        """Update viz config with new QC output path.

        Returns updated config path or None if no update needed.
        """
        if config_rel not in qc_mapping:
            return None

        qc_config_rel = qc_mapping[config_rel]
        if qc_config_rel not in self.qc_outputs:
            print(f"  Warning: QC output not found for {qc_config_rel}")
            return None

        qc_output_path = self.qc_outputs[qc_config_rel]
        final_data_path = qc_output_path / "10_final_data.csv"

        if not final_data_path.exists():
            print(f"  Warning: QC final data not found: {final_data_path}")
            return None

        # Load and update config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        config["data"]["csv_path"] = str(final_data_path)

        # Write updated config to temp file
        updated_path = self.run_dir / "viz" / f"_updated_{config_path.name}"
        with open(updated_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"  Updated data.csv_path -> {final_data_path}")
        return updated_path

    def _update_cross_platform_config(
        self,
        config_path: Path,
        config_rel: str,
        qc_mapping: dict[str, Any],
    ) -> Path | None:
        """Update cross-platform config with new QC output paths.

        Returns updated config path or None if no update needed.
        """
        if config_rel not in qc_mapping:
            return None

        mapping = qc_mapping[config_rel]
        if not isinstance(mapping, dict):
            return None

        # Load config
        with open(config_path) as f:
            config = yaml.safe_load(f)

        updated = False

        # Update exp1 path
        exp1_qc = mapping.get("exp1")
        if exp1_qc and exp1_qc in self.qc_outputs:
            qc_output = self.qc_outputs[exp1_qc]
            final_data = qc_output / "10_final_data.csv"
            if final_data.exists():
                config["exp1_data_path"] = str(final_data)
                print(f"  Updated exp1_data_path -> {final_data}")
                updated = True

        # Update exp2 path
        exp2_qc = mapping.get("exp2")
        if exp2_qc and exp2_qc in self.qc_outputs:
            qc_output = self.qc_outputs[exp2_qc]
            final_data = qc_output / "10_final_data.csv"
            if final_data.exists():
                config["exp2_data_path"] = str(final_data)
                print(f"  Updated exp2_data_path -> {final_data}")
                updated = True

        if not updated:
            return None

        # Write updated config
        updated_path = self.run_dir / "cross_platform" / f"_updated_{config_path.name}"
        with open(updated_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

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

    def _find_pipeline_output(
        self, output_dir: Path, config_path: Path
    ) -> Path | None:
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
            latest_path.symlink_to(
                self.run_timestamp, target_is_directory=True
            )
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

        # Write summary
        with open(summary_path, "w") as f:
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
        """Format QC results for summary."""
        lines = [
            "## QC Pipeline Results",
            "",
            "| Config | Status | Time | Output Path |",
            "|--------|--------|------|-------------|",
        ]

        for config, result in self.run_results["qc"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")
            lines.append(f"| {config} | {status} | {time_str} | `{output}` |")

        lines.extend(["", ""])
        return lines

    def _format_viz_summary(self) -> list[str]:
        """Format Viz results for summary."""
        lines = [
            "## Visualization Pipeline Results",
            "",
            "| Config | Status | Time | Output Path |",
            "|--------|--------|------|-------------|",
        ]

        for config, result in self.run_results["viz"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")
            lines.append(f"| {config} | {status} | {time_str} | `{output}` |")

        lines.extend(["", ""])
        return lines

    def _format_cross_platform_summary(self) -> list[str]:
        """Format Cross-Platform results for summary."""
        lines = [
            "## Cross-Platform Analysis Results",
            "",
            "| Config | Status | Time | Output Path |",
            "|--------|--------|------|-------------|",
        ]

        for config, result in self.run_results["cross_platform"].items():
            status = "Success" if result.get("success") else "Failed"
            time_str = f"{result.get('elapsed_seconds', 0):.1f}s"
            output = result.get("output_path", "N/A")
            lines.append(f"| {config} | {status} | {time_str} | `{output}` |")

        lines.extend(["", ""])
        return lines


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
