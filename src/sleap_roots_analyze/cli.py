"""Command-line interface for sleap-roots-analyze.

This module provides a CLI for running QC and Viz pipelines, validating configurations,
and accessing package utilities.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from sleap_roots_analyze.pipeline import (
    QCPipeline,
    VizPipeline,
    load_qc_config,
    load_viz_config,
)

console = Console()


def setup_logging(
    verbose: bool = False, quiet: bool = False, log_file: str | None = None
):
    """Configure logging based on CLI flags.

    Args:
        verbose: Enable DEBUG level logging
        quiet: Enable WARNING level logging
        log_file: Optional file path to save logs
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


@click.group()
@click.version_option(version="0.0.1", prog_name="sleap-roots-analyze")
def cli():
    """Statistical analysis tools for root trait data from SLEAP Roots.

    This CLI provides commands for running QC and visualization pipelines,
    validating configurations, and managing analysis workflows.

    Examples:
        sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
        sleap-roots-analyze viz configs/viz_example.yaml -o ./results
        sleap-roots-analyze config validate myconfig.yaml
    """
    pass


@cli.command()
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./qc_runs",
    help="Output directory for pipeline results (default: ./qc_runs)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose (DEBUG) logging",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Quiet mode - only show warnings and errors",
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Save logs to file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without running the pipeline",
)
def qc(
    config: Path,
    output_dir: Path,
    verbose: bool,
    quiet: bool,
    log_file: str | None,
    dry_run: bool,
):
    """Run QC pipeline on trait data.

    The QC pipeline performs:
    - Data loading and validation
    - Trait and sample cleanup
    - Exploratory analysis
    - Outlier detection and removal
    - Statistical analysis (ANOVA, heritability)
    - Summary generation

    Examples:
        sleap-roots-analyze qc configs/qc_turface_150genotypes.yaml
        sleap-roots-analyze qc myconfig.yaml -o /data/results --verbose
        sleap-roots-analyze qc myconfig.yaml --dry-run
    """
    setup_logging(verbose, quiet, log_file)
    logger = logging.getLogger(__name__)

    try:
        # Load and validate config
        console.print(f"[cyan]Loading configuration:[/cyan] {config}")
        cfg = load_qc_config(config)

        # Display config summary
        console.print(f"[cyan]Pipeline:[/cyan] {cfg.pipeline_name}")
        console.print(f"[cyan]Data:[/cyan] {cfg.data.csv_path}")
        console.print(f"[cyan]Output:[/cyan] {output_dir.absolute()}")

        if dry_run:
            console.print("\n[yellow]Dry run mode - validation complete[/yellow]")

            # Check if root core processing is enabled
            if cfg.root_core is not None:
                console.print("\n[cyan]Root Core Processing:[/cyan] ENABLED")
                console.print(f"  Sources: {len(cfg.root_core.sources)}")
                for i, src in enumerate(cfg.root_core.sources, 1):
                    console.print(
                        f"    {i}. {src.data_type}: {Path(src.csv_path).name}"
                    )
                console.print(
                    f"  Core QC: {'Enabled' if cfg.root_core.core_qc.enabled else 'Disabled'}"
                )

                console.print("\nWould execute QC pipeline with 15 steps:\n")

                # Root core steps (0a-0e)
                steps = [
                    (
                        "0a",
                        "LoadRootCoreData",
                        f"Load {len(cfg.root_core.sources)} root core data sources",
                    ),
                    (
                        "0b",
                        "TransformRootCoreData",
                        "Transform biomass/counting to long format",
                    ),
                    (
                        "0c",
                        "QCCoreLevel",
                        (
                            "Detect and remove outlier cores"
                            if cfg.root_core.core_qc.enabled
                            and cfg.root_core.core_qc.remove_outliers
                            else "Detect outlier cores"
                        ),
                    ),
                    (
                        "0d",
                        "AggregateCores",
                        f"Aggregate cores to replicate level ({cfg.root_core.sources[0].aggregation_method})",
                    ),
                    ("0e", "ReshapeForTraitQC", "Reshape to wide format with prefixes"),
                ]
            else:
                console.print("\nWould execute QC pipeline with 10 steps:\n")
                steps = []

            # Standard QC steps (1-10)
            steps.extend(
                [
                    ("1", "LoadData", "Load and validate CSV data"),
                    ("2", "CleanupTraits", "Remove problematic traits and samples"),
                    ("3", "ValidateClean", "Validate no NaN values remain"),
                    ("4", "ExploratoryAnalysis", "Generate EDA visualizations"),
                    ("5", "DetectOutliers", "Detect outliers using configured methods"),
                    ("6", "VisualizeOutliers", "Create outlier visualizations"),
                    ("7", "RemoveOutliers", "Remove outliers based on strategy"),
                    ("8", "StatisticalAnalysis", "Calculate ANOVA and heritability"),
                    (
                        "9",
                        "FilterHeritability",
                        (
                            "Filter low heritability traits"
                            if cfg.heritability.enabled
                            else "Skip (heritability filtering disabled)"
                        ),
                    ),
                    ("10", "GenerateSummary", "Generate complete pipeline summary"),
                ]
            )

            for num, name, desc in steps:
                console.print(f"  {num}. [cyan]{name}[/cyan] - {desc}")

            # Show key configuration
            console.print("\n[cyan]Key Configuration:[/cyan]")
            console.print(
                f"  Outlier detection: {', '.join(cfg.outlier_detection.traditional_methods + cfg.outlier_detection.clustering_methods) or 'None'}"
            )
            console.print(
                f"  Outlier removal: {cfg.outlier_removal.strategy} ({cfg.outlier_removal.method})"
            )
            console.print(
                f"  Heritability filtering: {'Enabled' if cfg.heritability.enabled else 'Disabled'}"
            )
            if cfg.heritability.enabled:
                console.print(f"  Heritability threshold: {cfg.heritability.threshold}")
            console.print(f"  Visualization DPI: {cfg.visualization.dpi}")
            console.print(f"  Figure format: {cfg.visualization.figure_format}")

            console.print("\n[green]Configuration is valid [OK][/green]")
            return

        # Create and run pipeline
        console.print("\n[cyan]Initializing QC pipeline...[/cyan]")
        pipeline = QCPipeline(config=cfg, output_dir=output_dir)

        console.print("[cyan]Running pipeline...[/cyan]")
        results = pipeline.run()

        # Display summary
        console.print("\n[green]Pipeline completed successfully![/green]")
        console.print(f"[green]Results saved to:[/green] {pipeline.run_dir}")
        console.print(f"[green]Steps completed:[/green] {len(results)}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        console.print(
            "[yellow]Hint: Check the config file path and data file paths in the config[/yellow]"
        )
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: Invalid configuration - {e}[/red]")
        console.print(
            "[yellow]Hint: Use 'sleap-roots-analyze config validate' to check your config[/yellow]"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        console.print(f"[red]Error: Pipeline failed - {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument(
    "config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(path_type=Path),
    default="./viz_runs",
    help="Output directory for visualization results (default: ./viz_runs)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose (DEBUG) logging",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Quiet mode - only show warnings and errors",
)
@click.option(
    "--log-file",
    type=str,
    default=None,
    help="Save logs to file",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate configuration without running the pipeline",
)
def viz(
    config: Path,
    output_dir: Path,
    verbose: bool,
    quiet: bool,
    log_file: str | None,
    dry_run: bool,
):
    """Run visualization pipeline on trait data.

    The visualization pipeline generates publication-quality figures including:
    - Trait distributions
    - Correlation plots
    - PCA analysis
    - Heritability plots
    - Custom visualizations based on config

    Examples:
        sleap-roots-analyze viz configs/viz_example.yaml
        sleap-roots-analyze viz myconfig.yaml -o /data/figures --verbose
        sleap-roots-analyze viz myconfig.yaml --dry-run
    """
    setup_logging(verbose, quiet, log_file)
    logger = logging.getLogger(__name__)

    try:
        # Load and validate config
        console.print(f"[cyan]Loading configuration:[/cyan] {config}")
        cfg = load_viz_config(config)

        # Display config summary
        console.print(f"[cyan]Pipeline:[/cyan] {cfg.pipeline_name}")
        console.print(f"[cyan]Data:[/cyan] {cfg.data.csv_path}")
        console.print(f"[cyan]Output:[/cyan] {output_dir.absolute()}")

        if dry_run:
            console.print("\n[yellow]Dry run mode - validation complete[/yellow]")
            console.print("\nWould execute Viz pipeline with configurable steps:\n")

            # Show which visualization steps would be executed
            console.print("  Data Loading:")
            console.print("    - Load trait data from CSV")
            console.print("    - Link images if image_dir specified")

            console.print("\n  Core Analysis:")
            console.print("    - PCA analysis (dimensionality reduction)")
            console.print("    - UMAP analysis (if enabled)")
            console.print("    - Statistical summaries")

            console.print("\n  Visualization Generation:")
            console.print("    - Trait distributions and correlations")
            console.print("    - PCA plots (scree, biplot, feature contributions)")
            console.print("    - Interactive plots (if enabled)")
            console.print("    - Custom publication figures")

            # Show key configuration
            console.print("\n[cyan]Key Configuration:[/cyan]")
            console.print(f"  Visualization DPI: {cfg.static_viz.dpi}")
            console.print(f"  Figure formats: {', '.join(cfg.static_viz.formats)}")

            console.print("\n[green]Configuration is valid [OK][/green]")
            return

        # Create and run pipeline
        console.print("\n[cyan]Initializing Viz pipeline...[/cyan]")
        pipeline = VizPipeline(config=cfg, output_dir=output_dir)

        console.print("[cyan]Running pipeline...[/cyan]")
        results = pipeline.run()

        # Display summary
        console.print("\n[green]Pipeline completed successfully![/green]")
        console.print(f"[green]Results saved to:[/green] {pipeline.run_dir}")
        console.print(f"[green]Steps completed:[/green] {len(results)}")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        console.print(
            "[yellow]Hint: Check the config file path and data file paths in the config[/yellow]"
        )
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: Invalid configuration - {e}[/red]")
        console.print(
            "[yellow]Hint: Use 'sleap-roots-analyze config validate' to check your config[/yellow]"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        console.print(f"[red]Error: Pipeline failed - {e}[/red]")
        sys.exit(1)


@cli.group()
def config():
    """Configuration management commands.

    Utilities for validating, inspecting, and listing configuration files.
    """
    pass


@config.command(name="validate")
@click.argument(
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def config_validate(config_file: Path):
    """Validate a configuration file.

    Checks that the config file:
    - Has valid YAML syntax
    - Contains all required fields
    - Has correct data types
    - References existing files (data paths)

    Examples:
        sleap-roots-analyze config validate configs/qc_turface_150genotypes.yaml
        sleap-roots-analyze config validate myconfig.yaml
    """
    try:
        console.print(f"[cyan]Validating:[/cyan] {config_file}")

        # Try to detect config type by checking for pipeline-specific fields
        # Try QC config first
        try:
            cfg = load_qc_config(config_file)
            config_type = "QC"
        except Exception:
            # Try Viz config
            cfg = load_viz_config(config_file)
            config_type = "Viz"

        # Display config info
        table = Table(title=f"{config_type} Configuration")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Pipeline Name", cfg.pipeline_name)
        table.add_row("Data Path", str(cfg.data.csv_path))
        if hasattr(cfg.data, "image_dir") and cfg.data.image_dir:
            table.add_row("Image Directory", str(cfg.data.image_dir))

        console.print(table)
        console.print("[green]Configuration is valid![/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: File not found - {e}[/red]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: Invalid configuration - {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: Failed to validate config - {e}[/red]")
        sys.exit(1)


@config.command(name="show")
@click.argument(
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def config_show(config_file: Path):
    """Display resolved configuration with all defaults.

    Shows the full configuration including default values that aren't
    explicitly set in the config file.

    Examples:
        sleap-roots-analyze config show configs/qc_turface_150genotypes.yaml
    """
    try:
        console.print(f"[cyan]Loading:[/cyan] {config_file}")

        # Try to detect config type
        try:
            cfg = load_qc_config(config_file)
            config_type = "QC"
        except Exception:
            cfg = load_viz_config(config_file)
            config_type = "Viz"

        console.print(f"\n[cyan]{config_type} Configuration:[/cyan]")

        # Use OmegaConf to display the config
        from omegaconf import OmegaConf

        cfg_dict = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
        console.print(OmegaConf.to_yaml(cfg_dict))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@config.command(name="list")
def config_list():
    """List example configuration files.

    Shows available example configs in the package's configs/ directory.
    """
    # Look for configs in the package directory
    package_root = Path(__file__).parent.parent.parent
    configs_dir = package_root / "configs"

    if not configs_dir.exists():
        console.print("[yellow]No configs directory found[/yellow]")
        return

    # Find all YAML files
    yaml_files = list(configs_dir.glob("*.yaml")) + list(configs_dir.glob("*.yml"))

    if not yaml_files:
        console.print("[yellow]No configuration files found[/yellow]")
        return

    table = Table(title="Example Configurations")
    table.add_column("File", style="cyan")
    table.add_column("Path", style="white")

    for yaml_file in sorted(yaml_files):
        table.add_row(yaml_file.name, str(yaml_file))

    console.print(table)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
