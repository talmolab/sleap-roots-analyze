"""Analyze correlations between biomass at different depths and above-ground traits.

This script performs correlation and regression analyses for root coring data:
1. Depth section independence: 0-30cm vs 30-60cm biomass
2. Cross-correlations: below-ground (biomass, root counts) vs above-ground (yield, biomass)
3. Specific regressions: yield vs deep biomass, stem biomass vs deep biomass
4. Comprehensive correlation table for supplementary materials

Usage:
    # With config file (recommended for publication-ready output)
    python scripts/analyze_biomass_depth_correlations.py --config configs/biomass_correlation_analysis.yaml --data <path_to_csv>

    # Simple mode (backward compatible)
    python scripts/analyze_biomass_depth_correlations.py --data <path_to_csv> --output <output_dir>

Example:
    python scripts/analyze_biomass_depth_correlations.py \\
        --config configs/biomass_correlation_analysis.yaml \\
        --data "qc_runs/EDPIE_field_qc_20251215/07_data_outliers_removed.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import yaml

from sleap_roots_analyze.visualization import (
    create_regression_plot,
    create_correlation_heatmap,
)


# Default configuration
DEFAULT_CONFIG = {
    "pipeline_name": "biomass_correlation_analysis",
    "publication": {
        "enabled": True,
        "sanitize_trait_names": True,
        "abbreviate": True,
        "custom_replacements": {},
        "trait_labels": {
            "biomass_0_30cm": "Root Biomass DW (g) 0-30cm",
            "biomass_30_60cm": "Root Biomass DW (g) 30-60cm",
            "grain_yield": "Grain Yield (g/m2)",
            "stem_biomass": "Stem Dry Weight (g)",
        },
    },
    "figures": {
        "formats": ["png", "pdf"],
        "dpi": 300,
        "figsize": {"regression": [8, 8], "heatmap": [14, 14]},
        "facecolor": "white",
    },
    "analysis": {
        "alpha": 0.05,
        "min_samples": 3,
        "nan_warning_threshold": 20.0,
    },
    "output": {
        "output_dir": "./biomass_correlation_analysis",
        "generate_summary": True,
    },
    "analyses": {
        "depth_independence": {"enabled": True},
        "correlation_heatmap": {"enabled": True, "include_root_counts": True},
        "yield_vs_deep_biomass": {"enabled": True},
        "stem_vs_deep_roots": {"enabled": True},
        "supplementary_tables": {"enabled": True},
    },
}


def load_config(config_path: Optional[Path]) -> Dict[str, Any]:
    """Load configuration from YAML file, merging with defaults.

    Args:
        config_path: Path to YAML config file, or None for defaults

    Returns:
        Merged configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()

    if config_path and config_path.exists():
        with open(config_path) as f:
            user_config = yaml.safe_load(f) or {}

        # Deep merge user config into defaults
        config = _deep_merge(config, user_config)

    return config


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def setup_output_dir(output_dir: Path) -> Path:
    """Create timestamped output directory.

    Args:
        output_dir: Base output directory path

    Returns:
        Path to timestamped output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"biomass_correlation_analysis_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (run_dir / "figures").mkdir(exist_ok=True)
    (run_dir / "tables").mkdir(exist_ok=True)

    return run_dir


def load_and_validate_data(
    csv_path: Path, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load data and validate required columns exist.

    Supports both notebook-style (c_0_30, c_30_60) and pipeline-style
    (Rootdw 15Cm, Rootdw 45Cm) column names.

    Args:
        csv_path: Path to CSV file
        config: Configuration dictionary

    Returns:
        Tuple of (DataFrame with validated data, publication label mapping)

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)

    # Define both naming conventions
    notebook_style = {
        "shallow_biomass": "c_0_30",
        "deep_biomass": "c_30_60",
        "yield": "GY_Calc_gm2",
        "stem_biomass": "StmDW_M_g",
    }

    pipeline_style = {
        "shallow_biomass": "Rootdw 15Cm",
        "deep_biomass": "Rootdw 45Cm",
        "yield": "Gy Calc Gm2",
        "stem_biomass": "Stmdw M (g)",
    }

    # Detect which style is present
    if all(col in df.columns for col in notebook_style.values()):
        print("  [OK] Detected notebook-style columns (c_0_30, c_30_60)")
        column_mapping = notebook_style
        data_source = "Nov 30 QC notebook"
    elif all(col in df.columns for col in pipeline_style.values()):
        print("  [OK] Detected pipeline-style columns (Rootdw 15Cm, Rootdw 45Cm)")
        column_mapping = pipeline_style
        data_source = "QC pipeline"
    else:
        # Try to provide helpful error message
        missing_notebook = [
            col for col in notebook_style.values() if col not in df.columns
        ]
        missing_pipeline = [
            col for col in pipeline_style.values() if col not in df.columns
        ]
        raise ValueError(
            f"Data does not match expected format.\n"
            f"Missing notebook-style columns: {missing_notebook}\n"
            f"Missing pipeline-style columns: {missing_pipeline}"
        )

    # Rename to standardized internal names for analysis
    rename_map = {
        column_mapping["shallow_biomass"]: "biomass_0_30cm",
        column_mapping["deep_biomass"]: "biomass_30_60cm",
        column_mapping["yield"]: "grain_yield",
        column_mapping["stem_biomass"]: "stem_biomass",
    }

    df = df.rename(columns=rename_map)

    # Store original names for reference
    df.attrs["column_mapping"] = column_mapping
    df.attrs["data_source"] = data_source

    print(f"  [OK] Data source: {data_source}")

    # Get publication labels from config
    pub_labels = config.get("publication", {}).get("trait_labels", {})

    return df, pub_labels


def get_publication_label(
    internal_name: str, pub_labels: Dict[str, str], config: Dict[str, Any]
) -> str:
    """Get publication-ready label for a trait.

    Args:
        internal_name: Internal column name
        pub_labels: Publication label mapping from config
        config: Full configuration dictionary

    Returns:
        Publication-ready label string
    """
    # First check explicit mapping
    if internal_name in pub_labels:
        return pub_labels[internal_name]

    # If publication mode enabled but no explicit mapping, apply sanitization
    if config.get("publication", {}).get("enabled", True):
        # Simple title case with underscores to spaces
        label = internal_name.replace("_", " ").title()
        # Fix common patterns
        label = label.replace("Cm", "cm")
        label = label.replace(" G ", " (g) ")
        return label

    return internal_name


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    filename: str,
    config: Dict[str, Any],
) -> List[Path]:
    """Save figure in configured formats.

    Args:
        fig: Matplotlib figure
        output_dir: Output directory
        filename: Base filename (without extension)
        config: Configuration dictionary

    Returns:
        List of saved file paths
    """
    fig_config = config.get("figures", {})
    formats = fig_config.get("formats", ["png", "pdf"])
    dpi = fig_config.get("dpi", 300)
    facecolor = fig_config.get("facecolor", "white")

    saved_paths = []
    for fmt in formats:
        path = output_dir / "figures" / f"{filename}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
        saved_paths.append(path)

    plt.close(fig)
    return saved_paths


def analyze_depth_independence(
    df: pd.DataFrame,
    output_dir: Path,
    pub_labels: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze independence between 0-30cm and 30-60cm biomass sections.

    Creates regression plot and returns correlation statistics.

    Args:
        df: DataFrame with biomass data
        output_dir: Directory for output files
        pub_labels: Publication label mapping
        config: Configuration dictionary

    Returns:
        Dictionary with R, R², p-value, and sample size
    """
    print("\n=== ANALYSIS 1: Depth Section Independence (0-30cm vs 30-60cm) ===")

    # Get publication labels
    x_label = get_publication_label("biomass_0_30cm", pub_labels, config)
    y_label = get_publication_label("biomass_30_60cm", pub_labels, config)

    # Get title from config or use default
    analysis_config = config.get("analyses", {}).get("depth_independence", {})
    title = analysis_config.get(
        "title", f"{y_label} vs {x_label}"
    )

    # Get figure size
    figsize = tuple(config.get("figures", {}).get("figsize", {}).get("regression", [8, 8]))

    # Create a copy with publication labels for plotting
    plot_df = df[["biomass_0_30cm", "biomass_30_60cm"]].copy()
    plot_df = plot_df.rename(columns={
        "biomass_0_30cm": x_label,
        "biomass_30_60cm": y_label,
    })

    # Create regression plot
    fig = create_regression_plot(
        plot_df,
        x_col=x_label,
        y_col=y_label,
        title=title,
        figsize=figsize,
    )

    # Save in configured formats
    save_figure(fig, output_dir, "depth_independence_regression", config)

    # Calculate statistics
    from scipy import stats

    valid_data = df[["biomass_0_30cm", "biomass_30_60cm"]].dropna()
    r, p = stats.pearsonr(valid_data["biomass_0_30cm"], valid_data["biomass_30_60cm"])

    results = {
        "comparison": f"{x_label} vs {y_label}",
        "r": float(r),
        "r_squared": float(r**2),
        "p_value": float(p),
        "n_samples": len(valid_data),
    }

    print(f"  Pearson R = {r:.3f}")
    print(f"  R² = {r**2:.3f}")
    print(f"  p-value = {p:.4f}")
    print(f"  n = {len(valid_data)}")

    alpha = config.get("analysis", {}).get("alpha", 0.05)
    if p > alpha:
        print(f"  [OK] No significant correlation (p > {alpha})")

    return results


def create_biomass_rootcount_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    pub_labels: Dict[str, str],
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Path]:
    """Create correlation heatmap for biomass and root counts vs above-ground traits.

    Args:
        df: DataFrame with all trait data
        output_dir: Directory for output files
        pub_labels: Publication label mapping
        config: Configuration dictionary

    Returns:
        Tuple of (correlation matrix DataFrame, path to saved figure)
    """
    print("\n=== ANALYSIS 2: Below-ground vs Above-ground Correlations ===")

    # Define trait groups using standardized names
    below_ground_biomass = ["biomass_0_30cm", "biomass_30_60cm"]

    # Get all root count columns (0-60cm range) if enabled
    analysis_config = config.get("analyses", {}).get("correlation_heatmap", {})
    include_root_counts = analysis_config.get("include_root_counts", True)

    root_count_cols = []
    if include_root_counts:
        root_count_cols = [
            col for col in df.columns if col.startswith("Rootcount") and "Cm" in col
        ]

    above_ground = [
        "grain_yield",  # Grain yield
        "stem_biomass",  # Stem dry weight
    ]

    # Add total biomass if available (may not be standardized)
    if "Bm Calc Gm2" in df.columns:
        above_ground.append("Bm Calc Gm2")

    # Combine below-ground and above-ground
    all_traits = below_ground_biomass + root_count_cols + above_ground

    # Filter to available columns
    available_traits = [col for col in all_traits if col in df.columns]

    print(f"  Below-ground biomass: {len(below_ground_biomass)} traits")
    print(f"  Root counts: {len(root_count_cols)} depths")
    print(f"  Above-ground: {len([c for c in above_ground if c in df.columns])} traits")

    # Create rename mapping for publication labels
    rename_for_plot = {}
    for col in available_traits:
        rename_for_plot[col] = get_publication_label(col, pub_labels, config)

    # Create a copy with publication labels
    plot_df = df[available_traits].copy()
    plot_df = plot_df.rename(columns=rename_for_plot)

    # Get figure size
    figsize = tuple(config.get("figures", {}).get("figsize", {}).get("heatmap", [14, 14]))

    # Create correlation heatmap
    fig = create_correlation_heatmap(
        plot_df,
        trait_cols=list(rename_for_plot.values()),
        figsize=figsize,
    )

    # Save
    save_figure(fig, output_dir, "biomass_rootcount_correlation_heatmap", config)

    # Calculate and save correlation matrix (with publication labels)
    corr_matrix = plot_df.corr()
    corr_matrix.to_csv(output_dir / "tables" / "correlation_matrix.csv")

    print("  [OK] Saved correlation heatmap and matrix")

    return corr_matrix, output_dir / "figures" / "biomass_rootcount_correlation_heatmap.png"


def analyze_yield_vs_deep_biomass(
    df: pd.DataFrame,
    output_dir: Path,
    pub_labels: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze relationship between grain yield and deep root biomass.

    Args:
        df: DataFrame with trait data
        output_dir: Directory for output files
        pub_labels: Publication label mapping
        config: Configuration dictionary

    Returns:
        Dictionary with correlation statistics
    """
    print("\n=== ANALYSIS 3: Yield vs Deep Biomass (30-60cm) ===")

    # Get publication labels
    x_label = get_publication_label("biomass_30_60cm", pub_labels, config)
    y_label = get_publication_label("grain_yield", pub_labels, config)

    # Get title from config or use default
    analysis_config = config.get("analyses", {}).get("yield_vs_deep_biomass", {})
    title = analysis_config.get("title", f"{y_label} vs {x_label}")

    # Get figure size
    figsize = tuple(config.get("figures", {}).get("figsize", {}).get("regression", [8, 8]))

    # Create a copy with publication labels for plotting
    plot_df = df[["biomass_30_60cm", "grain_yield"]].copy()
    plot_df = plot_df.rename(columns={
        "biomass_30_60cm": x_label,
        "grain_yield": y_label,
    })

    # Create regression plot
    fig = create_regression_plot(
        plot_df,
        x_col=x_label,
        y_col=y_label,
        title=title,
        figsize=figsize,
    )

    # Save
    save_figure(fig, output_dir, "yield_vs_deep_biomass", config)

    # Calculate statistics
    from scipy import stats

    valid_data = df[["biomass_30_60cm", "grain_yield"]].dropna()
    r, p = stats.pearsonr(valid_data["biomass_30_60cm"], valid_data["grain_yield"])

    results = {
        "comparison": f"{y_label} vs {x_label}",
        "r": float(r),
        "r_squared": float(r**2),
        "p_value": float(p),
        "n_samples": len(valid_data),
    }

    print(f"  Pearson R = {r:.3f}")
    print(f"  R² = {r**2:.3f}")
    print(f"  p-value = {p:.4f}")
    print(f"  n = {len(valid_data)}")

    alpha = config.get("analysis", {}).get("alpha", 0.05)
    if p > alpha:
        print(f"  [OK] No significant correlation (p > {alpha})")

    return results


def analyze_stem_biomass_vs_deep_roots(
    df: pd.DataFrame,
    output_dir: Path,
    pub_labels: Dict[str, str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Analyze relationship between stem biomass and deep root biomass.

    Args:
        df: DataFrame with trait data
        output_dir: Directory for output files
        pub_labels: Publication label mapping
        config: Configuration dictionary

    Returns:
        Dictionary with correlation statistics
    """
    print("\n=== ANALYSIS 4: Stem Biomass vs Deep Root Biomass (30-60cm) ===")

    # Get publication labels
    x_label = get_publication_label("biomass_30_60cm", pub_labels, config)
    y_label = get_publication_label("stem_biomass", pub_labels, config)

    # Get title from config or use default
    analysis_config = config.get("analyses", {}).get("stem_vs_deep_roots", {})
    title = analysis_config.get("title", f"{y_label} vs {x_label}")

    # Get figure size
    figsize = tuple(config.get("figures", {}).get("figsize", {}).get("regression", [8, 8]))

    # Create a copy with publication labels for plotting
    plot_df = df[["biomass_30_60cm", "stem_biomass"]].copy()
    plot_df = plot_df.rename(columns={
        "biomass_30_60cm": x_label,
        "stem_biomass": y_label,
    })

    # Create regression plot
    fig = create_regression_plot(
        plot_df,
        x_col=x_label,
        y_col=y_label,
        title=title,
        figsize=figsize,
    )

    # Save
    save_figure(fig, output_dir, "stem_biomass_vs_deep_roots", config)

    # Calculate statistics
    from scipy import stats

    valid_data = df[["biomass_30_60cm", "stem_biomass"]].dropna()
    r, p = stats.pearsonr(valid_data["biomass_30_60cm"], valid_data["stem_biomass"])

    results = {
        "comparison": f"{y_label} vs {x_label}",
        "r": float(r),
        "r_squared": float(r**2),
        "p_value": float(p),
        "n_samples": len(valid_data),
    }

    print(f"  Pearson R = {r:.3f}")
    print(f"  R² = {r**2:.3f}")
    print(f"  p-value = {p:.4f}")
    print(f"  n = {len(valid_data)}")

    alpha = config.get("analysis", {}).get("alpha", 0.05)
    if p > alpha:
        print(f"  [OK] No significant correlation (p > {alpha})")

    return results


def generate_supplementary_correlation_table(
    corr_matrix: pd.DataFrame,
    output_dir: Path,
    config: Dict[str, Any],
) -> Path:
    """Generate formatted correlation table for supplementary materials.

    Creates both full matrix and filtered version showing only below-ground
    vs above-ground correlations.

    Args:
        corr_matrix: Full correlation matrix (with publication labels)
        output_dir: Directory for output files
        config: Configuration dictionary

    Returns:
        Path to saved supplementary table
    """
    print("\n=== ANALYSIS 5: Supplementary Correlation Table ===")

    # Full correlation matrix (already saved in analysis 2)
    full_table_path = output_dir / "tables" / "correlation_matrix_full.csv"
    corr_matrix.to_csv(full_table_path, float_format="%.3f")

    # Extract below-ground vs above-ground subset
    # Use publication labels for detection
    below_ground = [col for col in corr_matrix.columns if "Root" in col or "Biomass" in col]
    above_ground = [
        col
        for col in corr_matrix.columns
        if any(x in col for x in ["Yield", "Stem", "Sheath", "Spike"])
    ]

    # Create subset showing cross-correlations
    if below_ground and above_ground:
        subset = corr_matrix.loc[below_ground, above_ground]
        subset_path = output_dir / "tables" / "correlation_belowground_vs_aboveground.csv"
        subset.to_csv(subset_path, float_format="%.3f")

        print(f"  [OK] Saved full correlation table: {full_table_path.name}")
        print(f"  [OK] Saved below-ground vs above-ground subset: {subset_path.name}")
        print(f"    - {len(below_ground)} below-ground traits")
        print(f"    - {len(above_ground)} above-ground traits")

        return subset_path

    return full_table_path


def generate_summary_report(
    all_results: List[Dict],
    output_dir: Path,
    config: Dict[str, Any],
    data_path: Path,
) -> None:
    """Generate markdown summary report of all analyses.

    Args:
        all_results: List of result dictionaries from all analyses
        output_dir: Directory for output files
        config: Configuration dictionary
        data_path: Path to input data file
    """
    summary_path = output_dir / "ANALYSIS_SUMMARY.md"
    alpha = config.get("analysis", {}).get("alpha", 0.05)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Biomass Depth Correlation Analysis Summary\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Input Data:** `{data_path}`\n")
        f.write(f"**Config:** `{config.get('pipeline_name', 'biomass_correlation_analysis')}`\n\n")

        f.write("## Key Findings\n\n")

        for result in all_results:
            f.write(f"### {result['comparison']}\n\n")
            f.write(f"- **Pearson R:** {result['r']:.3f}\n")
            f.write(f"- **R²:** {result['r_squared']:.3f}\n")
            f.write(f"- **p-value:** {result['p_value']:.4f}\n")
            f.write(f"- **Sample size:** {result['n_samples']}\n")

            if result["p_value"] > alpha:
                f.write(f"- **Interpretation:** No significant correlation (p > {alpha})\n")
            else:
                f.write(f"- **Interpretation:** Significant correlation (p < {alpha})\n")

            f.write("\n")

        # Figure formats
        formats = config.get("figures", {}).get("formats", ["png", "pdf"])
        dpi = config.get("figures", {}).get("dpi", 300)

        f.write("## Output Files\n\n")
        f.write(f"### Figures ({', '.join(fmt.upper() for fmt in formats)}, {dpi} DPI)\n")
        f.write("- `depth_independence_regression` - Shallow vs deep biomass correlation\n")
        f.write("- `biomass_rootcount_correlation_heatmap` - Full correlation heatmap\n")
        f.write("- `yield_vs_deep_biomass` - Grain yield vs deep root biomass\n")
        f.write("- `stem_biomass_vs_deep_roots` - Stem biomass vs deep root biomass\n\n")

        f.write("### Tables (CSV)\n")
        f.write("- `correlation_matrix_full.csv` - Complete correlation matrix\n")
        f.write("- `correlation_belowground_vs_aboveground.csv` - Below-ground vs above-ground subset\n")
        f.write("- `analysis_results.json` - Statistical results in JSON format\n\n")

        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        f.write(f"publication_mode: {config.get('publication', {}).get('enabled', True)}\n")
        f.write(f"figure_formats: {formats}\n")
        f.write(f"dpi: {dpi}\n")
        f.write(f"alpha: {alpha}\n")
        f.write("```\n")

    print(f"\n[OK] Summary report saved: {summary_path}")


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Analyze biomass depth correlations and above/below-ground relationships"
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Path to CSV file (use step 07 data with biomass columns intact)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML config file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine data path (CLI overrides config)
    data_path = args.data
    if data_path is None:
        data_path = config.get("data", {}).get("csv_path")
        if data_path:
            data_path = Path(data_path)

    if data_path is None:
        parser.error("--data is required (or set data.csv_path in config)")

    # Determine output directory (CLI overrides config)
    output_base = args.output
    if output_base is None:
        output_base = Path(config.get("output", {}).get("output_dir", "./biomass_correlation_analysis"))

    print("=" * 70)
    print("BIOMASS DEPTH CORRELATION ANALYSIS")
    if config.get("publication", {}).get("enabled", True):
        print("  [Publication Mode: ENABLED]")
    print("=" * 70)

    # Setup
    print(f"\nInput data: {data_path}")
    if args.config:
        print(f"Config: {args.config}")
    output_dir = setup_output_dir(output_base)
    print(f"Output directory: {output_dir}")

    # Save config used
    config_save_path = output_dir / "config_used.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"  [OK] Config saved: {config_save_path.name}")

    # Load data
    print("\nLoading and validating data...")
    df, pub_labels = load_and_validate_data(data_path, config)
    print(f"  [OK] Loaded {len(df)} samples with {len(df.columns)} columns")

    # Run analyses
    all_results = []
    analyses_config = config.get("analyses", {})

    # 1. Depth section independence
    if analyses_config.get("depth_independence", {}).get("enabled", True):
        result1 = analyze_depth_independence(df, output_dir, pub_labels, config)
        all_results.append(result1)

    # 2. Correlation heatmap
    corr_matrix = None
    if analyses_config.get("correlation_heatmap", {}).get("enabled", True):
        corr_matrix, _ = create_biomass_rootcount_correlation_heatmap(
            df, output_dir, pub_labels, config
        )

    # 3. Yield vs deep biomass
    if analyses_config.get("yield_vs_deep_biomass", {}).get("enabled", True):
        result3 = analyze_yield_vs_deep_biomass(df, output_dir, pub_labels, config)
        all_results.append(result3)

    # 4. Stem biomass vs deep roots
    if analyses_config.get("stem_vs_deep_roots", {}).get("enabled", True):
        result4 = analyze_stem_biomass_vs_deep_roots(df, output_dir, pub_labels, config)
        all_results.append(result4)

    # 5. Supplementary table
    if analyses_config.get("supplementary_tables", {}).get("enabled", True) and corr_matrix is not None:
        generate_supplementary_correlation_table(corr_matrix, output_dir, config)

    # Save JSON results
    results_path = output_dir / "tables" / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Statistical results saved: {results_path}")

    # Generate summary
    if config.get("output", {}).get("generate_summary", True):
        generate_summary_report(all_results, output_dir, config, data_path)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
