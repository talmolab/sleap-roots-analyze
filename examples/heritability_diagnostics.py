"""Example usage of heritability diagnostic functions.

This script demonstrates how to use the new diagnostic analysis and
visualization functions to understand why traits have low or zero heritability.
"""

import pandas as pd
from pathlib import Path

from sleap_roots_analyze.statistics import (
    calculate_heritability_estimates,
    analyze_trait_variance,
    diagnose_heritability_issues,
    compare_trait_heritabilities,
)
from sleap_roots_analyze.visualization import (
    create_variance_decomposition_plot,
    create_trait_by_genotype_boxplots,
    create_heritability_diagnostic_dashboard,
)

# Example: Analyzing why c_50_60_1 has H²=0.00 vs c_50_60_2 has H²=0.698

# Assuming you have your data loaded
# df = pd.read_csv("your_data.csv")
# traits_to_diagnose = ['c_50_60_1', 'c_50_60_2']

def diagnose_heritability_example(df, traits, genotype_col="geno", replicate_col="rep"):
    """Complete diagnostic workflow for heritability issues.

    Args:
        df: DataFrame with trait data
        traits: List of trait names to analyze
        genotype_col: Genotype column name
        replicate_col: Replicate column name
    """

    # Step 1: Calculate heritability for all traits
    print("=" * 80)
    print("STEP 1: Calculate Heritability")
    print("=" * 80)
    h2_results = calculate_heritability_estimates(
        df=df,
        trait_cols=traits,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
    )

    for trait in traits:
        h2 = h2_results[trait].get("heritability", "N/A")
        print(f"{trait}: H² = {h2:.3f}" if isinstance(h2, float) else f"{trait}: {h2}")

    # Step 2: Analyze variance components for each trait
    print("\n" + "=" * 80)
    print("STEP 2: Variance Decomposition")
    print("=" * 80)

    for trait in traits:
        print(f"\n{trait}:")
        var_analysis = analyze_trait_variance(
            df=df,
            trait=trait,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
        )

        print(f"  Total observations: {var_analysis['n_observations']}")
        print(f"  Number of genotypes: {var_analysis['n_genotypes']}")
        print(f"  Overall variance: {var_analysis['overall_variance']:.6f}")
        print(f"  Between-genotype variance: {var_analysis['between_genotype_variance']:.6f}")
        print(f"  Within-genotype variance: {var_analysis['within_genotype_variance']:.6f}")
        print(f"  % variance BETWEEN genotypes: {var_analysis['pct_variance_between_geno']:.1f}%")
        print(f"  Coefficient of variation: {var_analysis['trait_cv']:.1f}%")

        # KEY INSIGHT: Low % between-genotype variance = low heritability
        if var_analysis['pct_variance_between_geno'] < 30:
            print(f"  WARNING: Most variance is within-genotype (measurement noise/environment)")
        else:
            print(f"  OK: Substantial variance is between-genotype (genetic control)")

    # Step 3: Diagnose specific issues
    print("\n" + "=" * 80)
    print("STEP 3: Issue Diagnosis")
    print("=" * 80)

    for trait in traits:
        print(f"\n{trait}:")
        diagnosis = diagnose_heritability_issues(
            df=df,
            trait=trait,
            heritability_result=h2_results[trait],
            genotype_col=genotype_col,
            replicate_col=replicate_col,
        )

        print(f"  Has issues: {diagnosis['has_issues']}")
        print(f"  Severity: {diagnosis['severity']}")

        if diagnosis['issues']:
            print(f"  Issues identified:")
            for issue in diagnosis['issues']:
                print(f"    - {issue}")

        if diagnosis['recommendations']:
            print(f"  Recommendations:")
            for rec in diagnosis['recommendations']:
                print(f"    - {rec}")

    # Step 4: Compare multiple traits side-by-side
    print("\n" + "=" * 80)
    print("STEP 4: Multi-Trait Comparison")
    print("=" * 80)

    comparison_df = compare_trait_heritabilities(
        df=df,
        traits=traits,
        heritability_results=h2_results,
        genotype_col=genotype_col,
        replicate_col=replicate_col,
        sort_by="heritability",  # Sort by H² (lowest first)
    )

    print("\nComparison Table:")
    print(comparison_df[['trait', 'heritability', 'pct_var_between',
                        'var_genetic', 'var_residual', 'n_observations']])

    # Step 5: Create diagnostic visualizations
    print("\n" + "=" * 80)
    print("STEP 5: Create Diagnostic Plots")
    print("=" * 80)

    output_dir = Path("heritability_diagnostics")
    output_dir.mkdir(exist_ok=True)

    # Variance decomposition plot (4 panels)
    print("\nCreating variance decomposition plot...")
    fig1 = create_variance_decomposition_plot(
        comparison_df,
        output_path=output_dir / "variance_decomposition.png"
    )
    print(f"Saved to {output_dir / 'variance_decomposition.png'}")

    # Trait-by-genotype boxplots
    print("\nCreating trait boxplots by genotype...")
    fig2 = create_trait_by_genotype_boxplots(
        df=df,
        traits=traits,
        heritability_results=h2_results,
        genotype_col=genotype_col,
        output_path=output_dir / "trait_boxplots.png"
    )
    print(f"- Saved to {output_dir / 'trait_boxplots.png'}")

    # Comprehensive diagnostic dashboard
    print("\nCreating comprehensive diagnostic dashboard...")
    fig3 = create_heritability_diagnostic_dashboard(
        df=df,
        traits=traits,
        heritability_results=h2_results,
        comparison_df=comparison_df,
        output_path=output_dir / "diagnostic_dashboard.png"
    )
    print(f"- Saved to {output_dir / 'diagnostic_dashboard.png'}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)
    print(f"\nAll diagnostic outputs saved to: {output_dir.absolute()}")

    return h2_results, comparison_df


if __name__ == "__main__":
    # Example with simulated data
    import numpy as np

    np.random.seed(42)

    # Simulate data similar to your root coring experiment
    n_genotypes = 20
    n_reps = 5

    data = []
    for g in range(n_genotypes):
        # c_50_60_1: Low H² (high within-genotype noise)
        genetic_effect_1 = np.random.normal(0, 0.5)  # Small genetic effect

        # c_50_60_2: High H² (strong genetic control)
        genetic_effect_2 = np.random.normal(0, 3.0)  # Large genetic effect

        for r in range(n_reps):
            # c_50_60_1: High measurement noise
            env_noise_1 = np.random.normal(0, 10)  # Large environmental noise

            # c_50_60_2: Low measurement noise
            env_noise_2 = np.random.normal(0, 1)  # Small environmental noise

            data.append({
                'geno': f'G{g+1:02d}',
                'rep': r + 1,
                'Barcode': f'BC{g*n_reps + r:04d}',
                'c_50_60_1': 100 + genetic_effect_1 + env_noise_1,
                'c_50_60_2': 50 + genetic_effect_2 + env_noise_2,
            })

    df = pd.DataFrame(data)
    traits = ['c_50_60_1', 'c_50_60_2']

    print("Running example diagnostic workflow with simulated data...\n")
    h2_results, comparison = diagnose_heritability_example(df, traits)

    print("\n\nTo use with your own data:")
    print("  df = pd.read_csv('your_data.csv')")
    print("  traits = ['c_50_60_1', 'c_50_60_2']")
    print("  diagnose_heritability_example(df, traits)")