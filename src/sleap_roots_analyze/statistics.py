"""Statistical analysis utilities for trait heritability and ANOVA."""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.regression.mixed_linear_model import MixedLM
from scipy import stats
from scipy.stats import f_oneway
from typing import Dict, List, Tuple, Optional, Union

# Import for optional filtering
from .data_cleanup import remove_low_heritability_traits


def calculate_trait_statistics(df: pd.DataFrame, trait_cols: List[str]) -> Dict:
    """Calculate basic statistics for all trait columns.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names

    Returns:
        Dictionary with statistics for each trait
    """
    stats_dict = {}

    for trait in trait_cols:
        if trait in df.columns:
            data = df[trait].dropna()

            if len(data) == 0:
                stats_dict[trait] = {"error": "No valid data"}
                continue

            stats_dict[trait] = {
                "count": len(data),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
                "cv": float(data.std() / data.mean()) if data.mean() != 0 else np.inf,
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
            }

    return stats_dict


def perform_anova_by_genotype(
    df: pd.DataFrame, trait_cols: List[str], genotype_col: str = "geno"
) -> Dict:
    """Perform one-way ANOVA for each trait by genotype.

    One-way Analysis of Variance (ANOVA) tests whether group means differ significantly.
    It partitions total variance into between-group and within-group components.

    F-statistic = MS_between / MS_within

    Where:
    - MS_between = SS_between / (k-1)  [Mean Square Between Groups]
    - MS_within = SS_within / (N-k)    [Mean Square Within Groups]
    - k = number of groups, N = total sample size

    H₀: μ₁ = μ₂ = ... = μₖ (all group means are equal)
    H₁: At least one group mean differs

    Args:
        df: DataFrame with trait and genotype data
        trait_cols: List of trait column names
        genotype_col: Name of genotype column

    Returns:
        Dictionary with ANOVA results for each trait including:
        - f_statistic: F-test statistic
        - p_value: Probability of observing F-statistic under null hypothesis
        - significant: Whether p < 0.05
        - group_stats: Descriptive statistics for each genotype
    """
    anova_results = {}

    if genotype_col not in df.columns:
        return {"error": f"Genotype column '{genotype_col}' not found"}

    # Get unique genotypes
    genotypes = df[genotype_col].dropna().unique()

    if len(genotypes) < 2:
        return {"error": "Need at least 2 genotypes for ANOVA"}

    for trait in trait_cols:
        if trait not in df.columns:
            anova_results[trait] = {"error": f"Trait column '{trait}' not found"}
            continue

        # Group data by genotype
        groups = []
        group_stats = {}

        for geno in genotypes:
            geno_data = df[df[genotype_col] == geno][trait].dropna()

            if len(geno_data) > 0:
                groups.append(geno_data.values)
                group_stats[geno] = {
                    "n": len(geno_data),
                    "mean": float(geno_data.mean()),
                    "std": float(geno_data.std()),
                    "sem": float(geno_data.std() / np.sqrt(len(geno_data))),
                }

        # Need at least 2 groups with data
        if len(groups) < 2:
            anova_results[trait] = {"error": "Insufficient groups with data for ANOVA"}
            continue

        # Perform ANOVA
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f_stat, p_value = f_oneway(*groups)

            # Calculate effect size (eta-squared)
            total_data = df[trait].dropna()
            ss_between = sum(
                len(group) * (np.mean(group) - np.mean(total_data)) ** 2
                for group in groups
            )
            ss_total = sum((x - np.mean(total_data)) ** 2 for x in total_data)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0

            anova_results[trait] = {
                "f_statistic": float(f_stat),
                "p_value": float(p_value),
                "eta_squared": float(eta_squared),
                "significant": bool(p_value < 0.05),  # Ensure it's a Python bool
                "n_groups": len(groups),
                "total_n": sum(len(group) for group in groups),
                "group_stats": group_stats,
            }

        except Exception as e:
            anova_results[trait] = {"error": f"ANOVA failed: {str(e)}"}

    return anova_results


def calculate_heritability_estimates(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    replicate_col: str = "rep",
    force_method: Optional[str] = None,
    remove_low_h2: bool = False,
    h2_threshold: float = 0.3,
    barcode_col: str = "Barcode",
    additional_exclude: Optional[List[str]] = None,
) -> Union[Dict, Tuple[Dict, pd.DataFrame, List[str], Dict]]:
    """Calculate broad-sense heritability estimates for traits using mixed model approach.

    This implementation matches the R lme4 approach for calculating broad-sense heritability.
    It uses a linear mixed model with genotype as a random effect to properly partition
    variance components, especially for unbalanced designs.

    H² = σ²_G / (σ²_G + σ²_E / mean_n_reps)

    Where:
    - σ²_G = Genetic variance (between-genotype variance from random effects)
    - σ²_E = Environmental/residual variance (within-genotype variance)
    - mean_n_reps = Average number of replicates per genotype

    This formula accounts for unbalanced designs where genotypes may have different
    numbers of replicates, providing more accurate heritability estimates.

    H² ranges from 0 (no genetic contribution) to 1 (purely genetic).
    Values > 0.5 indicate traits with substantial genetic control.

    Args:
        df: DataFrame with trait, genotype, and replicate data
        trait_cols: List of trait column names
        genotype_col: Name of genotype column
        replicate_col: Name of replicate column
        force_method: Force a specific method ('mixed_model' or 'anova_based') for all traits.
                     If None or 'mixed_model', will use mixed model approach (default).
        remove_low_h2: If True, remove traits with low heritability and return filtered DataFrame
        h2_threshold: Heritability threshold for filtering (default: 0.3, only used if remove_low_h2=True)
        barcode_col: Name of barcode column (default: "Barcode", only used if remove_low_h2=True)
        additional_exclude: Additional columns to exclude from traits (only used if remove_low_h2=True)

    Returns:
        If remove_low_h2=False:
            Dictionary with heritability estimates including:
            - heritability: H² estimate (0-1)
            - var_genetic: Genetic variance component (σ²_G)
            - var_residual: Residual/environmental variance (σ²_E)
            - mean_n_reps: Average number of replicates per genotype
            - n_genotypes: Number of genotypes
            - n_observations: Total number of observations
            - model_type: Type of model used (mixed_model or anova_based)

        If remove_low_h2=True:
            Tuple of:
            - Dictionary with heritability estimates (as above)
            - DataFrame with low heritability traits removed
            - List of removed trait names
            - Dictionary with removal details
    """
    heritability_results = {}

    # Determine which method to use
    if force_method == "anova_based":
        use_mixed_model = False
        method_used = "anova_based"
        warnings.warn("Using ANOVA-based method as requested.")
    else:
        use_mixed_model = True
        method_used = "mixed_model"

    # Add metadata about method selection
    heritability_results["_metadata"] = {
        "method_used_for_all_traits": method_used,
        "method_consistency": True,
    }

    required_cols = [genotype_col, replicate_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}

    for trait in trait_cols:
        if trait not in df.columns:
            heritability_results[trait] = {"error": f"Trait column '{trait}' not found"}
            continue

        # Create a subset with complete data
        subset = df[[trait, genotype_col, replicate_col]].dropna()

        if len(subset) < 4:  # Need minimum data for variance estimation
            heritability_results[trait] = {
                "error": "Insufficient data for heritability estimation"
            }
            continue

        try:
            # Calculate mean number of replicates per genotype (for unbalanced design)
            reps_per_geno = subset.groupby(genotype_col).size()
            mean_n_reps = reps_per_geno.mean()

            # Check if all values are identical (no variance)
            if subset[trait].nunique() == 1:
                heritability_results[trait] = {
                    "heritability": 0.0,
                    "var_genetic": 0.0,
                    "var_residual": 0.0,
                    "mean_n_reps": float(mean_n_reps),
                    "n_genotypes": len(reps_per_geno),
                    "n_observations": len(subset),
                    "model_type": "no_variance",
                    "reps_per_geno_stats": {
                        "min": int(reps_per_geno.min()),
                        "max": int(reps_per_geno.max()),
                        "mean": float(mean_n_reps),
                        "std": (
                            float(reps_per_geno.std()) if len(reps_per_geno) > 1 else 0
                        ),
                    },
                }
                continue

            if use_mixed_model:
                # Use mixed model approach (matches R lme4)
                # Create a clean dataframe for the model
                model_data = subset.copy()
                model_data.columns = ["value", "genotype", "replicate"]

                # Fit mixed model: value ~ 1 + (1|genotype)
                # This matches the R code: lmer(value ~ (1 | ecot_id), data = data_H)
                try:
                    model = smf.mixedlm(
                        "value ~ 1", model_data, groups=model_data["genotype"]
                    )
                    result = model.fit(reml=True)  # Use REML like lme4 default

                    # Extract variance components
                    var_genetic = float(
                        result.cov_re.iloc[0, 0]
                    )  # Random effect variance
                    var_residual = float(result.scale)  # Residual variance

                    # Calculate heritability using the R formula
                    # H² = σ²_G / (σ²_G + σ²_E / mean_n_reps)
                    heritability = var_genetic / (
                        var_genetic + (var_residual / mean_n_reps)
                    )

                    model_type = "mixed_model"

                except Exception as e:
                    # If mixed model fails for this trait, record the error but keep going
                    heritability_results[trait] = {
                        "error": f"Mixed model failed: {str(e)}",
                        "model_type": "mixed_model_failed",
                    }
                    continue

            else:
                # Use ANOVA-based method for ALL traits (ensures consistency)
                # Calculate variance components from ANOVA
                grouped = subset.groupby(genotype_col)[trait]

                # Between-genotype variance
                geno_means = grouped.mean()
                geno_sizes = grouped.size()
                overall_mean = subset[trait].mean()

                # Calculate weighted sum of squares between groups
                ss_between = sum(
                    n * (mean - overall_mean) ** 2
                    for n, mean in zip(geno_sizes, geno_means)
                )
                df_between = len(geno_means) - 1
                ms_between = ss_between / df_between if df_between > 0 else 0

                # Within-genotype variance (pooled)
                ss_within = sum(
                    ((group_data - group_mean) ** 2).sum()
                    for (_, group_data), group_mean in zip(grouped, geno_means)
                )
                df_within = len(subset) - len(geno_means)
                ms_within = ss_within / df_within if df_within > 0 else 0

                # Estimate variance components
                var_residual = ms_within
                var_genetic = max(0, (ms_between - ms_within) / mean_n_reps)

                # Calculate heritability using the same formula as R
                heritability = var_genetic / (
                    var_genetic + (var_residual / mean_n_reps)
                )

                model_type = "anova_based"

            # Ensure heritability is between 0 and 1
            heritability = max(0, min(1, heritability))

            heritability_results[trait] = {
                "heritability": float(heritability),
                "var_genetic": float(var_genetic),
                "var_residual": float(var_residual),
                "mean_n_reps": float(mean_n_reps),
                "n_genotypes": len(reps_per_geno),
                "n_observations": len(subset),
                "model_type": model_type,
                "reps_per_geno_stats": {
                    "min": int(reps_per_geno.min()),
                    "max": int(reps_per_geno.max()),
                    "mean": float(mean_n_reps),
                    "std": float(reps_per_geno.std()) if len(reps_per_geno) > 1 else 0,
                },
            }

        except Exception as e:
            heritability_results[trait] = {
                "error": f"Heritability calculation failed: {str(e)}"
            }

    # Optionally filter low heritability traits
    if remove_low_h2:
        df_filtered, removed_traits, removal_details = remove_low_heritability_traits(
            df=df,
            heritability_results=heritability_results,
            heritability_threshold=h2_threshold,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
            additional_exclude=additional_exclude,
        )
        return heritability_results, df_filtered, removed_traits, removal_details

    return heritability_results


def identify_high_heritability_traits(
    heritability_results: Dict, threshold: float = 0.5
) -> List[str]:
    """Identify traits with high heritability.

    Args:
        heritability_results: Results from calculate_heritability_estimates
        threshold: Minimum heritability threshold

    Returns:
        List of trait names with high heritability
    """
    high_h2_traits = []

    for trait, results in heritability_results.items():
        if isinstance(results, dict) and "heritability" in results:
            if results["heritability"] >= threshold:
                high_h2_traits.append(trait)

    return high_h2_traits


def analyze_heritability_thresholds(
    heritability_results: Dict[str, Dict],
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Analyze how many traits would be retained at different heritability thresholds.

    Args:
        heritability_results: Dictionary with heritability results for each trait
        thresholds: Array of threshold values to test (default: 0 to 1 in 0.01 steps)

    Returns:
        Dictionary with:
            - 'thresholds': Array of threshold values
            - 'traits_retained': Number of traits retained at each threshold
            - 'traits_removed': Number of traits removed at each threshold
            - 'fraction_retained': Fraction of traits retained at each threshold
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)

    # Extract valid heritability values
    h2_values = []
    for trait, result in heritability_results.items():
        if isinstance(result, dict) and "heritability" in result:
            h2 = result["heritability"]
            if not np.isnan(h2):
                h2_values.append(h2)

    h2_values = np.array(h2_values)
    total_traits = len(h2_values)

    # Calculate retention at each threshold
    traits_retained = np.zeros(len(thresholds))
    for i, threshold in enumerate(thresholds):
        traits_retained[i] = np.sum(h2_values >= threshold)

    return {
        "thresholds": thresholds,
        "traits_retained": traits_retained,
        "traits_removed": total_traits - traits_retained,
        "fraction_retained": (
            traits_retained / total_traits if total_traits > 0 else traits_retained
        ),
        "total_traits": total_traits,
        "h2_values": h2_values,
    }
