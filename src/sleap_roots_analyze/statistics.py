"""Single-experiment trait statistics: heritability, ANOVA, and variance analysis.

This module computes statistics *within a single experiment* — broad-sense
heritability (H²) via mixed models, one-way ANOVA by genotype, and per-trait
variance decomposition and diagnostics. It operates on a tidy DataFrame whose rows
are individual observations (genotype × replicate) and whose columns are traits.

It is distinct from :mod:`sleap_roots_analyze.cross_experiment_analysis`, which
operates *across* experiments — aligning trait names, computing genotype-level
summaries, and correlating results between separate experiments/platforms. Use this
module when analyzing one experiment's replicated measurements; use
``cross_experiment_analysis`` when comparing or combining multiple experiments.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from scipy import stats
from scipy.stats import f_oneway
from typing import Any, Dict, List, Tuple, Optional, Union

# Import for optional filtering
from .data_cleanup import remove_low_heritability_traits


def calculate_trait_statistics(df: pd.DataFrame, trait_cols: List[str]) -> Dict:
    """Calculate basic statistics for all trait columns.

    Args:
        df: DataFrame with trait data
        trait_cols: List of trait column names

    Returns:
        Dictionary mapping each trait name to a dictionary of statistics. Columns
        listed in ``trait_cols`` but absent from ``df`` are skipped. For a trait
        with at least one non-NA value the inner dictionary contains:
            - count: Number of non-NA observations
            - mean: Arithmetic mean
            - std: Sample standard deviation
            - min: Minimum value
            - max: Maximum value
            - median: Median value
            - q25: 25th percentile
            - q75: 75th percentile
            - cv: Coefficient of variation (std / mean), or ``np.inf`` if mean is 0
            - skewness: Fisher-Pearson skewness
            - kurtosis: Excess kurtosis
        If a trait has no non-NA values, its entry is ``{"error": "No valid data"}``
        instead.
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
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    alpha: float = 0.05,
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
        alpha: Significance level for hypothesis testing (default: 0.05)

    Returns:
        Dictionary mapping each trait name to its ANOVA result. If ``genotype_col``
        is missing or fewer than two genotypes are present, a single
        ``{"error": ...}`` dictionary is returned instead of per-trait results. For a
        successfully analyzed trait the inner dictionary contains:
            - f_statistic: F-test statistic
            - p_value: Probability of observing the F-statistic under the null
            - eta_squared: Effect size (proportion of variance explained by genotype)
            - significant: Whether ``p_value < alpha`` (Python ``bool``)
            - n_groups: Number of genotype groups with data
            - total_n: Total number of observations across groups
            - group_stats: Per-genotype dictionary of ``n``, ``mean``, ``std``, and
              ``sem`` (standard error of the mean)
        A trait that cannot be analyzed (missing column, too few groups with data,
        or a computation failure) maps to an ``{"error": ...}`` dictionary.
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
                "significant": bool(p_value < alpha),  # Ensure it's a Python bool
                "n_groups": len(groups),
                "total_n": sum(len(group) for group in groups),
                "group_stats": group_stats,
            }

        except Exception as e:
            anova_results[trait] = {"error": f"ANOVA failed: {str(e)}"}

    return anova_results


def _marginal_intercept(
    result: Any, model_data: pd.DataFrame, fixed_effects: List[str]
) -> float:
    """Empirical, sample frequency-weighted intercept for a fixed-effects fit (#114).

    ``result.fe_params["Intercept"]`` alone represents the fitted value when
    every fixed effect is at patsy's reference level -- a naming artifact
    (the first level in sorted order, or the first declared category for a
    ``pandas.Categorical``), not a scientifically meaningful baseline. This
    computes an empirical frequency-weighted average across each fixed
    effect's *observed* levels instead: for each fixed effect, each level's
    fitted contribution (0.0 for the reference level; its own coefficient in
    ``result.fe_params`` for every other level) is weighted by that level's
    share of ``model_data`` rows, summed across levels within that fixed
    effect, then summed across all fixed effects and added to the base
    ``Intercept`` coefficient.

    This is a sample-margin quantity, not a population-typical or
    EMM/lsmeans-style equally-weighted marginal mean: it depends on each
    trait's own observed level frequencies (post-``dropna()``), so two
    traits sharing the same ``fixed_effects`` columns may get different
    values.

    Per-level coefficients are recovered by parsing ``result.fe_params``'s
    actual fitted parameter names, not by reconstructing the expected key
    string forward from each observed level's raw value -- the latter risks
    silently misattributing a real level to the reference level's implicit
    ``0.0`` on a dtype/formatting mismatch (e.g. a ``float64`` column).

    Args:
        result: The fitted ``MixedLMResults`` object.
        model_data: The DataFrame the model was fit on (post-``dropna()``),
            containing one column per name in ``fixed_effects``.
        fixed_effects: Names of the fixed-effect columns in the fitted
            formula.

    Returns:
        float: The empirical frequency-weighted intercept.

    Raises:
        ValueError: If a fixed effect's recovered coefficients don't form an
            exact identity with its observed levels -- either a coefficient
            doesn't match any observed level string (a dtype/formatting
            mismatch), or more than one observed level lacks a coefficient
            (more than one apparent "reference" level). Either case indicates
            a level failed to match back to its coefficient, which must not
            be silently defaulted to 0.0.
    """
    intercept = float(result.fe_params["Intercept"])
    for fe in fixed_effects:
        pattern = re.compile(rf"^C\({re.escape(fe)}\)\[T\.(.*)\]$")
        level_coefficients = {}
        for key in result.fe_params.index:
            match = pattern.match(key)
            if match:
                level_coefficients[match.group(1)] = float(result.fe_params[key])

        level_frequencies = model_data[fe].value_counts(normalize=True)
        observed_level_strs = {str(v) for v in level_frequencies.index}
        coefficient_keys = set(level_coefficients.keys())

        # An identity check, not just a count check: every recovered
        # coefficient must correspond to an actually-observed level (a
        # coefficient key that doesn't match any observed level string would
        # indicate a dtype/formatting mismatch, not a real reference level),
        # and exactly one observed level must lack a coefficient (the true
        # reference level patsy dropped). A count-only check (matching
        # len(level_coefficients) to n_levels - 1) can pass even when a real
        # level's string silently failed to match its own coefficient while
        # an unrelated mismatch happened to keep the count the same.
        if not coefficient_keys.issubset(observed_level_strs):
            raise ValueError(
                f"Fixed effect '{fe}' has fitted coefficient(s) for level(s) "
                f"not found in the fitted data: "
                f"{coefficient_keys - observed_level_strs}"
            )
        unmatched_levels = observed_level_strs - coefficient_keys
        if len(unmatched_levels) != 1:
            raise ValueError(
                f"Expected exactly one reference level (no fitted "
                f"coefficient) for fixed effect '{fe}', found "
                f"{len(unmatched_levels)}: {unmatched_levels}"
            )

        for level_value, frequency in level_frequencies.items():
            coefficient = level_coefficients.get(str(level_value), 0.0)
            intercept += float(frequency) * coefficient

    return intercept


def calculate_heritability_estimates(
    df: pd.DataFrame,
    trait_cols: List[str],
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    force_method: Optional[str] = None,
    remove_low_h2: bool = False,
    h2_threshold: float = 0.3,
    barcode_col: str = "Barcode",
    additional_exclude: Optional[List[str]] = None,
    fixed_effects: Optional[List[str]] = None,
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
        replicate_col: Name of replicate column, or None if the dataset has no
            replicate column. Replicate values are never used in the model
            (value ~ 1 + (1|genotype)); H² is identical whether this is set or None.
        force_method: Force a specific method ('mixed_model' or 'anova_based') for all traits.
                     If None or 'mixed_model', will use mixed model approach (default).
        remove_low_h2: If True, remove traits with low heritability and return filtered DataFrame
        h2_threshold: Heritability threshold for filtering (default: 0.3, only used if remove_low_h2=True)
        barcode_col: Name of barcode column (default: "Barcode", only used if remove_low_h2=True)
        additional_exclude: Additional columns to exclude from traits (only used if remove_low_h2=True)
        fixed_effects: Optional list of column names to add as fixed effects to
            the mixed model, changing the formula from ``value ~ 1`` to
            ``value ~ C(fe_1) + C(fe_2) + ...``. Every name is wrapped in
            ``C(...)`` unconditionally — always treated as categorical,
            regardless of pandas dtype, since fixed effects in this context
            are metadata-style confounders (experiment, wave, batch, scanner),
            not biological/phenotypic traits and not continuous covariates.
            Each name must be a valid Python identifier (validated with
            ``str.isidentifier()``); a name containing a patsy formula
            operator (``*``, ``:``, etc.) produces a structural error instead
            of being interpolated into the formula string, since that could
            otherwise silently misparse as an expression over other columns.
            Missing columns produce the same structural error as a missing
            ``genotype_col``. When set, the per-trait model subset additionally
            drops rows with a ``NaN`` in any fixed-effect column — this only
            changes behavior when ``fixed_effects`` is non-empty; ``None``
            (the default) reproduces this function's pre-existing behavior
            exactly, including the ANOVA-based path, which never uses
            ``fixed_effects`` in its own variance-component computation
            (only the row-filtering subset change applies to it). Also only
            when non-empty: a captured ``ConvergenceWarning`` during the fit
            (checked by category, not message text) is treated as a fit
            failure for that trait, since ``statsmodels`` does not always
            *raise* on a fixed effect confounded with genotype. This is a
            best-effort signal, not a guarantee: a fixed effect that is
            confounded with genotype but not enough to trigger a numerical
            convergence problem can converge cleanly with no warning at all,
            silently producing a non-identifiable variance-component split
            between the genotype random effect and the fixed effect. No
            upfront identifiability/collinearity pre-validation is performed
            — a fixed effect chosen as a metadata covariate should still be
            reviewed for how it's distributed across genotypes.
            ``fixed_effects`` is independent of ``replicate_col`` — a
            block/replicate fixed effect is expressed by naming that column
            here directly.

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
            - blup: BLUP (Best Linear Unbiased Prediction) per genotype, a
              dict[str, float] from the fitted mixed model's
              ``result.random_effects``. Present only when
              ``model_type == "mixed_model"`` — the ANOVA-based and
              no-variance paths never fit a mixedlm model, so they carry no
              ``blup``/``intercept`` keys.
            - intercept: The fixed-effect intercept from the same fit. When
              ``fixed_effects`` is empty/``None``, this is exactly
              ``result.fe_params["Intercept"]``. When ``fixed_effects`` is
              non-empty, this is instead the empirical, sample
              frequency-weighted intercept computed by
              :func:`_marginal_intercept` — a sample-composition-dependent
              value that can differ trait-to-trait (each trait computes its
              own ``dropna()`` subset), not a population-typical or
              EMM/lsmeans-style equally-weighted value. The genotype-adjusted
              mean is ``intercept + blup[genotype]`` either way. Present
              under the same condition as ``blup``.

        If remove_low_h2=True:
            Tuple of:
            - Dictionary with heritability estimates (as above)
            - DataFrame with low heritability traits removed
            - List of removed trait names
            - Dictionary with removal details

    Raises:
        Nothing propagates to the caller. When ``fixed_effects`` is
        non-empty, a captured ``ConvergenceWarning`` is converted to an
        internal ``RuntimeError`` to route it through the same handling as a
        raised model-fit exception — both are caught by this function's own
        per-trait ``try/except`` and recorded as that trait's
        ``{"error": ..., "model_type": "mixed_model_failed"}`` entry, never
        raised to the caller.
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
    heritability_results["__calculation_metadata__"] = {
        "method_used_for_all_traits": method_used,
        "method_consistency": True,
    }

    # replicate_col is optional and its values are never used in the model
    # (value ~ 1 + (1|genotype)). Only require the column when a (truthy) name was
    # provided; treat None or "" identically as "no replicate column" (issue #142).
    fixed_effects = fixed_effects or []
    required_cols = [genotype_col]
    if replicate_col:
        required_cols.append(replicate_col)
    required_cols.extend(fixed_effects)
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing required columns: {missing_cols}"}

    # genotype is a categorical label, never a quantity. A DataFrame that has
    # round-tripped through an intermediate CSV write/read (as pipeline steps
    # do between stages) can lose its original dtype: numeric-looking
    # accessions (e.g. "600824") get re-inferred as int/float on the next
    # read_csv while named cultivars stay str, producing a mixed-type column.
    # patsy/statsmodels then raises "'<' not supported between instances of
    # 'int' and 'str'" while building the mixed-model design (reproduced on a
    # real 3,343-row alfalfa GWAS dataset -- every one of 925 traits failed
    # with this exact error before this fix). Force a single string dtype
    # regardless of what the caller's DataFrame carries.
    df = df.copy()
    df[genotype_col] = df[genotype_col].astype(str)

    # Every fixed_effects name is interpolated directly into the formula
    # string below (issue #114); a name that isn't a valid identifier (e.g.
    # containing a patsy operator like `*` or `:`) could otherwise silently
    # misparse as an expression over other, differently-named columns. A
    # non-str element (e.g. an int-labeled CSV column) has no
    # .isidentifier() at all -- checked first so this returns the same
    # structural error instead of an uncaught AttributeError (PR #193
    # review).
    invalid_fe_names = [
        fe for fe in fixed_effects if not isinstance(fe, str) or not fe.isidentifier()
    ]
    if invalid_fe_names:
        return {"error": f"Invalid fixed_effects column name(s): {invalid_fe_names}"}

    # Naming genotype_col or replicate_col as a fixed effect too produces a
    # duplicate-column selection and a confusing pandas-internal error deep
    # inside the per-trait loop (e.g. "Grouper for 'geno' not 1-dimensional")
    # rather than a clear structural error -- reject it upfront instead.
    reused_names = [
        fe for fe in fixed_effects if fe == genotype_col or fe == replicate_col
    ]
    if reused_names:
        return {
            "error": (
                f"fixed_effects column(s) {reused_names} duplicate "
                f"genotype_col/replicate_col; name a different column"
            )
        }

    # A name repeated within fixed_effects itself (e.g. ["experiment",
    # "experiment"]) produces a duplicate C(...) term in the formula below,
    # which degrades to an obscure patsy failure deep inside the per-trait
    # try/except rather than a clear structural error (PR #193 review).
    seen_fe_names = set()
    duplicate_fe_names = []
    for fe in fixed_effects:
        if fe in seen_fe_names and fe not in duplicate_fe_names:
            duplicate_fe_names.append(fe)
        seen_fe_names.add(fe)
    if duplicate_fe_names:
        return {
            "error": f"Duplicate fixed_effects column name(s): {duplicate_fe_names}"
        }

    for trait in trait_cols:
        if trait not in df.columns:
            heritability_results[trait] = {"error": f"Trait column '{trait}' not found"}
            continue

        # Subset to the only columns the model uses. Replicate is deliberately
        # excluded: its values are never used, so including it (and any NaNs in it)
        # must not drop rows or change H² relative to replicate=None (issue #142).
        # fixed_effects columns ARE included (issue #114): a NaN in a named
        # fixed-effect column drops that row, applied identically regardless
        # of force_method (the ANOVA-based path below never uses
        # fixed_effects in its own formula, only in this shared row filter).
        subset = df[[trait, genotype_col] + fixed_effects].dropna()

        if len(subset) < 4:  # Need minimum data for variance estimation
            heritability_results[trait] = {
                "error": "Insufficient data for heritability estimation"
            }
            continue

        try:
            # Calculate mean number of replicates per genotype (for unbalanced design)
            reps_per_geno = subset.groupby(genotype_col).size()
            mean_n_reps = reps_per_geno.mean()

            # Heritability needs between-genotype contrast: with a single genotype
            # there is no estimable genetic variance, so report a structured error
            # instead of a meaningless H² from an unidentifiable model (issue #142).
            if len(reps_per_geno) < 2:
                heritability_results[trait] = {
                    "error": (
                        "Insufficient genotypes for heritability estimation "
                        "(need >= 2 genotypes)"
                    )
                }
                continue

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

            # blup/intercept are BLUPs (Best Linear Unbiased Predictions) extracted
            # from the mixed model fit below (issue #109). Only the mixed-model
            # branch has a fitted `result` to extract them from; the ANOVA-based
            # branch (below) computes variance components via groupby arithmetic
            # and never fits a model, so these stay None for that path.
            blup = None
            intercept = None

            if use_mixed_model:
                # Use mixed model approach (matches R lme4)
                # Rename to canonical value/genotype columns; fixed_effects
                # columns keep their original names since the formula below
                # references them by name (issue #114).
                model_data = subset.rename(
                    columns={trait: "value", genotype_col: "genotype"}
                )

                # Fit mixed model: value ~ 1 + (1|genotype), or
                # value ~ C(fe_1) + ... + (1|genotype) when fixed_effects is
                # set (issue #114). Every fixed effect is wrapped in C(...)
                # unconditionally -- always treated as categorical.
                # This matches the R code: lmer(value ~ (1 | ecot_id), data = data_H)
                if fixed_effects:
                    formula = "value ~ " + " + ".join(
                        f"C({fe})" for fe in fixed_effects
                    )
                else:
                    formula = "value ~ 1"
                try:
                    model = smf.mixedlm(
                        formula, model_data, groups=model_data["genotype"]
                    )
                    # statsmodels does not reliably *raise* on a fixed effect
                    # confounded with genotype -- it can instead emit a
                    # ConvergenceWarning and still return a plausible-looking
                    # result. Only when fixed_effects is set, capture
                    # warnings (forcing "always" so a repeat identical
                    # warning for a later trait isn't silently dropped by
                    # Python's default once-per-location filter) and treat a
                    # ConvergenceWarning as a fit failure via the same
                    # except block below -- gated on fixed_effects so
                    # fixed_effects=None callers see byte-for-byte identical
                    # behavior to before this parameter existed (issue #114).
                    if fixed_effects:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            result = model.fit(reml=True)
                        for w in caught:
                            if issubclass(w.category, ConvergenceWarning):
                                raise RuntimeError(f"Convergence warning: {w.message}")
                    else:
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

                    # Extract BLUPs (issue #109): result.random_effects is a lazy
                    # property, accessed exactly once here.
                    blup = {
                        str(geno): float(effect.iloc[0])
                        for geno, effect in result.random_effects.items()
                    }
                    if fixed_effects:
                        intercept = _marginal_intercept(
                            result, model_data, fixed_effects
                        )
                        # statsmodels' own convergence-warning check (above)
                        # is a confirmed false-negative for a fixed effect
                        # near-deterministically confounded with genotype --
                        # a fit can succeed cleanly with zero warnings (PR
                        # #193 review, 7.5). Surface an independent, cheap
                        # diagnostic instead of leaving that case silent: if
                        # every observation for a genotype sits in a single
                        # level of a fixed effect that has more than one
                        # level overall, that genotype contributes no
                        # within-genotype information for separating the two
                        # effects, inflating apparent heritability.
                        for fe in fixed_effects:
                            if model_data[fe].nunique() < 2:
                                continue
                            levels_per_genotype = model_data.groupby("genotype")[
                                fe
                            ].nunique()
                            confounded_genotypes = levels_per_genotype[
                                levels_per_genotype < 2
                            ].index.tolist()
                            if confounded_genotypes:
                                shown = confounded_genotypes[:5]
                                more = (
                                    f" (+{len(confounded_genotypes) - 5} more)"
                                    if len(confounded_genotypes) > 5
                                    else ""
                                )
                                warnings.warn(
                                    f"Trait '{trait}': fixed effect '{fe}' "
                                    f"may be confounded with genotype -- "
                                    f"{len(confounded_genotypes)} genotype(s) "
                                    f"appear in only one level of '{fe}': "
                                    f"{shown}{more}. Heritability may be "
                                    f"inflated by attributing '{fe}' "
                                    f"variation to genotype.",
                                    UserWarning,
                                )
                    else:
                        intercept = float(result.fe_params["Intercept"])

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
            # Additive BLUP keys (issue #109) — only present when the mixed
            # model actually fit (blup is None for the ANOVA-based path).
            if blup is not None:
                heritability_results[trait]["blup"] = blup
                heritability_results[trait]["intercept"] = intercept

        except Exception as e:
            heritability_results[trait] = {
                "error": f"Heritability calculation failed: {str(e)}"
            }

    # Optionally filter low heritability traits
    if remove_low_h2:
        # fixed_effects columns are metadata covariates, not candidate traits
        # (issue #114) -- without excluding them here, get_trait_columns()
        # (called internally by remove_low_heritability_traits) has no way to
        # know they aren't traits, since it was never told they were fit as
        # fixed effects rather than trait_cols. Left unexcluded, a
        # fixed_effects column with no entry in heritability_results (it was
        # never fit as a trait) gets silently removed from df_filtered with
        # reason "No heritability estimate available".
        combined_exclude = list(
            dict.fromkeys((additional_exclude or []) + fixed_effects)
        )
        df_filtered, removed_traits, removal_details = remove_low_heritability_traits(
            df=df,
            heritability_results=heritability_results,
            heritability_threshold=h2_threshold,
            barcode_col=barcode_col,
            genotype_col=genotype_col,
            replicate_col=replicate_col,
            additional_exclude=combined_exclude or None,
        )
        return heritability_results, df_filtered, removed_traits, removal_details

    return heritability_results


def extract_blup_table(heritability_results: Dict) -> pd.DataFrame:
    """Build a genotype x trait BLUP-adjusted-means table (issue #109).

    Consumes the dict returned by ``calculate_heritability_estimates`` (the
    ``remove_low_h2=False`` form, or the first element of the
    ``remove_low_h2=True`` tuple) and builds a table of
    ``adjusted_mean = intercept + blup[genotype]`` for every trait whose mixed
    model succeeded.

    A trait with no ``blup``/``intercept`` keys (the model failed, used the
    ANOVA-based or no-variance path, or was skipped) gets an entire ``NaN``
    column — not omitted from the table and not zero-filled. A genotype
    missing from one succeeded trait's ``blup`` dict but present in another's
    (traits compute their own genotype set independently, via a per-trait
    ``dropna()``) gets a cell-level ``NaN`` for that genotype/trait pair only.

    Does not mutate its input. Never raises: a run-level short-circuit dict
    (``{"error": "..."}``, no per-trait entries) produces an empty
    ``pd.DataFrame()``; a dict where every trait failed produces a zero-row
    table with one all-``NaN`` column per input trait.

    Args:
        heritability_results: The dict returned by
            ``calculate_heritability_estimates``.

    Returns:
        pd.DataFrame: Rows indexed by genotype (the union of every succeeded
        trait's ``blup`` keys), one column per trait (excluding
        ``__calculation_metadata__``), in the input's trait order.
    """
    run_level_error = heritability_results.get("error")
    if isinstance(run_level_error, str):
        return pd.DataFrame()

    trait_entries = {
        trait: entry
        for trait, entry in heritability_results.items()
        if trait != "__calculation_metadata__"
    }

    genotype_universe: set = set()
    for entry in trait_entries.values():
        blup = entry.get("blup") if isinstance(entry, dict) else None
        if blup is not None:
            genotype_universe.update(blup.keys())

    genotypes = sorted(genotype_universe)
    columns = {}
    for trait, entry in trait_entries.items():
        blup = entry.get("blup") if isinstance(entry, dict) else None
        intercept = entry.get("intercept") if isinstance(entry, dict) else None
        if blup is None or intercept is None:
            columns[trait] = [np.nan] * len(genotypes)
        else:
            columns[trait] = [
                intercept + blup[g] if g in blup else np.nan for g in genotypes
            ]

    return pd.DataFrame(columns, index=genotypes)


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


def analyze_trait_variance(
    df: pd.DataFrame,
    trait: str,
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> Dict[str, Any]:
    """Analyze variance components for a single trait.

    Decomposes trait variance into between-genotype and within-genotype
    components to support heritability diagnostics.

    Args:
        df: DataFrame with trait data
        trait: Name of trait column to analyze
        genotype_col: Name of genotype column (default: "geno")
        replicate_col: Name of replicate column (default: "rep"), or None if the
            dataset has no replicate column. Replicate values are not used in the
            variance decomposition.

    Returns:
        Dictionary containing:
            - n_observations: Total number of valid observations
            - n_genotypes: Number of unique genotypes
            - mean_reps_per_geno: Average replicates per genotype
            - min_reps_per_geno: Minimum replicates per genotype
            - max_reps_per_geno: Maximum replicates per genotype
            - overall_variance: Total variance of trait values
            - between_genotype_variance: Variance of genotype means
            - within_genotype_variance: Mean variance within genotypes
            - pct_variance_between_geno: Percentage of variance between genotypes
            - trait_mean: Mean of trait values
            - trait_std: Standard deviation of trait values
            - trait_cv: Coefficient of variation (%)

    Example:
        >>> df = pd.DataFrame({
        ...     'geno': ['G1', 'G1', 'G2', 'G2'],
        ...     'rep': [1, 2, 1, 2],
        ...     'trait1': [10.0, 12.0, 20.0, 22.0]
        ... })
        >>> result = analyze_trait_variance(df, 'trait1')
        >>> print(f"Between-genotype variance: {result['between_genotype_variance']:.2f}")
    """
    # A named-but-absent replicate column is a caller error: return a structured
    # error (mirroring calculate_heritability_estimates) rather than raising a raw
    # KeyError. A falsy replicate_col (None or "") means "no replicate column".
    if replicate_col and replicate_col not in df.columns:
        return {"error": f"Missing required columns: {[replicate_col]}"}

    # Subset to the only columns the decomposition uses (it groups by genotype).
    # Replicate is excluded so its presence/NaNs never change the result (issue #142).
    subset = df[[trait, genotype_col]].dropna()

    # Check for insufficient data
    if len(subset) < 3:
        return {
            "error": "Insufficient data for variance analysis",
            "n_observations": len(subset),
        }

    # Basic counts
    n_obs = len(subset)
    reps_per_geno = subset.groupby(genotype_col).size()
    n_genotypes = len(reps_per_geno)

    # Overall variance and statistics
    overall_var = subset[trait].var(ddof=1) if len(subset) > 1 else 0.0
    overall_mean = subset[trait].mean()
    overall_std = subset[trait].std(ddof=1) if len(subset) > 1 else 0.0

    # Coefficient of variation
    cv = (overall_std / overall_mean * 100) if overall_mean != 0 else np.inf

    # Between-genotype variance (variance of genotype means)
    geno_means = subset.groupby(genotype_col)[trait].mean()
    between_geno_var = geno_means.var(ddof=1) if len(geno_means) > 1 else 0.0

    # Within-genotype variance (mean of variances within each genotype)
    geno_vars = subset.groupby(genotype_col)[trait].var(ddof=1)
    # Filter out NaN values (groups with only 1 observation)
    geno_vars_valid = geno_vars.dropna()
    within_geno_var = geno_vars_valid.mean() if len(geno_vars_valid) > 0 else 0.0

    # Calculate percentage of variance between genotypes
    total_var = between_geno_var + within_geno_var
    if total_var > 0:
        pct_between = (between_geno_var / total_var) * 100
    else:
        pct_between = 0.0

    return {
        "n_observations": int(n_obs),
        "n_genotypes": int(n_genotypes),
        "mean_reps_per_geno": float(reps_per_geno.mean()),
        "min_reps_per_geno": int(reps_per_geno.min()),
        "max_reps_per_geno": int(reps_per_geno.max()),
        "trait_mean": float(overall_mean),
        "trait_std": float(overall_std),
        "trait_cv": float(cv),
        "overall_variance": float(overall_var),
        "between_genotype_variance": float(between_geno_var),
        "within_genotype_variance": float(within_geno_var),
        "pct_variance_between_geno": float(pct_between),
    }


def diagnose_heritability_issues(
    df: pd.DataFrame,
    trait: str,
    heritability_result: Dict[str, Any],
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
) -> Dict[str, Any]:
    """Identify specific causes of low or zero heritability with explanations.

    Args:
        df: DataFrame with trait data
        trait: Name of trait to diagnose
        heritability_result: Dictionary from calculate_heritability_estimates()
        genotype_col: Name of genotype column (default: "geno")
        replicate_col: Name of replicate column (default: "rep")

    Returns:
        Dictionary containing:
            - has_issues: Boolean indicating if problems detected
            - issues: List of issue descriptions
            - severity: "critical", "warning", or "info"
            - recommendations: List of suggested actions

    Example:
        >>> h2_results = calculate_heritability_estimates(df, ['trait1'])
        >>> diagnosis = diagnose_heritability_issues(df, 'trait1', h2_results['trait1'])
        >>> if diagnosis['has_issues']:
        ...     print(f"Issues: {', '.join(diagnosis['issues'])}")
    """
    issues = []
    recommendations = []
    severity = "info"

    # Check for errors in heritability calculation
    if "error" in heritability_result:
        return {
            "has_issues": True,
            "issues": [f"Model failure: {heritability_result['error']}"],
            "severity": "critical",
            "recommendations": [
                "Check data quality",
                "Ensure sufficient genotypes and replicates",
                "Try ANOVA-based method if mixed model failed",
            ],
        }

    # Get heritability value
    h2 = heritability_result.get("heritability", np.nan)
    if np.isnan(h2):
        return {
            "has_issues": True,
            "issues": ["Heritability could not be calculated"],
            "severity": "critical",
            "recommendations": ["Check input data for missing or invalid values"],
        }

    # Get variance analysis
    var_analysis = analyze_trait_variance(df, trait, genotype_col, replicate_col)

    if "error" in var_analysis:
        return {
            "has_issues": True,
            "issues": [var_analysis["error"]],
            "severity": "critical",
            "recommendations": ["Increase sample size", "Check for missing data"],
        }

    # Check for zero heritability
    if h2 == 0.0:
        severity = "critical"

        # Diagnose why
        if var_analysis["between_genotype_variance"] == 0.0:
            issues.append("No variation between genotype means")
            recommendations.append("Check if trait is constant across genotypes")
            recommendations.append("Verify genotype labels are correct")
        elif var_analysis["overall_variance"] == 0.0:
            issues.append("No variation in trait values (all identical)")
            recommendations.append("Check data quality and measurement accuracy")
        else:
            # High within-genotype variance
            ratio = (
                var_analysis["within_genotype_variance"]
                / var_analysis["between_genotype_variance"]
                if var_analysis["between_genotype_variance"] > 0
                else np.inf
            )
            if ratio > 10:
                issues.append(
                    f"Within-genotype variation >> between-genotype variation "
                    f"(ratio: {ratio:.1f}x)"
                )
                recommendations.append("Check for high measurement noise")
                recommendations.append("Consider improving experimental conditions")
                recommendations.append("Verify replicate quality")

    # Check sample size
    n_obs = var_analysis["n_observations"]
    if n_obs < 30:
        issues.append(f"Low sample size (n={n_obs})")
        severity = "warning" if severity == "info" else severity
        recommendations.append(
            f"Increase sample size (current: {n_obs}, recommended: >30)"
        )

    # Check design balance
    min_reps = var_analysis["min_reps_per_geno"]
    max_reps = var_analysis["max_reps_per_geno"]
    if max_reps > min_reps * 2:
        issues.append(f"Imbalanced design (reps per genotype: {min_reps}-{max_reps})")
        recommendations.append("Consider balancing replicate numbers across genotypes")

    # Check for high within-genotype variation (even if H² > 0)
    if h2 > 0:
        ratio = (
            var_analysis["within_genotype_variance"]
            / var_analysis["between_genotype_variance"]
            if var_analysis["between_genotype_variance"] > 0
            else 0
        )
        if ratio > 3:
            issues.append(
                f"High within-genotype variation (within/between ratio: {ratio:.1f})"
            )
            if severity == "info":
                severity = "warning"

    # Determine if issues exist
    has_issues = len(issues) > 0 or h2 < 0.3

    # If no specific issues but low H², provide general note
    if len(issues) == 0 and h2 < 0.3:
        issues.append(f"Low heritability (H²={h2:.3f})")
        severity = "warning"
        recommendations.append(
            "Trait may have weak genetic control or high environmental influence"
        )

    return {
        "has_issues": has_issues,
        "issues": issues,
        "severity": severity,
        "recommendations": recommendations if has_issues else [],
    }


def compare_trait_heritabilities(
    df: pd.DataFrame,
    traits: List[str],
    heritability_results: Dict[str, Dict[str, Any]],
    genotype_col: str = "geno",
    replicate_col: Optional[str] = "rep",
    sort_by: Optional[str] = None,
) -> pd.DataFrame:
    """Compare variance components and heritability metrics for multiple traits.

    Args:
        df: DataFrame with trait data
        traits: List of trait names to compare
        heritability_results: Dictionary mapping trait names to heritability results
        genotype_col: Name of genotype column (default: "geno")
        replicate_col: Name of replicate column (default: "rep")
        sort_by: Optional column name to sort by (default: None)

    Returns:
        DataFrame with one row per trait and columns for variance metrics

    Example:
        >>> traits = ['trait1', 'trait2', 'trait3']
        >>> h2_results = calculate_heritability_estimates(df, traits)
        >>> comparison = compare_trait_heritabilities(df, traits, h2_results)
        >>> print(comparison[['trait', 'heritability', 'pct_var_between']])
    """
    if len(traits) == 0:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(
            columns=[
                "trait",
                "heritability",
                "var_genetic",
                "var_residual",
                "between_geno_var",
                "within_geno_var",
                "pct_var_between",
                "n_observations",
                "n_genotypes",
                "mean_reps_per_geno",
                "trait_mean",
                "trait_cv",
                "model_type",
            ]
        )

    results = []

    for trait in traits:
        # Get heritability result
        h2_result = heritability_results.get(trait, {})

        # Get variance analysis
        var_analysis = analyze_trait_variance(df, trait, genotype_col, replicate_col)

        # Handle errors
        if "error" in h2_result or "error" in var_analysis:
            result = {
                "trait": trait,
                "heritability": np.nan,
                "var_genetic": np.nan,
                "var_residual": np.nan,
                "between_geno_var": np.nan,
                "within_geno_var": np.nan,
                "pct_var_between": np.nan,
                "n_observations": var_analysis.get("n_observations", 0),
                "n_genotypes": var_analysis.get("n_genotypes", 0),
                "mean_reps_per_geno": np.nan,
                "trait_mean": np.nan,
                "trait_cv": np.nan,
                "model_type": "error",
            }
        else:
            result = {
                "trait": trait,
                "heritability": h2_result.get("heritability", np.nan),
                "var_genetic": h2_result.get("var_genetic", np.nan),
                "var_residual": h2_result.get("var_residual", np.nan),
                "between_geno_var": var_analysis.get(
                    "between_genotype_variance", np.nan
                ),
                "within_geno_var": var_analysis.get("within_genotype_variance", np.nan),
                "pct_var_between": var_analysis.get(
                    "pct_variance_between_geno", np.nan
                ),
                "n_observations": var_analysis.get("n_observations", 0),
                "n_genotypes": var_analysis.get("n_genotypes", 0),
                "mean_reps_per_geno": var_analysis.get("mean_reps_per_geno", np.nan),
                "trait_mean": var_analysis.get("trait_mean", np.nan),
                "trait_cv": var_analysis.get("trait_cv", np.nan),
                "model_type": h2_result.get("model_type", "unknown"),
            }

        results.append(result)

    # Create DataFrame
    comparison_df = pd.DataFrame(results)

    # Sort if requested
    if sort_by and sort_by in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=sort_by)

    return comparison_df
