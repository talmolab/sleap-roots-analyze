"""Centralized pytest fixtures for test data."""

import json

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from scipy import stats
from omegaconf import OmegaConf, DictConfig


# ============================================================================
# PATH FIXTURES - Test data file paths
# ============================================================================


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def features_csv_path(test_data_dir):
    """Return the path to features.csv file."""
    return test_data_dir / "features.csv"


@pytest.fixture(scope="session")
def traits_11dag_csv_path(test_data_dir):
    """Return the path to traits_11DAG_cleaned_qc_scanner_independent.csv file."""
    return test_data_dir / "traits_11DAG_cleaned_qc_scanner_independent.csv"


@pytest.fixture(scope="session")
def traits_summary_csv_path(test_data_dir):
    """Return the path to traits_summary.csv file."""
    return test_data_dir / "traits_summary.csv"


@pytest.fixture(scope="session")
def traits_summary_lateral_csv_path(test_data_dir):
    """Return the path to traits_summary_lateral.csv file."""
    return test_data_dir / "traits_summary_lateral.csv"


@pytest.fixture(scope="session")
def turface_traits_csv_path(test_data_dir):
    """Return the path to Turface_all_traits_2024.csv file."""
    return test_data_dir / "Turface_all_traits_2024.csv"


@pytest.fixture(scope="session")
def turface_rsr_csv_path(test_data_dir):
    """Return the path to Turface_all_traits_2024_RSR.csv file."""
    return test_data_dir / "Turface_all_traits_2024_RSR.csv"


@pytest.fixture(scope="session")
def wheat_edpie_excel_path(test_data_dir):
    """Return the path to Wheat_EDPIE_cylinder_master_data.xlsx file."""
    return test_data_dir / "Wheat_EDPIE_cylinder_master_data.xlsx"


# ============================================================================
# DATAFRAME FIXTURES - Loaded CSV/Excel data
# ============================================================================


@pytest.fixture(scope="session")
def features_df(features_csv_path):
    """Load features.csv as a pandas DataFrame.

    This contains root system features including:
    - File.Name, Region.of.Interest
    - Root measurements (count, tips, length, depth, width, etc.)
    - Diameter ranges and volume measurements
    - Computation time and angle frequencies
    """
    return pd.read_csv(features_csv_path)


@pytest.fixture(scope="session")
def traits_11dag_df(traits_11dag_csv_path):
    """Load traits_11DAG_cleaned_qc_scanner_independent.csv as a pandas DataFrame.

    This contains 11 DAG (days after germination) trait data including:
    - Plant metadata (QR codes, genotype, replication, sterilization)
    - Scan information (scan_id, date, experiment details)
    - Crown and lateral root measurements
    - Network measurements and statistics
    """
    return pd.read_csv(traits_11dag_csv_path)


@pytest.fixture(scope="session")
def traits_summary_df(traits_summary_csv_path):
    """Load traits_summary.csv as a pandas DataFrame.

    This contains summarized trait data including:
    - Plant and scan identification
    - Species information
    - Crown root statistics (count, length, angles)
    - Scanline intersection counts
    - Network measurements
    """
    return pd.read_csv(traits_summary_csv_path)


@pytest.fixture(scope="session")
def traits_summary_lateral_df(traits_summary_lateral_csv_path):
    """Load traits_summary_lateral.csv as a pandas DataFrame.

    This contains lateral root specific summary data including:
    - Lateral root counts and lengths
    - Total lateral length measurements
    - Statistical summaries (min, max, mean, median, percentiles)
    """
    return pd.read_csv(traits_summary_lateral_csv_path)


@pytest.fixture(scope="session")
def turface_traits_df(turface_traits_csv_path):
    """Load Turface_all_traits_2024.csv as a pandas DataFrame.

    This contains 2024 Turface experiment trait data.
    """
    return pd.read_csv(turface_traits_csv_path)


@pytest.fixture(scope="session")
def wheat_edpie_excel_df(wheat_edpie_excel_path):
    """Load Wheat_EDPIE_cylinder_master_data.xlsx as a pandas DataFrame.

    This contains wheat EDPIE cylinder master data.
    Note: Returns the first sheet by default.
    """
    return pd.read_excel(wheat_edpie_excel_path)


# ============================================================================
# SAMPLE DATA FIXTURES - Small subsets for quick testing
# ============================================================================


@pytest.fixture
def features_sample(features_df):
    """Return first 10 rows of features data for quick testing."""
    return features_df.head(10).copy()


@pytest.fixture
def traits_11dag_sample(traits_11dag_df):
    """Return first 10 rows of traits_11DAG data for quick testing."""
    return traits_11dag_df.head(10).copy()


@pytest.fixture
def traits_summary_sample(traits_summary_df):
    """Return first 10 rows of traits_summary data for quick testing."""
    return traits_summary_df.head(10).copy()


@pytest.fixture
def traits_summary_lateral_sample(traits_summary_lateral_df):
    """Return first 10 rows of traits_summary_lateral data for quick testing."""
    return traits_summary_lateral_df.head(10).copy()


@pytest.fixture
def turface_traits_sample(turface_traits_df):
    """Return first 10 rows of turface_traits data for quick testing."""
    return turface_traits_df.head(10).copy()


# ============================================================================
# HERITABILITY TESTING FIXTURES - Data with known heritability values
# ============================================================================


@pytest.fixture
def heritability_data_known_h2():
    """Generate data with known heritability values for validation.

    Returns:
        tuple: (DataFrame, dict of expected h2 values)

    Expected H² calculations:
    - trait_high_h2: H² = 0.8 (σ²_G = 4.0, σ²_E = 1.0)
    - trait_moderate_h2: H² = 0.5 (σ²_G = 1.0, σ²_E = 1.0)
    - trait_low_h2: H² = 0.09 (σ²_G = 0.1, σ²_E = 1.0)
    """
    np.random.seed(42)

    n_genotypes = 20
    n_reps = 5

    # Known variance components
    genetic_vars = {"high": 4.0, "moderate": 1.0, "low": 0.1}
    env_var = 1.0

    # Expected heritabilities (broad-sense)
    expected_h2 = {
        "trait_high_h2": genetic_vars["high"] / (genetic_vars["high"] + env_var),
        "trait_moderate_h2": genetic_vars["moderate"]
        / (genetic_vars["moderate"] + env_var),
        "trait_low_h2": genetic_vars["low"] / (genetic_vars["low"] + env_var),
    }

    data = []
    for g in range(n_genotypes):
        # Genetic effects for each trait
        g_effect_high = np.random.normal(0, np.sqrt(genetic_vars["high"]))
        g_effect_mod = np.random.normal(0, np.sqrt(genetic_vars["moderate"]))
        g_effect_low = np.random.normal(0, np.sqrt(genetic_vars["low"]))

        for r in range(n_reps):
            # Environmental effects
            e_high = np.random.normal(0, np.sqrt(env_var))
            e_mod = np.random.normal(0, np.sqrt(env_var))
            e_low = np.random.normal(0, np.sqrt(env_var))

            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_high_h2": 100 + g_effect_high + e_high,
                    "trait_moderate_h2": 50 + g_effect_mod + e_mod,
                    "trait_low_h2": 25 + g_effect_low + e_low,
                }
            )

    df = pd.DataFrame(data)

    return df, expected_h2


@pytest.fixture
def heritability_perfect_data():
    """Generate perfect heritability data (H² = 1.0, no environmental variance).

    Returns:
        pd.DataFrame: Data where all variation is genetic
    """
    n_genotypes = 10
    n_reps = 4

    data = []
    for g in range(n_genotypes):
        # Each genotype has a fixed genetic value, no environmental variation
        genetic_value = 10 + g * 5

        for r in range(n_reps):
            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_perfect": genetic_value,  # No environmental noise
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_zero_data():
    """Generate zero heritability data (H² = 0.0, no genetic variance).

    Returns:
        pd.DataFrame: Data where all variation is environmental
    """
    np.random.seed(42)
    n_genotypes = 10
    n_reps = 4

    data = []
    for g in range(n_genotypes):
        for r in range(n_reps):
            # All variation is environmental, no genetic effects
            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_zero": 50
                    + np.random.normal(0, 5),  # Only environmental noise
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_data_unbalanced_reps():
    """Generate unbalanced-replication data for the BLUP shrinkage oracle.

    Known genetic variance sigma^2_G=4.0, residual variance sigma^2_E=1.0.
    10 "low-rep" genotypes (G01-G10) have n=2 reps each; 10 "high-rep"
    genotypes (G11-G20) have n=20 reps each. Low-rep genotypes' raw means
    are noisier and should shrink further toward the grand mean than
    high-rep genotypes' raw means once BLUPs are computed.

    Returns:
        tuple: (pd.DataFrame, dict) where the dict is
            {"low_rep_genotypes": [...], "high_rep_genotypes": [...],
            "trait": "trait_unbalanced"}. Raw means and the grand mean are
            deliberately not pre-computed here; tests derive them directly
            from the returned DataFrame so there is one source of truth.
    """
    np.random.seed(42)

    low_rep_genotypes = [f"G{g:02d}" for g in range(1, 11)]
    high_rep_genotypes = [f"G{g:02d}" for g in range(11, 21)]

    data = []
    barcode = 0
    for geno in low_rep_genotypes:
        genotype_effect = np.random.normal(0, 2.0)
        for r in range(2):
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_unbalanced": 50 + genotype_effect + np.random.normal(0, 1.0),
                }
            )
            barcode += 1
    for geno in high_rep_genotypes:
        genotype_effect = np.random.normal(0, 2.0)
        for r in range(20):
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_unbalanced": 50 + genotype_effect + np.random.normal(0, 1.0),
                }
            )
            barcode += 1

    df = pd.DataFrame(data)
    meta = {
        "low_rep_genotypes": low_rep_genotypes,
        "high_rep_genotypes": high_rep_genotypes,
        "trait": "trait_unbalanced",
    }
    return df, meta


@pytest.fixture
def heritability_data_batch_confounded():
    """Generate batch-confounded data for the fixed_effects H2 oracle.

    Mirrors issue #114's Bloom-experiment scenario: 20 genotypes, n=10
    reps each, evenly split into two groups whose reps are mostly (but not
    entirely) in one of two synthetic batches — genuinely partial
    per-genotype mixing, not full determinism. Combined with a large
    per-batch shift, the resulting between-genotype variance from the
    confound exceeds the within-genotype variance from the partial mixing,
    so uncorrected H2 (fixed_effects=None) comes out above corrected H2
    (fixed_effects=["experiment"]).

    Verified empirically (this exact construction, legacy np.random.seed
    API): seed=42 gives H2_uncorrected~=0.9405, H2_corrected~=0.7194,
    gap~=0.2211. Across 10 tested seeds (1,2,3,7,13,21,42,99,123,777) the
    minimum gap is 0.2138 -- comfortably above the 0.05 threshold tests
    assert. Do not change sigma_g/shift/rep counts without re-verifying;
    an earlier draft (sigma_g=3.0, shift=8.0, n=4 reps, a 16/4-genotype
    3:1/1:3 partial mix) was confirmed by simulation to produce the
    opposite sign, because with too few reps any genuinely partial mix
    injects more within-genotype variance than between-genotype variance,
    regardless of shift magnitude.

    Returns:
        tuple: (pd.DataFrame, dict) where the dict is
            {"trait": "trait_batch_confounded", "batch_col": "experiment"}.
    """
    np.random.seed(42)

    mostly_a = [f"G{g:02d}" for g in range(1, 11)]
    mostly_b = [f"G{g:02d}" for g in range(11, 21)]
    n_reps = 10

    data = []
    barcode = 0
    for geno in mostly_a:
        genotype_effect = np.random.normal(0, 0.4)
        batches = ["Bloom_A"] * 9 + ["Bloom_B"] * 1
        for r, batch in enumerate(batches):
            shift = 10.0 if batch == "Bloom_B" else 0.0
            value = 50 + genotype_effect + shift + np.random.normal(0, 1.0)
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_batch_confounded": value,
                    "experiment": batch,
                }
            )
            barcode += 1
    for geno in mostly_b:
        genotype_effect = np.random.normal(0, 0.4)
        batches = ["Bloom_A"] * 1 + ["Bloom_B"] * 9
        for r, batch in enumerate(batches):
            shift = 10.0 if batch == "Bloom_B" else 0.0
            value = 50 + genotype_effect + shift + np.random.normal(0, 1.0)
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_batch_confounded": value,
                    "experiment": batch,
                }
            )
            barcode += 1

    df = pd.DataFrame(data)
    meta = {"trait": "trait_batch_confounded", "batch_col": "experiment"}
    return df, meta


@pytest.fixture
def heritability_data_field_block():
    """Generate field-block data for the fixed_effects BLUP/shrinkage oracles.

    15 genotypes split into two replicate-count groups: 7 "low-rep"
    genotypes (G01-G07) with n=2 reps each, 8 "high-rep" genotypes
    (G08-G15) with n=10 reps each. A per-"block" shift is added, with
    block-composition skew applied as the same ~71% block_1-heavy ratio
    within *both* replicate-count groups -- orthogonal to which
    replicate-count group a genotype is in. This orthogonality is
    required: an earlier draft skewed block assignment by genotype ID in
    a way that correlated with the replicate-count grouping, and was
    confirmed by simulation to produce an unreliable shrinkage oracle
    (~40% failure rate on the naive comparison, even though the simpler
    BLUP-difference oracle was unaffected).

    Verified empirically (this exact construction, legacy np.random.seed
    API, 10 tested seeds): the BLUP-adjusted-means oracle (fixed_effects
    vs. none) never failed to show a difference, and the
    shrinkage-scales-with-replication oracle never failed *when compared
    against a block-detrended raw mean* (subtracting the fitted C(block)
    coefficient per observation before averaging within genotype) --
    comparing against the naive, non-detrended raw mean is NOT a valid
    shrinkage test here, since that naive mean is itself contaminated by
    each genotype's own block composition, the exact thing the fixed
    effect corrects for.

    Returns:
        tuple: (pd.DataFrame, dict) where the dict is
            {"trait": "trait_field_block", "low_rep_genotypes": [...],
            "high_rep_genotypes": [...]}.
    """
    np.random.seed(42)

    low_rep_genotypes = [f"G{g:02d}" for g in range(1, 8)]
    high_rep_genotypes = [f"G{g:02d}" for g in range(8, 16)]
    low_block1_heavy = {"G01", "G02", "G03", "G04", "G05"}
    high_block1_heavy = {"G08", "G09", "G10", "G11", "G12", "G13"}

    data = []
    barcode = 0
    for geno in low_rep_genotypes:
        n_reps = 2
        block1_frac = 0.8 if geno in low_block1_heavy else 0.2
        genotype_effect = np.random.normal(0, 2.0)
        n_b1 = round(block1_frac * n_reps)
        n_b2 = n_reps - n_b1
        blocks = ["block_1"] * n_b1 + ["block_2"] * n_b2
        for r, block in enumerate(blocks):
            shift = 5.0 if block == "block_2" else 0.0
            value = 50 + genotype_effect + shift + np.random.normal(0, 1.0)
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_field_block": value,
                    "block": block,
                }
            )
            barcode += 1
    for geno in high_rep_genotypes:
        n_reps = 10
        block1_frac = 0.8 if geno in high_block1_heavy else 0.2
        genotype_effect = np.random.normal(0, 2.0)
        n_b1 = round(block1_frac * n_reps)
        n_b2 = n_reps - n_b1
        blocks = ["block_1"] * n_b1 + ["block_2"] * n_b2
        for r, block in enumerate(blocks):
            shift = 5.0 if block == "block_2" else 0.0
            value = 50 + genotype_effect + shift + np.random.normal(0, 1.0)
            data.append(
                {
                    "geno": geno,
                    "rep": r + 1,
                    "Barcode": f"BC{barcode:04d}",
                    "trait_field_block": value,
                    "block": block,
                }
            )
            barcode += 1

    df = pd.DataFrame(data)
    meta = {
        "trait": "trait_field_block",
        "low_rep_genotypes": low_rep_genotypes,
        "high_rep_genotypes": high_rep_genotypes,
    }
    return df, meta


# ============================================================================
# HERITABILITY DIAGNOSTIC FIXTURES - Data for testing diagnostic functions
# ============================================================================


@pytest.fixture
def heritability_diagnostic_zero_variance():
    """Generate data with zero between-genotype variance for diagnostics.

    Returns:
        pd.DataFrame: All genotypes have identical trait values
    """
    n_genotypes = 8
    n_reps = 4

    data = []
    # All genotypes have same mean, only replicate variation
    constant_mean = 100.0
    for g in range(n_genotypes):
        for r in range(n_reps):
            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_zero_var": constant_mean + np.random.normal(0, 2),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_diagnostic_high_within_variance():
    """Generate data with high within-genotype (replicate) variance.

    Returns:
        pd.DataFrame: Within-genotype variance >> between-genotype variance
    """
    np.random.seed(42)
    n_genotypes = 10
    n_reps = 5

    data = []
    # Small genetic effects, large environmental noise
    for g in range(n_genotypes):
        # Small genetic effect (genotypes differ slightly)
        genetic_effect = np.random.normal(0, 0.5)

        for r in range(n_reps):
            # Large environmental noise (replicates vary widely)
            environmental_noise = np.random.normal(0, 10)

            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_high_within": 50 + genetic_effect + environmental_noise,
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_diagnostic_low_sample_size():
    """Generate data with minimal sample size for diagnostics.

    Returns:
        pd.DataFrame: Only 3 genotypes with 2 replicates each
    """
    n_genotypes = 3
    n_reps = 2

    data = []
    for g in range(n_genotypes):
        genetic_value = 10 + g * 5

        for r in range(n_reps):
            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_low_sample": genetic_value + np.random.normal(0, 1),
                }
            )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_diagnostic_mixed_quality():
    """Generate dataset with mix of good and bad quality traits.

    Returns:
        pd.DataFrame: Contains traits with varying heritability quality
    """
    np.random.seed(42)
    n_genotypes = 15
    n_reps = 4

    data = []
    for g in range(n_genotypes):
        # Different genetic effects for different trait types
        genetic_high = np.random.normal(0, 3)  # High H²
        genetic_low = np.random.normal(0, 0.2)  # Low H²
        genetic_zero = 0  # Zero variance between genotypes

        for r in range(n_reps):
            env_noise = np.random.normal(0, 1)

            data.append(
                {
                    "geno": f"G{g + 1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g * n_reps + r:04d}",
                    "trait_good": 100 + genetic_high + env_noise,
                    "trait_poor": 50 + genetic_low + env_noise * 5,
                    "trait_constant": 25 + genetic_zero + env_noise,
                }
            )

    return pd.DataFrame(data)


# ============================================================================
# ANOVA TESTING FIXTURES - Data with known group differences
# ============================================================================


@pytest.fixture
def anova_data_known_effects():
    """Generate data with known group effects for ANOVA testing.

    Returns:
        tuple: (DataFrame, expected ANOVA results)

    Expected results:
    - F-statistic should detect significant differences
    - p-value should be < 0.001
    """
    np.random.seed(42)

    # Three groups with different means
    group_means = {"A": 10, "B": 20, "C": 35}
    within_group_std = 2.0
    n_per_group = 30

    data = []
    for group, mean in group_means.items():
        for i in range(n_per_group):
            data.append(
                {
                    "geno": group,
                    "trait_anova": np.random.normal(mean, within_group_std),
                    "rep": i % 5 + 1,
                    "Barcode": f"BC_{group}_{i:03d}",
                }
            )

    df = pd.DataFrame(data)

    # Calculate expected F-statistic
    grand_mean = np.mean(list(group_means.values()))
    ssb = n_per_group * sum((mean - grand_mean) ** 2 for mean in group_means.values())
    msb = ssb / (len(group_means) - 1)
    msw = within_group_std**2
    expected_f = msb / msw

    expected_results = {
        "f_statistic": expected_f,
        "significant": True,
        "n_groups": 3,
    }

    return df, expected_results


@pytest.fixture
def anova_data_no_effect():
    """Generate data with no group differences (null hypothesis true).

    Returns:
        pd.DataFrame: Data where all groups have same distribution
    """
    np.random.seed(42)

    n_per_group = 30
    groups = ["A", "B", "C"]

    data = []
    for group in groups:
        for i in range(n_per_group):
            # All groups have same mean (50) and std (5)
            data.append(
                {
                    "geno": group,
                    "trait_null": np.random.normal(50, 5),
                    "rep": i % 5 + 1,
                    "Barcode": f"BC_{group}_{i:03d}",
                }
            )

    return pd.DataFrame(data)


# ============================================================================
# EDGE CASE FIXTURES - Boundary conditions and special cases
# ============================================================================


@pytest.fixture
def edge_case_nan_patterns():
    """Generate edge case data for NaN handling.

    Returns:
        dict: Multiple DataFrames with different NaN patterns
    """
    datasets = {}

    # All NaN trait
    n = 50
    datasets["all_nan"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "trait_all_nan": np.full(n, np.nan),
            "trait_normal": np.random.randn(n),
        }
    )

    # High NaN (>50%)
    high_nan_trait = np.random.randn(n)
    high_nan_trait[:30] = np.nan  # 60% NaN
    datasets["high_nan"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "trait_high_nan": high_nan_trait,
            "trait_normal": np.random.randn(n),
        }
    )

    # NaN in specific genotypes
    geno_nan_trait = np.random.randn(n)
    for i in range(n):
        if i % 5 == 0:  # All samples from G1 have NaN
            geno_nan_trait[i] = np.nan
    datasets["genotype_nan"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "trait_geno_nan": geno_nan_trait,
            "trait_normal": np.random.randn(n),
        }
    )

    return datasets


@pytest.fixture
def edge_case_zero_patterns():
    """Generate edge case data for zero handling.

    Returns:
        dict: Multiple DataFrames with different zero patterns
    """
    datasets = {}
    n = 100

    # All zeros
    datasets["all_zeros"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 4 + 1 for i in range(n)],
            "trait_all_zero": np.zeros(n),
            "trait_normal": np.random.randn(n),
        }
    )

    # High zeros (>50%)
    high_zero_trait = np.random.randn(n)
    high_zero_trait[:60] = 0  # 60% zeros
    datasets["high_zeros"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 4 + 1 for i in range(n)],
            "trait_high_zero": high_zero_trait,
            "trait_normal": np.random.randn(n),
        }
    )

    # Borderline zeros (exactly 50%)
    borderline_zero_trait = np.random.randn(n)
    borderline_zero_trait[:50] = 0  # Exactly 50% zeros
    datasets["borderline_zeros"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 4 + 1 for i in range(n)],
            "trait_borderline_zero": borderline_zero_trait,
            "trait_normal": np.random.randn(n),
        }
    )

    return datasets


@pytest.fixture
def edge_case_extreme_values():
    """Generate data with extreme value patterns for robustness testing.

    Returns:
        pd.DataFrame: Data with various extreme patterns
    """
    n = 100

    data = {
        "Barcode": [f"BC{i:04d}" for i in range(n)],
        "geno": [f"G{i % 5 + 1}" for i in range(n)],
        "rep": [i % 4 + 1 for i in range(n)],
        "trait_normal": np.random.normal(100, 15, n),
        "trait_inf": np.random.normal(50, 10, n),
        "trait_large_range": np.random.normal(1e6, 1e5, n),
        "trait_tiny_values": np.random.normal(1e-10, 1e-11, n),
        "trait_constant": np.full(n, 42.0),
        "trait_binary": np.random.choice([0, 1], n),
    }

    # Add infinity values
    data["trait_inf"][10] = np.inf
    data["trait_inf"][20] = -np.inf

    return pd.DataFrame(data)


@pytest.fixture
def edge_case_insufficient_data():
    """Generate datasets with insufficient data for analysis.

    Returns:
        dict: Multiple DataFrames with insufficient data patterns
    """
    datasets = {}

    # Single sample
    datasets["single_sample"] = pd.DataFrame(
        {
            "Barcode": ["BC001"],
            "geno": ["G1"],
            "rep": [1],
            "trait1": [1.0],
            "trait2": [2.0],
        }
    )

    # Single genotype
    datasets["single_genotype"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(10)],
            "geno": ["G1"] * 10,
            "rep": list(range(1, 11)),
            "trait1": np.random.randn(10),
        }
    )

    # No replicates
    datasets["no_replicates"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(5)],
            "geno": [f"G{i + 1}" for i in range(5)],
            "rep": [1] * 5,
            "trait1": np.random.randn(5),
        }
    )

    # Empty DataFrame
    datasets["empty"] = pd.DataFrame()

    return datasets


# ============================================================================
# OUTLIER DETECTION FIXTURES - Data with known outliers
# ============================================================================


@pytest.fixture
def outlier_data_with_known_indices():
    """Generate data with known outlier positions.

    Returns:
        tuple: (DataFrame, list of outlier indices)
    """
    np.random.seed(42)
    n = 100
    n_features = 5

    # Generate normal data
    data = np.random.normal(0, 1, (n, n_features))
    df = pd.DataFrame(data, columns=[f"feature_{i + 1}" for i in range(n_features)])

    # Add metadata
    df["Barcode"] = [f"BC{i:04d}" for i in range(n)]
    df["geno"] = [f"G{i % 5 + 1}" for i in range(n)]

    # Insert known outliers
    outlier_indices = [10, 25, 50, 75, 90]
    for idx in outlier_indices:
        # Make outliers extreme in multiple dimensions
        df.iloc[idx, :n_features] = np.random.normal(
            0, 1, n_features
        ) * 5 + np.random.choice([-10, 10])

    return df, outlier_indices


@pytest.fixture
def outlier_data_bimodal():
    """Generate bimodal data (not outliers, just different groups).

    Returns:
        pd.DataFrame: Bimodal distribution that should not be flagged as outliers
    """
    np.random.seed(42)
    n = 100

    # Two groups with different centers
    group1 = np.random.normal(-3, 0.5, n // 2)
    group2 = np.random.normal(3, 0.5, n // 2)

    df = pd.DataFrame(
        {
            "trait_bimodal": np.concatenate([group1, group2]),
            "trait_normal": np.random.normal(0, 1, n),
            "Barcode": [f"BC{i:04d}" for i in range(n)],
            "geno": [
                f"G{i % 2 + 1}" for i in range(n)
            ],  # Two genotypes corresponding to modes
        }
    )

    return df


# ============================================================================
# STATISTICAL DISTRIBUTION FIXTURES - Data with specific distributions
# ============================================================================


@pytest.fixture
def distribution_normal():
    """Generate perfectly normal distributed data.

    Returns:
        tuple: (DataFrame, distribution parameters)
    """
    np.random.seed(42)
    n = 500
    mean = 100
    std = 15

    df = pd.DataFrame(
        {
            "value": np.random.normal(mean, std, n),
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 10 + 1 for i in range(n)],
        }
    )

    params = {"mean": mean, "std": std, "distribution": "normal"}

    return df, params


@pytest.fixture
def distribution_lognormal():
    """Generate log-normal distributed data.

    Returns:
        tuple: (DataFrame, distribution parameters)
    """
    np.random.seed(42)
    n = 500
    mu = 3
    sigma = 0.5

    df = pd.DataFrame(
        {
            "value": np.random.lognormal(mu, sigma, n),
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 10 + 1 for i in range(n)],
        }
    )

    params = {"mu": mu, "sigma": sigma, "distribution": "lognormal"}

    return df, params


@pytest.fixture
def distribution_exponential():
    """Generate exponentially distributed data.

    Returns:
        tuple: (DataFrame, distribution parameters)
    """
    np.random.seed(42)
    n = 500
    scale = 10

    df = pd.DataFrame(
        {
            "value": np.random.exponential(scale, n),
            "geno": [f"G{i % 5 + 1}" for i in range(n)],
            "rep": [i % 10 + 1 for i in range(n)],
        }
    )

    params = {"scale": scale, "distribution": "exponential"}

    return df, params


# ============================================================================
# UTILITY FIXTURES
# ============================================================================


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def zero_inflated_data():
    """Create data with various levels of zero inflation."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "geno": ["G1"] * 10,
            "rep": list(range(1, 11)),
            "trait_all_zeros": [0] * 10,
            "trait_half_zeros": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            "trait_no_zeros": np.random.randint(1, 10, 10),
            "trait_normal": np.random.randn(10)
            + 5,  # Normal distribution, unlikely to have zeros
        }
    )


@pytest.fixture
def nan_data():
    """Create data with various levels of NaN values."""
    return pd.DataFrame(
        {
            "geno": ["G1"] * 10,
            "rep": list(range(1, 11)),
            "trait_all_nan": [np.nan] * 10,
            "trait_half_nan": [np.nan] * 5 + [1, 2, 3, 4, 5],
            "trait_some_nan": [np.nan, np.nan] + list(range(8)),
            "trait_no_nan": list(range(10)),
        }
    )


@pytest.fixture
def sparse_data():
    """Create data with various sample counts."""
    return pd.DataFrame(
        {
            "geno": ["G1"] * 10,
            "rep": list(range(1, 11)),
            "trait_sparse": [np.nan] * 7 + [1, 2, 3],  # Only 3 valid samples
            "trait_dense": list(range(10)),  # All 10 samples valid
            "trait_half": [np.nan] * 5 + [1, 2, 3, 4, 5],  # 5 valid samples
        }
    )


@pytest.fixture
def mixed_problem_data():
    """Create data with multiple quality issues."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(20)],
            "geno": ["G1"] * 10 + ["G2"] * 10,
            "rep": list(range(1, 11)) * 2,
            "trait_zero_inflated": [0] * 15 + list(range(5)),  # 75% zeros
            "trait_many_nans": [np.nan] * 8 + list(range(12)),  # 40% NaNs
            "trait_sparse": [np.nan] * 17 + [1, 2, 3],  # Only 3 valid samples
            "trait_good": np.random.randn(20) + 10,  # Good trait
            "trait_ok": [np.nan] * 3 + list(range(17)),  # 15% NaNs, should pass
        }
    )


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


# ============================================================================
# PCA TESTING FIXTURES - Data for PCA analysis
# ============================================================================


@pytest.fixture
def pca_simple_data():
    """Create simple 2D data with known PCA results.

    Returns:
        tuple: (data array, expected results dict)
    """
    np.random.seed(42)
    # Create data with clear principal components
    # Main variance along diagonal, less along anti-diagonal
    n_samples = 100

    # Generate data with known variance structure
    # PC1 should capture ~75% variance, PC2 ~25%
    t = np.random.randn(n_samples)
    x = 2 * t + 0.5 * np.random.randn(n_samples)
    y = 2 * t + 0.5 * np.random.randn(n_samples)

    data = np.column_stack([x, y])

    # Expected approximate results
    expected = {
        "n_components": 2,
        "variance_ratio_pc1_min": 0.7,  # PC1 should explain at least 70%
        "variance_ratio_pc1_max": 0.99,  # PC1 should explain at most 99%
        "total_variance": 2.0,  # For standardized data
    }

    return data, expected


@pytest.fixture
def pca_3d_data():
    """Create 3D data with known structure for PCA.

    Returns:
        tuple: (DataFrame, expected results dict)
    """
    np.random.seed(42)
    n_samples = 150

    # Create data with decreasing variance along each axis
    # PC1 ~60%, PC2 ~30%, PC3 ~10%
    pc1 = np.random.randn(n_samples) * 3
    pc2 = np.random.randn(n_samples) * 2
    pc3 = np.random.randn(n_samples) * 1

    # Rotate to make it less trivial
    data = np.column_stack([pc1 + 0.3 * pc2, pc2 + 0.2 * pc3, pc3 + 0.1 * pc1])

    df = pd.DataFrame(data, columns=["feature1", "feature2", "feature3"])

    expected = {
        "n_features": 3,
        "n_components_95": 2,  # 2 components should capture >95% variance
        "min_eigenvalue": 0.5,  # Smallest eigenvalue should be > 0.5
    }

    return df, expected


@pytest.fixture
def pca_high_dim_data():
    """Create high-dimensional data for PCA testing.

    Returns:
        tuple: (DataFrame, expected results dict)
    """
    np.random.seed(42)
    n_samples = 50
    n_features = 20

    # Create data where only first 5 features have signal
    signal_features = 5
    data = np.zeros((n_samples, n_features))

    # Add decreasing variance to first 5 features
    for i in range(signal_features):
        data[:, i] = np.random.randn(n_samples) * (5 - i)

    # Add small noise to remaining features
    data[:, signal_features:] = (
        np.random.randn(n_samples, n_features - signal_features) * 0.1
    )

    df = pd.DataFrame(data, columns=[f"feat_{i}" for i in range(n_features)])

    expected = {
        "n_features": n_features,
        "n_effective_components": signal_features,  # Should need ~5 components
        "variance_threshold_90": signal_features,  # 5 components for 90% variance
    }

    return df, expected


@pytest.fixture
def pca_perfect_correlation_data():
    """Create data with perfectly correlated features.

    Returns:
        pd.DataFrame: Data where some features are perfectly correlated
    """
    np.random.seed(42)
    n_samples = 100

    base = np.random.randn(n_samples)

    df = pd.DataFrame(
        {
            "feat1": base,
            "feat2": base * 2,  # Perfect correlation with feat1
            "feat3": base * -1,  # Perfect negative correlation
            "feat4": np.random.randn(n_samples),  # Independent
            "feat5": base + np.random.randn(n_samples) * 0.1,  # High correlation
        }
    )

    return df


@pytest.fixture
def pca_single_feature_data():
    """Create single feature data for edge case testing.

    Returns:
        pd.DataFrame: DataFrame with single feature
    """
    np.random.seed(42)
    return pd.DataFrame({"single_feature": np.random.randn(50)})


@pytest.fixture
def pca_constant_feature_data():
    """Create data with constant (zero variance) features.

    Returns:
        pd.DataFrame: Data with some constant features
    """
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "constant1": np.ones(n_samples),  # All ones
            "constant2": np.zeros(n_samples),  # All zeros
            "variable1": np.random.randn(n_samples),
            "variable2": np.random.randn(n_samples) * 2,
            "constant3": np.full(n_samples, 42),  # All same value
        }
    )

    return df


@pytest.fixture
def pca_standardized_data():
    """Create already standardized data (mean=0, std=1).

    Returns:
        tuple: (DataFrame, scaler used)
    """
    np.random.seed(42)
    from sklearn.preprocessing import StandardScaler

    n_samples = 100
    n_features = 5

    # Create raw data
    raw_data = np.random.randn(n_samples, n_features) * np.array(
        [1, 2, 3, 4, 5]
    ) + np.array([10, 20, 30, 40, 50])

    # Standardize
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw_data)

    df = pd.DataFrame(
        standardized, columns=[f"std_feat_{i}" for i in range(n_features)]
    )

    return df, scaler


# ============================================================================
# CROSS-EXPERIMENT ANALYSIS FIXTURES
# ============================================================================


@pytest.fixture
def cross_experiment_data_fixture():
    """Create sample data for cross-experiment analysis testing.

    Returns:
        tuple: (exp1_df, exp2_df) with common and unique genotypes
    """
    np.random.seed(42)

    # Common genotypes
    common_genotypes = ["Col-0", "Ler", "C24", "Ws", "Bay-0"]
    # Unique to exp1
    exp1_unique = ["Cvi", "Nd"]
    # Unique to exp2
    exp2_unique = ["Po", "Rld"]

    # Create experiment 1 data (e.g., cylinder experiment)
    exp1_genotypes = common_genotypes + exp1_unique
    exp1_data = []
    for geno in exp1_genotypes:
        for rep in range(1, 4):  # 3 replicates
            exp1_data.append(
                {
                    "Geno": geno,
                    "Rep": rep,
                    "primary_length_mm": np.random.normal(100, 20),
                    "lateral_length_mm": np.random.normal(50, 10),
                    "total_length_mm": np.random.normal(150, 25),
                    "root_depth_mm": np.random.normal(80, 15),
                }
            )
    exp1_df = pd.DataFrame(exp1_data)

    # Create experiment 2 data (e.g., turface experiment)
    exp2_genotypes = common_genotypes + exp2_unique
    exp2_data = []
    for geno in exp2_genotypes:
        for rep in range(1, 4):  # 3 replicates
            exp2_data.append(
                {
                    "geno": geno,
                    "rep": rep,
                    "network_length_mean": np.random.normal(120, 30),
                    "stem_length_mm": np.random.normal(60, 12),
                    "chull_area_mean": np.random.normal(200, 40),
                    "crown_lengths_mean_mean": np.random.normal(70, 15),
                }
            )
    exp2_df = pd.DataFrame(exp2_data)

    return exp1_df, exp2_df


@pytest.fixture
def cross_experiment_means_fixture():
    """Create genotype means DataFrames for cross-experiment testing.

    Returns:
        tuple: (exp1_means, exp2_means) DataFrames with genotype means
    """
    np.random.seed(42)

    genotypes = ["G1", "G2", "G3", "G4", "G5"]

    # Experiment 1 means
    exp1_means = pd.DataFrame(
        {
            "trait1": np.random.normal(100, 20, size=len(genotypes)),
            "trait2": np.random.normal(50, 10, size=len(genotypes)),
            "n_samples": [3, 3, 2, 3, 3],
        },
        index=genotypes,
    )

    # Experiment 2 means - correlated with exp1
    exp2_means = pd.DataFrame(
        {
            "trait3": exp1_means["trait1"] * 1.2
            + np.random.normal(0, 10, size=len(genotypes)),
            "trait4": exp1_means["trait2"] * 0.8
            + np.random.normal(0, 5, size=len(genotypes)),
            "n_samples": [3, 2, 3, 3, 2],
        },
        index=genotypes,
    )

    return exp1_means, exp2_means


@pytest.fixture
def pca_real_traits_data(traits_summary_df):
    """Use real trait data for PCA testing.

    Returns:
        tuple: (DataFrame of numeric traits, list of trait names)
    """
    # Select numeric columns, excluding metadata
    exclude_cols = ["Barcode", "geno", "rep", "species", "plant", "scan"]
    numeric_cols = traits_summary_df.select_dtypes(include=[np.number]).columns
    trait_cols = [col for col in numeric_cols if col not in exclude_cols]

    # Take subset with no NaNs for testing
    df_subset = traits_summary_df[trait_cols].dropna()

    # Ensure we have some data
    if df_subset.empty:
        # Create synthetic data if real data is all NaN
        np.random.seed(42)
        n_samples = 100
        n_features = min(10, len(trait_cols)) if trait_cols else 10
        synthetic_data = np.random.randn(n_samples, n_features)
        df_subset = pd.DataFrame(
            synthetic_data,
            columns=(
                trait_cols[:n_features]
                if trait_cols
                else [f"trait_{i}" for i in range(n_features)]
            ),
        )
        trait_cols = df_subset.columns.tolist()

    return df_subset, trait_cols


@pytest.fixture
def pca_nan_data():
    """Create data with NaN values for testing.

    Returns:
        pd.DataFrame: Data with some NaN values
    """
    np.random.seed(42)
    n_samples = 100

    data = np.random.randn(n_samples, 4)
    df = pd.DataFrame(data, columns=["feat1", "feat2", "feat3", "feat4"])

    # Add some NaNs
    df.iloc[10:15, 0] = np.nan  # NaN in first feature
    df.iloc[20:22, 1] = np.nan  # NaN in second feature
    df.iloc[30, :] = np.nan  # Entire row is NaN

    return df


@pytest.fixture
def pca_outlier_data():
    """Create data with outliers for robust PCA testing.

    Returns:
        tuple: (DataFrame, list of outlier indices)
    """
    np.random.seed(42)
    n_samples = 100

    # Normal data
    data = np.random.randn(n_samples, 3)

    # Add outliers
    outlier_indices = [10, 25, 50, 75]
    for idx in outlier_indices:
        data[idx, :] = np.random.randn(3) * 10  # Make outliers 10x larger

    df = pd.DataFrame(data, columns=["x", "y", "z"])

    return df, outlier_indices


@pytest.fixture
def pca_variance_threshold_data():
    """Create data for testing variance threshold selection.

    Returns:
        dict: Multiple datasets with different variance structures
    """
    np.random.seed(42)
    n_samples = 100

    datasets = {}

    # Dataset where 1 component captures >95% variance
    t = np.random.randn(n_samples)
    datasets["one_component"] = pd.DataFrame(
        {
            "x": t + np.random.randn(n_samples) * 0.1,
            "y": t + np.random.randn(n_samples) * 0.1,
            "z": t + np.random.randn(n_samples) * 0.1,
        }
    )

    # Dataset where 2 components needed for 95% variance
    datasets["two_components"] = pd.DataFrame(
        {
            "a": np.random.randn(n_samples) * 3,
            "b": np.random.randn(n_samples) * 2,
            "c": np.random.randn(n_samples) * 0.5,
            "d": np.random.randn(n_samples) * 0.3,
        }
    )

    # Dataset where all components needed
    datasets["all_components"] = pd.DataFrame(
        {
            "p": np.random.randn(n_samples),
            "q": np.random.randn(n_samples),
            "r": np.random.randn(n_samples),
        }
    )

    return datasets


@pytest.fixture
def pca_all_nan_data():
    """Create DataFrame with all NaN values for edge case testing.

    Returns:
        pd.DataFrame: Data with all NaN values
    """
    return pd.DataFrame(
        {"feat1": [np.nan] * 10, "feat2": [np.nan] * 10, "feat3": [np.nan] * 10}
    )


@pytest.fixture
def pca_single_sample_data():
    """Create DataFrame with only one sample for edge case testing.

    Returns:
        pd.DataFrame: Data with single row
    """
    return pd.DataFrame(
        {"feat1": [1.5], "feat2": [2.3], "feat3": [-0.8], "feat4": [4.2]}
    )


@pytest.fixture
def pca_zero_variance_all_columns():
    """Create DataFrame where all columns have zero variance.

    Returns:
        pd.DataFrame: Data where all values in each column are identical
    """
    n_samples = 50
    return pd.DataFrame(
        {
            "const1": [5.0] * n_samples,
            "const2": [10.0] * n_samples,
            "const3": [-2.5] * n_samples,
        }
    )


@pytest.fixture
def pca_1d_result_data():
    """Create data that will result in 1D PCA output.

    Returns:
        pd.DataFrame: High dimensional data with only 1 PC of significance
    """
    np.random.seed(42)
    n_samples = 100

    # Create data where all features are multiples of one underlying factor
    base = np.random.randn(n_samples)

    df = pd.DataFrame(
        {
            "feat1": base * 2,
            "feat2": base * 3,
            "feat3": base * -1,
            "feat4": base * 0.5,
            "feat5": base * 4,
        }
    )

    # Add tiny noise to avoid perfect correlation
    df += np.random.randn(*df.shape) * 1e-10

    return df


@pytest.fixture
def pca_zero_std_features():
    """Create data with some features having zero standard deviation.

    Returns:
        pd.DataFrame: Mixed data with some zero-std features
    """
    np.random.seed(42)
    n_samples = 75

    return pd.DataFrame(
        {
            "normal1": np.random.randn(n_samples),
            "zero_std1": [3.14159] * n_samples,  # Constant
            "normal2": np.random.randn(n_samples) * 2,
            "zero_std2": np.zeros(n_samples),  # All zeros
            "normal3": np.random.randn(n_samples) * 0.5,
        }
    )


@pytest.fixture
def pca_singular_covariance_data():
    """Create data with singular covariance matrix for edge case testing.

    Returns:
        pd.DataFrame: Data with linearly dependent features
    """
    np.random.seed(42)
    n_samples = 100

    # Create linearly dependent features
    x1 = np.random.randn(n_samples)
    x2 = np.random.randn(n_samples)

    df = pd.DataFrame(
        {
            "feat1": x1,
            "feat2": x2,
            "feat3": x1 + x2,  # Linear combination
            "feat4": 2 * x1 - x2,  # Another linear combination
            "feat5": x1 - x2 + x1,  # Yet another combination (2*x1 - x2)
        }
    )

    return df


@pytest.fixture
def pca_mixed_numeric_nonnumeric():
    """Create DataFrame with both numeric and non-numeric columns.

    Returns:
        pd.DataFrame: Mixed data types
    """
    np.random.seed(42)
    n_samples = 50

    return pd.DataFrame(
        {
            "barcode": [f"BC{i:04d}" for i in range(n_samples)],
            "value1": np.random.randn(n_samples),
            "category": np.random.choice(["A", "B", "C"], n_samples),
            "value2": np.random.randn(n_samples) * 2,
            "value3": np.random.randn(n_samples) * 0.5,
            "description": [f"Sample {i}" for i in range(n_samples)],
            "value4": np.random.randn(n_samples) * 3,
        }
    )


@pytest.fixture
def pca_empty_after_nan_removal():
    """Create DataFrame that becomes empty after NaN removal.

    Returns:
        pd.DataFrame: Data where every row has at least one NaN
    """
    np.random.seed(42)
    n_samples = 20

    df = pd.DataFrame(
        {
            "feat1": np.random.randn(n_samples),
            "feat2": np.random.randn(n_samples),
            "feat3": np.random.randn(n_samples),
        }
    )

    # Ensure every row has at least one NaN
    for i in range(n_samples):
        col_idx = i % 3
        df.iloc[i, col_idx] = np.nan

    return df


# ============================================================================
# PCA MATHEMATICAL VALIDATION FIXTURES
# ============================================================================


# ============================================================================
# VISUALIZATION TESTING FIXTURES - Data for visualization module
# ============================================================================


@pytest.fixture
def viz_sample_data():
    """Create sample data for basic visualization testing.

    Returns:
        pd.DataFrame: Sample data with traits and genotype
    """
    np.random.seed(42)
    n_samples = 100

    df = pd.DataFrame(
        {
            "trait1": np.random.normal(0, 1, n_samples),
            "trait2": np.random.normal(5, 2, n_samples),
            "trait3": np.random.uniform(0, 10, n_samples),
            "geno": np.random.choice(["A", "B", "C"], n_samples),
            "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
        }
    )

    return df


@pytest.fixture
def viz_data_with_nan():
    """Create data with NaN values for visualization testing.

    Returns:
        pd.DataFrame: Data containing various NaN patterns
    """
    np.random.seed(42)
    n_samples = 50

    df = pd.DataFrame(
        {
            "trait_complete": np.random.randn(n_samples),
            "trait_some_nan": np.concatenate([np.random.randn(40), [np.nan] * 10]),
            "trait_all_nan": [np.nan] * n_samples,
            "geno": np.random.choice(["Type1", "Type2"], n_samples),
            "Barcode": [f"BC{i:03d}" for i in range(n_samples)],
        }
    )

    return df


@pytest.fixture
def viz_empty_data():
    """Create empty DataFrame for edge case testing.

    Returns:
        pd.DataFrame: Empty DataFrame with expected columns
    """
    return pd.DataFrame(columns=["trait1", "trait2", "geno", "Barcode"])


@pytest.fixture
def viz_single_trait_data():
    """Create data with single trait for testing.

    Returns:
        pd.DataFrame: Data with one trait column
    """
    np.random.seed(42)
    return pd.DataFrame(
        {
            "single_trait": np.random.randn(75),
            "geno": np.random.choice(["G1", "G2", "G3"], 75),
            "Barcode": [f"BC{i:03d}" for i in range(75)],
        }
    )


@pytest.fixture
def viz_many_traits_data():
    """Create data with many traits for testing subplot layouts.

    Returns:
        pd.DataFrame: Data with 30+ trait columns
    """
    np.random.seed(42)
    n_samples = 50
    n_traits = 30

    data = {
        "Barcode": [f"BC{i:03d}" for i in range(n_samples)],
        "geno": np.random.choice(["A", "B", "C", "D"], n_samples),
    }

    for i in range(n_traits):
        data[f"trait_{i:02d}"] = np.random.randn(n_samples) * (i + 1)

    return pd.DataFrame(data)


@pytest.fixture
def viz_perfect_correlation_data():
    """Create data with perfectly correlated traits.

    Returns:
        pd.DataFrame: Data where some traits are perfectly correlated
    """
    np.random.seed(42)
    n_samples = 100
    base = np.random.randn(n_samples)

    df = pd.DataFrame(
        {
            "trait_a": base,
            "trait_b": base * 2,  # Perfect positive correlation
            "trait_c": -base,  # Perfect negative correlation
            "trait_d": np.random.randn(n_samples),  # Independent
            "geno": np.random.choice(["X", "Y"], n_samples),
        }
    )

    return df


@pytest.fixture
def viz_bimodal_data():
    """Create bimodal distributed data for visualization.

    Returns:
        pd.DataFrame: Data with bimodal distributions
    """
    np.random.seed(42)
    n_samples = 120

    # Create bimodal distribution
    group1 = np.random.normal(-2, 0.5, n_samples // 2)
    group2 = np.random.normal(2, 0.5, n_samples // 2)

    df = pd.DataFrame(
        {
            "trait_bimodal": np.concatenate([group1, group2]),
            "trait_normal": np.random.normal(0, 1, n_samples),
            "geno": ["GroupA"] * (n_samples // 2) + ["GroupB"] * (n_samples // 2),
        }
    )

    return df


@pytest.fixture
def viz_single_genotype_data():
    """Create data with only one genotype group.

    Returns:
        pd.DataFrame: Data with single genotype value
    """
    np.random.seed(42)
    n_samples = 60

    df = pd.DataFrame(
        {
            "trait1": np.random.randn(n_samples),
            "trait2": np.random.exponential(2, n_samples),
            "geno": ["SingleType"] * n_samples,
        }
    )

    return df


@pytest.fixture
def viz_constant_trait_data():
    """Create data with constant (zero variance) traits.

    Returns:
        pd.DataFrame: Data with some constant traits
    """
    np.random.seed(42)
    n_samples = 80

    df = pd.DataFrame(
        {
            "trait_constant": [42.0] * n_samples,
            "trait_variable": np.random.randn(n_samples),
            "trait_zero": np.zeros(n_samples),
            "geno": np.random.choice(["A", "B"], n_samples),
        }
    )

    return df


@pytest.fixture
def viz_eda_sample_data():
    """Create sample data for EDA plots testing with known metrics.

    Returns:
        pd.DataFrame: Data with various trait quality patterns
    """
    np.random.seed(42)
    n_samples = 100

    # Create traits with different quality issues
    data = {
        "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
        "geno": np.random.choice(["A", "B", "C"], n_samples),
        "rep": np.random.choice([1, 2, 3], n_samples),
        # Good trait - normal distribution
        "trait_good": np.random.normal(10, 2, n_samples),
        # High NaN trait (40% NaN)
        "trait_high_nan": np.where(
            np.random.random(n_samples) < 0.4, np.nan, np.random.normal(5, 1, n_samples)
        ),
        # High zero trait (60% zeros)
        "trait_high_zero": np.where(
            np.random.random(n_samples) < 0.6, 0, np.random.normal(3, 0.5, n_samples)
        ),
        # Low variance trait
        "trait_low_var": np.random.normal(50, 0.01, n_samples),
        # Outlier-prone trait
        "trait_outliers": np.concatenate(
            [
                np.random.normal(0, 1, 90),  # Normal values
                np.random.normal(10, 0.5, 10),  # Outliers
            ]
        ),
    }

    return pd.DataFrame(data)


@pytest.fixture
def viz_eda_thresholds():
    """Standard thresholds for EDA cleanup.

    Returns:
        dict: Thresholds for NaN, zero, and outlier fractions
    """
    return {
        "nan": 0.2,  # 20% maximum NaN (canonical QC default)
        "zero": 0.5,  # 50% maximum zeros
        "outlier": 0.1,  # 10% maximum outliers (though not used for trait removal)
    }


@pytest.fixture
def viz_eda_cleanup_log():
    """Sample cleanup log from apply_data_cleanup_filters.

    Returns:
        dict: Cleanup log with removed traits information
    """
    return {
        "removed_traits": [
            {
                "trait": "trait_high_nan",
                "reason": "too_many_nans",
                "nan_fraction": 0.4,
                "zero_fraction": 0.05,
                "valid_samples": 60,
            },
            {
                "trait": "trait_high_zero",
                "reason": "too_many_zeros",
                "nan_fraction": 0.02,
                "zero_fraction": 0.6,
                "valid_samples": 98,
            },
            {
                "trait": "trait_insufficient",
                "reason": "insufficient_samples",
                "nan_fraction": 0.85,
                "zero_fraction": 0.05,
                "valid_samples": 5,
            },
        ],
        "initial_traits": 10,
        "remaining_traits": 7,
        "traits_removed_high_nan": 1,
        "traits_removed_high_zero": 1,
        "traits_removed_low_samples": 1,
    }


@pytest.fixture
def viz_eda_data_with_extremes():
    """Create data with extreme values for EDA testing.

    Returns:
        pd.DataFrame: Data with various extreme patterns
    """
    np.random.seed(42)
    n_samples = 50

    df = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n_samples)],
            "geno": np.random.choice(["Type1", "Type2"], n_samples),
            # All NaN trait
            "trait_all_nan": [np.nan] * n_samples,
            # All zero trait
            "trait_all_zero": np.zeros(n_samples),
            # Single valid value
            "trait_single_valid": [np.nan] * (n_samples - 1) + [5.0],
            # Boundary case - exactly at threshold (30% NaN)
            "trait_boundary_nan": np.where(
                np.arange(n_samples) < 15, np.nan, np.random.normal(10, 1, n_samples)
            ),
            # Boundary case - exactly at threshold (50% zero)
            "trait_boundary_zero": np.where(
                np.arange(n_samples) < 25, 0, np.random.normal(5, 1, n_samples)
            ),
            # High variance trait
            "trait_high_var": np.random.normal(100, 50, n_samples),
            # Negative values
            "trait_negative": np.random.normal(-5, 2, n_samples),
        }
    )

    return df


@pytest.fixture
def viz_eda_empty_cleanup_log():
    """Empty cleanup log for testing.

    Returns:
        dict: Empty cleanup log
    """
    return {
        "removed_traits": [],
        "initial_traits": 5,
        "remaining_traits": 5,
        "traits_removed_high_nan": 0,
        "traits_removed_high_zero": 0,
        "traits_removed_low_samples": 0,
    }


@pytest.fixture
def viz_eda_many_traits_data():
    """Create data with many traits for comprehensive EDA.

    Returns:
        pd.DataFrame: Data with 50+ traits
    """
    np.random.seed(42)
    n_samples = 100
    n_traits = 50

    data = {
        "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
        "geno": np.random.choice(["G1", "G2", "G3"], n_samples),
        "rep": np.random.choice([1, 2, 3, 4], n_samples),
    }

    # Add traits with various prefixes and patterns
    prefixes = ["root", "lateral", "crown", "network", "depth"]
    for i in range(n_traits):
        prefix = prefixes[i % len(prefixes)]
        trait_name = f"{prefix}_{i:02d}"

        # Vary the quality of traits
        if i % 5 == 0:
            # Some high NaN traits
            data[trait_name] = np.where(
                np.random.random(n_samples) < 0.35, np.nan, np.random.randn(n_samples)
            )
        elif i % 7 == 0:
            # Some high zero traits
            data[trait_name] = np.where(
                np.random.random(n_samples) < 0.55, 0, np.random.randn(n_samples)
            )
        else:
            # Normal traits with varying variance
            data[trait_name] = np.random.randn(n_samples) * (i + 1)

    return pd.DataFrame(data)


@pytest.fixture
def controlled_spectrum_data():
    """Create data with known eigenvalue spectrum using low-rank structure.

    Returns:
        pd.DataFrame: Data with 3 strong components and noise
    """
    rng = np.random.default_rng(0)
    n_samples = 50
    n_features = 7

    # Create data with known covariance structure
    # Low-rank structure: 3 strong components, rest noise
    latent_dim = 3
    W = rng.standard_normal((n_features, latent_dim))
    Z = rng.standard_normal((n_samples, latent_dim))
    noise = rng.standard_normal((n_samples, n_features)) * 0.1
    X = Z @ W.T + noise

    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])


@pytest.fixture
def diagonal_covariance_data():
    """Create data with diagonal covariance (known eigenvalues).

    Returns:
        tuple: (DataFrame, known eigenvalues array)
    """
    rng = np.random.default_rng(42)
    n_samples = 100
    # Eigenvalues: [5, 3, 2, 1, 0.5, 0.2, 0.1]
    eigenvalues = np.array([5, 3, 2, 1, 0.5, 0.2, 0.1])
    n_features = len(eigenvalues)

    # Generate independent features with specified variances
    X = rng.standard_normal((n_samples, n_features))
    X *= np.sqrt(eigenvalues)

    return (
        pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)]),
        eigenvalues,
    )


@pytest.fixture
def correlated_pairs_data():
    """Create data with pairs of correlated features.

    Returns:
        pd.DataFrame: Data with 3 pairs of correlated features
    """
    rng = np.random.default_rng(123)
    n_samples = 80

    # Create 3 pairs of correlated features
    data = []
    for _ in range(3):
        z = rng.standard_normal(n_samples)
        noise1 = rng.standard_normal(n_samples) * 0.3
        noise2 = rng.standard_normal(n_samples) * 0.3
        data.append(z + noise1)
        data.append(z + noise2)

    X = np.column_stack(data)
    return pd.DataFrame(X, columns=[f"feat_{i}" for i in range(6)])


# ============================================================================
# OUTLIER DETECTION FIXTURES - Data with known outlier patterns
# ============================================================================


@pytest.fixture
def outlier_data_with_known_outliers():
    """Generate data with known outliers at specific positions.

    Returns:
        tuple: (DataFrame, list of outlier indices, dict with metadata)
    """
    rng = np.random.default_rng(42)
    n_normal = 100
    n_outliers = 5
    n_features = 6

    # Generate normal data from multivariate normal
    mean = np.zeros(n_features)
    cov = np.eye(n_features)
    # Add some correlation structure
    cov[0, 1] = cov[1, 0] = 0.5
    cov[2, 3] = cov[3, 2] = 0.3

    normal_data = rng.multivariate_normal(mean, cov, n_normal)

    # Generate outliers far from the mean
    outlier_data = []
    outlier_directions = [
        np.array([5, 5, 0, 0, 0, 0]),  # Outlier in first two dims
        np.array([0, 0, -6, 0, 0, 0]),  # Outlier in third dim
        np.array([0, 0, 0, 0, 4, 4]),  # Outlier in last two dims
        np.array([3, 3, 3, 3, 3, 3]),  # Global outlier
        np.array([-4, 4, -4, 4, -4, 4]),  # Alternating outlier
    ]

    for direction in outlier_directions:
        outlier_data.append(direction + rng.normal(0, 0.1, n_features))

    # Combine data
    all_data = np.vstack([normal_data, np.array(outlier_data)])
    outlier_indices = list(range(n_normal, n_normal + n_outliers))

    df = pd.DataFrame(all_data, columns=[f"trait_{i}" for i in range(n_features)])

    metadata = {
        "n_normal": n_normal,
        "n_outliers": n_outliers,
        "n_features": n_features,
        "outlier_types": [
            "correlated_pair",
            "single_dim",
            "correlated_pair",
            "global",
            "alternating",
        ],
        "expected_mahalanobis_min": 3.0,  # Minimum expected distance for outliers
    }

    return df, outlier_indices, metadata


@pytest.fixture
def outlier_data_high_dimensional():
    """Generate high-dimensional data for testing PCA-based outlier detection.

    Returns:
        tuple: (DataFrame, list of outlier indices, dict with metadata)
    """
    rng = np.random.default_rng(123)
    n_samples = 50
    n_features = 20  # High dimensional
    n_outliers = 3

    # Generate data with intrinsic lower dimensionality (rank ~5)
    n_components = 5
    U = rng.standard_normal((n_samples - n_outliers, n_components))
    V = rng.standard_normal((n_components, n_features))
    normal_data = U @ V + rng.normal(0, 0.1, (n_samples - n_outliers, n_features))

    # Add outliers in the high-dimensional space
    outlier_data = rng.normal(0, 5, (n_outliers, n_features))

    all_data = np.vstack([normal_data, outlier_data])
    outlier_indices = list(range(n_samples - n_outliers, n_samples))

    df = pd.DataFrame(all_data, columns=[f"feature_{i}" for i in range(n_features)])

    metadata = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_outliers": n_outliers,
        "intrinsic_dim": n_components,
        "variance_threshold_suggested": 0.9,
    }

    return df, outlier_indices, metadata


@pytest.fixture
def outlier_data_multimodal():
    """Generate multimodal distribution to test robustness.

    Returns:
        tuple: (DataFrame, list of outlier indices, dict with metadata)
    """
    rng = np.random.default_rng(456)

    # Create two clusters
    n_cluster1 = 40
    n_cluster2 = 40
    n_outliers = 5
    n_features = 4

    # Cluster 1 centered at origin
    cluster1 = rng.multivariate_normal(
        [0, 0, 0, 0], np.eye(n_features) * 0.5, n_cluster1
    )

    # Cluster 2 offset
    cluster2 = rng.multivariate_normal(
        [3, 3, 0, 0], np.eye(n_features) * 0.5, n_cluster2
    )

    # Outliers between and beyond clusters
    outliers = np.array(
        [
            [1.5, 1.5, 4, 0],  # Between clusters but off in dim 3
            [6, 6, 0, 0],  # Beyond cluster 2
            [-3, -3, 0, 0],  # Beyond cluster 1
            [0, 0, 0, 5],  # Off in dim 4
            [1.5, 1.5, -3, 3],  # Between clusters, off in dims 3&4
        ]
    )

    all_data = np.vstack([cluster1, cluster2, outliers])
    outlier_indices = list(
        range(n_cluster1 + n_cluster2, n_cluster1 + n_cluster2 + n_outliers)
    )

    df = pd.DataFrame(all_data, columns=[f"metric_{i}" for i in range(n_features)])

    metadata = {
        "n_cluster1": n_cluster1,
        "n_cluster2": n_cluster2,
        "n_outliers": n_outliers,
        "n_features": n_features,
        "cluster_separation": 3.0,
        "robust_covariance_recommended": True,
    }

    return df, outlier_indices, metadata


@pytest.fixture
def outlier_data_with_clusters():
    """Generate data with distinct clusters for testing cluster-aware outlier detection.

    Returns:
        tuple: (DataFrame with cluster labels, list of outlier indices, dict with metadata)
    """
    rng = np.random.default_rng(789)

    # Create three distinct clusters
    n_per_cluster = 30
    n_outliers = 4
    n_features = 3

    clusters = []
    cluster_centers = [
        [0, 0, 0],
        [5, 0, 0],
        [2.5, 4, 0],
    ]

    for i, center in enumerate(cluster_centers):
        cluster_data = rng.multivariate_normal(
            center, np.eye(n_features) * 0.3, n_per_cluster
        )
        clusters.append(cluster_data)

    # Outliers at various positions
    outliers = np.array(
        [
            [2.5, 2, 5],  # Above the triangle of clusters
            [2.5, 2, -5],  # Below the triangle of clusters
            [10, 0, 0],  # Far right
            [-5, 0, 0],  # Far left
        ]
    )

    all_data = np.vstack(clusters + [outliers])

    # Create cluster labels
    cluster_labels = []
    for i in range(3):
        cluster_labels.extend([f"cluster_{i}"] * n_per_cluster)
    cluster_labels.extend(["outlier"] * n_outliers)

    df = pd.DataFrame(all_data, columns=[f"dim_{i}" for i in range(n_features)])
    df["cluster"] = cluster_labels

    outlier_indices = list(range(n_per_cluster * 3, n_per_cluster * 3 + n_outliers))

    metadata = {
        "n_clusters": 3,
        "n_per_cluster": n_per_cluster,
        "n_outliers": n_outliers,
        "n_features": n_features,
        "cluster_column": "cluster",
        "outlier_label": "outlier",
    }

    return df, outlier_indices, metadata


@pytest.fixture
def outlier_data_edge_cases():
    """Generate edge case data for outlier detection.

    Returns:
        dict: Dictionary of edge case datasets
    """
    rng = np.random.default_rng(101)

    edge_cases = {}

    # Single sample (cannot compute covariance)
    edge_cases["single_sample"] = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])

    # Two samples (minimum for covariance)
    edge_cases["two_samples"] = pd.DataFrame(
        [[1, 2, 3], [2, 3, 4]], columns=["a", "b", "c"]
    )

    # All identical samples (zero variance)
    edge_cases["identical_samples"] = pd.DataFrame(
        [[1, 2, 3]] * 10, columns=["a", "b", "c"]
    )

    # One constant feature
    data = rng.standard_normal((20, 3))
    data[:, 1] = 5.0  # Make middle column constant
    edge_cases["constant_feature"] = pd.DataFrame(
        data, columns=["var1", "constant", "var2"]
    )

    # High correlation (near singular covariance)
    n = 30
    x = rng.standard_normal(n)
    edge_cases["high_correlation"] = pd.DataFrame(
        {
            "x": x,
            "y": x + rng.normal(0, 0.01, n),  # Almost perfectly correlated
            "z": x + rng.normal(0, 0.01, n),  # Almost perfectly correlated
        }
    )

    # Data with NaN values
    data_with_nan = rng.standard_normal((20, 4))
    data_with_nan[5:8, 1] = np.nan
    data_with_nan[10, :] = np.nan
    edge_cases["with_nan"] = pd.DataFrame(
        data_with_nan, columns=[f"col_{i}" for i in range(4)]
    )

    # Empty dataframe
    edge_cases["empty"] = pd.DataFrame()

    # Single feature (1D)
    edge_cases["single_feature"] = pd.DataFrame({"value": rng.standard_normal(25)})

    return edge_cases


@pytest.fixture
def pca_reconstruction_data_low_rank():
    """Generate low-rank data with noise for reconstruction error testing.

    Creates data that primarily lies in a 2D subspace of 5D space,
    with small noise added. Outliers are created by adding large noise.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create low-rank structure (rank 2)
    # Generate data in 2D subspace
    latent = np.random.randn(n_samples, 2)

    # Create loading matrix (5x2)
    W = np.random.randn(n_features, 2)

    # Generate low-rank data
    X_low_rank = latent @ W.T

    # Add small noise to most samples
    noise = np.random.randn(n_samples, n_features) * 0.1
    X_noisy = X_low_rank + noise

    # Add outliers with large reconstruction error
    outlier_indices = [95, 96, 97, 98, 99]
    for idx in outlier_indices:
        # Add large noise perpendicular to the manifold
        X_noisy[idx, :] += np.random.randn(n_features) * 3.0

    df = pd.DataFrame(X_noisy, columns=[f"feature_{i}" for i in range(n_features)])

    return (
        df,
        outlier_indices,
        {
            "true_rank": 2,
            "n_features": n_features,
            "n_outliers": len(outlier_indices),
            "noise_level": 0.1,
            "outlier_noise": 3.0,
        },
    )


@pytest.fixture
def pca_reconstruction_perfect_low_rank():
    """Generate perfect low-rank data (no noise) for testing edge cases."""
    np.random.seed(42)
    n_samples = 50
    n_features = 6

    # Create perfect rank-3 data
    latent = np.random.randn(n_samples, 3)
    W = np.random.randn(n_features, 3)
    X = latent @ W.T

    df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(n_features)])

    return df, {
        "true_rank": 3,
        "n_features": n_features,
        "expected_reconstruction_error": 0.0,  # Perfect reconstruction with 3 components
    }


@pytest.fixture
def pca_reconstruction_varying_errors():
    """Generate data with varying reconstruction errors for threshold testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 4

    # Create base low-rank structure
    latent = np.random.randn(n_samples, 2)
    W = np.random.randn(n_features, 2)
    X = latent @ W.T

    # Add varying levels of noise to create gradient of reconstruction errors
    # First 70: small noise (normal samples)
    X[:70, :] += np.random.randn(70, n_features) * 0.2

    # Next 20: medium noise (borderline)
    X[70:90, :] += np.random.randn(20, n_features) * 1.0

    # Last 10: large noise (clear outliers)
    X[90:, :] += np.random.randn(10, n_features) * 3.0

    df = pd.DataFrame(X, columns=[f"var_{i}" for i in range(n_features)])

    return df, {
        "n_normal": 70,
        "n_borderline": 20,
        "n_outliers": 10,
        "outlier_indices": list(range(90, 100)),
    }


@pytest.fixture
def isolation_forest_data_with_anomalies():
    """Generate data with clear anomalies suitable for Isolation Forest detection.

    Isolation Forest works well with:
    - Global outliers (far from all normal data)
    - Local outliers (isolated in specific feature subspaces)
    - Anomalies that are easy to isolate with few splits
    """
    np.random.seed(42)
    n_normal = 100
    n_anomalies = 10
    n_features = 5

    # Generate normal data (compact cluster)
    normal_data = np.random.randn(n_normal, n_features) * 0.5

    # Generate anomalies (scattered, easy to isolate)
    anomalies = []
    for i in range(n_anomalies):
        # Each anomaly is isolated in different ways
        anomaly = np.zeros(n_features)
        if i < 3:
            # Global outliers - far from origin
            anomaly = (
                np.random.randn(n_features) * 3.0
                + np.sign(np.random.randn(n_features)) * 5.0
            )
        elif i < 6:
            # Feature-specific outliers
            anomaly = np.random.randn(n_features) * 0.5
            anomaly[i % n_features] = np.random.choice(
                [-7, 7]
            )  # Extreme in one dimension
        else:
            # Mixed outliers
            anomaly = np.random.randn(n_features) * 2.0
            anomaly[0:2] *= 3.0  # Extreme in subset of dimensions
        anomalies.append(anomaly)

    anomalies = np.array(anomalies)

    # Combine data
    X = np.vstack([normal_data, anomalies])

    # Shuffle while keeping track of anomaly indices
    original_anomaly_indices = list(range(n_normal, n_normal + n_anomalies))
    shuffle_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffle_indices]

    # Track where anomalies ended up after shuffling
    anomaly_indices = []
    for i, idx in enumerate(shuffle_indices):
        if idx in original_anomaly_indices:
            anomaly_indices.append(i)

    df = pd.DataFrame(X_shuffled, columns=[f"feature_{i}" for i in range(n_features)])

    return (
        df,
        anomaly_indices,
        {
            "n_normal": n_normal,
            "n_anomalies": n_anomalies,
            "contamination": n_anomalies / (n_normal + n_anomalies),
        },
    )


@pytest.fixture
def isolation_forest_multimodal_data():
    """Generate multimodal data where Isolation Forest should excel.

    Isolation Forest handles multimodal distributions well because it doesn't
    assume a single center or distribution shape.
    """
    np.random.seed(42)
    n_per_cluster = 40
    n_features = 4
    n_anomalies = 8

    # Create three distinct clusters
    cluster1 = np.random.randn(n_per_cluster, n_features) * 0.3 + np.array(
        [-3, -3, 0, 0]
    )
    cluster2 = np.random.randn(n_per_cluster, n_features) * 0.3 + np.array(
        [3, -3, 0, 0]
    )
    cluster3 = np.random.randn(n_per_cluster, n_features) * 0.3 + np.array([0, 3, 0, 0])

    # Add anomalies between and outside clusters
    anomalies = []
    for i in range(n_anomalies):
        if i < 3:
            # Anomalies between clusters
            anomaly = np.array([0, 0, 0, 0]) + np.random.randn(n_features) * 0.5
        else:
            # Anomalies far from all clusters
            anomaly = np.random.randn(n_features) * 5.0
        anomalies.append(anomaly)

    anomalies = np.array(anomalies)

    # Combine all data
    X = np.vstack([cluster1, cluster2, cluster3, anomalies])
    expected_anomaly_indices = list(range(3 * n_per_cluster, len(X)))

    df = pd.DataFrame(X, columns=[f"dim_{i}" for i in range(n_features)])

    return (
        df,
        expected_anomaly_indices,
        {
            "n_clusters": 3,
            "n_per_cluster": n_per_cluster,
            "n_anomalies": n_anomalies,
            "contamination": n_anomalies / len(X),
        },
    )


@pytest.fixture
def isolation_forest_high_dimensional_sparse():
    """Generate high-dimensional sparse data where Isolation Forest performs well.

    In high dimensions, Isolation Forest can efficiently isolate anomalies
    without suffering from the curse of dimensionality as much as distance-based methods.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 20  # High dimensional
    n_anomalies = 5
    sparsity = 0.7  # 70% of values will be near zero

    # Generate sparse normal data
    normal_data = np.random.randn(n_samples - n_anomalies, n_features) * 0.1
    mask = np.random.random((n_samples - n_anomalies, n_features)) < sparsity
    normal_data[mask] = np.random.randn(np.sum(mask)) * 0.01  # Very small values

    # Generate anomalies (dense or with different sparsity pattern)
    anomalies = []
    for i in range(n_anomalies):
        if i < 2:
            # Dense anomalies (not sparse)
            anomaly = np.random.randn(n_features) * 0.5
        else:
            # Different sparsity pattern
            anomaly = np.zeros(n_features)
            active_features = np.random.choice(n_features, size=5, replace=False)
            anomaly[active_features] = np.random.randn(5) * 2.0
        anomalies.append(anomaly)

    anomalies = np.array(anomalies)

    # Combine and create DataFrame
    X = np.vstack([normal_data, anomalies])
    expected_anomaly_indices = list(range(n_samples - n_anomalies, n_samples))

    df = pd.DataFrame(X, columns=[f"feat_{i:02d}" for i in range(n_features)])

    return (
        df,
        expected_anomaly_indices,
        {
            "n_features": n_features,
            "sparsity": sparsity,
            "n_anomalies": n_anomalies,
            "contamination": n_anomalies / n_samples,
        },
    )


# =============================================================================
# Outlier Visualization Fixtures
# =============================================================================


@pytest.fixture
def outlier_viz_sample_data():
    """Sample data with genotypes for outlier visualization testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    n_genotypes = 4

    # Create data with some structure
    data = np.random.randn(n_samples, n_features)

    # Add some outliers
    outlier_indices = [10, 25, 40, 55, 70, 85]
    for idx in outlier_indices:
        data[idx, :] *= 3  # Make these samples outliers

    df = pd.DataFrame(data, columns=[f"trait_{i}" for i in range(n_features)])

    # Add genotype column
    genotypes = ["Geno_A", "Geno_B", "Geno_C", "Geno_D"]
    df["geno"] = np.random.choice(genotypes, size=n_samples)

    return df, outlier_indices


@pytest.fixture
def outlier_viz_isolation_results():
    """Isolation Forest results for visualization testing."""
    np.random.seed(42)
    n_samples = 100

    # Generate anomaly scores (lower = more anomalous)
    scores = np.random.beta(5, 2, n_samples)  # Most scores high
    outlier_indices = [5, 15, 25, 35, 45]

    # Make outliers have lower scores
    for idx in outlier_indices:
        scores[idx] = np.random.uniform(0, 0.3)

    return {
        "method": "IsolationForest",
        "anomaly_scores": scores.tolist(),
        "outlier_indices": outlier_indices,
        "data_indices": list(range(n_samples)),
        "contamination": 0.05,
        "n_outliers": len(outlier_indices),
    }


@pytest.fixture
def outlier_viz_mahalanobis_results():
    """Mahalanobis results for visualization testing."""
    np.random.seed(42)
    n_samples = 100
    n_components = 3

    # Generate distances (chi-squared distributed for normal points)
    from scipy.stats import chi2

    distances = np.sqrt(chi2.rvs(df=n_components, size=n_samples))
    outlier_indices = [8, 18, 28, 38, 48, 58]

    # Make outliers have larger distances
    for idx in outlier_indices:
        distances[idx] = np.random.uniform(3, 5)

    # Generate PCA components for visualization
    pca_components = np.random.randn(n_samples, n_components)

    # Feature importance data
    n_features = 10
    feature_names = [f"trait_{i}" for i in range(n_features)]
    explained_var_ratio = np.random.dirichlet(np.ones(n_components))
    explained_var_per_feature = np.random.uniform(0.5, 1.0, n_features)

    return {
        "method": "Mahalanobis",
        "mahalanobis_distances": distances.tolist(),
        "outlier_indices": outlier_indices,
        "data_indices": list(range(n_samples)),
        "n_components": n_components,
        "threshold_type": "chi_squared",
        "threshold_value": chi2.ppf(0.975, df=n_components),
        "chi2_percentile": 97.5,
        "pca_components": pca_components.tolist(),
        "explained_variance_ratio": explained_var_ratio.tolist(),
        "explained_variance_ratio_per_feature": explained_var_per_feature.tolist(),
        "feature_names": feature_names,
        "variance_threshold": 0.95,
        "n_outliers": len(outlier_indices),
    }


@pytest.fixture
def outlier_viz_all_methods_results(
    outlier_viz_isolation_results, outlier_viz_mahalanobis_results
):
    """Combined results from multiple outlier detection methods."""
    # PCA results
    pca_results = {
        "method": "PCA",
        "outlier_indices": [5, 15, 25, 35],
        "reconstruction_errors": np.random.exponential(1, 100).tolist(),
        "n_outliers": 4,
    }

    # Combined results
    combined_results = {
        "method": "Combined",
        "consensus_outliers": [5, 15, 25],  # Overlap between methods
        "n_consensus_outliers": 3,
        "consensus_threshold": 0.5,
        "n_methods": 3,
        "pca_outliers": pca_results["outlier_indices"],
        "isolation_forest_outliers": outlier_viz_isolation_results["outlier_indices"],
        "mahalanobis_outliers": outlier_viz_mahalanobis_results["outlier_indices"],
        "agreement_summary": {
            "methods_compared": ["pca", "isolation_forest", "mahalanobis"],
            "total_methods": 3,
            "consensus_rule": "Agreed by at least 50% of methods (2 out of 3)",
        },
    }

    return {
        "pca": pca_results,
        "isolation_forest": outlier_viz_isolation_results,
        "mahalanobis": outlier_viz_mahalanobis_results,
        "combined": combined_results,
    }


@pytest.fixture
def outlier_viz_error_results():
    """Results with errors for testing error handling."""
    return {
        "isolation_forest": {
            "error": "Not enough samples for Isolation Forest",
            "outlier_indices": [],
        },
        "mahalanobis": {
            "error": "Singular covariance matrix",
            "outlier_indices": [],
        },
    }


@pytest.fixture
def outlier_viz_empty_results():
    """Empty results for edge case testing."""
    return {
        "pca": {
            "method": "PCA",
            "outlier_indices": [],
            "n_outliers": 0,
        },
        "isolation_forest": {
            "method": "IsolationForest",
            "outlier_indices": [],
            "anomaly_scores": [],
            "n_outliers": 0,
        },
        "mahalanobis": {
            "method": "Mahalanobis",
            "outlier_indices": [],
            "mahalanobis_distances": [],
            "n_outliers": 0,
        },
    }


@pytest.fixture
def outlier_viz_pca_results():
    """PCA outlier detection results for visualization testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 15
    n_components_used = 5
    n_total_components = 10

    # Generate explained variance ratios
    raw_variances = np.random.exponential(1, n_total_components)
    raw_variances = np.sort(raw_variances)[::-1]  # Sort descending
    explained_var_ratio = raw_variances / raw_variances.sum()
    cumulative_variance = np.cumsum(explained_var_ratio)

    # Generate reconstruction errors
    reconstruction_errors = np.random.exponential(0.5, n_samples)
    outlier_indices = [5, 15, 25, 35, 45, 55]

    # Make outliers have larger errors
    for idx in outlier_indices:
        reconstruction_errors[idx] = np.random.uniform(2, 4)

    # Generate PCA components
    pca_components = np.random.randn(n_samples, n_components_used)

    # Feature names and their explained variance
    feature_names = [f"trait_{i}" for i in range(n_features)]
    explained_var_per_feature = np.random.uniform(0.3, 1.0, n_features)

    # Generate loadings
    loadings = np.random.randn(n_features, n_total_components)
    eigenvalues = raw_variances

    return {
        "method": "PCA",
        "reconstruction_errors": reconstruction_errors.tolist(),
        "outlier_indices": outlier_indices,
        "data_indices": list(range(n_samples)),
        "n_components": n_components_used,
        "explained_variance_threshold": 0.95,
        "threshold_value": 1.8,
        "outlier_threshold": 2.5,
        "pca_components": pca_components.tolist(),
        "explained_variance_ratio": explained_var_ratio.tolist(),
        "cumulative_variance": cumulative_variance.tolist(),
        "explained_variance_ratio_per_feature": explained_var_per_feature.tolist(),
        "feature_names": feature_names,
        "loadings": loadings.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "n_outliers": len(outlier_indices),
    }


# ============================================================================
# HERITABILITY FIXTURES
# ============================================================================


@pytest.fixture
def heritability_results_basic():
    """Basic heritability results for testing visualization."""
    np.random.seed(42)

    # Generate heritability values with different ranges
    trait_names = [
        "root_depth",
        "root_width",
        "lateral_count",
        "primary_length",
        "total_length",
        "convex_area",
        "network_area",
        "perimeter",
        "avg_radius",
        "max_radius",
        "stem_width",
        "density",
    ]

    # Mix of high, medium, and low heritability values
    h2_values = [
        0.85,
        0.72,
        0.68,
        0.65,  # High heritability
        0.55,
        0.48,
        0.42,
        0.38,  # Medium heritability
        0.25,
        0.18,
        0.12,
        0.08,  # Low heritability
    ]

    results = {}
    for trait, h2 in zip(trait_names, h2_values):
        results[trait] = {
            "heritability": h2,
            "variance_components": {
                "genetic": h2 * 100,
                "environmental": (1 - h2) * 100,
                "total": 100,
            },
            "confidence_interval": [max(0, h2 - 0.1), min(1, h2 + 0.1)],
            "n_genotypes": 50,
            "n_observations": 150,
        }

    return results


@pytest.fixture
def heritability_results_empty():
    """Empty heritability results for edge case testing."""
    return {}


@pytest.fixture
def heritability_results_invalid():
    """Heritability results with invalid/missing data."""
    return {
        "trait_1": {"variance": 100},  # Missing heritability key
        "trait_2": {"heritability": None},  # None value
        "trait_3": "not_a_dict",  # Invalid format
        "trait_4": {"heritability": -0.1},  # Invalid value (negative)
        "trait_5": {"heritability": 1.5},  # Invalid value (>1)
    }


@pytest.fixture
def heritability_threshold_analysis():
    """Threshold analysis results for heritability threshold plot."""
    thresholds = np.linspace(0, 1, 101)
    total_traits = 50

    # Simulate retention curve - exponential decay
    traits_retained = np.zeros_like(thresholds)
    for i, thresh in enumerate(thresholds):
        # Count traits above threshold
        traits_retained[i] = total_traits * np.exp(-3 * thresh)

    # Ensure integer counts and reasonable values
    traits_retained = np.round(traits_retained).astype(int)
    traits_retained = np.maximum(0, traits_retained)
    traits_retained = np.minimum(total_traits, traits_retained)

    fraction_retained = traits_retained / total_traits

    return {
        "thresholds": thresholds,
        "traits_retained": traits_retained,
        "fraction_retained": fraction_retained,
        "total_traits": total_traits,
        "trait_names_by_threshold": {
            0.3: ["trait_" + str(i) for i in range(35)],
            0.5: ["trait_" + str(i) for i in range(18)],
            0.7: ["trait_" + str(i) for i in range(7)],
        },
    }


@pytest.fixture
def heritability_threshold_analysis_empty():
    """Empty threshold analysis for edge case testing."""
    return {
        "thresholds": np.array([]),
        "traits_retained": np.array([]),
        "fraction_retained": np.array([]),
        "total_traits": 0,
    }


# ============================================================================
# PCA VISUALIZATION FIXTURES
# ============================================================================


@pytest.fixture
def pca_viz_results():
    """PCA results for visualization testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    n_components = 5

    # Create fake PCA results
    X_transformed = np.random.randn(n_samples, n_components)

    # Create loadings matrix
    loadings = np.random.randn(n_features, n_components)
    # Normalize to make orthonormal
    loadings, _ = np.linalg.qr(loadings)

    # Create eigenvalues (decreasing)
    eigenvalues = np.array([5.0, 3.0, 2.0, 1.0, 0.5])

    # Calculate explained variance
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Feature names
    feature_names = [f"trait_{i}" for i in range(n_features)]

    # Create feature contributions DataFrame with feature names as index
    # This matches the new standard key name
    total_contributions = np.sum(loadings**2 * eigenvalues, axis=1)
    fractional_contributions = total_contributions / np.sum(total_contributions)

    feature_contributions = pd.DataFrame(
        {
            "total_contribution": total_contributions,
            "fractional_contribution": fractional_contributions,
        },
        index=feature_names,
    )
    feature_contributions = feature_contributions.sort_values(
        "total_contribution", ascending=False
    )

    return {
        "transformed_data": X_transformed,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": cumulative_variance,
        "n_components_selected": n_components,
        "feature_names": feature_names,
        "n_features": n_features,
        "feature_contributions": feature_contributions,  # Standard key for feature contributions
    }


@pytest.fixture
def pca_viz_dataframe():
    """DataFrame with PCA-ready data and metadata."""
    np.random.seed(42)
    n_samples = 100

    # Create trait data
    trait_data = {}
    for i in range(10):
        trait_data[f"trait_{i}"] = np.random.randn(n_samples) * (i + 1)

    # Add metadata
    df = pd.DataFrame(trait_data)
    df["Barcode"] = [f"Sample_{i:03d}" for i in range(n_samples)]
    df["geno"] = np.random.choice(["A", "B", "C", "D", "E"], n_samples)
    df["treatment"] = np.random.choice(["Control", "Treated"], n_samples)

    return df


@pytest.fixture
def umap_viz_results():
    """UMAP embedding results for visualization testing."""
    np.random.seed(42)
    n_samples = 100

    # Create 2D UMAP embedding with some structure
    # Three clusters
    cluster_centers = np.array([[0, 0], [5, 5], [-5, 2]])
    labels = np.random.choice(3, n_samples)

    umap_embedding = np.zeros((n_samples, 2))
    for i in range(n_samples):
        center = cluster_centers[labels[i]]
        umap_embedding[i] = center + np.random.randn(2) * 0.5

    # Return in the format expected from perform_umap_analysis
    return {
        "embedding": umap_embedding,
        "n_neighbors": 15,
        "min_dist": 0.1,
        "reducer": None,  # Would be the UMAP object in real usage
        "scaler": None,  # Would be StandardScaler in real usage
    }


@pytest.fixture
def extreme_samples_data():
    """Data with known extreme samples for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Create normal data
    X = np.random.randn(n_samples, n_features)

    # Add extreme samples
    X[0, :] = 5  # Extreme high in all dimensions
    X[1, :] = -5  # Extreme low in all dimensions
    X[10, 0] = 4  # Extreme in first PC
    X[20, 1] = -4  # Extreme in second PC

    # Create PCA results
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_features)
    X_transformed = pca.fit_transform(X)

    df = pd.DataFrame(X, columns=[f"trait_{i}" for i in range(n_features)])
    df["Barcode"] = [f"Sample_{i:03d}" for i in range(n_samples)]
    df["geno"] = np.random.choice(["A", "B", "C"], n_samples)

    pca_results = {
        "transformed_data": X_transformed,
        "loadings": pca.components_.T,
        "eigenvalues": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "n_components_selected": n_features,
    }

    return df, pca_results


@pytest.fixture
def genotype_pc_data():
    """Data with genotype-specific PC patterns."""
    np.random.seed(42)
    n_samples_per_geno = 20
    genotypes = ["Geno_A", "Geno_B", "Geno_C", "Geno_D", "Geno_E"]

    data_list = []
    geno_list = []

    # Create genotype-specific patterns
    for i, geno in enumerate(genotypes):
        # Each genotype has a different mean in PC space
        mean_shift = np.array([i - 2, 2 - i, 0, 0, 0])
        data = np.random.randn(n_samples_per_geno, 5) + mean_shift
        data_list.append(data)
        geno_list.extend([geno] * n_samples_per_geno)

    X = np.vstack(data_list)

    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"trait_{i}" for i in range(5)])
    df["geno"] = geno_list
    df["Barcode"] = [f"Sample_{i:03d}" for i in range(len(df))]

    # Create PCA results
    from sklearn.decomposition import PCA

    pca = PCA(n_components=5)
    X_transformed = pca.fit_transform(X)

    pca_results = {
        "transformed_data": X_transformed,
        "loadings": pca.components_.T,
        "eigenvalues": pca.explained_variance_,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "n_components_selected": 5,
    }

    return df, pca_results


@pytest.fixture
def pca_results_with_feature_importance():
    """Create PCA results with feature importance for heatmap testing."""
    np.random.seed(42)
    n_features = 30
    n_components = 10

    # Create mock PCA results
    loadings = np.random.randn(n_features, n_components) * 0.3

    # Make some features have strong loadings on first few PCs
    loadings[0:5, 0] = np.random.uniform(0.7, 0.9, 5)  # Strong on PC1
    loadings[5:10, 1] = np.random.uniform(0.6, 0.8, 5)  # Strong on PC2
    loadings[10:15, 2] = np.random.uniform(0.5, 0.7, 5)  # Strong on PC3

    # Create feature importance DataFrame
    feature_names = [f"trait_{i}" for i in range(n_features)]
    feature_importance = pd.DataFrame(
        loadings[:, :5],  # First 5 components
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(5)],
    )

    # Add total contribution
    feature_importance["total_contribution"] = np.abs(feature_importance).sum(axis=1)

    # Calculate eigenvalues (explained variance)
    # For a dataset with total variance of 30 (30 features with variance 1 each if standardized)
    explained_variance_ratio = np.array(
        [0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01]
    )
    eigenvalues = explained_variance_ratio * n_features  # Scale by total variance

    pca_results = {
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "feature_importance": feature_importance,
        "feature_contributions": feature_importance,  # Add alias for new code
        "n_components_selected": 5,
        "feature_names": feature_names,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance_ratio": np.cumsum(explained_variance_ratio),
    }

    return pca_results


@pytest.fixture
def phenotype_variation_data():
    """Create data for testing phenotype variation plots."""
    np.random.seed(42)

    # Create groups with different means and variances
    groups = []
    for i, geno in enumerate(["G1", "G2", "G3", "G4", "G5"]):
        n_samples = np.random.randint(5, 15)
        mean = i * 2 - 4  # Means: -4, -2, 0, 2, 4
        std = 0.5 + i * 0.2  # Increasing variance
        values = np.random.normal(mean, std, n_samples)

        for j, val in enumerate(values):
            groups.append(
                {
                    "geno": geno,
                    "rep": j + 1,
                    "trait_A": val,
                    "trait_B": val * 0.5 + np.random.normal(0, 0.2),
                }
            )

    # Add extreme groups
    for val in np.random.normal(10, 0.5, 8):  # Extremely high
        groups.append(
            {
                "geno": "G_high",
                "rep": len([g for g in groups if g["geno"] == "G_high"]) + 1,
                "trait_A": val,
                "trait_B": val * 0.3,
            }
        )

    for val in np.random.normal(-10, 0.5, 8):  # Extremely low
        groups.append(
            {
                "geno": "G_low",
                "rep": len([g for g in groups if g["geno"] == "G_low"]) + 1,
                "trait_A": val,
                "trait_B": val * 0.3,
            }
        )

    df = pd.DataFrame(groups)

    # Add some NaN values
    df.loc[df.index[::10], "trait_B"] = np.nan

    return df


@pytest.fixture
def pca_export_data():
    """Create data for testing PCA export functionality."""
    np.random.seed(42)

    # Create sample data with known structure
    n_samples = 50
    n_features = 10

    # Generate correlated features with known variance structure
    # First 3 features have high variance, next 3 medium, last 4 low
    data = []
    for i in range(n_samples):
        sample = []
        # High variance features
        base1 = np.random.normal(0, 3)
        sample.extend([base1 + np.random.normal(0, 0.5) for _ in range(3)])
        # Medium variance features
        base2 = np.random.normal(0, 1.5)
        sample.extend([base2 + np.random.normal(0, 0.3) for _ in range(3)])
        # Low variance features
        base3 = np.random.normal(0, 0.5)
        sample.extend([base3 + np.random.normal(0, 0.1) for _ in range(4)])
        data.append(sample)

    # Create DataFrame with metadata
    trait_cols = [f"trait_{i}" for i in range(n_features)]
    df = pd.DataFrame(data, columns=trait_cols)

    # Add metadata columns
    df["Barcode"] = [f"Sample_{i:03d}" for i in range(n_samples)]
    df["geno"] = [f"G{i % 5}" for i in range(n_samples)]  # 5 genotypes
    df["rep"] = [(i // 5) % 3 + 1 for i in range(n_samples)]  # 3 replicates

    # Reorder to put metadata first
    df = df[["Barcode", "geno", "rep"] + trait_cols]

    return df, trait_cols


# ============================================================================
# CONFIGURATION FIXTURES - Pipeline configuration objects
# ============================================================================


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for pipeline testing."""
    return {
        "data": {
            "cleaned_data_path": "test_data.csv",
            "image_dir": "test_images/",
            "barcode_col": "Barcode",
            "genotype_col": "geno",
            "replicate_col": "rep",
        },
        "output": {
            "base_dir": "./test_runs",
            "subdirs": [
                "figures",
                "publication_figures",
                "interactive_plots",
                "analysis_outputs",
            ],
            "figure_format": "png",
            "figure_dpi": 150,
            "save_publication": True,
        },
        "analysis": {
            "pca": {
                "variance_threshold": 0.95,
                "n_components": None,
                "standardize": True,
                "n_features_show": 15,
            },
            "umap": {"n_neighbors": 8, "min_dist": 0.1, "n_components": 2},
            "heritability": {"threshold": 0.6, "min_samples_per_genotype": 3},
            "outliers": {
                "phenotype_n_std": 2.0,
                "pc_space_n_std": 2.5,
                "n_pcs_check": 3,
            },
        },
        "visualization": {
            "interactive": True,
            "figsize": {
                "standard": [10, 8],
                "correlation": [12, 10],
                "scree": [12, 5],
                "biplot": [10, 8],
            },
            "colors": {"scheme": "tab20"},
            "scatter": {"point_size": 50, "alpha": 0.7},
        },
        "logging": {
            "level": "INFO",
            "file": "pipeline.log",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
    }


@pytest.fixture
def sample_omegaconf_config(sample_config_dict):
    """OmegaConf configuration object for pipeline testing."""
    return OmegaConf.create(sample_config_dict)


@pytest.fixture
def minimal_config_dict():
    """Minimal configuration with only required fields."""
    return {
        "data": {
            "cleaned_data_path": "data.csv",
            "genotype_col": "geno",
            "replicate_col": "rep",
        },
        "output": {"base_dir": "./runs"},
    }


@pytest.fixture
def minimal_omegaconf_config(minimal_config_dict):
    """Minimal OmegaConf configuration object."""
    return OmegaConf.create(minimal_config_dict)


@pytest.fixture
def invalid_config_dict():
    """Invalid configuration missing required fields."""
    return {
        "output": {"base_dir": "./runs"}
        # Missing data section
    }


@pytest.fixture
def config_with_env_vars():
    """Configuration with environment variable interpolation."""
    return OmegaConf.create(
        {
            "data": {
                "cleaned_data_path": "${oc.env:DATA_PATH,default_data.csv}",
                "image_dir": "${oc.env:IMAGE_DIR,./images}",
                "genotype_col": "geno",
                "replicate_col": "rep",
            },
            "output": {"base_dir": "${oc.env:OUTPUT_DIR,./runs}"},
        }
    )


@pytest.fixture
def sample_trait_data():
    """Create sample trait data for testing."""
    np.random.seed(42)
    n_samples = 50
    n_traits = 10

    # Create sample data
    data = pd.DataFrame(
        {
            "geno": np.random.choice(["Geno_A", "Geno_B", "Geno_C"], n_samples),
            "rep": np.random.choice([1, 2, 3], n_samples),
            "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
        }
    )

    # Add trait columns
    for i in range(n_traits):
        data[f"trait_{i + 1}"] = np.random.normal(100 + i * 10, 10, n_samples)

    # Add some metadata columns
    data["QC_status"] = "pass"
    data["scan_date"] = "2024-01-01"

    return data


@pytest.fixture
def simple_cluster_data():
    """Create simple synthetic data with clear clusters for clustering tests."""
    np.random.seed(42)

    # Create 3 well-separated clusters
    cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(30, 5) + np.array([5, 5, 5, 5, 5])
    cluster3 = np.random.randn(30, 5) + np.array([-5, -5, -5, -5, -5])

    data = np.vstack([cluster1, cluster2, cluster3])
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(5)])

    return df


@pytest.fixture
def multimodal_data():
    """Create data with multiple modes for GMM testing."""
    np.random.seed(42)

    # Create 2 overlapping Gaussian distributions
    mode1 = np.random.randn(50, 3) * 0.5 + np.array([0, 0, 0])
    mode2 = np.random.randn(50, 3) * 0.5 + np.array([2, 2, 2])

    data = np.vstack([mode1, mode2])
    df = pd.DataFrame(data, columns=["x", "y", "z"])

    return df


# ============================================================================
# CLUSTERING AND VISUALIZATION FIXTURES
# ============================================================================


@pytest.fixture
def kmeans_cluster_result(simple_cluster_data):
    """Create a sample K-Means clustering result for visualization testing.

    Returns a dictionary matching the structure from perform_kmeans_clustering().
    """
    from sleap_roots_analyze.clustering import perform_kmeans_clustering

    result = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=3, random_state=42
    )
    return result


@pytest.fixture
def gmm_cluster_result(multimodal_data):
    """Create a sample GMM clustering result for visualization testing.

    Returns a dictionary matching the structure from perform_gmm_clustering().
    """
    from sleap_roots_analyze.clustering import perform_gmm_clustering

    result = perform_gmm_clustering(multimodal_data, n_components=2, random_state=42)
    return result


@pytest.fixture
def hierarchical_cluster_result(simple_cluster_data):
    """Create a sample hierarchical clustering result for dendrogram visualization.

    Returns a dictionary matching the structure from perform_hierarchical_clustering().
    """
    from sleap_roots_analyze.clustering import perform_hierarchical_clustering

    result = perform_hierarchical_clustering(
        simple_cluster_data, method="ward", metric="euclidean"
    )
    return result


@pytest.fixture
def pca_result_for_clustering(simple_cluster_data):
    """Create PCA result for cluster visualization testing.

    Returns PCA results with enough components for 2D visualization.
    """
    from sleap_roots_analyze.pca import perform_pca_analysis

    result = perform_pca_analysis(
        simple_cluster_data, n_components=3, standardize=True, random_state=42
    )
    return result


@pytest.fixture
def minimal_pca_result():
    """Create PCA result with only 1 component for edge case testing."""
    np.random.seed(42)
    data = pd.DataFrame({"x": np.random.randn(20), "y": np.random.randn(20)})

    from sleap_roots_analyze.pca import perform_pca_analysis

    result = perform_pca_analysis(
        data, n_components=1, standardize=True, random_state=42
    )
    return result


@pytest.fixture
def distance_array():
    """Create sample distance array for distance distribution plots."""
    np.random.seed(42)
    # Mix of normal and some outliers
    distances = np.concatenate(
        [
            np.random.gamma(2, 2, 80),  # Normal distances
            np.random.uniform(15, 20, 20),  # Outlier distances
        ]
    )
    return distances


@pytest.fixture
def bic_aic_data():
    """Create sample BIC/AIC data for model comparison plots."""
    k_range = list(range(2, 11))
    np.random.seed(42)

    # BIC typically decreases then increases (elbow around k=4)
    bic = [1000 - 50 * k + 5 * k**2 for k in k_range]
    # AIC similar pattern but different scale
    aic = [900 - 45 * k + 4.5 * k**2 for k in k_range]

    return {"k_range": k_range, "bic": bic, "aic": aic, "optimal_k": 4}


@pytest.fixture
def silhouette_data(simple_cluster_data):
    """Create silhouette coefficient data for silhouette plots."""
    from sleap_roots_analyze.clustering import perform_kmeans_clustering
    from sklearn.metrics import silhouette_samples

    result = perform_kmeans_clustering(
        simple_cluster_data, n_clusters=3, random_state=42
    )

    silhouette_vals = silhouette_samples(
        result["data_processed"], result["cluster_labels"]
    )

    return {
        "silhouette_values": silhouette_vals,
        "cluster_labels": result["cluster_labels"],
        "n_clusters": result["n_clusters"],
        "silhouette_avg": result["silhouette_score"],
    }


@pytest.fixture
def edge_case_cluster_data():
    """Create edge case clustering data for error testing."""
    np.random.seed(42)

    return {
        "empty": pd.DataFrame(),
        "all_nan": pd.DataFrame({"x": [np.nan] * 10, "y": [np.nan] * 10}),
        "insufficient_samples": pd.DataFrame({"x": [1], "y": [2]}),
        "two_samples": pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        "single_feature": pd.DataFrame({"x": np.random.randn(50)}),
        "identical_values": pd.DataFrame(
            {"x": [1.0] * 20, "y": [2.0] * 20, "z": [3.0] * 20}
        ),
    }


@pytest.fixture
def cluster_mixed_constant_and_nonnumeric_data():
    """Create clustering data mixing a constant column and a non-numeric column.

    Covers the half of the feature_names bug that pca_constant_feature_data
    does not: a non-numeric (string) column silently dropped by
    select_dtypes inside standardize_data.
    """
    np.random.seed(42)
    n_samples = 90

    df = pd.DataFrame(
        {
            "genotype_id": [f"G{i % 3}" for i in range(n_samples)],
            "constant_trait": np.full(n_samples, 7.0),
            "variable1": np.random.randn(n_samples),
            "variable2": np.random.randn(n_samples) * 2,
        }
    )

    return df


@pytest.fixture
def cluster_nonnumeric_only_data():
    """Create clustering data with a non-numeric column but no constant column.

    cluster_mixed_constant_and_nonnumeric_data always pairs the non-numeric
    column with a constant column; this isolates the select_dtypes half of
    the filter on its own.
    """
    np.random.seed(42)
    n_samples = 90

    df = pd.DataFrame(
        {
            "genotype_id": [f"G{i % 3}" for i in range(n_samples)],
            "variable1": np.random.randn(n_samples),
            "variable2": np.random.randn(n_samples) * 2,
        }
    )

    return df


@pytest.fixture
def cluster_separated_data_with_constant():
    """Well-separated 3-cluster 2D data plus one constant column.

    Clusters are separated enough (10 units apart, 0.3 std) that GMM's soft
    assignments collapse to hard assignments to machine precision, making
    each cluster's true center exactly recoverable from cluster_labels +
    data_processed. Needed to test that cluster_centers/means (not just
    data_processed) are positionally aligned with feature_names -- unlike
    pca_constant_feature_data, whose variable1/variable2 have no true
    cluster structure, so GMM's per-component means only approximately
    match a hard-assignment recomputation there.
    """
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) * 0.3 + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) * 0.3 + np.array([10, 10])
    cluster3 = np.random.randn(30, 2) * 0.3 + np.array([-10, 10])
    data = np.vstack([cluster1, cluster2, cluster3])
    df = pd.DataFrame(data, columns=["trait_a", "trait_b"])
    df["constant_trait"] = 7.0
    return df


@pytest.fixture
def linkage_matrix_small():
    """Create a small linkage matrix for hierarchical clustering edge cases."""
    from scipy.cluster.hierarchy import linkage

    # Create minimal data (3 samples)
    np.random.seed(42)
    data = np.random.randn(3, 2)

    matrix = linkage(data, method="ward")
    return matrix


@pytest.fixture
def cluster_result_with_nan():
    """Create clustering data with NaN values for robustness testing."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 4, 6, np.nan, 10, 12, 14, 16, 18, 20],
            "z": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )
    return df


@pytest.fixture
def gmm_convergence_data():
    """Create data that might cause GMM convergence issues."""
    np.random.seed(42)

    # Create well-separated but small clusters
    cluster1 = np.random.randn(5, 2) + np.array([0, 0])
    cluster2 = np.random.randn(5, 2) + np.array([10, 10])

    data = np.vstack([cluster1, cluster2])
    df = pd.DataFrame(data, columns=["x", "y"])

    return df


@pytest.fixture
def hierarchical_edge_cases():
    """Create edge cases for hierarchical clustering method validation."""
    np.random.seed(42)

    return {
        "ward_euclidean": pd.DataFrame(np.random.randn(20, 3)),  # Valid
        "ward_manhattan": pd.DataFrame(np.random.randn(20, 3)),  # Invalid combo
        "complete_manhattan": pd.DataFrame(np.random.randn(20, 3)),  # Valid
        "single_cosine": pd.DataFrame(np.random.randn(20, 3)),  # Valid
    }


@pytest.fixture
def optimal_k_test_data():
    """Create data for testing optimal k selection algorithms."""
    np.random.seed(42)

    # Create data with obvious k=4 clusters
    clusters = []
    centers = [[0, 0], [10, 0], [0, 10], [10, 10]]

    for center in centers:
        cluster = np.random.randn(15, 2) * 0.5 + np.array(center)
        clusters.append(cluster)

    data = np.vstack(clusters)
    df = pd.DataFrame(data, columns=["x", "y"])

    return {"data": df, "true_k": 4, "min_k": 2, "max_k": 8}


@pytest.fixture
def cluster_sizes_data():
    """Create cluster size data for bar plot testing."""
    return {
        "cluster_sizes": [45, 30, 15, 5, 3],
        "n_clusters": 5,
        "labels": ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4", "Cluster 5"],
    }


@pytest.fixture
def highlight_indices_data(simple_cluster_data):
    """Create sample indices to highlight in cluster scatter plots."""
    # Select some samples from different clusters
    np.random.seed(42)

    total_samples = len(simple_cluster_data)
    n_highlights = 10

    highlight_idx = np.random.choice(
        total_samples, n_highlights, replace=False
    ).tolist()

    return {
        "indices": highlight_idx,
        "empty": [],
        "no_match": [999, 1000, 1001],  # Indices not in data
        "all": list(range(total_samples)),
    }


# ============================================================================
# VISUALIZATION PIPELINE FIXTURES
# ============================================================================


@pytest.fixture
def viz_config_minimal():
    """Minimal VizPipelineConfig for testing."""
    from sleap_roots_analyze.pipeline.config import VizPipelineConfig

    config = VizPipelineConfig(pipeline_name="test_viz_minimal")
    config.data.csv_path = "dummy.csv"
    config.statistics.calculate_anova = False
    config.statistics.calculate_heritability = False
    config.umap.enabled = False
    config.clustering.enabled = False
    config.heritability.enabled = False
    config.interesting_genotypes.enabled = False
    config.static_viz.enabled = False
    config.interactive_viz.enabled = False
    config.dashboard.enabled = False
    return config


@pytest.fixture
def viz_config_with_stats():
    """VizPipelineConfig with statistics enabled."""
    from sleap_roots_analyze.pipeline.config import VizPipelineConfig

    config = VizPipelineConfig(pipeline_name="test_viz_stats")
    config.data.csv_path = "dummy.csv"
    config.statistics.calculate_anova = True
    config.statistics.calculate_heritability = True
    # Disable optional features
    config.umap.enabled = False
    config.clustering.enabled = False
    config.heritability.enabled = False
    config.interesting_genotypes.enabled = False
    config.static_viz.enabled = False
    config.interactive_viz.enabled = False
    config.dashboard.enabled = False
    return config


@pytest.fixture
def adaptive_sizing_config():
    """Default AdaptiveSizingConfig for testing."""
    from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig

    return AdaptiveSizingConfig()


@pytest.fixture
def adaptive_sizing_config_disabled():
    """AdaptiveSizingConfig with sizing disabled."""
    from sleap_roots_analyze.pipeline.config import AdaptiveSizingConfig

    config = AdaptiveSizingConfig()
    config.enabled = False
    return config


# ============================================================================
# ROOT CORE ANALYSIS FIXTURES
# ============================================================================


@pytest.fixture
def create_test_root_core_data():
    """Create sample root core data for testing.

    Returns a DataFrame with root core count data in the expected format:
    - Metadata columns: Plot, geno, Rep, core_n
    - Depth columns: c_<start>_<end>_<subcore> (e.g., c_0_10_1, c_0_10_2)

    Data includes:
    - 2 genotypes (GH_7386, GH_7418)
    - 1 replicate each
    - 3 cores per plot
    - 4 depth ranges (0-10, 10-20, 20-30, 30-40 cm)
    - 2 subcores per depth range

    Returns:
        pd.DataFrame: Root core count data with known values for testing
    """
    data = {
        "Plot": [1, 1, 1, 2, 2, 2],
        "geno": ["GH_7386", "GH_7386", "GH_7386", "GH_7418", "GH_7418", "GH_7418"],
        "Ent": [1, 1, 1, 2, 2, 2],
        "Rep": [1, 1, 1, 1, 1, 1],
        "Sub": [1, 1, 1, 1, 1, 1],
        "core_n": [1, 2, 3, 1, 2, 3],
        # Depth 0-10cm (2 subcores)
        "c_0_10_1": [78, 89, 87, 120, 47, 115],
        "c_0_10_2": [62, 96, 42, 134, 67, 98],
        # Depth 10-20cm (2 subcores)
        "c_10_20_1": [44, 49, 36, 78, 56, 79],
        "c_10_20_2": [26, 56, 32, 38, 38, 43],
        # Depth 20-30cm (2 subcores)
        "c_20_30_1": [26, 32, 22, 27, 36, 46],
        "c_20_30_2": [23, 4, 21, 16, 17, 32],
        # Depth 30-40cm (2 subcores)
        "c_30_40_1": [16, 6, 7, 11, 8, 7],
        "c_30_40_2": [5, 9, 3, 9, 5, 10],
    }
    return pd.DataFrame(data)


# ============================================================================
# CROSS-PLATFORM ANALYSIS FIXTURES
# ============================================================================


@pytest.fixture
def cross_platform_exp1_df():
    """Generate experiment 1 DataFrame for cross-platform analysis testing.

    Simulates cylinder experiment data with:
    - 18 genotypes (15 common + 3 unique)
    - 4-6 replicates per genotype
    - 50 numeric traits
    - Some NaN values to test handling

    Returns:
        pd.DataFrame: Experiment 1 data with genotype, replicate, and trait columns
    """
    np.random.seed(42)

    # Common genotypes across experiments
    common_genotypes = [f"Geno{i:02d}" for i in range(1, 16)]
    # Unique to exp1
    unique_genotypes = ["GenoX1", "GenoX2", "GenoX3"]
    all_genotypes = common_genotypes + unique_genotypes

    data = []
    for geno in all_genotypes:
        n_reps = np.random.randint(4, 7)  # 4-6 replicates
        for rep in range(1, n_reps + 1):
            row = {
                "plant_qr_code": f"{geno}_R{rep}",
                "Geno": geno,
                "rep": rep,
            }

            # Add 50 numeric traits with genotype-specific means
            geno_idx = all_genotypes.index(geno)
            for trait_idx in range(50):
                # Create traits with genotype effects and some correlation
                base_value = 100 + geno_idx * 5 + trait_idx * 2
                noise = np.random.normal(0, 10)
                value = base_value + noise

                # Add some NaN values (5% chance)
                if np.random.random() < 0.05:
                    value = np.nan

                row[f"exp1_trait_{trait_idx:02d}"] = value

            data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def cross_platform_exp2_df():
    """Generate experiment 2 DataFrame for cross-platform analysis testing.

    Simulates turface experiment data with:
    - 18 genotypes (15 common + 3 unique)
    - 3-5 replicates per genotype
    - 12 numeric traits (fewer than exp1)
    - Some NaN values to test handling

    Returns:
        pd.DataFrame: Experiment 2 data with genotype, replicate, and trait columns
    """
    np.random.seed(123)

    # Common genotypes across experiments (same as exp1)
    common_genotypes = [f"Geno{i:02d}" for i in range(1, 16)]
    # Unique to exp2
    unique_genotypes = ["GenoY1", "GenoY2", "GenoY3"]
    all_genotypes = common_genotypes + unique_genotypes

    data = []
    for geno in all_genotypes:
        n_reps = np.random.randint(3, 6)  # 3-5 replicates
        for rep in range(1, n_reps + 1):
            row = {
                "Barcode": f"{geno}_T{rep}",
                "geno": geno,
                "rep": rep,
            }

            # Add 12 numeric traits with genotype-specific means
            # Some traits correlated with exp1, some uncorrelated
            geno_idx = all_genotypes.index(geno)
            for trait_idx in range(12):
                if trait_idx < 6:
                    # Positively correlated with exp1 traits
                    base_value = 100 + geno_idx * 5 + trait_idx * 2
                else:
                    # Uncorrelated or negatively correlated
                    base_value = 200 - geno_idx * 3 + trait_idx

                noise = np.random.normal(0, 8)
                value = base_value + noise

                # Add some NaN values (5% chance)
                if np.random.random() < 0.05:
                    value = np.nan

                row[f"exp2_trait_{trait_idx:02d}"] = value

            data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def cross_platform_config_dict():
    """Generate valid configuration dictionary for cross-platform analysis.

    Returns:
        dict: Configuration with all required fields for CrossPlatformConfig
    """
    return {
        "exp1_data_path": "exp1_data.csv",
        "exp1_name": "Cylinder",
        "exp1_genotype_col": "Geno",
        "exp2_data_path": "exp2_data.csv",
        "exp2_name": "Turface",
        "exp2_genotype_col": "geno",
        "correlation_method": "spearman",
        "min_samples_per_genotype": 3,
        "significance_level": 0.05,
        "top_n_correlations": 20,
        "top_n_joint_plots": 6,
        "top_n_boxplots": 6,
        "figsize_summary": (14, 12),
        "figsize_joint": (10, 10),
        "figsize_boxplot": (14, 6),
    }


@pytest.fixture
def cross_platform_correlation_results():
    """Generate sample correlation results DataFrame for testing visualizations.

    Returns:
        pd.DataFrame: Correlation results with traits, correlation values, and p-values
    """
    np.random.seed(42)

    n_correlations = 100
    data = []

    for i in range(n_correlations):
        # Generate realistic correlation values
        rho = np.random.uniform(-0.5, 0.5)

        # P-values tend to be larger for small correlations
        if abs(rho) < 0.2:
            p_value = np.random.uniform(0.1, 0.9)
        elif abs(rho) < 0.35:
            p_value = np.random.uniform(0.01, 0.2)
        else:
            p_value = np.random.uniform(0.0001, 0.05)

        data.append(
            {
                "cylinder_trait": f"exp1_trait_{i % 50:02d}",
                "turface_trait": f"exp2_trait_{i % 12:02d}",
                "spearman_r": rho,
                "spearman_p": p_value,
                "n_genotypes": 15,
                "abs_spearman": abs(rho),
            }
        )

    df = pd.DataFrame(data)
    return df.sort_values("abs_spearman", ascending=False).reset_index(drop=True)


@pytest.fixture(scope="session")
def cross_platform_turface_df(test_data_dir):
    """Load Turface_all_traits_2024.csv for real cross-platform testing.

    Returns:
        pd.DataFrame: Real turface experiment data with root traits
    """
    return pd.read_csv(test_data_dir / "Turface_all_traits_2024.csv")


@pytest.fixture(scope="session")
def cross_platform_field_df(test_data_dir):
    """Load Field_2024_clean.csv for real cross-platform testing.

    Returns:
        pd.DataFrame: Real field experiment data with above-ground and root core data
    """
    return pd.read_csv(test_data_dir / "Field_2024_clean.csv")


# ============================================================================
# PIPELINE REPRODUCTION FIXTURES (#120)
# Golden wheat-EDPIE fixtures backing the full pipeline (QC -> viz ->
# cross-platform). Loaded once per session and shared across the per-stage
# reproduction tests in test_pipeline_reproduction.py. See
# tests/fixtures/README.md for layout, curation, and tolerance/regenerate policy.
# ============================================================================


# The four EDPIE platforms whose golden fixtures are committed under
# tests/fixtures/real/wheat_edpie/expected/{qc,viz}/<platform>/.
EDPIE_PLATFORMS = ("turface_19", "turface_150", "cylinder", "root_core")


@pytest.fixture(scope="session")
def edpie_platforms():
    """Return the tuple of EDPIE platform keys with committed golden fixtures."""
    return EDPIE_PLATFORMS


@pytest.fixture(scope="session")
def repro_fixtures_dir():
    """Return the root of the pipeline reproduction fixture tree."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def harness_dir(repro_fixtures_dir):
    """Return the harness directory (runnable EDPIE recipe)."""
    return repro_fixtures_dir / "harness"


@pytest.fixture(scope="session")
def edpie_real_dir(repro_fixtures_dir):
    """Return the real wheat-EDPIE fixture directory."""
    return repro_fixtures_dir / "real" / "wheat_edpie"


@pytest.fixture(scope="session")
def final_data_by_platform(edpie_real_dir):
    """Load each platform's post-QC ``10_final_data.csv`` golden table (once)."""
    return {
        p: pd.read_csv(edpie_real_dir / "expected" / "qc" / p / "10_final_data.csv")
        for p in EDPIE_PLATFORMS
    }


@pytest.fixture(scope="session")
def qc_heritability_by_platform(edpie_real_dir):
    """Load each platform's QC heritability-filter summary JSON."""
    return {
        p: json.loads(
            (
                edpie_real_dir
                / "expected"
                / "qc"
                / p
                / "09_heritability_filter_summary.json"
            ).read_text()
        )
        for p in EDPIE_PLATFORMS
    }


@pytest.fixture(scope="session")
def qc_removed_counts_by_platform(edpie_real_dir):
    """Map each platform to its removed outlier/trait/sample row counts."""
    detail = {
        "outliers": "07_removed_outliers_detail.csv",
        "traits": "01_removed_traits_detail.csv",
        "samples": "02_removed_samples_detail.csv",
    }
    out = {}
    for p in EDPIE_PLATFORMS:
        qc = edpie_real_dir / "expected" / "qc" / p
        out[p] = {k: len(pd.read_csv(qc / f)) for k, f in detail.items()}
    return out


@pytest.fixture(scope="session")
def viz_summary_by_platform(edpie_real_dir):
    """Load each platform's viz summary JSON (headline metrics)."""
    return {
        p: json.loads(
            (edpie_real_dir / "expected" / "viz" / p / "summary.json").read_text()
        )
        for p in EDPIE_PLATFORMS
    }


@pytest.fixture(scope="session")
def viz_pca_by_platform(edpie_real_dir):
    """Load each platform's curated viz PCA metadata (trait_cols, explained var)."""
    return {
        p: json.loads(
            (
                edpie_real_dir / "expected" / "viz" / p / "viz_pca_metadata.json"
            ).read_text()
        )
        for p in EDPIE_PLATFORMS
    }


@pytest.fixture(scope="session")
def viz_umap_by_platform(edpie_real_dir):
    """Map each platform with a UMAP embedding to its golden Nx2 array.

    Platforms whose viz run produced no UMAP (e.g. ``root_core``) are absent.
    """
    out = {}
    for p in EDPIE_PLATFORMS:
        f = edpie_real_dir / "expected" / "viz" / p / "viz_umap_embedding.csv"
        if f.is_file():
            out[p] = pd.read_csv(f).to_numpy()
    return out


@pytest.fixture(scope="session")
def crossplatform_dir(edpie_real_dir):
    """Return the directory holding the cross-platform golden pairings."""
    return edpie_real_dir / "expected" / "cross_platform"


@pytest.fixture(scope="session")
def numerical_stability_golden():
    """Load the committed numerical-stability golden artifacts + provenance.

    Mirrors the ``*_by_platform`` reproduction loaders. Returns a dict with the golden
    UMAP embedding, cluster labels, per-genotype trait summary, and the provenance
    record the golden was generated under. See ``tests/test_numerical_stability.py``.
    """
    from tests.numerical_stability_recompute import (
        GOLDEN_EMBEDDING,
        GOLDEN_LABELS,
        GOLDEN_PROVENANCE,
        GOLDEN_TRAIT_SUMMARY,
    )

    return {
        "embedding": pd.read_csv(GOLDEN_EMBEDDING),
        "labels": pd.read_csv(GOLDEN_LABELS),
        "trait_summary": pd.read_csv(GOLDEN_TRAIT_SUMMARY, index_col=0),
        "provenance": json.loads(GOLDEN_PROVENANCE.read_text()),
    }
