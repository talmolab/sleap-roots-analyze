"""Centralized pytest fixtures for test data."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from scipy import stats


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
                    "geno": f"G{g+1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g*n_reps + r:04d}",
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
                    "geno": f"G{g+1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g*n_reps + r:04d}",
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
                    "geno": f"G{g+1:02d}",
                    "rep": r + 1,
                    "Barcode": f"BC{g*n_reps + r:04d}",
                    "trait_zero": 50
                    + np.random.normal(0, 5),  # Only environmental noise
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
        "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i+1}" for i in range(5)],
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
    df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(n_features)])

    # Add metadata
    df["Barcode"] = [f"BC{i:04d}" for i in range(n)]
    df["geno"] = [f"G{i%5+1}" for i in range(n)]

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
                f"G{i%2+1}" for i in range(n)
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
            "geno": [f"G{i%5+1}" for i in range(n)],
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
    
    df = pd.DataFrame({
        'trait1': np.random.normal(0, 1, n_samples),
        'trait2': np.random.normal(5, 2, n_samples),
        'trait3': np.random.uniform(0, 10, n_samples),
        'geno': np.random.choice(['A', 'B', 'C'], n_samples),
        'Barcode': [f'BC{i:04d}' for i in range(n_samples)]
    })
    
    return df


@pytest.fixture
def viz_data_with_nan():
    """Create data with NaN values for visualization testing.
    
    Returns:
        pd.DataFrame: Data containing various NaN patterns
    """
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        'trait_complete': np.random.randn(n_samples),
        'trait_some_nan': np.concatenate([np.random.randn(40), [np.nan]*10]),
        'trait_all_nan': [np.nan] * n_samples,
        'geno': np.random.choice(['Type1', 'Type2'], n_samples),
        'Barcode': [f'BC{i:03d}' for i in range(n_samples)]
    })
    
    return df


@pytest.fixture
def viz_empty_data():
    """Create empty DataFrame for edge case testing.
    
    Returns:
        pd.DataFrame: Empty DataFrame with expected columns
    """
    return pd.DataFrame(columns=['trait1', 'trait2', 'geno', 'Barcode'])


@pytest.fixture  
def viz_single_trait_data():
    """Create data with single trait for testing.
    
    Returns:
        pd.DataFrame: Data with one trait column
    """
    np.random.seed(42)
    return pd.DataFrame({
        'single_trait': np.random.randn(75),
        'geno': np.random.choice(['G1', 'G2', 'G3'], 75),
        'Barcode': [f'BC{i:03d}' for i in range(75)]
    })


@pytest.fixture
def viz_many_traits_data():
    """Create data with many traits for testing subplot layouts.
    
    Returns:
        pd.DataFrame: Data with 30+ trait columns
    """
    np.random.seed(42)
    n_samples = 50
    n_traits = 30
    
    data = {'Barcode': [f'BC{i:03d}' for i in range(n_samples)],
            'geno': np.random.choice(['A', 'B', 'C', 'D'], n_samples)}
    
    for i in range(n_traits):
        data[f'trait_{i:02d}'] = np.random.randn(n_samples) * (i+1) 
    
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
    
    df = pd.DataFrame({
        'trait_a': base,
        'trait_b': base * 2,  # Perfect positive correlation
        'trait_c': -base,     # Perfect negative correlation  
        'trait_d': np.random.randn(n_samples),  # Independent
        'geno': np.random.choice(['X', 'Y'], n_samples)
    })
    
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
    group1 = np.random.normal(-2, 0.5, n_samples//2)
    group2 = np.random.normal(2, 0.5, n_samples//2) 
    
    df = pd.DataFrame({
        'trait_bimodal': np.concatenate([group1, group2]),
        'trait_normal': np.random.normal(0, 1, n_samples),
        'geno': ['GroupA']*(n_samples//2) + ['GroupB']*(n_samples//2)
    })
    
    return df


@pytest.fixture
def viz_single_genotype_data():
    """Create data with only one genotype group.
    
    Returns:
        pd.DataFrame: Data with single genotype value
    """
    np.random.seed(42)
    n_samples = 60
    
    df = pd.DataFrame({
        'trait1': np.random.randn(n_samples),
        'trait2': np.random.exponential(2, n_samples),
        'geno': ['SingleType'] * n_samples
    })
    
    return df


@pytest.fixture
def viz_constant_trait_data():
    """Create data with constant (zero variance) traits.
    
    Returns:
        pd.DataFrame: Data with some constant traits
    """
    np.random.seed(42)
    n_samples = 80
    
    df = pd.DataFrame({
        'trait_constant': [42.0] * n_samples,
        'trait_variable': np.random.randn(n_samples),
        'trait_zero': np.zeros(n_samples),
        'geno': np.random.choice(['A', 'B'], n_samples)
    })
    
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
        'Barcode': [f'BC{i:04d}' for i in range(n_samples)],
        'geno': np.random.choice(['A', 'B', 'C'], n_samples),
        'rep': np.random.choice([1, 2, 3], n_samples),
        # Good trait - normal distribution
        'trait_good': np.random.normal(10, 2, n_samples),
        # High NaN trait (40% NaN)
        'trait_high_nan': np.where(np.random.random(n_samples) < 0.4, np.nan, 
                                   np.random.normal(5, 1, n_samples)),
        # High zero trait (60% zeros)
        'trait_high_zero': np.where(np.random.random(n_samples) < 0.6, 0, 
                                    np.random.normal(3, 0.5, n_samples)),
        # Low variance trait
        'trait_low_var': np.random.normal(50, 0.01, n_samples),
        # Outlier-prone trait
        'trait_outliers': np.concatenate([
            np.random.normal(0, 1, 90),  # Normal values
            np.random.normal(10, 0.5, 10)  # Outliers
        ])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def viz_eda_thresholds():
    """Standard thresholds for EDA cleanup.
    
    Returns:
        dict: Thresholds for NaN, zero, and outlier fractions
    """
    return {
        'nan': 0.3,    # 30% maximum NaN
        'zero': 0.5,   # 50% maximum zeros
        'outlier': 0.1  # 10% maximum outliers (though not used for trait removal)
    }


@pytest.fixture
def viz_eda_cleanup_log():
    """Sample cleanup log from apply_data_cleanup_filters.
    
    Returns:
        dict: Cleanup log with removed traits information
    """
    return {
        'removed_traits': [
            {
                'trait': 'trait_high_nan',
                'reason': 'too_many_nans',
                'nan_fraction': 0.4,
                'zero_fraction': 0.05,
                'valid_samples': 60
            },
            {
                'trait': 'trait_high_zero',
                'reason': 'too_many_zeros',
                'nan_fraction': 0.02,
                'zero_fraction': 0.6,
                'valid_samples': 98
            },
            {
                'trait': 'trait_insufficient',
                'reason': 'insufficient_samples',
                'nan_fraction': 0.85,
                'zero_fraction': 0.05,
                'valid_samples': 5
            }
        ],
        'initial_traits': 10,
        'remaining_traits': 7,
        'traits_removed_high_nan': 1,
        'traits_removed_high_zero': 1,
        'traits_removed_low_samples': 1
    }


@pytest.fixture
def viz_eda_data_with_extremes():
    """Create data with extreme values for EDA testing.
    
    Returns:
        pd.DataFrame: Data with various extreme patterns
    """
    np.random.seed(42)
    n_samples = 50
    
    df = pd.DataFrame({
        'Barcode': [f'BC{i:03d}' for i in range(n_samples)],
        'geno': np.random.choice(['Type1', 'Type2'], n_samples),
        # All NaN trait
        'trait_all_nan': [np.nan] * n_samples,
        # All zero trait
        'trait_all_zero': np.zeros(n_samples),
        # Single valid value
        'trait_single_valid': [np.nan] * (n_samples - 1) + [5.0],
        # Boundary case - exactly at threshold (30% NaN)
        'trait_boundary_nan': np.where(np.arange(n_samples) < 15, np.nan,
                                       np.random.normal(10, 1, n_samples)),
        # Boundary case - exactly at threshold (50% zero)
        'trait_boundary_zero': np.where(np.arange(n_samples) < 25, 0,
                                        np.random.normal(5, 1, n_samples)),
        # High variance trait
        'trait_high_var': np.random.normal(100, 50, n_samples),
        # Negative values
        'trait_negative': np.random.normal(-5, 2, n_samples)
    })
    
    return df


@pytest.fixture
def viz_eda_empty_cleanup_log():
    """Empty cleanup log for testing.
    
    Returns:
        dict: Empty cleanup log
    """
    return {
        'removed_traits': [],
        'initial_traits': 5,
        'remaining_traits': 5,
        'traits_removed_high_nan': 0,
        'traits_removed_high_zero': 0,
        'traits_removed_low_samples': 0
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
        'Barcode': [f'BC{i:04d}' for i in range(n_samples)],
        'geno': np.random.choice(['G1', 'G2', 'G3'], n_samples),
        'rep': np.random.choice([1, 2, 3, 4], n_samples)
    }
    
    # Add traits with various prefixes and patterns
    prefixes = ['root', 'lateral', 'crown', 'network', 'depth']
    for i in range(n_traits):
        prefix = prefixes[i % len(prefixes)]
        trait_name = f'{prefix}_{i:02d}'
        
        # Vary the quality of traits
        if i % 5 == 0:
            # Some high NaN traits
            data[trait_name] = np.where(np.random.random(n_samples) < 0.35,
                                       np.nan, np.random.randn(n_samples))
        elif i % 7 == 0:
            # Some high zero traits
            data[trait_name] = np.where(np.random.random(n_samples) < 0.55,
                                       0, np.random.randn(n_samples))
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
    outlier_viz_isolation_results,
    outlier_viz_mahalanobis_results
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
        "root_depth", "root_width", "lateral_count", "primary_length",
        "total_length", "convex_area", "network_area", "perimeter",
        "avg_radius", "max_radius", "stem_width", "density"
    ]
    
    # Mix of high, medium, and low heritability values
    h2_values = [
        0.85, 0.72, 0.68, 0.65,  # High heritability
        0.55, 0.48, 0.42, 0.38,  # Medium heritability
        0.25, 0.18, 0.12, 0.08   # Low heritability
    ]
    
    results = {}
    for trait, h2 in zip(trait_names, h2_values):
        results[trait] = {
            "heritability": h2,
            "variance_components": {
                "genetic": h2 * 100,
                "environmental": (1 - h2) * 100,
                "total": 100
            },
            "confidence_interval": [max(0, h2 - 0.1), min(1, h2 + 0.1)],
            "n_genotypes": 50,
            "n_observations": 150
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
            0.7: ["trait_" + str(i) for i in range(7)]
        }
    }


@pytest.fixture
def heritability_threshold_analysis_empty():
    """Empty threshold analysis for edge case testing."""
    return {
        "thresholds": np.array([]),
        "traits_retained": np.array([]),
        "fraction_retained": np.array([]),
        "total_traits": 0
    }
