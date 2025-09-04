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
