"""Centralized pytest fixtures for test data."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


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


# DataFrame fixtures with caching at session scope
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


# Sample data fixtures (returns first N rows for quick testing)
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


# Column name fixtures for validation
@pytest.fixture
def features_columns():
    """Return expected column names for features data."""
    return [
        "File.Name",
        "Region.of.Interest",
        "Median.Number.of.Roots",
        "Maximum.Number.of.Roots",
        "Number.of.Root.Tips",
        "Total.Root.Length.mm",
        "Depth.mm",
        "Maximum.Width.mm",
        "Width-to-Depth.Ratio",
        "Network.Area.mm2",
        "Convex.Area.mm2",
        "Solidity",
        "Lower.Root.Area.mm2",
        "Average.Diameter.mm",
        "Median.Diameter.mm",
        "Maximum.Diameter.mm",
        "Perimeter.mm",
        "Volume.mm3",
        "Surface.Area.mm2",
        "Holes",
        "Average.Hole.Size.mm2",
        "Computation.Time.s",
        "Average.Root.Orientation.deg",
        "Shallow.Angle.Frequency",
        "Medium.Angle.Frequency",
        "Steep.Angle.Frequency",
        "Root.Length.Diameter.Range.1.mm",
        "Root.Length.Diameter.Range.2.mm",
        "Root.Length.Diameter.Range.3.mm",
        "Projected.Area.Diameter.Range.1.mm2",
        "Projected.Area.Diameter.Range.2.mm2",
        "Projected.Area.Diameter.Range.3.mm2",
        "Surface.Area.Diameter.Range.1.mm2",
        "Surface.Area.Diameter.Range.2.mm2",
        "Surface.Area.Diameter.Range.3.mm2",
        "Volume.Diameter.Range.1.mm3",
        "Volume.Diameter.Range.2.mm3",
        "Volume.Diameter.Range.3.mm3",
    ]


@pytest.fixture
def traits_summary_key_columns():
    """Return key column names for traits_summary data."""
    return [
        "scan_id",
        "plant_qr_code",
        "scan_path",
        "scanner_id",
        "species_id",
        "species_name",
        "species_genus",
        "species_species",
        "wave_id",
        "wave_number",
        "wave_name",
        "accession_id",
        "date_scanned",
        "experiment_id",
        "experiment_name",
        "plant_age_days",
        "plant_id",
        "crown_count_mean",
        "crown_lengths_mean_mean",
        "network_length_mean",
    ]


# Mock data fixtures for unit testing
@pytest.fixture
def mock_features_data():
    """Create mock features data for unit testing."""
    return pd.DataFrame(
        {
            "File.Name": ["test1.png", "test2.png", "test3.png"],
            "Region.of.Interest": ["Full", "Full", "Full"],
            "Median.Number.of.Roots": [8, 7, 6],
            "Maximum.Number.of.Roots": [25, 17, 30],
            "Number.of.Root.Tips": [102, 105, 120],
            "Total.Root.Length.mm": [27316.67, 29748.59, 32372.98],
            "Depth.mm": [1924, 2385, 2385],
            "Maximum.Width.mm": [2103, 2044, 2435],
            "Width-to-Depth.Ratio": [1.093, 0.857, 1.021],
            "Network.Area.mm2": [230348, 256798, 285617],
            "Convex.Area.mm2": [2916281, 2894284, 3392015],
            "Solidity": [0.079, 0.089, 0.084],
        }
    )


@pytest.fixture
def mock_traits_summary_data():
    """Create mock traits summary data for unit testing."""
    return pd.DataFrame(
        {
            "scan_id": [10199192, 10199193, 10199194],
            "plant_qr_code": ["08J3VRNDEW", "08J3VRNDEX", "08J3VRNDEY"],
            "species_name": ["Wheat", "Wheat", "Wheat"],
            "species_genus": ["Triticum", "Triticum", "Triticum"],
            "species_species": ["aestivum", "aestivum", "aestivum"],
            "plant_age_days": [11, 11, 11],
            "crown_count_mean": [4.21, 3.89, 4.56],
            "crown_count_median": [4.0, 4.0, 5.0],
            "crown_lengths_mean_mean": [368.98, 342.15, 401.23],
            "network_length_mean": [1095.45, 1023.67, 1198.34],
        }
    )


@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame()


@pytest.fixture
def rng():
    """Random number generator with fixed seed for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_trait_data(rng):
    """Generate sample trait data for testing.

    Returns:
        pd.DataFrame: DataFrame with 100 samples, 5 traits, and genotype/rep columns
    """
    n_samples = 100
    n_traits = 5
    n_genotypes = 5
    n_reps = 4

    # Generate base trait data
    trait_data = rng.randn(n_samples, n_traits)
    trait_names = [f"trait_{i+1}" for i in range(n_traits)]

    # Add genotype effects to some traits
    genotypes = []
    for g in range(n_genotypes):
        genotypes.extend([f"G{g+1}"] * (n_samples // n_genotypes))

    # Add some structure to traits
    for i, geno in enumerate(genotypes):
        geno_idx = int(geno[1:]) - 1
        # Trait 1: strong genotype effect
        trait_data[i, 0] += geno_idx * 2
        # Trait 2: moderate genotype effect
        trait_data[i, 1] += geno_idx * 0.5
        # Traits 3-5: no genotype effect (pure noise)

    # Create DataFrame
    df = pd.DataFrame(trait_data, columns=trait_names)
    df["genotype"] = genotypes
    df["geno"] = genotypes  # Alias for compatibility
    df["rep"] = list(range(1, n_reps + 1)) * (n_samples // n_reps)
    df["Barcode"] = [f"BC{i:04d}" for i in range(n_samples)]

    return df


@pytest.fixture
def nan_data(rng):
    """Generate data with controlled NaN patterns for testing.

    Returns:
        pd.DataFrame: Data with specific NaN patterns
    """
    n_samples = 50
    n_traits = 5

    # Start with complete data
    data = rng.randn(n_samples, n_traits)
    df = pd.DataFrame(data, columns=[f"trait_{i+1}" for i in range(n_traits)])

    # Add metadata
    df["geno"] = [f"G{i%5 + 1}" for i in range(n_samples)]
    df["Barcode"] = [f"BC{i:04d}" for i in range(n_samples)]

    # Insert NaN patterns
    # Samples 0-4: Have NaN in trait_1
    df.loc[0:4, "trait_1"] = np.nan

    # Samples 5-9: Have NaN in multiple traits
    df.loc[5:9, ["trait_2", "trait_3"]] = np.nan

    # Trait_4: Has 30% NaN (15 samples)
    nan_indices = rng.choice(range(n_samples), size=15, replace=False)
    df.loc[nan_indices, "trait_4"] = np.nan

    return df


@pytest.fixture
def zero_inflated_data(rng):
    """Generate data with controlled zero inflation for testing.

    Returns:
        pd.DataFrame: Data with specific zero patterns
    """
    n_samples = 100
    n_traits = 4

    # Generate base data
    data = np.abs(rng.randn(n_samples, n_traits)) * 10

    # Trait 1: 60% zeros (should be removed)
    zero_mask = rng.random(n_samples) < 0.6
    data[zero_mask, 0] = 0

    # Trait 2: 30% zeros (should be kept)
    zero_mask = rng.random(n_samples) < 0.3
    data[zero_mask, 1] = 0

    # Traits 3-4: Few zeros

    df = pd.DataFrame(data, columns=[f"trait_{i+1}" for i in range(n_traits)])
    df["geno"] = [f"G{i%5 + 1}" for i in range(n_samples)]

    return df


@pytest.fixture
def image_data():
    """Generate mock image link data for testing.

    Returns:
        dict: Image links by barcode
    """
    barcodes = [f"BC{i:04d}" for i in range(10)]
    image_links = {}

    for bc in barcodes:
        image_links[bc] = {
            "features.png": f"/path/to/{bc}_features.png",
            "seg.png": f"/path/to/{bc}_seg.png",
        }

    # Add some missing images
    image_links["BC0005"]["seg.png"] = None
    image_links["BC0008"] = {}

    return image_links


@pytest.fixture
def outlier_data(rng):
    """Generate data with known outliers for testing outlier detection.

    Returns:
        tuple: (data, true_outlier_indices)
    """
    n_samples = 100
    n_features = 10
    n_outliers = 5

    # Generate normal data
    data = rng.randn(n_samples, n_features)

    # Add outliers at known positions
    outlier_indices = [10, 25, 50, 75, 90]
    for idx in outlier_indices:
        # Make these samples extreme in multiple dimensions
        data[idx, :] = rng.randn(n_features) * 5 + rng.choice([-10, 10])

    df = pd.DataFrame(data, columns=[f"feature_{i+1}" for i in range(n_features)])

    return df, outlier_indices


@pytest.fixture
def multivariate_normal_data(rng):
    """Generate multivariate normal data with known covariance structure.

    Returns:
        tuple: (data, mean, covariance)
    """
    n_samples = 200
    n_features = 5

    # Create correlation matrix
    A = rng.randn(n_features, n_features)
    cov = np.dot(A, A.T)
    mean = rng.randn(n_features) * 2

    # Generate data
    data = rng.multivariate_normal(mean, cov, n_samples)
    df = pd.DataFrame(data, columns=[f"var_{i+1}" for i in range(n_features)])

    return df, mean, cov


@pytest.fixture
def clustered_data(rng):
    """Generate data with known clusters for K-means testing.

    Returns:
        tuple: (data, true_labels, centers)
    """
    n_samples_per_cluster = 30
    n_clusters = 3
    n_features = 4

    centers = rng.randn(n_clusters, n_features) * 5

    data = []
    labels = []

    for i in range(n_clusters):
        cluster_data = rng.randn(n_samples_per_cluster, n_features) + centers[i]
        data.append(cluster_data)
        labels.extend([i] * n_samples_per_cluster)

    data = np.vstack(data)
    df = pd.DataFrame(data, columns=[f"feat_{i+1}" for i in range(n_features)])

    return df, labels, centers


@pytest.fixture
def heritability_data(rng):
    """Generate data with known heritability for testing.

    Returns:
        tuple: (data, true_h2_values)
    """
    n_genotypes = 10
    n_reps = 5
    n_traits = 3

    # True variance components
    genetic_variances = [4.0, 1.0, 0.1]  # High, moderate, low
    environmental_variance = 1.0

    # True heritabilities
    true_h2 = [g / (g + environmental_variance) for g in genetic_variances]

    data = []
    genotypes = []
    reps = []

    for g in range(n_genotypes):
        # Genetic effects
        genetic_effects = [rng.normal(0, np.sqrt(var)) for var in genetic_variances]

        for r in range(n_reps):
            # Environmental effects
            env_effects = rng.normal(0, np.sqrt(environmental_variance), n_traits)

            # Observed values
            values = np.array(genetic_effects) + env_effects
            data.append(values)
            genotypes.append(f"G{g+1}")
            reps.append(r + 1)

    df = pd.DataFrame(data, columns=[f"trait_{i+1}" for i in range(n_traits)])
    df["geno"] = genotypes
    df["rep"] = reps

    return df, true_h2


@pytest.fixture
def anova_data(rng):
    """Generate data with known group differences for ANOVA testing.

    Returns:
        tuple: (data, expected_results)
    """
    # Three groups with different means
    group_means = [0, 2, 5]
    group_sizes = [20, 20, 20]
    within_group_std = 1.0

    data = []
    groups = []

    for mean, size, group_name in zip(group_means, group_sizes, ["A", "B", "C"]):
        group_data = rng.normal(mean, within_group_std, size)
        data.extend(group_data)
        groups.extend([group_name] * size)

    df = pd.DataFrame({"value": data, "group": groups})

    # Calculate expected F-statistic
    n_groups = len(group_means)
    n_total = sum(group_sizes)

    # Between-group variance
    grand_mean = np.mean(data)
    ssb = sum(
        size * (mean - grand_mean) ** 2 for size, mean in zip(group_sizes, group_means)
    )
    msb = ssb / (n_groups - 1)

    # Within-group variance
    msw = within_group_std**2

    expected_f = msb / msw
    expected_p = 1 - stats.f.cdf(expected_f, n_groups - 1, n_total - n_groups)

    expected_results = {
        "f_statistic": expected_f,
        "p_value": expected_p,
        "significant": expected_p < 0.05,
    }

    return df, expected_results


@pytest.fixture
def extreme_value_data(rng):
    """Generate data with extreme value patterns for robustness testing.

    Returns:
        pd.DataFrame: Data with various extreme patterns
    """
    n_samples = 100

    # Create base data
    data = {
        "normal_trait": rng.normal(100, 15, n_samples),
        "inf_trait": rng.normal(50, 10, n_samples),
        "large_range_trait": rng.normal(1e6, 1e5, n_samples),
        "tiny_values_trait": rng.normal(1e-10, 1e-11, n_samples),
        "constant_trait": np.full(n_samples, 42.0),
        "binary_trait": rng.choice([0, 1], n_samples),
        "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
        "rep": [i % 4 + 1 for i in range(n_samples)],
        "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
    }

    # Add some infinity values
    data["inf_trait"][10] = np.inf
    data["inf_trait"][20] = -np.inf

    return pd.DataFrame(data)


@pytest.fixture
def edge_case_trait_cleanup_data():
    """Generate edge case data for testing trait cleanup functions.

    Returns:
        dict: Multiple datasets with different edge cases
    """
    datasets = {}

    # Case 1: All zeros trait
    n = 50
    datasets["all_zeros"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i%5+1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "all_zero_trait": np.zeros(n),
            "normal_trait": np.random.randn(n),
        }
    )

    # Case 2: All NaN trait
    datasets["all_nan"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i%5+1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "all_nan_trait": np.full(n, np.nan),
            "normal_trait": np.random.randn(n),
        }
    )

    # Case 3: High zeros (>50%)
    high_zero_trait = np.random.randn(n)
    high_zero_trait[:30] = 0  # 60% zeros
    datasets["high_zeros"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i%5+1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "high_zero_trait": high_zero_trait,
            "normal_trait": np.random.randn(n),
        }
    )

    # Case 4: High NaN (>30%)
    high_nan_trait = np.random.randn(n)
    high_nan_trait[:20] = np.nan  # 40% NaN
    datasets["high_nan"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i%5+1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "high_nan_trait": high_nan_trait,
            "normal_trait": np.random.randn(n),
        }
    )

    # Case 5: Insufficient samples (< 10 valid)
    n_small = 8
    datasets["insufficient_samples"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n_small)],
            "geno": [f"G{i%2+1}" for i in range(n_small)],
            "rep": [i % 2 + 1 for i in range(n_small)],
            "small_trait": np.random.randn(n_small),
            "normal_trait": np.random.randn(n_small),
        }
    )

    # Case 6: Mixed problematic traits
    mixed_trait1 = np.random.randn(n)
    mixed_trait1[:15] = 0  # 30% zeros
    mixed_trait1[30:40] = np.nan  # 20% NaN

    mixed_trait2 = np.random.randn(n)
    mixed_trait2[:28] = 0  # 56% zeros (should be removed)

    datasets["mixed_problems"] = pd.DataFrame(
        {
            "Barcode": [f"BC{i:03d}" for i in range(n)],
            "geno": [f"G{i%5+1}" for i in range(n)],
            "rep": [i % 3 + 1 for i in range(n)],
            "borderline_trait": mixed_trait1,  # On the edge, shouldn't be removed
            "bad_trait": mixed_trait2,  # Should be removed
            "good_trait": np.random.randn(n),
        }
    )

    # Case 7: Empty dataframe
    datasets["empty"] = pd.DataFrame()

    # Case 8: Single sample
    datasets["single_sample"] = pd.DataFrame(
        {
            "Barcode": ["BC001"],
            "geno": ["G1"],
            "rep": [1],
            "trait1": [1.0],
            "trait2": [2.0],
        }
    )

    return datasets


@pytest.fixture
def edge_case_outlier_data():
    """Generate edge case data for testing outlier detection.

    Returns:
        dict: Multiple datasets with different outlier patterns
    """
    datasets = {}
    np.random.seed(42)

    # Case 1: No outliers (normal distribution)
    n = 100
    datasets["no_outliers"] = pd.DataFrame(
        {
            "trait1": np.random.normal(0, 1, n),
            "trait2": np.random.normal(5, 2, n),
            "trait3": np.random.normal(-2, 0.5, n),
        }
    )

    # Case 2: Single extreme outlier
    data = np.random.normal(0, 1, n)
    data[50] = 100  # Extreme outlier
    datasets["single_outlier"] = pd.DataFrame(
        {
            "trait1": data,
            "trait2": np.random.normal(0, 1, n),
        }
    )

    # Case 3: Multiple outliers (10%)
    data = np.random.normal(0, 1, n)
    outlier_indices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    for idx in outlier_indices:
        data[idx] = np.random.choice([-10, 10]) + np.random.randn()
    datasets["multiple_outliers"] = pd.DataFrame(
        {
            "trait1": data,
            "trait2": np.random.normal(0, 1, n),
        }
    )

    # Case 4: Bimodal distribution (not outliers, just different groups)
    group1 = np.random.normal(-3, 0.5, n // 2)
    group2 = np.random.normal(3, 0.5, n // 2)
    datasets["bimodal"] = pd.DataFrame(
        {
            "trait1": np.concatenate([group1, group2]),
            "trait2": np.random.normal(0, 1, n),
        }
    )

    # Case 5: Heavy-tailed distribution
    datasets["heavy_tailed"] = pd.DataFrame(
        {
            "trait1": np.random.standard_t(df=3, size=n),  # t-distribution with df=3
            "trait2": np.random.normal(0, 1, n),
        }
    )

    # Case 6: Constant values (no variance)
    datasets["constant"] = pd.DataFrame(
        {
            "trait1": np.full(n, 5.0),
            "trait2": np.full(n, 10.0),
        }
    )

    # Case 7: Near-constant with one different value
    data = np.full(n, 5.0)
    data[50] = 5.1  # Slightly different
    datasets["near_constant"] = pd.DataFrame(
        {
            "trait1": data,
            "trait2": np.full(n, 10.0),
        }
    )

    # Case 8: Correlated outliers (outliers in multiple dimensions)
    base = np.random.normal(0, 1, n)
    data1 = base + np.random.normal(0, 0.1, n)
    data2 = base * 2 + np.random.normal(0, 0.1, n)
    # Add correlated outliers
    for idx in [25, 50, 75]:
        data1[idx] += 10
        data2[idx] += 20
    datasets["correlated_outliers"] = pd.DataFrame(
        {
            "trait1": data1,
            "trait2": data2,
        }
    )

    # Case 9: High-dimensional with few samples
    n_small = 10
    n_features = 20
    datasets["high_dim_few_samples"] = pd.DataFrame(
        np.random.randn(n_small, n_features),
        columns=[f"trait{i+1}" for i in range(n_features)],
    )

    return datasets


@pytest.fixture
def typical_pipeline_data():
    """Generate typical data that would go through the full pipeline.

    Returns:
        pd.DataFrame: Realistic trait data with some issues
    """
    np.random.seed(42)
    n = 150
    n_genotypes = 10
    n_traits = 15

    # Base trait data
    data = {}

    # Metadata columns
    data["Barcode"] = [f"BC{i:04d}" for i in range(n)]
    data["geno"] = [f"G{i%n_genotypes+1}" for i in range(n)]
    data["rep"] = [i % 5 + 1 for i in range(n)]

    # Generate traits with different characteristics
    for i in range(n_traits):
        if i < 10:  # Normal traits
            trait = np.random.normal(50 + i * 5, 10, n)
            # Add genotype effects
            for g in range(n_genotypes):
                mask = np.array(data["geno"]) == f"G{g+1}"
                trait[mask] += np.random.normal(g * 2, 1)
        elif i == 10:  # Trait with some zeros (20%)
            trait = np.random.normal(50, 10, n)
            trait[np.random.choice(n, size=30, replace=False)] = 0
        elif i == 11:  # Trait with some NaN (15%)
            trait = np.random.normal(60, 8, n)
            trait[np.random.choice(n, size=22, replace=False)] = np.nan
        elif i == 12:  # Trait with high zeros (55% - should be removed)
            trait = np.random.normal(40, 5, n)
            trait[np.random.choice(n, size=83, replace=False)] = 0
        elif i == 13:  # Low variance trait
            trait = np.random.normal(100, 0.1, n)
        else:  # Trait with outliers
            trait = np.random.normal(75, 12, n)
            trait[np.random.choice(n, size=5, replace=False)] = np.random.choice(
                [150, -20], size=5
            )

        data[f"trait_{i+1}"] = trait

    df = pd.DataFrame(data)

    # Add a few samples with many NaN values
    for idx in [10, 50, 100]:
        for col in df.columns:
            if col.startswith("trait_") and np.random.rand() > 0.3:
                df.loc[idx, col] = np.nan

    return df


@pytest.fixture
def extreme_value_data(rng):
    """Generate data with extreme value patterns for robustness testing.

    Returns:
        pd.DataFrame: Data with various extreme patterns
    """
    n_samples = 100

    # Create base data
    data = {
        "normal_trait": rng.normal(100, 15, n_samples),
        "inf_trait": rng.normal(50, 10, n_samples),
        "large_range_trait": rng.normal(1e6, 1e5, n_samples),
        "tiny_values_trait": rng.normal(1e-10, 1e-11, n_samples),
        "constant_trait": np.full(n_samples, 42.0),
        "binary_trait": rng.choice([0, 1], n_samples),
        "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
        "rep": [i % 4 + 1 for i in range(n_samples)],
        "Barcode": [f"BC{i:04d}" for i in range(n_samples)],
    }

    # Add some infinity values
    data["inf_trait"][10] = np.inf
    data["inf_trait"][20] = -np.inf

    return pd.DataFrame(data)


@pytest.fixture
def pca_edge_case_data(rng):
    """Generate data with edge cases for PCA analysis.

    Returns:
        dict: Dictionary of edge case datasets
    """
    edge_cases = {}

    # Case 1: Perfect multicollinearity
    n = 50
    base = rng.randn(n)
    edge_cases["perfect_collinearity"] = pd.DataFrame(
        {
            "trait_1": base,
            "trait_2": base * 2,  # Perfect linear relationship
            "trait_3": base * -1.5,  # Another perfect linear relationship
            "trait_4": rng.randn(n),  # Independent
            "geno": [f"G{i%3 + 1}" for i in range(n)],
        }
    )

    # Case 2: Wide data (more features than samples)
    edge_cases["wide_data"] = pd.DataFrame(
        rng.randn(10, 30),  # 10 samples, 30 features
        columns=[f"trait_{i}" for i in range(30)],
    )
    edge_cases["wide_data"]["geno"] = [f"G{i%2 + 1}" for i in range(10)]

    # Case 3: All features have zero variance except one
    edge_cases["single_varying_feature"] = pd.DataFrame(
        {
            "const_1": np.ones(100),
            "const_2": np.zeros(100),
            "const_3": np.full(100, 5.5),
            "varying": rng.randn(100),
            "geno": [f"G{i%5 + 1}" for i in range(100)],
        }
    )

    # Case 4: Extreme correlation structure
    n = 80
    correlation_matrix = np.eye(5)
    correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.999  # Nearly perfect
    correlation_matrix[2, 3] = correlation_matrix[3, 2] = -0.95  # Strong negative
    correlation_matrix[0, 4] = correlation_matrix[4, 0] = 0.01  # Nearly independent

    # Generate correlated data
    mean = np.zeros(5)
    cov = np.diag([1, 1, 1, 1, 1])
    # Convert correlation to covariance (assuming unit variance)
    for i in range(5):
        for j in range(5):
            if i != j:
                cov[i, j] = correlation_matrix[i, j]

    correlated_data = rng.multivariate_normal(mean, cov, n)
    edge_cases["extreme_correlations"] = pd.DataFrame(
        correlated_data, columns=[f"trait_{i}" for i in range(5)]
    )
    edge_cases["extreme_correlations"]["geno"] = [f"G{i%4 + 1}" for i in range(n)]

    return edge_cases


@pytest.fixture
def statistical_distribution_data(rng):
    """Generate data following specific statistical distributions.

    Returns:
        dict: Dictionary of datasets with different distributions
    """
    n_samples = 200
    distributions = {}

    # Normal distribution
    distributions["normal"] = pd.DataFrame(
        {
            "value": rng.normal(100, 15, n_samples),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    # Log-normal distribution
    distributions["lognormal"] = pd.DataFrame(
        {
            "value": rng.lognormal(3, 0.5, n_samples),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    # Exponential distribution
    distributions["exponential"] = pd.DataFrame(
        {
            "value": rng.exponential(scale=10, size=n_samples),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    # Uniform distribution
    distributions["uniform"] = pd.DataFrame(
        {
            "value": rng.uniform(0, 100, n_samples),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    # Bimodal distribution (mixture of two normals)
    mode1 = rng.normal(30, 5, n_samples // 2)
    mode2 = rng.normal(70, 5, n_samples - n_samples // 2)
    distributions["bimodal"] = pd.DataFrame(
        {
            "value": np.concatenate([mode1, mode2]),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    # Heavy-tailed distribution (t-distribution with low df)
    from scipy.stats import t as t_dist

    distributions["heavy_tailed"] = pd.DataFrame(
        {
            "value": t_dist.rvs(
                df=3, loc=50, scale=10, size=n_samples, random_state=rng
            ),
            "geno": [f"G{i%5 + 1}" for i in range(n_samples)],
            "rep": [i % 4 + 1 for i in range(n_samples)],
        }
    )

    return distributions


@pytest.fixture
def temporal_data(rng):
    """Generate data with temporal/batch effects for testing.

    Returns:
        pd.DataFrame: Data with time/batch structure
    """
    n_genotypes = 6
    n_timepoints = 4
    n_reps_per_timepoint = 3

    data = []

    # Add temporal drift
    time_effect = np.array([0, 2, 5, 8])  # Increasing trend

    for t in range(n_timepoints):
        for g in range(n_genotypes):
            genetic_effect = rng.normal(g * 2, 0.5)

            for r in range(n_reps_per_timepoint):
                # Value = genetic + time + noise
                value = genetic_effect + time_effect[t] + rng.normal(0, 1)

                data.append(
                    {
                        "value": value,
                        "geno": f"G{g+1}",
                        "timepoint": f"T{t+1}",
                        "rep": r + 1,
                        "batch": f"B{t//2 + 1}",  # Two timepoints per batch
                    }
                )

    return pd.DataFrame(data)


@pytest.fixture
def interaction_effects_data(rng):
    """Generate data with genotype x environment interactions.

    Returns:
        pd.DataFrame: Data with G×E interactions
    """
    n_genotypes = 8
    n_environments = 3
    n_reps = 5

    # Define main effects
    genotype_effects = rng.normal(0, 2, n_genotypes)
    environment_effects = np.array([-5, 0, 5])  # Low, medium, high stress

    # Define interaction effects (some genotypes respond differently to environments)
    # G1-G3: stable across environments
    # G4-G6: sensitive to environment (large G×E)
    # G7-G8: inverse response to environment
    interaction_matrix = np.zeros((n_genotypes, n_environments))
    interaction_matrix[3:6, :] = rng.normal(0, 3, (3, n_environments))  # Large G×E
    interaction_matrix[6:8, :] = rng.normal(0, 2, (2, n_environments)) * np.array(
        [-1, 0, 1]
    )  # Inverse response

    data = []

    for g in range(n_genotypes):
        for e in range(n_environments):
            for r in range(n_reps):
                # Y = μ + G + E + G×E + ε
                value = (
                    50  # Overall mean
                    + genotype_effects[g]
                    + environment_effects[e]
                    + interaction_matrix[g, e]
                    + rng.normal(0, 1)
                )  # Error

                data.append(
                    {"yield": value, "geno": f"G{g+1}", "env": f"E{e+1}", "rep": r + 1}
                )

    return pd.DataFrame(data)


@pytest.fixture
def heritability_balanced_data(rng):
    """Generate perfectly balanced data for heritability testing.

    Returns:
        tuple: (data, expected_heritability_values)
    """
    n_genotypes = 8
    n_reps = 6  # Same number of reps for all genotypes

    # Define traits with different heritability levels
    trait_configs = [
        {"name": "high_h2", "var_g": 10.0, "var_e": 2.0},  # H² ≈ 0.83
        {"name": "moderate_h2", "var_g": 5.0, "var_e": 5.0},  # H² = 0.50
        {"name": "low_h2", "var_g": 1.0, "var_e": 9.0},  # H² = 0.10
        {"name": "zero_h2", "var_g": 0.0, "var_e": 5.0},  # H² = 0.00
    ]

    data = []
    genotypes = []
    reps = []

    for g in range(n_genotypes):
        # Generate genetic effects for this genotype
        genetic_effects = {
            config["name"]: rng.normal(0, np.sqrt(config["var_g"]))
            for config in trait_configs
        }

        for r in range(n_reps):
            row_data = {}
            for config in trait_configs:
                # Add environmental noise
                env_effect = rng.normal(0, np.sqrt(config["var_e"]))
                row_data[config["name"]] = genetic_effects[config["name"]] + env_effect

            data.append(row_data)
            genotypes.append(f"G{g+1}")
            reps.append(r + 1)

    df = pd.DataFrame(data)
    df["geno"] = genotypes
    df["rep"] = reps

    # Calculate expected heritabilities using the formula
    expected_h2 = {}
    for config in trait_configs:
        var_g = config["var_g"]
        var_e = config["var_e"]
        # H² = σ²_G / (σ²_G + σ²_E / n_reps)
        expected_h2[config["name"]] = var_g / (var_g + var_e / n_reps)

    return df, expected_h2


@pytest.fixture
def heritability_unbalanced_data(rng):
    """Generate unbalanced data for heritability testing.

    Returns:
        tuple: (data, expected_heritability_values, rep_counts)
    """
    # Unbalanced design: different numbers of reps per genotype
    genotype_configs = [
        {"name": "G1", "n_reps": 10},
        {"name": "G2", "n_reps": 8},
        {"name": "G3", "n_reps": 6},
        {"name": "G4", "n_reps": 4},
        {"name": "G5", "n_reps": 3},
        {"name": "G6", "n_reps": 2},
        {"name": "G7", "n_reps": 1},
        {"name": "G8", "n_reps": 12},
    ]

    # Single trait with known variance components
    var_g = 8.0
    var_e = 4.0

    data = []
    genotypes = []
    reps = []
    rep_counts = {}

    for geno_config in genotype_configs:
        geno_name = geno_config["name"]
        n_reps = geno_config["n_reps"]
        rep_counts[geno_name] = n_reps

        # Genetic effect for this genotype
        genetic_effect = rng.normal(0, np.sqrt(var_g))

        for r in range(n_reps):
            # Environmental effect
            env_effect = rng.normal(0, np.sqrt(var_e))
            value = genetic_effect + env_effect

            data.append({"trait": value})
            genotypes.append(geno_name)
            reps.append(r + 1)

    df = pd.DataFrame(data)
    df["geno"] = genotypes
    df["rep"] = reps

    # Calculate mean number of reps
    mean_n_reps = np.mean(list(rep_counts.values()))

    # Expected heritability with unbalanced design
    expected_h2 = var_g / (var_g + var_e / mean_n_reps)

    return df, expected_h2, rep_counts


@pytest.fixture
def heritability_edge_cases(rng):
    """Generate edge case data for heritability testing.

    Returns:
        dict: Dictionary of edge case DataFrames
    """
    edge_cases = {}

    # Case 1: All genotypes have identical values (no genetic variance)
    n_geno = 5
    n_rep = 4
    identical_data = []
    for g in range(n_geno):
        for r in range(n_rep):
            identical_data.append(
                {"trait": 10.0, "geno": f"G{g+1}", "rep": r + 1}  # All identical
            )
    edge_cases["identical_values"] = pd.DataFrame(identical_data)

    # Case 2: Single replicate per genotype
    single_rep_data = []
    for g in range(10):
        single_rep_data.append(
            {"trait": rng.normal(g, 1), "geno": f"G{g+1}", "rep": 1}  # Different means
        )
    edge_cases["single_replicate"] = pd.DataFrame(single_rep_data)

    # Case 3: Very small dataset (minimum viable)
    edge_cases["minimal_data"] = pd.DataFrame(
        {
            "trait": [1.0, 1.1, 2.0, 2.1],
            "geno": ["G1", "G1", "G2", "G2"],
            "rep": [1, 2, 1, 2],
        }
    )

    # Case 4: Large environmental variance, small genetic variance
    large_env_data = []
    for g in range(5):
        genetic_effect = rng.normal(0, 0.1)  # Small genetic variance
        for r in range(10):
            env_effect = rng.normal(0, 10)  # Large environmental variance
            large_env_data.append(
                {"trait": genetic_effect + env_effect, "geno": f"G{g+1}", "rep": r + 1}
            )
    edge_cases["large_environmental"] = pd.DataFrame(large_env_data)

    # Case 5: Missing data patterns
    missing_data = []
    for g in range(6):
        for r in range(5):
            value = rng.normal(g, 1)
            # Introduce missing values in a pattern
            if (g == 2 and r > 2) or (g == 4 and r < 2):
                value = np.nan
            missing_data.append({"trait": value, "geno": f"G{g+1}", "rep": r + 1})
    edge_cases["missing_values"] = pd.DataFrame(missing_data)

    # Case 6: Extreme outliers
    outlier_data = []
    for g in range(5):
        genetic_effect = rng.normal(0, 1)
        for r in range(6):
            if g == 2 and r == 3:
                value = 1000  # Extreme outlier
            else:
                value = genetic_effect + rng.normal(0, 1)
            outlier_data.append({"trait": value, "geno": f"G{g+1}", "rep": r + 1})
    edge_cases["extreme_outliers"] = pd.DataFrame(outlier_data)

    return edge_cases


@pytest.fixture
def heritability_mixed_model_data(rng):
    """Generate data specifically for testing mixed model implementation.

    Returns:
        tuple: (data, variance_components, expected_results)
    """
    # Parameters matching those used in mixed model literature
    n_genotypes = 15
    n_blocks = 3  # Could represent different environments/blocks
    reps_per_block = [3, 4, 2]  # Unbalanced within blocks

    # True variance components
    var_g = 12.0  # Genetic variance
    var_e = 6.0  # Residual variance
    var_block = 2.0  # Block effect variance (fixed in our model)

    data = []
    genotypes = []
    reps = []
    blocks = []

    # Block effects (fixed)
    block_effects = rng.normal(0, np.sqrt(var_block), n_blocks)

    for g in range(n_genotypes):
        # Random genetic effect
        genetic_effect = rng.normal(0, np.sqrt(var_g))

        for b in range(n_blocks):
            for r in range(reps_per_block[b]):
                # Phenotype = overall mean + block effect + genetic effect + residual
                residual = rng.normal(0, np.sqrt(var_e))
                value = 50 + block_effects[b] + genetic_effect + residual

                data.append(
                    {
                        "trait": value,
                        "geno": f"G{g+1}",
                        "rep": r + 1,
                        "block": f"B{b+1}",
                    }
                )

    df = pd.DataFrame(data)

    # Calculate expected heritability
    total_reps = sum(reps_per_block)
    mean_reps = total_reps / n_blocks
    expected_h2 = var_g / (var_g + var_e / mean_reps)

    variance_components = {
        "var_genetic": var_g,
        "var_residual": var_e,
        "var_block": var_block,
        "mean_reps": mean_reps,
    }

    expected_results = {
        "heritability": expected_h2,
        "n_genotypes": n_genotypes,
        "n_observations": len(df),
    }

    return df, variance_components, expected_results


@pytest.fixture
def heritability_comparison_data(rng):
    """Generate data for comparing heritability calculation methods.

    Returns:
        dict: Dictionary with different trait types for method comparison
    """
    n_genotypes = 12
    n_reps = 5

    comparison_data = {}

    # Trait 1: High heritability, normal distribution
    data1 = []
    for g in range(n_genotypes):
        genetic_effect = rng.normal(0, 3)
        for r in range(n_reps):
            env_effect = rng.normal(0, 1)
            data1.append(
                {"value": genetic_effect + env_effect, "geno": f"G{g+1}", "rep": r + 1}
            )
    comparison_data["normal_high_h2"] = pd.DataFrame(data1)

    # Trait 2: Low heritability, skewed distribution
    data2 = []
    for g in range(n_genotypes):
        genetic_effect = rng.gamma(2, 0.5) - 1  # Skewed genetic effects
        for r in range(n_reps):
            env_effect = rng.gamma(5, 1) - 5  # Skewed environmental effects
            data2.append(
                {"value": genetic_effect + env_effect, "geno": f"G{g+1}", "rep": r + 1}
            )
    comparison_data["skewed_low_h2"] = pd.DataFrame(data2)

    # Trait 3: Bimodal distribution
    data3 = []
    for g in range(n_genotypes):
        # Half genotypes from one mode, half from another
        if g < n_genotypes // 2:
            genetic_effect = rng.normal(-5, 1)
        else:
            genetic_effect = rng.normal(5, 1)

        for r in range(n_reps):
            env_effect = rng.normal(0, 2)
            data3.append(
                {"value": genetic_effect + env_effect, "geno": f"G{g+1}", "rep": r + 1}
            )
    comparison_data["bimodal"] = pd.DataFrame(data3)

    # Trait 4: Count data (Poisson-like)
    data4 = []
    for g in range(n_genotypes):
        genetic_lambda = np.exp(rng.normal(2, 0.5))  # Log-normal genetic effects
        for r in range(n_reps):
            # Poisson with overdispersion
            value = rng.poisson(genetic_lambda * rng.gamma(2, 0.5))
            data4.append({"value": value, "geno": f"G{g+1}", "rep": r + 1})
    comparison_data["count_data"] = pd.DataFrame(data4)

    return comparison_data
