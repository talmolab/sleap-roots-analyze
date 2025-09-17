"""Outlier detection using Mahalanobis distance on PCA-transformed data."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    build_feature_metrics_df,
)


def detect_outliers_mahalanobis(
    data: Union[pd.DataFrame, np.ndarray],
    standardize: bool = True,
    variance_threshold: float = 0.95,
    use_chi_squared: bool = True,
    chi2_percentile: float = 97.5,
    distance_threshold: Optional[float] = None,
    robust_covariance: bool = False,
    random_state: int = 42,
) -> Dict:
    """Detect outliers using Mahalanobis distance on PCA-transformed data.

    Mahalanobis distance measures how many standard deviations away a point is
    from the mean, accounting for the covariance structure of the data.

    Args:
        data: Input data as DataFrame or array
        standardize: Whether to standardize data before PCA
        variance_threshold: Cumulative variance threshold for PC selection (0-1)
        use_chi_squared: Use chi-squared distribution threshold
        chi2_percentile: Percentile for chi-squared threshold (0-100)
        distance_threshold: Direct Mahalanobis distance threshold (if not using chi-squared)
        robust_covariance: Use robust covariance estimation (MinCovDet)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with outlier detection results including:
        - outlier_indices: List of outlier sample indices
        - mahalanobis_distances: Distance for each sample
        - n_outliers: Number of outliers detected
        - n_components: Number of PCA components used
        - threshold_type: Type of threshold used
        - threshold_value: Threshold value used
        - feature_names: List of feature names
        - error: Error message if detection failed
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(data.shape[1])])
        original_indices = list(range(len(data)))
    else:
        df = data.copy()
        original_indices = df.index.tolist()

    # Check if data is empty
    if df.empty or df.shape[0] == 0:
        return {
            "method": "Mahalanobis",
            "outlier_indices": [],
            "error": "Empty data provided",
        }

    # Check for NaN values
    if df.isna().any().any():
        return {
            "method": "Mahalanobis",
            "outlier_indices": [],
            "error": "Data contains NaN values. Please remove NaN samples before outlier detection.",
        }

    try:
        # Perform PCA analysis with the specified variance threshold
        pca_result = perform_pca_analysis(
            df,
            standardize=standardize,
            explained_variance_threshold=variance_threshold,
            n_components=None,
            random_state=random_state,
        )

        # Check if PCA was successful
        if "error" in pca_result:
            return {
                "method": "Mahalanobis",
                "outlier_indices": [],
                "error": f"PCA failed: {pca_result['error']}",
            }

        # Use all selected components for Mahalanobis distance
        X_pca = pca_result["transformed_data"]
        n_components = pca_result["n_components_selected"]

        # Calculate Mahalanobis distances
        distances, mean_pca, cov_matrix = calculate_mahalanobis_distances(
            X_pca, robust=robust_covariance
        )

        # Calculate threshold
        threshold, threshold_type = calculate_outlier_threshold(
            n_components=n_components,
            use_chi_squared=use_chi_squared,
            chi2_percentile=chi2_percentile,
            distance_threshold=distance_threshold,
        )

        # Identify outliers
        outlier_result = identify_outliers_from_distances(
            distances=distances,
            threshold=threshold,
            threshold_type=threshold_type,
            indices=pd.Index(original_indices),
        )

        # Get per-feature metrics directly from PCA result
        # Since we're using all selected components, we can use the metrics as-is
        pca_metrics = calculate_pca_metrics(
            pca_result["pca"],
            X_pca,
            X_fitted=pca_result.get("data_processed"),
            ddof_for_feature_var=1,
        )

        # Extract the metrics we need (using correct keys from calculate_pca_metrics)
        feature_var_explained = pca_metrics.get(
            "explained_variance_per_feature", np.zeros(len(pca_result["feature_names"]))
        )
        feature_fraction_explained = pca_metrics.get(
            "explained_variance_ratio_per_feature", feature_var_explained
        )

        # Compile results
        result = {
            "method": "Mahalanobis",
            "variance_threshold": variance_threshold,
            "n_components": n_components,
            "cumulative_variance_explained": float(
                pca_result["cumulative_variance_ratio"][-1]
            ),
            "threshold_type": threshold_type,
            "threshold_value": float(threshold),
            "chi2_percentile": chi2_percentile if use_chi_squared else None,
            "distance_threshold": distance_threshold if not use_chi_squared else None,
            "mahalanobis_distances": distances.tolist(),
            "outlier_indices": outlier_result["outlier_indices"],
            "n_outliers": outlier_result["n_outliers"],
            "degrees_of_freedom": n_components,
            "explained_variance_ratio": pca_result["explained_variance_ratio"].tolist(),
            "pca_loadings": pca_result["loadings"].tolist(),
            "eigenvalues": pca_result["eigenvalues"].tolist(),
            "pca_components": X_pca.tolist(),
            "feature_names": pca_result["feature_names"],
            "data_indices": original_indices,
            "feature_variance_explained": feature_var_explained.tolist(),
            "feature_fraction_explained": feature_fraction_explained.tolist(),
            "robust_covariance": robust_covariance,
        }

        return result

    except Exception as e:
        return {
            "method": "Mahalanobis",
            "outlier_indices": [],
            "error": f"Mahalanobis distance calculation failed: {str(e)}",
        }


def calculate_outlier_threshold(
    n_components: int,
    use_chi_squared: bool = True,
    chi2_percentile: float = 97.5,
    distance_threshold: Optional[float] = None,
) -> Tuple[float, str]:
    """Calculate threshold for outlier detection.

    Args:
        n_components: Number of PCA components (degrees of freedom)
        use_chi_squared: Whether to use chi-squared distribution
        chi2_percentile: Percentile for chi-squared threshold (0-100)
        distance_threshold: Direct distance threshold

    Returns:
        Tuple of (threshold value, threshold type string)

    Raises:
        ValueError: If parameters are invalid
    """
    # Validate inputs
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")

    if use_chi_squared:
        if not 0 < chi2_percentile < 100:
            raise ValueError(
                f"chi2_percentile must be between 0 and 100, got {chi2_percentile}"
            )

        # Calculate chi-squared threshold
        threshold = stats.chi2.ppf(chi2_percentile / 100, n_components)
        threshold_type = "chi_squared"
    else:
        # Use direct distance threshold
        if distance_threshold is None:
            # Default to 3 standard deviations
            distance_threshold = 3.0
        elif distance_threshold < 0:
            raise ValueError(
                f"distance_threshold must be non-negative, got {distance_threshold}"
            )

        threshold = distance_threshold
        threshold_type = "distance"

    return float(threshold), threshold_type


def identify_outliers_from_distances(
    distances: np.ndarray,
    threshold: float,
    threshold_type: str = "chi_squared",
    indices: Optional[pd.Index] = None,
) -> Dict:
    """Identify outliers from Mahalanobis distances.

    Args:
        distances: Array of Mahalanobis distances
        threshold: Threshold value for outlier detection
        threshold_type: Type of threshold ("chi_squared" or "distance")
        indices: Optional custom indices for the samples

    Returns:
        Dictionary with:
        - outlier_mask: Boolean mask of outliers
        - outlier_indices: List of outlier indices
        - n_outliers: Number of outliers
    """
    distances = np.asarray(distances)

    # Handle empty distances
    if len(distances) == 0:
        return {
            "outlier_mask": np.array([], dtype=bool),
            "outlier_indices": [],
            "n_outliers": 0,
        }

    # Determine outlier mask based on threshold type
    if threshold_type == "chi_squared":
        # For chi-squared, compare squared distances
        outlier_mask = (distances**2) > threshold
    else:
        # For direct distance, compare distances
        outlier_mask = distances > threshold

    # Get outlier indices
    if indices is not None:
        outlier_indices = indices[outlier_mask].tolist()
    else:
        outlier_indices = np.where(outlier_mask)[0].tolist()

    return {
        "outlier_mask": outlier_mask,
        "outlier_indices": outlier_indices,
        "n_outliers": int(np.sum(outlier_mask)),
    }
