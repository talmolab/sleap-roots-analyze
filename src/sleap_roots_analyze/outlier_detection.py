"""Outlier detection using Mahalanobis distance on PCA-transformed data."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
    calculate_mahalanobis_distances,
    calculate_pca_metrics,
    build_feature_metrics_df,
    calculate_pca_reconstruction_error,
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
    # Convert to DataFrame to handle indices consistently
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(data.shape[1])])
    else:
        df = data.copy()
    
    # Track indices - PCA will drop NaN rows, we need to track valid indices
    # perform_pca_analysis handles NaN removal internally
    original_indices = df.dropna().index.tolist()

    try:
        # Perform PCA analysis with the specified variance threshold
        # This will handle NaN removal internally
        pca_result = perform_pca_analysis(
            df,
            standardize=standardize,
            explained_variance_threshold=variance_threshold,
            n_components=None,
            random_state=random_state,
        )

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


def detect_outliers_pca(
    data: Union[pd.DataFrame, np.ndarray],
    n_components: Optional[int] = None,
    explained_variance_threshold: float = 0.95,
    outlier_threshold: float = 2.5,
) -> Dict:
    """Detect outliers using PCA reconstruction error.

    Principal Component Analysis (PCA) reduces data dimensionality while preserving
    variance. Outliers are detected by reconstruction error - samples that cannot
    be well-reconstructed from the principal components.

    Reconstruction Error = Σ(X_original - X_reconstructed)²

    Args:
        data: DataFrame with numeric trait data or numpy array
        n_components: Number of PCA components (auto-determined if None)
        explained_variance_threshold: Cumulative variance threshold for auto-selection (0-1)
        outlier_threshold: Threshold for outlier detection (standard deviations)

    Returns:
        Dictionary with outlier detection results including:
        - outlier_indices: List of row indices identified as outliers
        - n_components: Number of components used
        - reconstruction_errors: Per-sample reconstruction errors
        - explained_variance_ratio: Variance explained by each component
        - cumulative_variance: Cumulative variance explained
        - explained_variance_per_feature: Variance explained for each original feature
        - explained_variance_ratio_per_feature: Fraction of each feature's variance explained
        - error: Error message if detection failed
    """
    # Convert to DataFrame to handle indices consistently
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(data.shape[1])])
    else:
        df = data.copy()
    
    # Track indices - PCA will drop NaN rows, we need to track valid indices
    original_indices = df.dropna().index.tolist()

    try:
        # Perform PCA analysis using our simplified API
        pca_result = perform_pca_analysis(
            df,
            standardize=True,  # Always standardize for outlier detection
            explained_variance_threshold=explained_variance_threshold,
            n_components=n_components,
            random_state=42,
        )

        # Get processed data for reconstruction error calculation
        X_processed = pca_result.get("data_processed")
        if X_processed is None:
            # This shouldn't happen with our current implementation
            raise ValueError("Unable to get processed data for reconstruction")

        # Calculate reconstruction errors
        reconstruction_errors = calculate_pca_reconstruction_error(
            X_processed, pca_result
        )

        # Detect outliers using z-score of reconstruction errors
        error_mean = np.mean(reconstruction_errors)
        error_std = np.std(reconstruction_errors)

        if error_std == 0:
            # All samples have the same reconstruction error
            outlier_indices = []
            threshold_value = error_mean
        else:
            # Calculate threshold
            threshold_value = error_mean + outlier_threshold * error_std

            # Identify outliers
            outlier_mask = reconstruction_errors > threshold_value
            outlier_indices = [original_indices[i] for i in np.where(outlier_mask)[0]]

        # Get per-feature metrics using calculate_pca_metrics
        pca_metrics = calculate_pca_metrics(
            pca_result["pca"],
            pca_result["transformed_data"],
            X_fitted=X_processed,
            ddof_for_feature_var=1,
        )

        # Compile results
        result = {
            "method": "PCA",
            "n_components": pca_result["n_components_selected"],
            "explained_variance_ratio": pca_result["explained_variance_ratio"].tolist(),
            "cumulative_variance": pca_result["cumulative_variance_ratio"].tolist(),
            "total_variance_explained": float(
                pca_result["cumulative_variance_ratio"][-1]
            ),
            "explained_variance_threshold": explained_variance_threshold,
            "outlier_threshold": outlier_threshold,
            "threshold_value": float(threshold_value),
            "reconstruction_errors": reconstruction_errors.tolist(),
            "outlier_indices": outlier_indices,
            "n_outliers": len(outlier_indices),
            "pca_components": pca_result["transformed_data"].tolist(),
            "loadings": pca_result["loadings"].tolist(),
            "eigenvalues": pca_result["eigenvalues"].tolist(),
            "feature_names": pca_result["feature_names"],
            "data_indices": original_indices,
            "explained_variance_per_feature": pca_metrics.get(
                "explained_variance_per_feature",
                np.zeros(len(pca_result["feature_names"])),
            ).tolist(),
            "explained_variance_ratio_per_feature": pca_metrics.get(
                "explained_variance_ratio_per_feature",
                np.zeros(len(pca_result["feature_names"])),
            ).tolist(),
        }

        return result

    except Exception as e:
        return {
            "method": "PCA",
            "outlier_indices": [],
            "error": f"PCA reconstruction outlier detection failed: {str(e)}",
        }


def detect_outliers_isolation_forest(
    data: Union[pd.DataFrame, np.ndarray],
    contamination: float = 0.1,
    random_state: int = 42,
) -> Dict:
    """Detect outliers using Isolation Forest.

    Isolation Forest isolates anomalies by randomly selecting features and split values.
    Outliers are data points that require fewer splits to isolate, indicating they are
    different from the majority of the data.

    Anomaly Score = 2^(-E(h(x))/c(n))

    Where E(h(x)) is the average path length of sample x in isolation trees,
    and c(n) is the average path length of unsuccessful search in a BST with n points.

    Args:
        data: DataFrame with numeric trait data or numpy array
        contamination: Expected proportion of outliers (0-0.5)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with outlier detection results including:
        - outlier_indices: List of row indices identified as outliers
        - anomaly_scores: Per-sample anomaly scores (more negative = more anomalous)
        - contamination: Contamination parameter used
        - outlier_labels: -1 for outliers, 1 for inliers
        - data_indices: Original indices of the data
        - error: Error message if detection failed
    """
    # Track original indices
    if isinstance(data, np.ndarray):
        original_indices = list(range(len(data)))
        df = pd.DataFrame(data, columns=[f"Feature_{i}" for i in range(data.shape[1])])
    else:
        original_indices = data.index.tolist()
        df = data.copy()

    # Basic validation
    if df.empty or df.shape[0] == 0:
        return {
            "method": "IsolationForest",
            "outlier_indices": [],
            "error": "Empty data provided",
        }

    if df.isna().any().any():
        return {
            "method": "IsolationForest",
            "outlier_indices": [],
            "error": "Data contains NaN values. Please remove NaN samples before outlier detection.",
        }

    try:
        # Standardize data for consistency with other methods
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,  # Default number of trees
        )
        
        # Fit and predict outliers
        outlier_labels = iso_forest.fit_predict(X_scaled)

        # Get outlier indices (Isolation Forest returns -1 for outliers, 1 for inliers)
        outlier_mask = outlier_labels == -1
        outlier_indices = [original_indices[i] for i in np.where(outlier_mask)[0]]

        # Get anomaly scores (more negative = more anomalous)
        # decision_function returns the opposite of anomaly scores
        anomaly_scores = iso_forest.decision_function(X_scaled)

        return {
            "method": "IsolationForest",
            "contamination": contamination,
            "outlier_indices": outlier_indices,
            "n_outliers": len(outlier_indices),
            "anomaly_scores": anomaly_scores.tolist(),
            "outlier_labels": outlier_labels.tolist(),
            "data_indices": original_indices,
        }
        
    except Exception as e:
        return {
            "method": "IsolationForest",
            "outlier_indices": [],
            "error": f"Isolation Forest outlier detection failed: {str(e)}",
        }
