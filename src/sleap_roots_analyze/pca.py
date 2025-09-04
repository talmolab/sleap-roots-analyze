"""Modular PCA analysis functions for consistent use across the codebase."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple, Union


def select_n_components(
    X: np.ndarray,
    explained_variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> int:
    """Determine the optimal number of PCA components.

    Args:
        X: Data array of shape (n_samples, n_features)
        explained_variance_threshold: Cumulative variance threshold for component selection
        n_components: If specified, overrides automatic selection
        random_state: Random state for reproducibility

    Returns:
        Number of components to use
    """
    n_samples, n_features = X.shape

    # Handle edge case: single sample
    if n_samples <= 1:
        return 0  # Can't do PCA with single sample

    if n_components is not None:
        # Use specified number, but ensure it's valid
        return max(1, min(n_components, n_features, n_samples - 1))

    # Auto-select based on explained variance threshold
    max_components = min(n_features, n_samples - 1)

    # Fit PCA with all components to find optimal number
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(X)

    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    if cumulative_variance[-1] >= explained_variance_threshold:
        # Find the minimum number of components needed
        n_components = (
            np.argmax(cumulative_variance >= explained_variance_threshold) + 1
        )
    else:
        # Use all available components
        n_components = max_components

    return max(1, min(n_components, max_components))


def fit_pca(
    X: np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> Tuple[PCA, np.ndarray]:
    """Fit PCA with specified number of components.

    Args:
        X: Data array of shape (n_samples, n_features)
        n_components: Number of components to use
        random_state: Random state for reproducibility

    Returns:
        Tuple of:
            - Fitted PCA object
            - Transformed data (n_samples, n_components)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_transformed = pca.fit_transform(X)
    return pca, X_transformed


def calculate_pca_metrics(
    pca: PCA,
    X_transformed: np.ndarray,
) -> Dict:
    """Calculate comprehensive PCA metrics.

    Args:
        pca: Fitted PCA object
        X_transformed: Transformed data (n_samples, n_components)

    Returns:
        Dictionary containing all PCA metrics
    """
    # Get loadings (eigenvectors) - shape: (n_features, n_components)
    loadings = pca.components_.T

    # Get eigenvalues (explained variance)
    eigenvalues = pca.explained_variance_

    # Calculate explained variance per feature
    explained_variance_per_feature = np.sum(
        (loadings**2) * eigenvalues[np.newaxis, :], axis=1
    )

    # For standardized data, total variance per feature is 1.0
    explained_variance_ratio_per_feature = explained_variance_per_feature

    return {
        "pca": pca,
        "n_components_selected": pca.n_components_,
        "transformed_data": X_transformed,
        "loadings": loadings,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        "total_variance_explained": np.sum(pca.explained_variance_ratio_),
        "explained_variance_per_feature": explained_variance_per_feature,
        "explained_variance_ratio_per_feature": explained_variance_ratio_per_feature,
    }


def perform_pca_with_variance_threshold(
    X: np.ndarray,
    explained_variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """Perform PCA analysis with automatic component selection based on variance threshold.

    This is the legacy function maintained for backward compatibility.
    New code should use perform_pca_analysis() instead.

    Args:
        X: Standardized data array of shape (n_samples, n_features)
        explained_variance_threshold: Cumulative variance threshold for component selection
        n_components: If specified, overrides automatic selection
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing PCA results
    """
    n_samples, n_features = X.shape

    # Select number of components
    n_components_selected = select_n_components(
        X, explained_variance_threshold, n_components, random_state
    )

    # Fit PCA
    pca, X_transformed = fit_pca(X, n_components_selected, random_state)

    # Calculate metrics
    return calculate_pca_metrics(pca, X_transformed)


def calculate_pca_reconstruction_error(
    X_scaled: np.ndarray, pca_result: Dict
) -> np.ndarray:
    """Calculate reconstruction error for each sample.

    Args:
        X_scaled: Standardized original data (n_samples, n_features)
        pca_result: Result dictionary from perform_pca_with_variance_threshold

    Returns:
        Array of reconstruction errors (n_samples,)
    """
    pca = pca_result["pca"]
    X_transformed = pca_result["transformed_data"]

    # Reconstruct the data
    X_reconstructed = pca.inverse_transform(X_transformed)

    # Calculate reconstruction errors (sum of squared errors per sample)
    reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)

    return reconstruction_errors


def calculate_mahalanobis_distances(
    X_transformed: np.ndarray, robust: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Mahalanobis distances in PCA space.

    Args:
        X_transformed: PCA-transformed data (n_samples, n_components)
        robust: If True, use robust covariance estimation

    Returns:
        Tuple of:
            - distances: Mahalanobis distances (n_samples,)
            - mean: Mean of transformed data (n_components,)
            - covariance: Covariance matrix (n_components, n_components)
    """
    # Handle 1D case
    if X_transformed.ndim == 1:
        X_transformed = X_transformed.reshape(-1, 1)
    elif X_transformed.shape[1] == 1:
        # Already proper shape for 1D
        pass

    n_samples, n_features = X_transformed.shape

    if robust:
        from sklearn.covariance import MinCovDet

        robust_cov = MinCovDet(random_state=42).fit(X_transformed)
        mean = robust_cov.location_
        covariance = robust_cov.covariance_
    else:
        mean = np.mean(X_transformed, axis=0)
        if n_features == 1:
            # Special case for 1D
            covariance = np.array([[np.var(X_transformed[:, 0])]])
        else:
            covariance = np.cov(X_transformed, rowvar=False)
            # Ensure it's 2D even for single feature
            if covariance.ndim == 0:
                covariance = np.array([[covariance]])

    # Calculate Mahalanobis distances
    diff = X_transformed - mean
    try:
        if n_features == 1:
            # For 1D, just use standard deviation
            std = np.sqrt(covariance[0, 0])
            if std > 0:
                distances = np.abs(diff[:, 0]) / std
            else:
                distances = np.zeros(n_samples)
        else:
            inv_cov = np.linalg.inv(covariance)
            distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        inv_cov = np.linalg.pinv(covariance)
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))

    return distances, mean, covariance


def perform_pca_analysis(
    data: Union[pd.DataFrame, np.ndarray],
    standardize: bool = True,
    explained_variance_threshold: float = 0.95,
    n_components: Optional[int] = None,
    random_state: int = 42,
) -> Dict:
    """Perform complete PCA analysis pipeline with optional standardization.

    This is the main entry point for PCA analysis in the codebase.

    Args:
        data: Input data as DataFrame or array
        standardize: Whether to standardize the data (default: True)
        explained_variance_threshold: Cumulative variance threshold for component selection
        n_components: If specified, overrides automatic selection
        random_state: Random state for reproducibility

    Returns:
        Dictionary containing:
            - All metrics from calculate_pca_metrics
            - scaler: StandardScaler if standardize=True, else None
            - data_processed: Processed data (standardized or cleaned)
            - feature_names: List of feature names used
    """
    from typing import Union

    # Handle DataFrame input
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Empty DataFrame provided")

        # Remove non-numeric columns
        df_numeric = data.select_dtypes(include=[np.number])

        if df_numeric.empty:
            raise ValueError("No numeric columns found")

        # Drop rows with any NaN values for PCA
        df_numeric = df_numeric.dropna()

        if df_numeric.empty:
            raise ValueError("No valid samples after removing NaN values")

        # Drop columns with zero variance (single sample has zero variance)
        if len(df_numeric) > 1:
            variances = df_numeric.var()
            non_zero_var_cols = variances[variances > 0].index
            df_clean = df_numeric[non_zero_var_cols]

            if df_clean.empty:
                raise ValueError("No numeric columns with non-zero variance found")
        else:
            # Single sample - can't compute variance meaningfully
            df_clean = df_numeric

        feature_names = df_clean.columns.tolist()
        X = df_clean.values
    else:
        # Assume it's already a numpy array
        X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError(f"Input array must be 2D, got shape {X.shape}")

        # Check for zero variance columns
        variances = np.var(X, axis=0)
        non_zero_var_mask = variances > 0
        X = X[:, non_zero_var_mask]

        if X.shape[1] == 0:
            raise ValueError("No columns with non-zero variance found")

        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    # Standardize if requested
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(X)
    else:
        X_processed = X

    # Select number of components
    n_components_selected = select_n_components(
        X_processed, explained_variance_threshold, n_components, random_state
    )

    # Fit PCA
    pca, X_transformed = fit_pca(X_processed, n_components_selected, random_state)

    # Calculate metrics
    result = calculate_pca_metrics(pca, X_transformed)

    # Add additional information
    result.update(
        {
            "scaler": scaler,
            "data_processed": X_processed,
            "feature_names": feature_names,
        }
    )

    return result


def standardize_data(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """Standardize numeric data for PCA analysis.

    Args:
        df: DataFrame with numeric features

    Returns:
        Tuple of:
            - X_scaled: Standardized data array
            - scaler: Fitted StandardScaler
            - df_clean: DataFrame with non-numeric columns removed
    """
    # Remove non-numeric columns
    df_clean = df.select_dtypes(include=[np.number])

    # Drop columns with zero variance
    variances = df_clean.var()
    non_zero_var_cols = variances[variances > 0].index
    df_clean = df_clean[non_zero_var_cols]

    if df_clean.empty:
        raise ValueError("No numeric columns with non-zero variance found")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)

    return X_scaled, scaler, df_clean
