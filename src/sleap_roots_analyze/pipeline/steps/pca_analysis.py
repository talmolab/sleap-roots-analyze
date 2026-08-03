"""Step 3: PCA analysis and top feature identification."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
    select_n_features_by_variance,
    select_top_features_from_pca,
)
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class PCAAnalysisStep(BaseStep):
    """Perform PCA analysis and identify top contributing features.

    This step:
    1. Performs PCA with optional standardization
    2. Selects top features based on configured strategy
    3. Saves PCA results and feature rankings

    Input: DataFrame with trait data
    Output: DataFrame + metadata with PCA results and top features
    """

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute the PCA analysis step.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration.
            run_dir: Directory for this pipeline run.
            prev_result: Result from previous step (calculate_statistics).

        Returns:
            StepResult with DataFrame and PCA results metadata.
        """
        # Try multiple metadata keys for trait columns (compatibility with different pipeline stages)
        trait_cols = (
            prev_result.metadata.get("trait_names")
            or prev_result.metadata.get("valid_trait_names")
            or prev_result.metadata.get("trait_cols")
        )
        if trait_cols is None:
            raise KeyError(
                "Could not find trait column names in previous step metadata. "
                "Expected 'trait_names', 'valid_trait_names', or 'trait_cols'."
            )

        logger.info(f"Performing PCA on {len(trait_cols)} traits")

        # Clean data before PCA (drop rows with NaN in trait columns)
        # This ensures we have matching indices for PC scores
        data_clean = data[trait_cols].dropna()
        logger.info(f"Using {len(data_clean)} samples after dropping NaN values")

        # Perform PCA analysis (only on trait columns)
        pca_results = perform_pca_analysis(
            data_clean,
            standardize=config.pca.standardize,
            explained_variance_threshold=(
                config.pca.n_components if config.pca.n_components < 1 else None
            ),
            n_components=(
                int(config.pca.n_components) if config.pca.n_components >= 1 else None
            ),
        )

        n_components = pca_results["n_components_selected"]
        explained_var = pca_results["cumulative_variance_ratio"][n_components - 1]
        logger.info(
            f"PCA complete: {n_components} components explain {explained_var:.1%} variance"
        )

        # Use the post-filtering feature names from PCA (zero-variance traits removed)
        feature_names = pca_results["feature_names"]
        excluded_traits = sorted(set(trait_cols) - set(feature_names))

        if excluded_traits:
            logger.info(
                f"Excluded {len(excluded_traits)} zero-variance traits: "
                f"{excluded_traits}"
            )
            if len(excluded_traits) / len(trait_cols) > 0.5:
                warnings.warn(
                    f"{len(excluded_traits)} of {len(trait_cols)} traits "
                    f"({len(excluded_traits) / len(trait_cols):.0%}) have zero variance "
                    f"and were excluded from PCA. This may indicate data quality issues.",
                    UserWarning,
                    stacklevel=2,
                )

        logger.info(f"PCA used {len(feature_names)} of {len(trait_cols)} traits")

        # Resolve n_features_to_select per feature_selection_strategy (issue #206).
        # pc_indices is always explicit, scoped to every retained PC (issue #203) —
        # never relies on select_top_features_from_pca()'s [0, 1] default.
        strategy = config.pca.feature_selection_strategy
        if strategy == "extreme":
            n_features_to_select = 1
            logger.info(
                "feature_selection_strategy='extreme': pca.n_top_features is not "
                "read for this method (always 1 most-positive + 1 most-negative "
                "loading feature per retained PC)."
            )
        elif strategy == "top_variance" and config.pca.n_top_features < 1:
            n_features_to_select = select_n_features_by_variance(
                pca_results["feature_contributions"], config.pca.n_top_features
            )
        else:
            n_features_to_select = int(config.pca.n_top_features)

        # Select top features using filtered feature names
        top_feature_indices = select_top_features_from_pca(
            loadings=pca_results["loadings"],
            eigenvalues=pca_results["eigenvalues"],
            n_features_total=len(feature_names),
            n_features_to_select=n_features_to_select,
            method=strategy,
            pc_indices=list(range(n_components)),
        )

        top_features = [feature_names[i] for i in top_feature_indices]

        logger.info(
            f"Selected {len(top_features)} top features using {config.pca.feature_selection_strategy} method"
        )

        # Save PCA results (to data/pca/ per VIZ-OUTPUT-001)
        pca_dir = run_dir / "data" / "pca"
        pca_dir.mkdir(parents=True, exist_ok=True)

        # Save PC scores (use cleaned data index)
        pc_scores_df = pd.DataFrame(
            pca_results["transformed_data"],
            index=data_clean.index,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
        pc_scores_df.to_csv(pca_dir / "pc_scores.csv")

        # Save loadings (use filtered feature names, not original trait_cols)
        loadings_df = pd.DataFrame(
            pca_results["loadings"],
            index=feature_names,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
        loadings_df.to_csv(pca_dir / "loadings.csv")

        # Save explained variance
        explained_var_df = pd.DataFrame(
            {
                "explained_variance": pca_results["eigenvalues"],
                "explained_variance_ratio": pca_results["explained_variance_ratio"],
                "cumulative_variance_ratio": pca_results["cumulative_variance_ratio"],
            },
            index=[f"PC{i + 1}" for i in range(n_components)],
        )
        explained_var_df.to_csv(pca_dir / "explained_variance.csv")

        # Save top features
        top_features_df = pd.DataFrame(top_features)
        top_features_df.to_csv(pca_dir / "top_features.csv", index=False)

        logger.info(f"Saved PCA results to: {pca_dir}")

        # Create serializable version of pca_results (exclude sklearn PCA object)
        # Downstream steps need transformed_data, loadings, eigenvalues, etc.
        # but not the PCA model object itself
        pca_results_serializable = {k: v for k, v in pca_results.items() if k != "pca"}

        # Update metadata. `trait_names`/`valid_trait_names` are corrected to
        # the post-filtering feature set here, matching the "traits still in
        # play" contract every other filtering step maintains (cleanup_traits,
        # filter_heritability, remove_outliers, detect_outliers) — downstream
        # steps that read `trait_names` must see the traits PCA actually used,
        # not the pre-filter list. `original_trait_names` preserves the
        # pre-filter list for traceability.
        metadata = {
            **prev_result.metadata,
            "pca_results": pca_results_serializable,
            "top_features": top_features,
            "n_pca_components": n_components,
            "pca_explained_variance": explained_var,
            "excluded_zero_variance_traits": excluded_traits,
            "n_traits_after_filtering": len(feature_names),
            "trait_names": feature_names,
            "valid_trait_names": feature_names,
            "original_trait_names": list(trait_cols),
        }

        return StepResult(
            data=data,
            metadata=metadata,
        )
