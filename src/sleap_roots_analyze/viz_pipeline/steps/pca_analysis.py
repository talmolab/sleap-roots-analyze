"""Step 3: PCA analysis and top feature identification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.pca import (
    perform_pca_analysis,
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
        trait_cols = prev_result.metadata["trait_cols"]

        logger.info(f"Performing PCA on {len(trait_cols)} traits")

        # Perform PCA analysis (only on trait columns)
        pca_results = perform_pca_analysis(
            data[trait_cols],
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

        # Select top features
        top_feature_indices = select_top_features_from_pca(
            loadings=pca_results["loadings"],
            eigenvalues=pca_results["eigenvalues"],
            n_features_total=len(trait_cols),
            n_features_to_select=config.pca.n_top_features,
            method=config.pca.feature_selection_strategy,
        )

        top_features = [trait_cols[i] for i in top_feature_indices]

        logger.info(
            f"Selected {len(top_features)} top features using {config.pca.feature_selection_strategy} method"
        )

        # Save PCA results
        pca_dir = run_dir / "pca"
        pca_dir.mkdir(exist_ok=True)

        # Save PC scores
        pc_scores_df = pd.DataFrame(
            pca_results["transformed_data"],
            index=data.index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        pc_scores_df.to_csv(pca_dir / "pc_scores.csv")

        # Save loadings
        loadings_df = pd.DataFrame(
            pca_results["loadings"],
            index=trait_cols,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        loadings_df.to_csv(pca_dir / "loadings.csv")

        # Save explained variance
        explained_var_df = pd.DataFrame(
            {
                "explained_variance": pca_results["eigenvalues"],
                "explained_variance_ratio": pca_results["explained_variance_ratio"],
                "cumulative_variance_ratio": pca_results["cumulative_variance_ratio"],
            },
            index=[f"PC{i+1}" for i in range(n_components)],
        )
        explained_var_df.to_csv(pca_dir / "explained_variance.csv")

        # Save top features
        top_features_df = pd.DataFrame(top_features)
        top_features_df.to_csv(pca_dir / "top_features.csv", index=False)

        logger.info(f"Saved PCA results to: {pca_dir}")

        # Update metadata
        metadata = {
            **prev_result.metadata,
            "pca_results": pca_results,
            "top_features": top_features,
            "n_pca_components": n_components,
            "pca_explained_variance": explained_var,
        }

        return StepResult(
            data=data,
            metadata=metadata,
        )
