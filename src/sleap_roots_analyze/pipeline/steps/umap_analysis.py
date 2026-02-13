"""Step 4: UMAP dimensionality reduction for trait visualization."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.umap import UMAP_AVAILABLE, perform_umap_analysis

logger = logging.getLogger(__name__)


class UMAPAnalysisStep(BaseStep):
    """Perform UMAP dimensionality reduction for trait visualization.

    This step:
    1. Performs UMAP dimensionality reduction on trait data
    2. Saves UMAP embedding to data/umap/ directory
    3. Passes umap_results to downstream visualization steps

    Input: DataFrame with trait data + PCA results from previous step
    Output: DataFrame (unchanged) + metadata with umap_results
    """

    def __init__(self):
        """Initialize the UMAP analysis step."""
        super().__init__(
            step_name="UMAPAnalysis",
            description="Perform UMAP dimensionality reduction for trait visualization",
        )

    def execute(
        self,
        data: pd.DataFrame,
        config,
        run_dir: Path,
        prev_result: StepResult,
    ) -> StepResult:
        """Execute UMAP analysis.

        Args:
            data: DataFrame with trait data.
            config: Pipeline configuration with umap settings.
            run_dir: Directory for this pipeline run.
            prev_result: Result from previous step (PCAAnalysisStep).

        Returns:
            StepResult with DataFrame and umap_results metadata.
        """
        # Check if UMAP is disabled in config
        if not config.umap.enabled:
            logger.info("UMAP analysis disabled, skipping")
            return StepResult(
                data=data,
                metadata={
                    **prev_result.metadata,
                    "umap_status": "disabled",
                },
            )

        # Check if UMAP is installed
        if not UMAP_AVAILABLE:
            logger.warning(
                "umap-learn package not installed. "
                "Install with: pip install umap-learn"
            )
            return StepResult(
                data=data,
                metadata={
                    **prev_result.metadata,
                    "umap_status": "skipped_not_installed",
                },
            )

        # Get trait columns from previous step metadata
        # (using same pattern as PCAAnalysisStep for compatibility)
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

        logger.info(f"Performing UMAP on {len(trait_cols)} traits")

        # Clean data before UMAP (drop rows with NaN in trait columns)
        # This ensures we have matching indices for embeddings
        data_clean = data[trait_cols].dropna()
        logger.info(f"Using {len(data_clean)} samples after dropping NaN values")

        # Get UMAP parameters from config
        n_neighbors = config.umap.n_neighbors
        min_dist = config.umap.min_dist
        random_state = config.umap.random_state

        # Perform UMAP analysis
        umap_results = perform_umap_analysis(
            df=data_clean,
            feature_cols=trait_cols,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,  # Always 2D for visualization
            random_state=random_state,
        )

        logger.info(
            f"UMAP complete: {len(data_clean)} samples embedded to 2D "
            f"(n_neighbors={umap_results['n_neighbors']}, min_dist={min_dist})"
        )

        # Save UMAP results to data/umap/ directory
        umap_dir = run_dir / "data" / "umap"
        umap_dir.mkdir(parents=True, exist_ok=True)

        # Save embedding as CSV (with sample IDs as index)
        barcode_col = config.columns.barcode
        # Use barcode as index if available, otherwise use original index
        if barcode_col in data.columns:
            index_values = data.loc[data_clean.index, barcode_col].values
        else:
            index_values = data_clean.index
        embedding_df = pd.DataFrame(
            umap_results["embedding"],
            index=index_values,
            columns=["UMAP1", "UMAP2"],
        )
        embedding_df.to_csv(umap_dir / "umap_embedding.csv")

        # Save parameters as JSON
        params = {
            "n_neighbors": umap_results["n_neighbors"],
            "min_dist": min_dist,
            "random_state": random_state,
            "n_samples": len(data_clean),
            "n_traits": len(trait_cols),
        }
        with open(umap_dir / "umap_parameters.json", "w") as f:
            json.dump(params, f, indent=2)

        logger.info(f"Saved UMAP results to: {umap_dir}")

        # Create serializable version of umap_results
        # (exclude sklearn/umap objects that can't be JSON serialized)
        umap_results_serializable = {
            "embedding": umap_results["embedding"],
            "n_neighbors": umap_results["n_neighbors"],
            "min_dist": min_dist,
            "random_state": random_state,
            "n_samples": len(data_clean),
            "clean_indices": data_clean.index.tolist(),
        }

        # Update metadata (preserving all previous metadata)
        metadata = {
            **prev_result.metadata,
            "umap_results": umap_results_serializable,
        }

        return StepResult(
            data=data,
            metadata=metadata,
        )
