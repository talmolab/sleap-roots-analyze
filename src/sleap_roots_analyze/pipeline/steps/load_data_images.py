"""Step 1: Load trait data and link images."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from sleap_roots_analyze.data_cleanup import get_trait_columns
from sleap_roots_analyze.data_utils import link_rhizovision_images_to_samples
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult

logger = logging.getLogger(__name__)


class LoadDataAndImagesStep(BaseStep):
    """Load trait data from CSV and optionally link images.

    This step:
    1. Loads trait data from CSV file
    2. Identifies trait columns vs metadata columns
    3. Links images to samples if image_dir is provided
    4. Validates required columns are present

    Input: None (reads from config.data.csv_path)
    Output: DataFrame with trait data + metadata dict containing image paths
    """

    def execute(
        self,
        data: Optional[pd.DataFrame],
        config,
        run_dir: Path,
        prev_result: Optional[StepResult],
    ) -> StepResult:
        """Execute the load data and images step.

        Args:
            data: Not used (None).
            config: Pipeline configuration.
            run_dir: Directory for this pipeline run.
            prev_result: Not used (None - this is the first step).

        Returns:
            StepResult with loaded DataFrame and metadata.
        """
        csv_path = Path(config.data.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        logger.info(f"Loading trait data from: {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")

        # Validate required columns exist
        required_cols = [config.columns.barcode, config.columns.genotype]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Identify trait columns
        trait_cols = get_trait_columns(
            df,
            barcode_col=config.columns.barcode,
            genotype_col=config.columns.genotype,
            replicate_col=config.columns.replicate,
            additional_exclude=(
                config.data.additional_exclude_cols
                if hasattr(config.data, "additional_exclude_cols")
                else None
            ),
        )
        logger.info(f"Identified {len(trait_cols)} trait columns")

        # Apply trait filters if specified
        if config.data.traits_to_include is not None:
            trait_cols = [t for t in trait_cols if t in config.data.traits_to_include]
            logger.info(
                f"Filtered to {len(trait_cols)} traits based on traits_to_include"
            )
        elif config.data.traits_to_exclude:
            trait_cols = [
                t for t in trait_cols if t not in config.data.traits_to_exclude
            ]
            logger.info(
                f"Filtered to {len(trait_cols)} traits after excluding {len(config.data.traits_to_exclude)} traits"
            )

        # Link images if image_dir provided
        image_paths = None
        if config.data.image_dir is not None:
            image_dir = Path(config.data.image_dir)
            if image_dir.exists():
                logger.info(f"Linking images from: {image_dir}")
                image_paths = link_rhizovision_images_to_samples(
                    df,
                    image_dir=image_dir,
                    barcode_col=config.columns.barcode,
                )
                logger.info(f"Linked {len(image_paths)} images to samples")
            else:
                logger.warning(f"Image directory not found: {image_dir}")

        # Prepare metadata
        metadata = {
            "n_samples": len(df),
            "n_traits": len(trait_cols),
            "trait_cols": trait_cols,
            "valid_trait_names": trait_cols,  # Add for compatibility with downstream steps
            "trait_names": trait_cols,  # Add for compatibility with downstream steps
            "barcode_col": config.columns.barcode,
            "genotype_col": config.columns.genotype,
            "replicate_col": config.columns.replicate,
            "n_images_linked": len(image_paths) if image_paths else 0,
        }

        if image_paths:
            metadata["image_paths"] = image_paths

        return StepResult(
            data=df,
            metadata=metadata,
        )
