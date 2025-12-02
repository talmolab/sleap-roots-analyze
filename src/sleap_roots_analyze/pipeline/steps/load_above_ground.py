"""Step 11: Load Above-Ground Traits.

This step loads above-ground phenotype data and validates that it can be merged
with the root trait data from the QC pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sleap_roots_analyze.pipeline.config import QCPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult


class LoadAboveGroundTraitsStep:
    """Load and validate above-ground trait data for merging.

    This step loads the above-ground phenotype CSV specified in the config and
    validates that:
    - The file exists and can be loaded
    - Required join keys are present (must match those configured in merge_traits.join_keys)
    - No duplicate samples exist (must have exactly one row per join key combination)
    - Data types are appropriate

    IMPORTANT: The above-ground CSV must contain all columns listed in merge_traits.join_keys.
    Common configurations:
    - join_keys: ["Rep", "geno"] - For replicate-level above-ground data (most common)
    - join_keys: ["Plot", "Rep", "geno"] - For plot-level above-ground data

    The loaded data is prepared for merging with root core traits in the next step.
    """

    def execute(
        self,
        data: Any,
        config: QCPipelineConfig,
        run_dir: Path,
        prev_result: StepResult | None = None,
    ) -> StepResult:
        """Execute Step 11: Load Above-Ground Traits.

        Args:
            data: Unused (this step loads from file).
            config: Pipeline configuration with merge_traits settings.
            run_dir: Directory for outputs.
            prev_result: Result from previous step (unused).

        Returns:
            StepResult containing:
                - data: DataFrame with above-ground traits
                - metadata: Loading statistics and validation info

        Raises:
            ValueError: If config is invalid or data validation fails.
            FileNotFoundError: If CSV file doesn't exist.
        """
        # Validate merge_traits config exists
        if config.root_core is None or config.root_core.merge_traits is None:
            raise ValueError(
                "merge_traits configuration is required in root_core config"
            )

        merge_config = config.root_core.merge_traits

        # Load CSV
        csv_path = Path(merge_config.above_ground_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Above-ground traits CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # Validate not empty
        if len(df) == 0:
            raise ValueError("Above-ground CSV is empty")

        # Validate join keys present
        missing_keys = [k for k in merge_config.join_keys if k not in df.columns]
        if missing_keys:
            raise ValueError(
                f"Missing join keys in above-ground CSV: {missing_keys}. "
                f"Available columns: {list(df.columns)}"
            )

        # Check for duplicates on join keys
        duplicates = df[df.duplicated(subset=merge_config.join_keys, keep=False)]
        if len(duplicates) > 0:
            dup_keys = duplicates[merge_config.join_keys].drop_duplicates()
            raise ValueError(
                f"Duplicate samples found in above-ground CSV. "
                f"Duplicate key combinations:\n{dup_keys}"
            )

        # Count trait columns (exclude join keys)
        trait_cols = [c for c in df.columns if c not in merge_config.join_keys]

        # Create metadata
        metadata = {
            "csv_path": str(csv_path),
            "num_samples": len(df),
            "num_traits": len(trait_cols),
            "trait_columns": trait_cols,
            "join_keys": merge_config.join_keys,
            "join_key_unique_counts": {
                k: df[k].nunique() for k in merge_config.join_keys
            },
        }

        return StepResult(data=df, metadata=metadata)
