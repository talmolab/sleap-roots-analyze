"""Step 12: Merge All Traits.

This step merges root trait data (from QC pipeline) with above-ground phenotype data,
handling duplicate columns and generating processing metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from sleap_roots_analyze.pipeline.config import QCPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult


class MergeAllTraitsStep:
    """Merge root traits with above-ground traits.

    This step performs the merge operation and handles:
    - Different join types (inner, left, right, outer)
    - Duplicate column detection and resolution
    - Output file generation
    - Processing metadata documentation
    """

    def execute(
        self,
        data: Dict[str, pd.DataFrame],
        config: QCPipelineConfig,
        run_dir: Path,
        prev_result: StepResult | None = None,
    ) -> StepResult:
        """Execute Step 12: Merge All Traits.

        Args:
            data: Dictionary with "qc_data" (root traits) and "above_ground" keys.
            config: Pipeline configuration with merge_traits settings.
            run_dir: Directory for outputs.
            prev_result: Result from previous step (unused).

        Returns:
            StepResult containing:
                - data: Merged DataFrame with all traits
                - metadata: Merge statistics and column info

        Raises:
            ValueError: If duplicate columns found and strategy is "fail".
        """
        # Get data
        qc_data = data["qc_data"]
        above_ground = data["above_ground"]

        merge_config = config.root_core.merge_traits
        join_keys = merge_config.join_keys

        # Identify trait columns (non-join-key columns)
        qc_traits = [c for c in qc_data.columns if c not in join_keys]
        ag_traits = [c for c in above_ground.columns if c not in join_keys]

        # Check for duplicate columns
        qc_trait_set = set(qc_traits)
        ag_trait_set = set(ag_traits)
        duplicates = qc_trait_set & ag_trait_set

        if duplicates and merge_config.duplicate_strategy == "fail":
            raise ValueError(
                f"Duplicate columns found between root and above-ground traits: {duplicates}. "
                f"Use duplicate_strategy='skip' or 'suffix' to handle this."
            )

        # Handle duplicates based on strategy
        if duplicates:
            if merge_config.duplicate_strategy == "skip":
                # Drop duplicate columns from above-ground data
                above_ground = above_ground.drop(columns=list(duplicates))
                ag_traits = [c for c in above_ground.columns if c not in join_keys]

            elif merge_config.duplicate_strategy == "suffix":
                # Add suffixes to duplicate columns
                for col in duplicates:
                    qc_data = qc_data.rename(columns={col: f"{col}_root"})
                    above_ground = above_ground.rename(columns={col: f"{col}_ag"})

                # Update trait lists
                qc_traits = [c for c in qc_data.columns if c not in join_keys]
                ag_traits = [c for c in above_ground.columns if c not in join_keys]

        # Perform merge
        merged = pd.merge(
            qc_data,
            above_ground,
            on=join_keys,
            how=merge_config.join_type,
            suffixes=("_root", "_ag"),  # Backup suffixes if pandas adds any
        )

        # Create metadata
        metadata = {
            "merge_type": merge_config.join_type,
            "join_keys": join_keys,
            "num_samples": len(merged),
            "num_root_samples": len(qc_data),
            "num_above_ground_samples": len(above_ground),
            "num_root_traits": len(qc_traits),
            "num_above_ground_traits": len(ag_traits),
            "num_total_traits": len(qc_traits) + len(ag_traits),
            "root_trait_columns": qc_traits,
            "above_ground_trait_columns": ag_traits,
            "duplicate_columns_found": list(duplicates) if duplicates else [],
            "duplicate_strategy_used": merge_config.duplicate_strategy
            if duplicates
            else None,
        }

        # Save merged CSV
        output_path = Path(merge_config.output_path)
        if not output_path.is_absolute():
            output_path = run_dir / output_path

        merged.to_csv(output_path, index=False)

        # Save metadata JSON
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        metadata["output_csv"] = str(output_path)
        metadata["metadata_json"] = str(metadata_path)

        return StepResult(data=merged, metadata=metadata)