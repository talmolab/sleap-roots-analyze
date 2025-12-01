"""Load root core data from CSV files (Step 0a)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class LoadRootCoreDataStep(BaseStep):
    """Load root core experimental data from CSV files.

    This step loads one or more root core data sources (biomass and/or counting)
    and performs initial validation.

    For each source:
    1. Loads the CSV file
    2. Validates required columns exist
    3. Creates sample identifiers if needed
    4. Saves loaded data

    Outputs:
        - 00a_root_core_biomass_loaded.csv: Loaded biomass data (if present)
        - 00a_root_core_counting_loaded.csv: Loaded counting data (if present)
        - 00a_root_core_metadata.json: Metadata about loaded sources
    """

    def __init__(self):
        """Initialize LoadRootCoreDataStep."""
        super().__init__(
            step_name="LoadRootCoreData",
            description="Load root core data from CSV files",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the root core data loading step.

        Args:
            data: Not used (this is the first root core step).
            config: Pipeline configuration with root_core.sources list.
            run_dir: Directory to save outputs.
            prev_result: Not used (this is the first root core step).

        Returns:
            StepResult with dict of loaded DataFrames keyed by data_type.

        Raises:
            ValueError: If required columns are missing or no sources configured.
        """
        if not config.root_core or not config.root_core.sources:
            raise ValueError("No root core sources configured")

        loaded_data = {}
        files = []
        metadata = {
            "sources": [],
            "total_samples": 0,
        }

        for idx, source in enumerate(config.root_core.sources):
            # Load CSV
            csv_path = Path(source.csv_path)
            if not csv_path.exists():
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            df = pd.read_csv(csv_path)

            # Validate required columns based on data type
            required_cols = ["Plot", "Rep"]
            if source.data_type == "biomass":
                required_cols.append("Core_Replicate")
            elif source.data_type == "counting":
                required_cols.append("core_n")
            else:
                raise ValueError(
                    f"Invalid data_type: {source.data_type}. Must be 'biomass' or 'counting'"
                )

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {csv_path.name}: {missing_cols}"
                )

            # Handle genotype column renaming for standardization
            if source.genotype_column not in df.columns:
                raise ValueError(
                    f"Genotype column '{source.genotype_column}' not found in {csv_path.name}. "
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Rename genotype column to standard 'geno' if it's different
            if source.genotype_column != "geno":
                df = df.rename(columns={source.genotype_column: "geno"})

            # Create sample identifier if not present
            if "sample_id" not in df.columns:
                df = self._create_sample_identifier(df, source.data_type)

            # Store in result dict
            loaded_data[source.data_type] = df

            # Save to file
            filename = f"00a_root_core_{source.data_type}_loaded.csv"
            output_path = self.save_dataframe(df, filename, run_dir)
            files.append(output_path)

            # Track metadata
            metadata["sources"].append(
                {
                    "index": idx,
                    "data_type": source.data_type,
                    "csv_path": csv_path.as_posix(),
                    "samples": len(df),
                    "columns": list(df.columns),
                }
            )
            metadata["total_samples"] += len(df)

        # Save metadata
        metadata_path = self.save_json(metadata, "00a_root_core_metadata.json", run_dir)
        files.append(metadata_path)

        return StepResult(data=loaded_data, metadata=metadata, files_generated=files)

    def _create_sample_identifier(
        self, df: pd.DataFrame, data_type: str
    ) -> pd.DataFrame:
        """Create unique sample identifiers for core-level data.

        Args:
            df: DataFrame with Plot, Rep, geno columns.
            data_type: Type of data ("biomass" or "counting").

        Returns:
            DataFrame with added 'sample_id' column.
        """
        df = df.copy()

        # Determine core column name
        core_col = "Core_Replicate" if data_type == "biomass" else "core_n"

        # Create identifier: plot{Plot}_rep{Rep}_{geno}_core{N}
        # Convert numeric columns to int first to avoid ".0" in identifiers
        df["sample_id"] = (
            "plot"
            + df["Plot"].astype(int).astype(str)
            + "_rep"
            + df["Rep"].astype(int).astype(str)
            + "_"
            + df["geno"].astype(str)
            + "_core"
            + df[core_col].astype(int).astype(str)
        )

        return df
