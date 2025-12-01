"""Tests for MergeAllTraitsStep."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import QCPipelineConfig, RootCoreConfig
from sleap_roots_analyze.pipeline.config.components import MergeTraitsConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.merge_all_traits import MergeAllTraitsStep


@pytest.fixture
def root_trait_data():
    """Create sample root trait data (output from QC pipeline)."""
    data = []
    for plot in [1, 2, 3]:
        for rep in [1, 2]:
            for geno in ["GH_7386", "GH_7420", "Control"]:
                data.append(
                    {
                        "Plot": plot,
                        "Rep": rep,
                        "geno": geno,
                        "Barcode": f"{plot}-{rep}",
                        "RootDW_15cm": np.random.uniform(1.0, 3.0),
                        "RootDW_45cm": np.random.uniform(0.5, 2.0),
                        "RootCount_0cm": np.random.randint(10, 50),
                        "RootCount_5cm": np.random.randint(5, 30),
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def above_ground_data():
    """Create sample above-ground trait data."""
    data = []
    for plot in [1, 2, 3]:
        for rep in [1, 2]:
            for geno in ["GH_7386", "GH_7420", "Control"]:
                data.append(
                    {
                        "Plot": plot,
                        "Rep": rep,
                        "geno": geno,
                        "Height_cm": np.random.uniform(20.0, 40.0),
                        "Biomass_g": np.random.uniform(10.0, 30.0),
                    }
                )
    return pd.DataFrame(data)


@pytest.fixture
def config_with_merge(tmp_path):
    """Create config with merge_traits settings."""
    return QCPipelineConfig(
        pipeline_name="test_merge",
        root_core=RootCoreConfig(
            sources=[],
            merge_traits=MergeTraitsConfig(
                above_ground_csv=str(tmp_path / "above_ground.csv"),
                join_keys=["Plot", "Rep", "geno"],
                join_type="inner",
                duplicate_strategy="fail",
            ),
        ),
    )


def test_merge_inner_join(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test inner join merge (default)."""
    # Mock previous step results
    prev_qc_result = StepResult(data=root_trait_data, metadata={"step": "qc"})
    prev_ag_result = StepResult(data=above_ground_data, metadata={"step": "load_ag"})

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Check result
    assert isinstance(result.data, pd.DataFrame)
    assert len(result.data) == 18  # All samples match

    # Check columns from both sources
    assert "RootDW_15cm" in result.data.columns
    assert "Height_cm" in result.data.columns
    assert "Barcode" in result.data.columns

    # Check metadata
    assert "num_root_traits" in result.metadata
    assert "num_above_ground_traits" in result.metadata
    assert "merge_type" in result.metadata
    assert result.metadata["merge_type"] == "inner"


def test_merge_left_join(root_trait_data, above_ground_data, config_with_merge, tmp_path):
    """Test left join keeps all root samples even if above-ground missing."""
    # Remove one above-ground sample
    above_ground_partial = above_ground_data[
        ~((above_ground_data["Plot"] == 1) & (above_ground_data["Rep"] == 1))
    ]

    config_with_merge.root_core.merge_traits.join_type = "left"

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_partial},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Should keep all 18 root samples
    assert len(result.data) == 18

    # Check NaNs in above-ground traits for missing sample
    missing_samples = result.data[
        (result.data["Plot"] == 1) & (result.data["Rep"] == 1)
    ]
    assert pd.isna(missing_samples["Height_cm"].iloc[0])


def test_merge_duplicate_column_fail(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test that duplicate columns (besides join keys) cause error."""
    # Add duplicate column to above-ground
    above_ground_data["RootDW_15cm"] = 999.0  # Conflicts with root trait

    step = MergeAllTraitsStep()

    with pytest.raises(ValueError, match="Duplicate columns found"):
        step.execute(
            data={"qc_data": root_trait_data, "above_ground": above_ground_data},
            config=config_with_merge,
            run_dir=tmp_path,
        )


def test_merge_duplicate_column_suffix(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test suffix strategy for duplicate columns."""
    # Add duplicate column
    above_ground_data["RootDW_15cm"] = 999.0

    # Change strategy to suffix
    config_with_merge.root_core.merge_traits.duplicate_strategy = "suffix"

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Should have both versions with suffixes
    assert "RootDW_15cm_root" in result.data.columns
    assert "RootDW_15cm_ag" in result.data.columns


def test_merge_duplicate_column_skip(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test skip strategy keeps root version and drops above-ground duplicate."""
    # Add duplicate column
    above_ground_data["RootDW_15cm"] = 999.0

    # Change strategy to skip
    config_with_merge.root_core.merge_traits.duplicate_strategy = "skip"

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Should have root version only
    assert "RootDW_15cm" in result.data.columns
    # Check it's the root value (not 999.0)
    assert result.data["RootDW_15cm"].max() < 999.0


def test_merge_saves_output_csv(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test that merged data is saved to configured output path."""
    output_path = tmp_path / "merged_output.csv"
    config_with_merge.root_core.merge_traits.output_path = str(output_path)

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Check file was created
    assert output_path.exists()

    # Check content matches result
    df_saved = pd.read_csv(output_path)
    assert len(df_saved) == len(result.data)


def test_merge_generates_metadata_json(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test that processing metadata JSON is generated."""
    output_path = tmp_path / "merged_output.csv"
    config_with_merge.root_core.merge_traits.output_path = str(output_path)

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Check metadata JSON exists
    metadata_path = tmp_path / "merged_output_metadata.json"
    assert metadata_path.exists()

    # Check metadata contains key info
    import json

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert "num_samples" in metadata
    assert "num_root_traits" in metadata
    assert "num_above_ground_traits" in metadata
    assert "join_keys" in metadata


def test_merge_outer_join(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test outer join includes all samples from both datasets."""
    # Add extra sample to above-ground
    extra_sample = pd.DataFrame(
        [{"Plot": 99, "Rep": 1, "geno": "Extra", "Height_cm": 25.0}]
    )
    above_ground_extended = pd.concat(
        [above_ground_data, extra_sample], ignore_index=True
    )

    config_with_merge.root_core.merge_traits.join_type = "outer"

    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_extended},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Should have 19 samples (18 + 1 extra)
    assert len(result.data) == 19

    # Check extra sample has NaN root traits
    extra = result.data[result.data["Plot"] == 99]
    assert pd.isna(extra["RootDW_15cm"].iloc[0])


def test_merge_preserves_barcode(
    root_trait_data, above_ground_data, config_with_merge, tmp_path
):
    """Test that Barcode column is preserved from root data."""
    step = MergeAllTraitsStep()
    result = step.execute(
        data={"qc_data": root_trait_data, "above_ground": above_ground_data},
        config=config_with_merge,
        run_dir=tmp_path,
    )

    # Check Barcode exists and has expected values
    assert "Barcode" in result.data.columns
    assert "1-1" in result.data["Barcode"].values
    assert "2-1" in result.data["Barcode"].values
