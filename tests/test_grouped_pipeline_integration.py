"""Integration tests for grouped pipeline execution with real data fixtures.

These tests use actual test data and run real (though minimal) pipelines to ensure
the grouping functionality works end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sleap_roots_analyze.pipeline.config import get_default_qc_config
from sleap_roots_analyze.pipeline.pipelines.qc_pipeline import QCPipeline
from sleap_roots_analyze.pipeline.utils import run_grouped_pipelines


@pytest.mark.integration
def test_grouped_qc_pipeline_with_real_data(traits_summary_csv_path, tmp_path):
    """Test QC pipeline grouping with real test data fixture.

    Uses traits_summary.csv which has plant_age_days column with values [5, 11, 14].
    This is a faster integration test that disables slow steps.
    """
    import pandas as pd

    # Load and subset data to just first 30 samples for faster test
    df = pd.read_csv(traits_summary_csv_path)
    df_subset = df.head(30)
    subset_csv = tmp_path / "subset_data.csv"
    df_subset.to_csv(subset_csv, index=False)

    # Create config with grouping by plant_age_days
    config = get_default_qc_config()
    config.data.csv_path = str(subset_csv)
    config.data.group_by = "plant_age_days"
    config.columns.barcode = "plant_qr_code"
    config.columns.genotype = "accession_id"
    config.columns.replicate = "plant_id"
    config.cleanup.min_samples_per_trait = 2  # Lower threshold for small subset
    config.outlier_detection.traditional_methods = []  # Disable outlier detection for speed
    config.visualization.create_eda_figures = False  # Disable EDA
    config.heritability.enabled = False  # Disable heritability for speed

    # Run grouped pipelines
    results = run_grouped_pipelines(
        config=config,
        output_dir=tmp_path / "qc_grouped",
        pipeline_class=QCPipeline,
        validate=False,  # Skip validation for test
    )

    # With first 30 samples, we should get at least one group (likely just day 5)
    assert isinstance(results, dict)
    assert len(results) >= 1  # At least one group should be processed

    # Get the groups that were actually processed
    processed_groups = list(results.keys())
    assert all(isinstance(g, (int, float)) for g in processed_groups)

    # Check that each processed group has correct output structure
    for group_value in processed_groups:
        group_results = results[group_value]
        assert "pipeline" in group_results
        assert "output_dir" in group_results
        assert "results" in group_results

        # Verify output directory exists and is named correctly
        output_dir = Path(group_results["output_dir"])
        assert output_dir.exists()
        assert f"plant_age_days_{group_value}_" in output_dir.name

        # Verify key output files exist
        assert (output_dir / "config.yaml").exists()
        assert (output_dir / "pipeline_summary.json").exists()

        # Verify 10_final_data.csv exists and has correct group
        final_data_csv = output_dir / "10_final_data.csv"
        assert final_data_csv.exists()

        # Read and verify data
        import pandas as pd
        final_df = pd.read_csv(final_data_csv)
        assert len(final_df) > 0
        # All rows should have the correct plant_age_days value
        if "plant_age_days" in final_df.columns:
            assert all(final_df["plant_age_days"] == group_value)


@pytest.mark.integration
def test_ungrouped_qc_pipeline_with_real_data(traits_summary_csv_path, tmp_path):
    """Test QC pipeline WITHOUT grouping to ensure normal operation still works."""
    import pandas as pd

    # Load and subset data for faster test
    df = pd.read_csv(traits_summary_csv_path)
    df_subset = df.head(30)
    subset_csv = tmp_path / "subset_data.csv"
    df_subset.to_csv(subset_csv, index=False)

    # Create config WITHOUT group_by
    config = get_default_qc_config()
    config.data.csv_path = str(subset_csv)
    # No group_by set
    config.columns.barcode = "plant_qr_code"
    config.columns.genotype = "accession_id"
    config.columns.replicate = "plant_id"
    config.outlier_detection.traditional_methods = []  # Disable for speed
    config.visualization.create_eda_figures = False  # Disable EDA
    config.heritability.enabled = False  # Disable for speed

    # Run single pipeline
    results = run_grouped_pipelines(
        config=config,
        output_dir=tmp_path / "qc_ungrouped",
        pipeline_class=QCPipeline,
        validate=False,
    )

    # Should return single result with None key
    assert isinstance(results, dict)
    assert len(results) == 1
    assert None in results

    # Verify output structure
    output_dir = Path(results[None]["output_dir"])
    assert output_dir.exists()
    assert "plant_age_days" not in output_dir.name  # No group-specific naming

    # Verify outputs exist
    assert (output_dir / "config.yaml").exists()
    assert (output_dir / "pipeline_summary.json").exists()
    assert (output_dir / "10_final_data.csv").exists()
