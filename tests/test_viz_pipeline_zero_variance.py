"""Integration tests: VizPipeline with zero-variance traits.

Verifies that the full visualization pipeline runs end-to-end when
the input dataset contains constant (zero-variance) trait columns.
Addresses GitHub Issue #74.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import VizPipelineConfig
from sleap_roots_analyze.pipeline.pipelines.viz_pipeline import VizPipeline


@pytest.fixture
def csv_with_zero_variance_traits(tmp_path):
    """Create a CSV file with a mix of variable and constant traits.

    Mimics the Alfalfa GWAS scenario from Issue #74:
    - 4 variable traits with realistic plant phenotyping variation
    - 4 constant traits (zero variance) simulating early-stage measurements
    - 60 samples across 4 genotypes with 5 replicates each
    """
    np.random.seed(42)
    n_genotypes = 4
    n_reps = 5
    n_samples = n_genotypes * n_reps * 3  # 60 samples

    genotypes = []
    reps = []
    for g in range(n_genotypes):
        for r in range(n_reps):
            for _ in range(3):
                genotypes.append(f"GEN_{g:02d}")
                reps.append(r + 1)

    data = {
        "Barcode": [f"S{i:03d}" for i in range(n_samples)],
        "Genotype": genotypes,
        "Replicate": reps,
        # 4 variable traits with genotype-dependent means
        "primary_root_length": [
            np.random.normal(10 + int(g.split("_")[1]) * 2, 2.0) for g in genotypes
        ],
        "lateral_root_density": [
            np.random.normal(3 + int(g.split("_")[1]) * 0.5, 0.8) for g in genotypes
        ],
        "network_area": [
            np.random.normal(100 + int(g.split("_")[1]) * 10, 15.0) for g in genotypes
        ],
        "root_depth": [
            np.random.normal(8 + int(g.split("_")[1]), 1.5) for g in genotypes
        ],
        # 4 constant traits (zero variance) — all seedlings identical
        "lateral_count_day0": np.full(n_samples, 1.0),
        "lateral_length_day0": np.full(n_samples, 0.0),
        "secondary_count_day0": np.full(n_samples, 0.0),
        "tip_angle_day0": np.full(n_samples, 45.0),
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_zero_variance.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def viz_config_for_zero_variance(csv_with_zero_variance_traits):
    """VizPipelineConfig with PCA enabled, optional steps disabled."""
    config = VizPipelineConfig(pipeline_name="test_zero_variance_viz")
    config.data.csv_path = str(csv_with_zero_variance_traits)
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"

    # PCA: request 2 components, top 3 features
    config.pca.n_components = 2
    config.pca.standardize = True
    config.pca.n_top_features = 3
    config.pca.feature_selection_strategy = "top_absolute"

    # Statistics needed for PCA step dependency
    config.statistics.calculate_anova = True
    config.statistics.calculate_heritability = True

    # Disable optional steps to keep test fast and focused
    config.umap.enabled = False
    config.clustering.enabled = False
    config.heritability.enabled = False
    config.interesting_genotypes.enabled = False
    config.interactive_viz.enabled = False
    config.dashboard.enabled = False

    # Enable static viz to test PCA figures don't crash
    config.static_viz.enabled = True

    return config


class TestVizPipelineZeroVariance:
    """Integration tests: VizPipeline with zero-variance traits (Issue #74)."""

    def test_viz_pipeline_completes_with_zero_variance_traits(
        self, viz_config_for_zero_variance, tmp_path
    ):
        """Full viz pipeline succeeds when some traits have zero variance."""
        pipeline = VizPipeline(
            viz_config_for_zero_variance,
            output_dir=tmp_path / "viz_runs",
            validate=False,
        )
        results = pipeline.run()

        # All 12 steps should complete
        assert len(results) == 12
        for step_name, result in results.items():
            assert result.data is not None, f"{step_name} returned no data"

        summary = pipeline.get_summary()
        assert summary.status == "success"

    def test_viz_pipeline_pca_metadata_propagates_to_figures(
        self, viz_config_for_zero_variance, tmp_path
    ):
        """PCA metadata (filtered feature names) propagates correctly to figure step."""
        pipeline = VizPipeline(
            viz_config_for_zero_variance,
            output_dir=tmp_path / "viz_runs",
            validate=False,
        )
        results = pipeline.run()

        # PCA step metadata should have the new fields
        pca_result = results["03_pca_analysis"].data
        assert "excluded_zero_variance_traits" in pca_result.metadata
        assert "n_traits_after_filtering" in pca_result.metadata

        excluded = pca_result.metadata["excluded_zero_variance_traits"]
        n_after = pca_result.metadata["n_traits_after_filtering"]

        assert len(excluded) == 4  # 4 constant traits
        assert n_after == 4  # 4 variable traits remain

        # The figure generation step should have completed successfully
        # (it would have crashed if it received mismatched dimensions)
        figures_result = results["09_generate_static_figures"]
        assert figures_result.data is not None

        # Figures directory should exist and contain PCA plots
        figures_dir = pipeline.run_dir / "figures"
        assert figures_dir.exists()

    def test_viz_pipeline_loadings_csv_dimensions_match(
        self, viz_config_for_zero_variance, tmp_path
    ):
        """Loadings CSV has rows for variable traits only, not all original traits."""
        pipeline = VizPipeline(
            viz_config_for_zero_variance,
            output_dir=tmp_path / "viz_runs",
            validate=False,
        )
        pipeline.run()

        # Find the loadings CSV in the run directory
        loadings_path = pipeline.run_dir / "data" / "pca" / "loadings.csv"
        assert loadings_path.exists(), f"loadings.csv not found at {loadings_path}"

        loadings = pd.read_csv(loadings_path, index_col=0)

        # Should have 4 rows (variable traits), not 8 (all original traits)
        assert loadings.shape[0] == 4, (
            f"Expected 4 rows (variable traits only), got {loadings.shape[0]}. "
            f"Index: {list(loadings.index)}"
        )

        # Index should be the variable trait names
        variable_traits = {
            "primary_root_length",
            "lateral_root_density",
            "network_area",
            "root_depth",
        }
        assert set(loadings.index) == variable_traits

        # Should have 2 columns (PC1, PC2) matching config.pca.n_components
        assert loadings.shape[1] == 2
        assert list(loadings.columns) == ["PC1", "PC2"]
