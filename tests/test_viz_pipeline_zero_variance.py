"""Integration tests: VizPipeline with zero-variance traits.

Verifies that the full visualization pipeline runs end-to-end when
the input dataset contains constant (zero-variance) trait columns.
Addresses GitHub Issue #74.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import VizPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult
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


@pytest.fixture
def csv_with_interleaved_zero_variance_traits(tmp_path):
    """CSV with the same trait composition as `csv_with_zero_variance_traits`.

    4 variable + 4 constant traits, but with the constant traits interleaved
    among the variable ones instead of trailing.

    The trailing-only fixture above cannot distinguish "filtered by value"
    from "filtered by trailing-slice coincidence" — this one can, since a
    naive `trait_names[:n_features]` slice of the original column order
    would wrongly keep the first excluded (constant) trait instead of
    raising an error or silently substituting the correct one.
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
        # constant trait first
        "lateral_count_day0": np.full(n_samples, 1.0),
        "primary_root_length": [
            np.random.normal(10 + int(g.split("_")[1]) * 2, 2.0) for g in genotypes
        ],
        "lateral_length_day0": np.full(n_samples, 0.0),
        "lateral_root_density": [
            np.random.normal(3 + int(g.split("_")[1]) * 0.5, 0.8) for g in genotypes
        ],
        "secondary_count_day0": np.full(n_samples, 0.0),
        "network_area": [
            np.random.normal(100 + int(g.split("_")[1]) * 10, 15.0) for g in genotypes
        ],
        "tip_angle_day0": np.full(n_samples, 45.0),
        "root_depth": [
            np.random.normal(8 + int(g.split("_")[1]), 1.5) for g in genotypes
        ],
    }

    df = pd.DataFrame(data)
    csv_path = tmp_path / "test_interleaved_zero_variance.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def viz_config_for_interleaved_zero_variance(csv_with_interleaved_zero_variance_traits):
    """VizPipelineConfig with PCA and UMAP enabled, on the interleaved fixture."""
    config = VizPipelineConfig(pipeline_name="test_interleaved_zero_variance_viz")
    config.data.csv_path = str(csv_with_interleaved_zero_variance_traits)
    config.columns.barcode = "Barcode"
    config.columns.genotype = "Genotype"
    config.columns.replicate = "Replicate"

    config.pca.n_components = 2
    config.pca.standardize = True
    config.pca.n_top_features = 3
    config.pca.feature_selection_strategy = "top_absolute"

    config.statistics.calculate_anova = True
    config.statistics.calculate_heritability = True

    # UMAP enabled here (unlike viz_config_for_zero_variance) to exercise
    # the direct PCA -> UMAP metadata spread end-to-end.
    config.umap.enabled = True
    config.clustering.enabled = False
    config.heritability.enabled = False
    config.interesting_genotypes.enabled = False
    config.interactive_viz.enabled = False
    config.dashboard.enabled = False

    config.static_viz.enabled = True

    return config


class TestVizPipelineInterleavedZeroVariance:
    """Regression tests for Issue #80.

    `PCAAnalysisStep` must update `metadata["trait_names"]` to the
    zero-variance-filtered set, and that correction must reach both
    `UMAPAnalysisStep` (direct metadata spread) and
    `GenerateStaticFiguresStep` (via the orchestrator's PCA-branch
    cherry-pick in `_run_generate_static_figures`).
    """

    @pytest.mark.slow
    def test_umap_receives_pca_filtered_trait_count(
        self, viz_config_for_interleaved_zero_variance, tmp_path
    ):
        """Verify UMAP receives the PCA-filtered trait count, not the original.

        `umap_parameters.json`'s `n_traits` reflects the PCA-filtered count
        (4), not the original count (8), even with interleaved
        zero-variance traits.
        """
        pipeline = VizPipeline(
            viz_config_for_interleaved_zero_variance,
            output_dir=tmp_path / "viz_runs",
            validate=False,
        )
        pipeline.run()

        umap_params_path = pipeline.run_dir / "data" / "umap" / "umap_parameters.json"
        assert umap_params_path.exists()
        with open(umap_params_path) as f:
            params = json.load(f)

        assert params["n_traits"] == 4, (
            f"UMAP should receive the 4 PCA-filtered traits, not the "
            f"original 8. Got n_traits={params['n_traits']}"
        )

    def test_static_figures_receive_pca_filtered_trait_names(
        self, viz_config_for_interleaved_zero_variance, tmp_path
    ):
        """Verify static figures receive PCA's filtered trait_names.

        `GenerateStaticFiguresStep`'s effective `trait_names` must equal
        PCA's filtered `feature_names`, not the pre-PCA list relayed from
        the heritability/aggregation branch.
        """
        pipeline = VizPipeline(
            viz_config_for_interleaved_zero_variance,
            output_dir=tmp_path / "viz_runs",
            validate=False,
        )
        results = pipeline.run()

        pca_result = results["03_pca_analysis"].data
        figures_result = results["09_generate_static_figures"].data

        expected = pca_result.metadata["pca_results"]["feature_names"]
        actual = figures_result.metadata.get("trait_names")

        assert expected == [
            "primary_root_length",
            "lateral_root_density",
            "network_area",
            "root_depth",
        ]
        assert actual == expected, (
            f"Static figures step should receive PCA-filtered trait_names "
            f"{expected}, got {actual} (likely relayed unchanged from the "
            f"pre-PCA heritability/aggregation branch instead)"
        )


class TestGenerateStaticFiguresMetadataMergeGuard:
    """Regression guard for Issue #80's fix.

    The orchestrator's PCA-branch metadata merge must not clobber
    `trait_names` when no PCA task result is available. There is no
    `config.pca.enabled` flag in this codebase (PCA always executes when
    scheduled), so this models the DAG executor never producing a
    `"03_pca_analysis"` result (e.g. after an upstream failure) by calling
    the orchestrator method directly rather than a full run.
    """

    def test_keeps_aggregation_branch_trait_names_when_pca_result_absent(
        self, tmp_path
    ):
        """Guard against overwriting trait_names with a missing value.

        Merging PCA-branch metadata must not overwrite `trait_names` when
        the PCA task result is absent from kwargs.
        """
        config = VizPipelineConfig(pipeline_name="test_pca_result_absent_guard")
        # Skip actual figure generation — GenerateStaticFiguresStep then
        # returns prev_result.metadata verbatim, so we can assert on the
        # merged metadata directly without needing real trait/PCA data.
        config.static_viz.enabled = False

        pipeline = VizPipeline(config, output_dir=tmp_path / "viz_runs", validate=False)

        aggregation_result = StepResult(
            data=pd.DataFrame({"a": [1, 2, 3]}),
            metadata={
                "trait_names": ["trait_a", "trait_b"],
                "valid_trait_names": ["trait_a", "trait_b"],
            },
            files_generated=[],
        )

        class _FakeTaskResult:
            def __init__(self, data):
                self.data = data

        result = pipeline._run_generate_static_figures(
            config=config,
            run_dir=tmp_path / "run",
            logger=logging.getLogger("test_pca_result_absent_guard"),
            **{"08_genotype_aggregation": _FakeTaskResult(aggregation_result)},
            # "03_pca_analysis" and "04_umap_analysis" deliberately omitted
        )

        assert result.metadata["trait_names"] == ["trait_a", "trait_b"]
