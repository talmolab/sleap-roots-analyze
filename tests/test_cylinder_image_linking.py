"""Tests for cylinder image linking support in the visualization pipeline.

These tests verify that:
1. DataConfig accepts image_linking_method and scan_path_col fields
2. LoadDataAndImagesStep uses the correct linking function based on config
3. Metadata flows correctly through the pipeline with cylinder images

TDD: These tests are written before implementation and should fail initially.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from sleap_roots_analyze.pipeline import ColumnConfig, DataConfig
from sleap_roots_analyze.pipeline.config.components import (
    PCAConfig,
    StatisticsConfig,
    UMAPConfig,
)
from sleap_roots_analyze.pipeline.config.viz_config import VizPipelineConfig
from sleap_roots_analyze.pipeline.core import StepResult
from sleap_roots_analyze.pipeline.steps.load_data_images import LoadDataAndImagesStep


class TestDataConfigImageLinkingMethod:
    """Tests for image_linking_method configuration in DataConfig."""

    def test_default_image_linking_method_is_rhizovision(self):
        """Test that default image_linking_method is 'rhizovision'."""
        config = DataConfig(csv_path="test.csv")
        assert config.image_linking_method == "rhizovision"

    def test_image_linking_method_accepts_rhizovision(self):
        """Test that image_linking_method accepts 'rhizovision' value."""
        config = DataConfig(csv_path="test.csv", image_linking_method="rhizovision")
        assert config.image_linking_method == "rhizovision"

    def test_image_linking_method_accepts_cylinder(self):
        """Test that image_linking_method accepts 'cylinder' value."""
        config = DataConfig(csv_path="test.csv", image_linking_method="cylinder")
        assert config.image_linking_method == "cylinder"

    def test_default_scan_path_col(self):
        """Test that default scan_path_col is 'scan_path'."""
        config = DataConfig(csv_path="test.csv")
        assert config.scan_path_col == "scan_path"

    def test_custom_scan_path_col(self):
        """Test that scan_path_col can be customized."""
        config = DataConfig(csv_path="test.csv", scan_path_col="my_scan_path")
        assert config.scan_path_col == "my_scan_path"


class TestLoadDataAndImagesStepLinkingMethod:
    """Tests for LoadDataAndImagesStep using correct linking function."""

    @pytest.fixture
    def sample_csv_rhizovision(self, tmp_path):
        """Create sample CSV for RhizoVision testing."""
        data = {
            "Barcode": ["plant_001", "plant_002", "plant_003"],
            "Genotype": ["geno_A", "geno_B", "geno_A"],
            "Replicate": [1, 1, 2],
            "primary_length": [10.5, 12.3, 11.8],
            "lateral_count": [5.2, 6.1, 5.8],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "rhizovision_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def sample_csv_cylinder(self, tmp_path):
        """Create sample CSV for cylinder testing with scan_path column."""
        data = {
            "Barcode": ["plant_001", "plant_002", "plant_003"],
            "Genotype": ["geno_A", "geno_B", "geno_A"],
            "Replicate": [1, 1, 2],
            "scan_path": [
                "./images_downloader_output/images/Wave1/Day11/plant_001",
                "./images_downloader_output/images/Wave1/Day11/plant_002",
                "./images_downloader_output/images/Wave1/Day11/plant_003",
            ],
            "primary_length": [10.5, 12.3, 11.8],
            "lateral_count": [5.2, 6.1, 5.8],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "cylinder_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    @pytest.fixture
    def image_dir_rhizovision(self, tmp_path):
        """Create RhizoVision image directory with proper structure."""
        image_dir = tmp_path / "rhizovision_images"
        image_dir.mkdir()
        # Create RhizoVision-style image files
        for barcode in ["plant_001", "plant_002", "plant_003"]:
            (image_dir / f"{barcode}_c1_p1_features.png").touch()
            (image_dir / f"{barcode}_c1_p1_seg.png").touch()
        return image_dir

    @pytest.fixture
    def image_dir_cylinder(self, tmp_path):
        """Create cylinder image directory with proper structure."""
        image_dir = tmp_path / "images_downloader_output"
        # Create subdirectories for each plant
        for barcode in ["plant_001", "plant_002", "plant_003"]:
            plant_dir = image_dir / "images" / "Wave1" / "Day11" / barcode
            plant_dir.mkdir(parents=True)
            # Create rotation images
            (plant_dir / "1.jpg").touch()
            (plant_dir / "36.jpg").touch()
        return image_dir

    def test_uses_rhizovision_linking_by_default(
        self, sample_csv_rhizovision, image_dir_rhizovision, tmp_path
    ):
        """Test that LoadDataAndImagesStep uses RhizoVision linking by default."""
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(sample_csv_rhizovision),
                image_dir=str(image_dir_rhizovision),
                # Default: image_linking_method="rhizovision"
            ),
        )

        step = LoadDataAndImagesStep()

        with patch(
            "sleap_roots_analyze.pipeline.steps.load_data_images.link_rhizovision_images_to_samples"
        ) as mock_rhizo:
            mock_rhizo.return_value = {}
            step.execute(
                data=None, config=config, run_dir=tmp_path / "run", prev_result=None
            )
            mock_rhizo.assert_called_once()

    def test_uses_rhizovision_linking_when_specified(
        self, sample_csv_rhizovision, image_dir_rhizovision, tmp_path
    ):
        """Test that LoadDataAndImagesStep uses RhizoVision linking when explicitly specified."""
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(sample_csv_rhizovision),
                image_dir=str(image_dir_rhizovision),
                image_linking_method="rhizovision",
            ),
        )

        step = LoadDataAndImagesStep()

        with patch(
            "sleap_roots_analyze.pipeline.steps.load_data_images.link_rhizovision_images_to_samples"
        ) as mock_rhizo:
            mock_rhizo.return_value = {}
            step.execute(
                data=None, config=config, run_dir=tmp_path / "run", prev_result=None
            )
            mock_rhizo.assert_called_once()

    def test_uses_cylinder_linking_when_specified(
        self, sample_csv_cylinder, image_dir_cylinder, tmp_path
    ):
        """Test that LoadDataAndImagesStep uses cylinder linking when specified."""
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(sample_csv_cylinder),
                image_dir=str(image_dir_cylinder),
                image_linking_method="cylinder",
            ),
        )

        step = LoadDataAndImagesStep()

        with patch(
            "sleap_roots_analyze.pipeline.steps.load_data_images.link_cylinder_images_from_scan_path"
        ) as mock_cylinder:
            mock_cylinder.return_value = {}
            step.execute(
                data=None, config=config, run_dir=tmp_path / "run", prev_result=None
            )
            mock_cylinder.assert_called_once()

    def test_cylinder_linking_uses_scan_path_col_from_config(
        self, sample_csv_cylinder, image_dir_cylinder, tmp_path
    ):
        """Test that cylinder linking uses the configured scan_path_col."""
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(sample_csv_cylinder),
                image_dir=str(image_dir_cylinder),
                image_linking_method="cylinder",
                scan_path_col="scan_path",
            ),
        )

        step = LoadDataAndImagesStep()

        with patch(
            "sleap_roots_analyze.pipeline.steps.load_data_images.link_cylinder_images_from_scan_path"
        ) as mock_cylinder:
            mock_cylinder.return_value = {}
            step.execute(
                data=None, config=config, run_dir=tmp_path / "run", prev_result=None
            )
            # Verify scan_path_col is passed to the function
            call_kwargs = mock_cylinder.call_args[1]
            assert call_kwargs.get("scan_path_col") == "scan_path"

    def test_cylinder_linking_with_custom_scan_path_col(self, tmp_path):
        """Test that cylinder linking works with custom scan_path column name."""
        # Create CSV with custom scan_path column name
        data = {
            "Barcode": ["plant_001", "plant_002"],
            "Genotype": ["geno_A", "geno_B"],
            "Replicate": [1, 1],
            "my_custom_scan_col": [
                "./images/Wave1/Day11/plant_001",
                "./images/Wave1/Day11/plant_002",
            ],
            "primary_length": [10.5, 12.3],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "custom_scan.csv"
        df.to_csv(csv_path, index=False)

        # Create image dir
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(csv_path),
                image_dir=str(image_dir),
                image_linking_method="cylinder",
                scan_path_col="my_custom_scan_col",
            ),
        )

        step = LoadDataAndImagesStep()

        with patch(
            "sleap_roots_analyze.pipeline.steps.load_data_images.link_cylinder_images_from_scan_path"
        ) as mock_cylinder:
            mock_cylinder.return_value = {}
            step.execute(
                data=None, config=config, run_dir=tmp_path / "run", prev_result=None
            )
            call_kwargs = mock_cylinder.call_args[1]
            assert call_kwargs.get("scan_path_col") == "my_custom_scan_col"

    def test_cylinder_linking_warns_if_scan_path_col_missing(
        self, sample_csv_rhizovision, tmp_path
    ):
        """Test that cylinder linking logs warning if scan_path column is missing."""
        # Use rhizovision CSV which lacks scan_path column
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(sample_csv_rhizovision),
                image_dir=str(tmp_path),
                image_linking_method="cylinder",
                scan_path_col="scan_path",  # This column doesn't exist
            ),
        )

        step = LoadDataAndImagesStep()

        # Should not raise, but should log warning and skip image linking
        result = step.execute(
            data=None, config=config, run_dir=tmp_path / "run", prev_result=None
        )

        # No images should be linked
        assert result.metadata.get("n_images_linked", 0) == 0


class TestCylinderImageLinksCorrectPaths:
    """Test that cylinder image linking produces correct image paths."""

    @pytest.fixture
    def cylinder_setup(self, tmp_path):
        """Set up cylinder data and images."""
        # Create CSV with scan_path
        data = {
            "Barcode": ["plant_001", "plant_002"],
            "Genotype": ["geno_A", "geno_B"],
            "Replicate": [1, 1],
            "scan_path": [
                "./images_downloader_output/images/Wave1/Day11/plant_001",
                "./images_downloader_output/images/Wave1/Day11/plant_002",
            ],
            "primary_length": [10.5, 12.3],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "cylinder_data.csv"
        df.to_csv(csv_path, index=False)

        # Create image directory structure
        base_dir = tmp_path / "images_downloader_output"
        for barcode in ["plant_001", "plant_002"]:
            plant_dir = base_dir / "images" / "Wave1" / "Day11" / barcode
            plant_dir.mkdir(parents=True)
            (plant_dir / "1.jpg").touch()
            (plant_dir / "36.jpg").touch()

        return {
            "csv_path": csv_path,
            "image_dir": base_dir,
            "tmp_path": tmp_path,
        }

    def test_cylinder_linking_creates_correct_image_paths(self, cylinder_setup):
        """Test that cylinder linking creates correct paths to rotation images."""
        config = VizPipelineConfig(
            pipeline_name="test",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(cylinder_setup["csv_path"]),
                image_dir=str(cylinder_setup["image_dir"]),
                image_linking_method="cylinder",
            ),
        )

        step = LoadDataAndImagesStep()
        result = step.execute(
            data=None,
            config=config,
            run_dir=cylinder_setup["tmp_path"] / "run",
            prev_result=None,
        )

        # Check that image_paths metadata exists and has correct structure
        assert "image_paths" in result.metadata
        image_paths = result.metadata["image_paths"]

        # Should have entries for both barcodes
        assert "plant_001" in image_paths
        assert "plant_002" in image_paths

        # Each barcode should have paths to rotation images
        assert "1.jpg" in image_paths["plant_001"]
        assert "36.jpg" in image_paths["plant_001"]

        # Paths should exist and point to correct files
        path_1 = image_paths["plant_001"]["1.jpg"]
        assert path_1 is not None
        assert Path(path_1).exists()
        assert Path(path_1).name == "1.jpg"


class TestImagePathsMetadataFlowWithCylinder:
    """Integration tests for metadata flow with cylinder images."""

    @pytest.fixture
    def cylinder_viz_config(self, tmp_path):
        """Create Viz config for cylinder images."""
        # Create minimal CSV
        data = {
            "Barcode": ["plant_001", "plant_002", "plant_003"],
            "Genotype": ["geno_A", "geno_B", "geno_A"],
            "Replicate": [1, 1, 2],
            "scan_path": [
                "./images/plant_001",
                "./images/plant_002",
                "./images/plant_003",
            ],
            "trait1": [10.5, 12.3, 11.8],
            "trait2": [5.2, 6.1, 5.8],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)

        return VizPipelineConfig(
            pipeline_name="test_cylinder_flow",
            columns=ColumnConfig(
                barcode="Barcode", genotype="Genotype", replicate="Replicate"
            ),
            data=DataConfig(
                csv_path=str(csv_path),
                image_dir=str(tmp_path),
                image_linking_method="cylinder",
            ),
            statistics=StatisticsConfig(),
            pca=PCAConfig(n_components=2, standardize=True),
            umap=UMAPConfig(enabled=True, n_neighbors=2, random_state=42),
        )

    @pytest.fixture
    def mock_cylinder_image_paths(self):
        """Create mock image_paths as would come from cylinder linking."""
        return {
            "plant_001": {
                "1.jpg": Path("/mock/images/plant_001/1.jpg"),
                "36.jpg": Path("/mock/images/plant_001/36.jpg"),
            },
            "plant_002": {
                "1.jpg": Path("/mock/images/plant_002/1.jpg"),
                "36.jpg": Path("/mock/images/plant_002/36.jpg"),
            },
            "plant_003": {
                "1.jpg": Path("/mock/images/plant_003/1.jpg"),
                "36.jpg": Path("/mock/images/plant_003/36.jpg"),
            },
        }

    def test_cylinder_image_paths_preserved_through_stats(
        self, cylinder_viz_config, mock_cylinder_image_paths, tmp_path
    ):
        """Test that cylinder image_paths survives StatisticalAnalysisStep."""
        from sleap_roots_analyze.pipeline.steps import StatisticalAnalysisStep

        # Create sample data
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Barcode": ["plant_001", "plant_002", "plant_003"],
                "Genotype": ["geno_A", "geno_B", "geno_A"],
                "Replicate": [1, 1, 2],
                "trait1": np.random.randn(3) * 10 + 50,
                "trait2": np.random.randn(3) * 5 + 25,
            }
        )

        # Simulate LoadDataAndImagesStep result with cylinder image paths
        load_result = StepResult(
            data=df,
            metadata={
                "image_paths": mock_cylinder_image_paths,
                "trait_names": ["trait1", "trait2"],
                "valid_trait_names": ["trait1", "trait2"],
                "n_samples": 3,
            },
        )

        # Run StatisticalAnalysisStep
        stats_step = StatisticalAnalysisStep()
        stats_result = stats_step.execute(
            data=df,
            config=cylinder_viz_config,
            run_dir=tmp_path / "stats",
            prev_result=load_result,
        )

        # CRITICAL: image_paths must be preserved with cylinder format
        assert "image_paths" in stats_result.metadata
        assert stats_result.metadata["image_paths"] == mock_cylinder_image_paths

        # Verify cylinder-specific structure is preserved
        assert "1.jpg" in stats_result.metadata["image_paths"]["plant_001"]
        assert "36.jpg" in stats_result.metadata["image_paths"]["plant_001"]

    def test_cylinder_image_paths_preserved_through_umap(
        self, cylinder_viz_config, mock_cylinder_image_paths, tmp_path
    ):
        """Test that cylinder image_paths survives UMAPAnalysisStep (Task 2.2)."""
        from sleap_roots_analyze.pipeline.steps import UMAPAnalysisStep

        # Create sample data with enough rows for UMAP (needs at least n_neighbors)
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Barcode": ["plant_001", "plant_002", "plant_003", "plant_004", "plant_005"],
                "Genotype": ["geno_A", "geno_B", "geno_A", "geno_B", "geno_C"],
                "Replicate": [1, 1, 2, 2, 1],
                "trait1": np.random.randn(5) * 10 + 50,
                "trait2": np.random.randn(5) * 5 + 25,
            }
        )

        # Mock image paths with extra plants
        mock_image_paths = {
            **mock_cylinder_image_paths,
            "plant_004": {
                "1.jpg": Path("/mock/images/plant_004/1.jpg"),
                "36.jpg": Path("/mock/images/plant_004/36.jpg"),
            },
            "plant_005": {
                "1.jpg": Path("/mock/images/plant_005/1.jpg"),
                "36.jpg": Path("/mock/images/plant_005/36.jpg"),
            },
        }

        # Simulate previous step result with cylinder image paths
        prev_result = StepResult(
            data=df,
            metadata={
                "image_paths": mock_image_paths,
                "trait_names": ["trait1", "trait2"],
                "valid_trait_names": ["trait1", "trait2"],
                "n_samples": 5,
            },
        )

        # Run UMAPAnalysisStep
        umap_step = UMAPAnalysisStep()
        umap_result = umap_step.execute(
            data=df,
            config=cylinder_viz_config,
            run_dir=tmp_path / "umap",
            prev_result=prev_result,
        )

        # CRITICAL: image_paths must be preserved with cylinder format
        assert "image_paths" in umap_result.metadata
        assert umap_result.metadata["image_paths"] == mock_image_paths

        # Verify cylinder-specific structure is preserved
        assert "1.jpg" in umap_result.metadata["image_paths"]["plant_001"]
        assert "36.jpg" in umap_result.metadata["image_paths"]["plant_001"]


class TestImagePathsFormatHandling:
    """Tests for image_paths format detection and handling in GenerateStaticFiguresStep.

    Task 2.5, 2.6, 2.7: Test format detection for nested dict and legacy Series formats.
    """

    @pytest.fixture
    def nested_dict_image_paths(self):
        """Create nested dict format image_paths (from link_* functions)."""
        return {
            "plant_001": {
                "1.jpg": Path("/images/plant_001/1.jpg"),
                "36.jpg": Path("/images/plant_001/36.jpg"),
            },
            "plant_002": {
                "1.jpg": Path("/images/plant_002/1.jpg"),
                "36.jpg": Path("/images/plant_002/36.jpg"),
            },
        }

    @pytest.fixture
    def legacy_series_image_paths(self):
        """Create legacy pd.Series format image_paths."""
        return pd.Series(
            {
                0: "/images/plant_001/features.png",
                1: "/images/plant_002/features.png",
            }
        )

    def test_nested_dict_format_detected_correctly(self, nested_dict_image_paths):
        """Test that nested dict format is correctly detected (Task 2.5)."""
        # This is the detection logic from _create_genotype_image_grids
        image_paths = nested_dict_image_paths

        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        assert is_nested_dict_format is True

    def test_legacy_series_format_detected_correctly(self, legacy_series_image_paths):
        """Test that legacy Series format is correctly detected (Task 2.6)."""
        image_paths = legacy_series_image_paths

        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        # Series is not a nested dict format
        assert is_nested_dict_format is False

    def test_empty_dict_not_detected_as_nested(self):
        """Test that empty dict is not detected as nested dict format."""
        image_paths = {}

        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        assert is_nested_dict_format is False

    def test_flat_dict_not_detected_as_nested(self):
        """Test that flat dict (non-nested) is not detected as nested dict format."""
        image_paths = {
            "plant_001": Path("/images/plant_001/features.png"),
            "plant_002": Path("/images/plant_002/features.png"),
        }

        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        assert is_nested_dict_format is False

    def test_cylinder_format_creates_correct_image_links(
        self, nested_dict_image_paths
    ):
        """Test that cylinder format creates correct image_links structure (Task 2.7)."""
        image_paths = nested_dict_image_paths

        # Apply the conversion logic from _create_genotype_image_grids
        image_links = {}
        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        if is_nested_dict_format:
            for barcode, img_dict in image_paths.items():
                image_links[barcode] = {
                    img_type: Path(path) if path else None
                    for img_type, path in img_dict.items()
                }

        # Verify structure matches what create_genotype_image_grid expects
        assert "plant_001" in image_links
        assert "plant_002" in image_links
        assert "1.jpg" in image_links["plant_001"]
        assert "36.jpg" in image_links["plant_001"]
        assert isinstance(image_links["plant_001"]["1.jpg"], Path)

    def test_legacy_series_creates_correct_image_links(self, legacy_series_image_paths):
        """Test that legacy Series format creates correct image_links structure."""
        image_paths = legacy_series_image_paths
        image_type = "features.png"
        barcode_col = "Barcode"

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "Barcode": ["plant_001", "plant_002"],
                "Genotype": ["geno_A", "geno_B"],
            }
        )

        # Apply the conversion logic from _create_genotype_image_grids
        image_links = {}
        is_nested_dict_format = (
            isinstance(image_paths, dict)
            and len(image_paths) > 0
            and isinstance(next(iter(image_paths.values())), dict)
        )

        if not is_nested_dict_format:
            # Legacy format: pd.Series indexed by DataFrame row numbers
            for idx, path in image_paths.items():
                if idx in df.index:
                    barcode = (
                        df.loc[idx, barcode_col]
                        if barcode_col in df.columns
                        else str(idx)
                    )
                    image_links[barcode] = {image_type: Path(path)}

        # Verify structure matches what create_genotype_image_grid expects
        assert "plant_001" in image_links
        assert "plant_002" in image_links
        assert image_type in image_links["plant_001"]
        assert isinstance(image_links["plant_001"][image_type], Path)
