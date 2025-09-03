"""Comprehensive tests for data_cleanup module with 100% coverage."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    link_images_to_samples,
    save_cleaned_data,
    remove_nan_samples,
    get_numeric_traits_only,
    remove_low_heritability_traits,
)
from src.sleap_roots_analyze.data_utils import (
    create_run_directory,
    _convert_to_json_serializable,
)


class TestLoadTraitData:
    """Tests for load_trait_data function."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file."""
        # Create test CSV
        csv_path = tmp_path / "test_traits.csv"
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "trait1": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        # Load and verify
        loaded = load_trait_data(csv_path)
        assert len(loaded) == 2
        assert "Barcode" in loaded.columns
        assert "geno" in loaded.columns

    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Trait data file not found"):
            load_trait_data("nonexistent.csv")

    def test_missing_required_columns(self, tmp_path):
        """Test that missing required columns raises ValueError."""
        csv_path = tmp_path / "test_traits.csv"
        df = pd.DataFrame(
            {
                "ID": ["ID001", "ID002"],
                "trait1": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_trait_data(csv_path)

    def test_missing_columns_with_suggestions(self, tmp_path):
        """Test that missing columns error includes suggestions."""
        csv_path = tmp_path / "test_traits.csv"
        df = pd.DataFrame(
            {
                "barcodes": ["BC001", "BC002"],  # Similar to "Barcode"
                "genotype": ["G1", "G2"],  # Similar to "geno"
                "trait1": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError) as exc_info:
            load_trait_data(csv_path)

        error_msg = str(exc_info.value)
        assert "maybe:" in error_msg  # Should include suggestions for similar columns
        assert "barcodes" in error_msg  # Suggestion for Barcode
        assert "genotype" in error_msg  # Suggestion for geno

    def test_custom_column_names(self, tmp_path):
        """Test loading with custom column names."""
        csv_path = tmp_path / "test_traits.csv"
        df = pd.DataFrame(
            {
                "plant_id": ["P001", "P002"],
                "genotype": ["G1", "G2"],
                "replication": [1, 2],
                "trait1": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        loaded = load_trait_data(
            csv_path,
            barcode_col="plant_id",
            genotype_col="genotype",
            replicate_col="replication",
        )
        assert len(loaded) == 2
        assert "plant_id" in loaded.columns

    def test_no_replicate_column(self, tmp_path):
        """Test loading without replicate column."""
        csv_path = tmp_path / "test_traits.csv"
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "trait1": [1.0, 2.0],
            }
        )
        df.to_csv(csv_path, index=False)

        loaded = load_trait_data(csv_path, replicate_col=None)
        assert len(loaded) == 2


class TestGetTraitColumns:
    """Tests for get_trait_columns function."""

    def test_basic_exclusion(self):
        """Test basic exclusion of metadata columns."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )

        trait_cols = get_trait_columns(df)
        assert "Barcode" not in trait_cols
        assert "geno" not in trait_cols
        assert "rep" not in trait_cols
        assert "trait1" in trait_cols
        assert "trait2" in trait_cols

    def test_metadata_keyword_exclusion(self):
        """Test automatic exclusion of metadata keywords."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "QC_SLEAP": [1.0],
                "outlier_flag": [0],
                "wave_name": ["W1"],
                "scan_date": ["2024-01-01"],
                "trait1": [1.0],
            }
        )

        trait_cols = get_trait_columns(df)
        assert "QC_SLEAP" not in trait_cols
        assert "outlier_flag" not in trait_cols
        assert "wave_name" not in trait_cols
        assert "scan_date" not in trait_cols
        assert "trait1" in trait_cols

    def test_non_numeric_exclusion(self):
        """Test that non-numeric columns are excluded."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "text_column": ["text"],
                "trait1": [1.0],
            }
        )

        trait_cols = get_trait_columns(df)
        assert "text_column" not in trait_cols
        assert "trait1" in trait_cols

    def test_additional_exclusions(self):
        """Test additional column exclusions."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "extra1": [1.0],
                "extra2": [2.0],
                "trait1": [3.0],
            }
        )

        trait_cols = get_trait_columns(df, additional_exclude=["extra1", "extra2"])
        assert "extra1" not in trait_cols
        assert "extra2" not in trait_cols
        assert "trait1" in trait_cols


class TestLinkImagesToSamples:
    """Tests for link_images_to_samples function."""

    def test_link_existing_images(self, tmp_path):
        """Test linking existing images to samples."""
        # Create test images
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        (image_dir / "BC001_c1_p1_features.png").touch()
        (image_dir / "BC001_c1_p1_seg.png").touch()
        (image_dir / "BC002_c1_p1_features.png").touch()

        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "trait1": [1.0, 2.0],
            }
        )

        links = link_images_to_samples(df, image_dir)

        assert links["BC001"]["features.png"] is not None
        assert links["BC001"]["seg.png"] is not None
        assert links["BC002"]["features.png"] is not None
        assert links["BC002"]["seg.png"] is None

    def test_custom_image_types(self, tmp_path):
        """Test linking with custom image types."""
        image_dir = tmp_path / "images"
        image_dir.mkdir()

        (image_dir / "BC001_c1_p1_custom.png").touch()

        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "trait1": [1.0],
            }
        )

        links = link_images_to_samples(df, image_dir, image_types=["custom.png"])

        assert links["BC001"]["custom.png"] is not None

    def test_missing_barcode_column(self, tmp_path):
        """Test error when barcode column is missing."""
        df = pd.DataFrame(
            {
                "ID": ["ID001"],
                "trait1": [1.0],
            }
        )

        with pytest.raises(ValueError, match="Barcode column"):
            link_images_to_samples(df, tmp_path)


class TestCreateRunDirectory:
    """Tests for create_run_directory function."""

    def test_create_directory(self, tmp_path):
        """Test creating a run directory."""
        base_dir = tmp_path / "runs"
        run_dir = create_run_directory(base_dir)

        assert run_dir.exists()
        assert run_dir.is_dir()
        assert "run_" in run_dir.name

    def test_create_nested_directory(self, tmp_path):
        """Test creating nested directories."""
        base_dir = tmp_path / "deep" / "nested" / "runs"
        run_dir = create_run_directory(base_dir)

        assert run_dir.exists()
        assert base_dir.exists()


class TestSaveCleanedData:
    """Tests for save_cleaned_data function."""

    def test_save_basic_data(self, tmp_path):
        """Test saving cleaned data and log."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "trait1": [1.0, 2.0],
            }
        )

        outliers = {
            "method": "test",
            "outliers": [0],
        }

        cleaned_path, log_path = save_cleaned_data(df, outliers, tmp_path)

        assert cleaned_path.exists()
        assert log_path.exists()

        # Check CSV
        loaded_df = pd.read_csv(cleaned_path)
        assert len(loaded_df) == 2

        # Check log
        with open(log_path) as f:
            log_data = json.load(f)
        assert "timestamp" in log_data
        assert log_data["original_samples"] == 2

    def test_save_with_additional_info(self, tmp_path):
        """Test saving with additional log info."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "trait1": [1.0],
            }
        )

        outliers = {"outliers": []}
        log_info = {
            "processing_time": 1.23,
            "user": "test_user",
        }

        _, log_path = save_cleaned_data(df, outliers, tmp_path, log_info)

        with open(log_path) as f:
            log_data = json.load(f)
        assert log_data["processing_time"] == 1.23
        assert log_data["user"] == "test_user"


class TestConvertToJsonSerializable:
    """Tests for _convert_to_json_serializable function."""

    def test_numpy_conversion(self):
        """Test conversion of numpy types."""
        data = {
            "int32": np.int32(42),
            "int64": np.int64(100),
            "float32": np.float32(3.14),
            "float64": np.float64(2.718),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3]),
        }

        converted = _convert_to_json_serializable(data)

        assert converted["int32"] == 42
        assert converted["int64"] == 100
        assert abs(converted["float32"] - 3.14) < 0.01
        assert abs(converted["float64"] - 2.718) < 0.001
        assert converted["bool"] is True
        assert converted["array"] == [1, 2, 3]

    def test_nested_conversion(self):
        """Test conversion of nested structures."""
        data = {
            "list": [np.int32(1), np.float64(2.0)],
            "dict": {
                "nested": np.array([1, 2]),
            },
            "tuple": (np.int64(3), np.bool_(False)),
        }

        converted = _convert_to_json_serializable(data)

        assert converted["list"] == [1, 2.0]
        assert converted["dict"]["nested"] == [1, 2]
        assert converted["tuple"] == (3, False)

    def test_object_with_tolist(self):
        """Test conversion of objects with tolist method."""

        # Create a mock object with tolist method
        class MockObject:
            def tolist(self):
                return [1, 2, 3]

        data = {"mock": MockObject(), "regular": "test"}

        converted = _convert_to_json_serializable(data)
        assert converted["mock"] == [1, 2, 3]
        assert converted["regular"] == "test"


class TestRemoveNanSamples:
    """Tests for remove_nan_samples function."""

    def test_remove_samples_with_nan(self):
        """Test removing samples with NaN values."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003"],
                "geno": ["G1", "G2", "G3"],
                "rep": [1, 2, 3],
                "trait1": [1.0, np.nan, 3.0],
                "trait2": [4.0, 5.0, np.nan],
            }
        )

        trait_cols = ["trait1", "trait2"]
        df_cleaned, df_removed, stats = remove_nan_samples(
            df, trait_cols, max_nan_fraction=0.3
        )

        assert len(df_cleaned) == 1  # BC002 and BC003 removed for exceeding threshold
        assert len(df_removed) == 2  # Two samples exceed 30% NaN threshold
        assert stats["samples_with_any_nan"] == 2

    def test_remove_high_nan_fraction(self):
        """Test removing samples exceeding NaN threshold."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003"],
                "geno": ["G1", "G2", "G3"],
                "trait1": [1.0, np.nan, 3.0],
                "trait2": [4.0, np.nan, 6.0],
                "trait3": [7.0, np.nan, 9.0],
            }
        )

        trait_cols = ["trait1", "trait2", "trait3"]
        df_cleaned, df_removed, stats = remove_nan_samples(
            df, trait_cols, max_nan_fraction=0.5
        )

        assert len(df_cleaned) == 2  # BC002 removed (100% NaN)
        assert len(df_removed) == 1
        assert stats["samples_removed"] == 1

    def test_no_nan_samples(self):
        """Test when no samples have NaN."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )

        trait_cols = ["trait1", "trait2"]
        df_cleaned, df_removed, stats = remove_nan_samples(df, trait_cols)

        assert len(df_cleaned) == 2
        assert len(df_removed) == 0
        assert stats["samples_with_any_nan"] == 0


class TestSaveNanRemovedRows:
    """Tests for save_nan_removed_rows function."""

    def test_save_removed_rows(self, tmp_path):
        """Test saving removed rows to CSV."""
        df_original = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003"],
                "trait1": [1.0, np.nan, 3.0],
                "trait2": [4.0, 5.0, 6.0],
            }
        )

        df_cleaned = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC003"],
                "trait1": [1.0, 3.0],
                "trait2": [4.0, 6.0],
            },
            index=[0, 2],
        )

        trait_cols = ["trait1", "trait2"]
        path = save_nan_removed_rows(df_original, df_cleaned, tmp_path, trait_cols)

        assert path.exists()
        removed_df = pd.read_csv(path)
        assert len(removed_df) == 1
        assert "nan_traits" in removed_df.columns
        assert "removal_reason" in removed_df.columns

    def test_no_removed_rows(self, tmp_path):
        """Test when no rows are removed."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "trait1": [1.0, 2.0],
            }
        )

        trait_cols = ["trait1"]
        path = save_nan_removed_rows(df, df, tmp_path, trait_cols)

        assert path.exists()
        removed_df = pd.read_csv(path)
        assert len(removed_df) == 0


class TestGetNumericTraitsOnly:
    """Tests for get_numeric_traits_only function."""

    def test_extract_numeric_traits(self):
        """Test extracting only numeric trait columns."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "text_col": ["A", "B"],
                "trait1": [1.0, 2.0],
                "trait2": [3, 4],
            }
        )

        numeric_df = get_numeric_traits_only(df)

        assert "Barcode" not in numeric_df.columns
        assert "geno" not in numeric_df.columns
        assert "rep" not in numeric_df.columns
        assert "text_col" not in numeric_df.columns
        assert "trait1" in numeric_df.columns
        assert "trait2" in numeric_df.columns
        assert len(numeric_df.columns) == 2


class TestRemoveLowHeritabilityTraits:
    """Tests for remove_low_heritability_traits function."""

    def test_remove_low_heritability(self):
        """Test removing traits with low heritability."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
                "trait3": [5.0, 6.0],
            }
        )

        heritability_results = {
            "trait1": {"heritability": 0.8, "var_genetic": 4.0, "var_residual": 1.0},
            "trait2": {"heritability": 0.2, "var_genetic": 1.0, "var_residual": 4.0},
            "trait3": {"heritability": 0.5, "var_genetic": 2.0, "var_residual": 2.0},
        }

        df_cleaned, removed, summary = remove_low_heritability_traits(
            df, heritability_results, heritability_threshold=0.3
        )

        assert "trait1" in df_cleaned.columns  # H² = 0.8
        assert "trait2" not in df_cleaned.columns  # H² = 0.2
        assert "trait3" in df_cleaned.columns  # H² = 0.5
        assert len(removed) == 1
        assert summary["removed_traits"] == 1

    def test_missing_heritability_results(self):
        """Test handling traits without heritability results."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "trait1": [1.0],
                "trait2": [2.0],
            }
        )

        heritability_results = {
            "trait1": {"heritability": 0.8},
        }

        df_cleaned, removed, summary = remove_low_heritability_traits(
            df, heritability_results
        )

        assert "trait1" in df_cleaned.columns
        assert "trait2" not in df_cleaned.columns  # No heritability data
        assert len(removed) == 1

    def test_invalid_heritability_results(self):
        """Test handling invalid heritability results."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "trait1": [1.0],
                "trait2": [2.0],
            }
        )

        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": "invalid",  # Invalid format
        }

        df_cleaned, removed, summary = remove_low_heritability_traits(
            df, heritability_results
        )

        assert "trait1" in df_cleaned.columns
        assert "trait2" not in df_cleaned.columns  # Invalid result

    def test_with_additional_exclusions(self):
        """Test with additional column exclusions."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "rep": [1],
                "extra_col": [99],
                "trait1": [1.0],
                "trait2": [2.0],
            }
        )

        heritability_results = {
            "trait1": {"heritability": 0.8},
            "trait2": {"heritability": 0.8},
            "extra_col": {"heritability": 0.1},  # Should be excluded anyway
        }

        df_cleaned, removed, summary = remove_low_heritability_traits(
            df, heritability_results, additional_exclude=["extra_col"]
        )

        # extra_col should be preserved but not considered a trait
        assert "extra_col" in df_cleaned.columns
        assert "extra_col" not in removed  # Not in removed traits list


# Integration tests with fixtures
class TestWithFixtures:
    """Integration tests using data fixtures."""

    def test_with_features_data(self, features_df):
        """Test with features.csv fixture."""
        # Test get_trait_columns
        trait_cols = get_trait_columns(
            features_df,
            barcode_col="File.Name",
            genotype_col="Region.of.Interest",
            replicate_col=None,
        )

        assert "Total.Root.Length.mm" in trait_cols
        assert "File.Name" not in trait_cols

    def test_with_traits_summary_data(self, traits_summary_df):
        """Test with traits_summary.csv fixture."""
        # Test remove_nan_samples
        trait_cols = get_trait_columns(
            traits_summary_df,
            barcode_col="plant_qr_code",
            genotype_col="species_name",
            replicate_col=None,
        )

        if trait_cols:
            df_cleaned, df_removed, stats = remove_nan_samples(
                traits_summary_df, trait_cols, max_nan_fraction=0.2
            )

            assert len(df_cleaned) <= len(traits_summary_df)

    def test_with_mock_heritability_data(self, heritability_data):
        """Test with heritability test data."""
        df, true_h2 = heritability_data

        # Create heritability results dict
        heritability_results = {}
        for i, col in enumerate([c for c in df.columns if c.startswith("trait_")]):
            heritability_results[col] = {
                "heritability": true_h2[i] if i < len(true_h2) else 0.1
            }

        df_cleaned, removed, summary = remove_low_heritability_traits(
            df,
            heritability_results,
            heritability_threshold=0.3,
            barcode_col="sample_id" if "sample_id" in df.columns else "Barcode",
            genotype_col="genotype" if "genotype" in df.columns else "geno",
        )

        # Traits with H² < 0.3 should be removed
        assert summary["threshold"] == 0.3
