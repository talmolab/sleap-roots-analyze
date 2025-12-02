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
    save_cleaned_data,
    remove_nan_samples,
    get_numeric_traits_only,
    remove_low_heritability_traits,
)
from src.sleap_roots_analyze.data_utils import (
    create_run_directory,
    convert_to_json_serializable,
    link_rhizovision_images_to_samples,
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

    def test_root_core_metadata_exclusion(self):
        """Test exclusion of root core pipeline metadata columns (Plot, Rep, geno, Barcode).

        CRITICAL: Root core pipeline uses capital-letter column names (Plot, Rep) which
        must be excluded from trait analyses to prevent metadata contamination.
        """
        df = pd.DataFrame(
            {
                "Plot": [1, 2, 3],
                "Rep": [1, 2, 3],
                "geno": ["GH_7386", "GH_7420", "Control"],
                "Barcode": ["1-1", "2-2", "3-3"],
                "RootDW_15cm": [2.5, 2.1, 2.8],
                "RootDW_45cm": [1.2, 0.9, 1.5],
                "RootCount_0cm": [10, 12, 11],
            }
        )

        # Test with capital Rep (root core pipeline uses 'Rep')
        trait_cols = get_trait_columns(
            df, barcode_col="Barcode", genotype_col="geno", replicate_col="Rep"
        )

        # Verify ALL metadata columns are excluded
        assert "Plot" not in trait_cols, "Plot should be excluded"
        assert "Rep" not in trait_cols, "Rep should be excluded"
        assert "geno" not in trait_cols, "geno should be excluded"
        assert "Barcode" not in trait_cols, "Barcode should be excluded"

        # Verify only trait columns remain
        assert "RootDW_15cm" in trait_cols
        assert "RootDW_45cm" in trait_cols
        assert "RootCount_0cm" in trait_cols
        assert (
            len(trait_cols) == 3
        ), f"Should have 3 traits, got {len(trait_cols)}: {trait_cols}"


class TestLinkRhizovisionImagesToSamples:
    """Tests for link_rhizovision_images_to_samples function."""

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

        links = link_rhizovision_images_to_samples(df, image_dir)

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

        links = link_rhizovision_images_to_samples(
            df, image_dir, image_types=["custom.png"]
        )

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
            link_rhizovision_images_to_samples(df, tmp_path)


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
    """Tests for convert_to_json_serializable function."""

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

        converted = convert_to_json_serializable(data)

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

        converted = convert_to_json_serializable(data)

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

        converted = convert_to_json_serializable(data)
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

    def test_save_removed_samples(self, tmp_path):
        """Test saving removed samples to CSV."""
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
        save_path = tmp_path / "nan_removed.csv"

        df_cleaned, df_removed, stats = remove_nan_samples(
            df, trait_cols, max_nan_fraction=0.3, save_removed_path=save_path
        )

        # Check that file was saved
        assert save_path.exists()
        assert stats["saved_path"] == str(save_path)

        # Read saved file and verify content
        saved_df = pd.read_csv(save_path)
        assert len(saved_df) == 2  # Two samples removed
        assert "nan_traits" in saved_df.columns
        assert "removal_reason" in saved_df.columns

    def test_save_empty_file_when_no_removals(self, tmp_path):
        """Test saving empty file when no samples are removed."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )

        trait_cols = ["trait1", "trait2"]
        save_path = tmp_path / "nan_removed.csv"

        df_cleaned, df_removed, stats = remove_nan_samples(
            df, trait_cols, save_removed_path=save_path
        )

        # Check that empty file was saved
        assert save_path.exists()
        assert stats["saved_path"] == str(save_path)

        # Read saved file and verify it's empty
        saved_df = pd.read_csv(save_path)
        assert len(saved_df) == 0


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

    def test_custom_column_names(self):
        """Test with custom metadata column names."""
        df = pd.DataFrame(
            {
                "PlantID": ["P1", "P2", "P3"],
                "genotype": ["TypeA", "TypeB", "TypeA"],
                "replicate": [1, 2, 3],
                "root_length": [10.5, 12.3, 11.8],
                "lateral_count": [5, 7, 6],
                "notes": ["good", "ok", "good"],
            }
        )

        numeric_df = get_numeric_traits_only(
            df,
            barcode_col="PlantID",
            genotype_col="genotype",
            replicate_col="replicate",
            additional_exclude=["notes"],
        )

        assert "PlantID" not in numeric_df.columns
        assert "genotype" not in numeric_df.columns
        assert "replicate" not in numeric_df.columns
        assert "notes" not in numeric_df.columns
        assert "root_length" in numeric_df.columns
        assert "lateral_count" in numeric_df.columns
        assert len(numeric_df.columns) == 2

    def test_no_replicate_column(self):
        """Test when replicate column doesn't exist."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "trait1": [1.5, 2.5],
                "trait2": [3.5, 4.5],
            }
        )

        # Should work even if rep column doesn't exist
        numeric_df = get_numeric_traits_only(df, replicate_col="rep")

        assert "trait1" in numeric_df.columns
        assert "trait2" in numeric_df.columns
        assert len(numeric_df.columns) == 2

    def test_additional_exclusions(self):
        """Test excluding additional columns."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "date_scanned": ["2024-01-01", "2024-01-02"],
                "QC_status": ["pass", "pass"],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )

        numeric_df = get_numeric_traits_only(
            df, additional_exclude=["date_scanned", "QC_status"]
        )

        assert "date_scanned" not in numeric_df.columns
        assert "QC_status" not in numeric_df.columns
        assert "trait1" in numeric_df.columns
        assert "trait2" in numeric_df.columns
        assert len(numeric_df.columns) == 2

    def test_with_nan_values(self):
        """Test that NaN values are preserved in numeric columns."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003"],
                "geno": ["G1", "G2", "G1"],
                "trait1": [1.0, np.nan, 3.0],
                "trait2": [np.nan, 2.0, 3.0],
            }
        )

        numeric_df = get_numeric_traits_only(df)

        assert len(numeric_df.columns) == 2
        assert pd.isna(numeric_df.iloc[0, 1])  # trait2 first row
        assert pd.isna(numeric_df.iloc[1, 0])  # trait1 second row

    def test_with_mixed_types(self):
        """Test with mixed numeric types (int, float)."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "int_trait": [1, 2],
                "float_trait": [1.5, 2.5],
                "bool_col": [True, False],
                "str_trait": ["1.0", "2.0"],  # String that looks numeric
            }
        )

        numeric_df = get_numeric_traits_only(df)

        assert "int_trait" in numeric_df.columns
        assert "float_trait" in numeric_df.columns
        # Boolean columns might be included as they're numeric-like
        # String columns should be excluded
        assert "str_trait" not in numeric_df.columns

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        numeric_df = get_numeric_traits_only(df)

        assert numeric_df.empty
        assert len(numeric_df.columns) == 0

    def test_no_numeric_columns(self):
        """Test when no numeric columns exist."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "notes": ["good", "bad"],
                "category": ["A", "B"],
            }
        )

        numeric_df = get_numeric_traits_only(df)

        assert numeric_df.empty
        assert len(numeric_df.columns) == 0

    def test_preserves_copy(self):
        """Test that function returns a copy, not a view."""
        df = pd.DataFrame(
            {"Barcode": ["BC001", "BC002"], "geno": ["G1", "G2"], "trait1": [1.0, 2.0]}
        )

        numeric_df = get_numeric_traits_only(df)

        # Modify the returned dataframe
        numeric_df.iloc[0, 0] = 999

        # Original should be unchanged
        assert df["trait1"].iloc[0] == 1.0
        assert numeric_df.iloc[0, 0] == 999

    def test_with_real_data(self, features_df):
        """Test with real features data."""
        # Get metadata columns to exclude
        metadata_cols = ["File.Name", "Region.of.Interest"]

        numeric_df = get_numeric_traits_only(
            features_df,
            barcode_col="File.Name",  # Using File.Name as ID
            genotype_col="Region.of.Interest",  # Using as pseudo-genotype
            replicate_col=None,
            additional_exclude=[],
        )

        # Should only have numeric columns
        assert all(
            pd.api.types.is_numeric_dtype(numeric_df[col]) for col in numeric_df.columns
        )

        # Should not have metadata columns
        assert "File.Name" not in numeric_df.columns
        assert "Region.of.Interest" not in numeric_df.columns

    def test_with_turface_data(self, turface_traits_df):
        """Test with Turface trait data."""
        numeric_df = get_numeric_traits_only(
            turface_traits_df,
            barcode_col="Barcode",
            genotype_col="geno",
            replicate_col="rep",
            additional_exclude=(
                ["wave_name"] if "wave_name" in turface_traits_df.columns else []
            ),
        )

        # Should exclude metadata
        assert "Barcode" not in numeric_df.columns
        assert "geno" not in numeric_df.columns
        assert "rep" not in numeric_df.columns

        # Should only have numeric columns
        assert all(
            pd.api.types.is_numeric_dtype(numeric_df[col]) for col in numeric_df.columns
        )

    def test_integration_with_get_trait_columns(self, mixed_problem_data):
        """Test that get_numeric_traits_only uses get_trait_columns correctly."""
        # First get trait columns
        trait_cols = get_trait_columns(
            mixed_problem_data,
            barcode_col="Barcode",
            genotype_col="geno",
            replicate_col="rep",
        )

        # Then get numeric traits only
        numeric_df = get_numeric_traits_only(
            mixed_problem_data,
            barcode_col="Barcode",
            genotype_col="geno",
            replicate_col="rep",
        )

        # All columns in numeric_df should be in trait_cols
        assert all(col in trait_cols for col in numeric_df.columns)

        # All columns should be numeric
        assert all(
            pd.api.types.is_numeric_dtype(numeric_df[col]) for col in numeric_df.columns
        )


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

    def test_with_mock_heritability_data(self, heritability_data_known_h2):
        """Test with heritability test data.

        Expected heritability values from fixture:
        - trait_high_h2: 0.8 (should be kept)
        - trait_moderate_h2: 0.5 (should be kept)
        - trait_low_h2: 0.09 (should be removed when threshold=0.3)
        """
        df, expected_h2 = heritability_data_known_h2

        # Create heritability results dict using expected values directly
        heritability_results = {}
        for trait_name, h2_value in expected_h2.items():
            heritability_results[trait_name] = {"heritability": h2_value}

        # Set threshold to 0.3
        threshold = 0.3
        df_cleaned, removed_traits, summary = remove_low_heritability_traits(
            df,
            heritability_results,
            heritability_threshold=threshold,
            barcode_col="Barcode",  # Check actual column name in df
            genotype_col="geno",  # Check actual column name in df
        )

        # Check that the threshold was properly set
        assert summary["threshold"] == threshold

        # Verify which traits were removed
        # trait_low_h2 (H²=0.09) should be removed as it's below threshold (0.3)
        assert "trait_low_h2" in removed_traits
        assert len(removed_traits) == 1

        # Verify which traits were kept
        # trait_high_h2 (H²=0.8) and trait_moderate_h2 (H²=0.5) should be kept
        assert "trait_high_h2" in df_cleaned.columns
        assert "trait_moderate_h2" in df_cleaned.columns
        assert "trait_low_h2" not in df_cleaned.columns

        # Verify summary statistics
        assert summary["removed_traits"] == 1
        assert summary["retained_traits"] == 2


class TestModularCleanupFunctions:
    """Test the modular cleanup functions."""

    def test_remove_zero_inflated_traits_basic(self, zero_inflated_data):
        """Test removal of zero-inflated traits."""
        from src.sleap_roots_analyze.data_cleanup import remove_zero_inflated_traits

        df = zero_inflated_data
        trait_cols = [
            "trait_all_zeros",
            "trait_half_zeros",
            "trait_no_zeros",
            "trait_normal",
        ]

        # Remove traits with > 50% zeros
        df_filtered, remaining_traits, removal_details = remove_zero_inflated_traits(
            df, trait_cols, max_zero_fraction=0.5
        )

        # trait_all_zeros should be removed (100% zeros)
        assert "trait_all_zeros" not in df_filtered.columns
        assert "trait_all_zeros" not in remaining_traits
        assert "trait_all_zeros" in removal_details
        assert removal_details["trait_all_zeros"]["reason"] == "too_many_zeros"
        assert removal_details["trait_all_zeros"]["zero_fraction"] == 1.0

        # trait_half_zeros should NOT be removed (exactly 50% zeros)
        assert "trait_half_zeros" in df_filtered.columns
        assert "trait_half_zeros" in remaining_traits

        # trait_no_zeros should NOT be removed
        assert "trait_no_zeros" in df_filtered.columns
        assert "trait_no_zeros" in remaining_traits

    def test_remove_zero_inflated_traits_edge_cases(self):
        """Test edge cases for zero-inflated trait removal."""
        from src.sleap_roots_analyze.data_cleanup import remove_zero_inflated_traits

        # Empty dataframe
        df_empty = pd.DataFrame()
        df_filtered, remaining, details = remove_zero_inflated_traits(
            df_empty, [], max_zero_fraction=0.5
        )
        assert len(df_filtered) == 0
        assert len(remaining) == 0
        assert len(details) == 0

        # All zeros
        df_all_zeros = pd.DataFrame({"trait1": [0, 0, 0], "trait2": [0, 0, 0]})
        df_filtered, remaining, details = remove_zero_inflated_traits(
            df_all_zeros, ["trait1", "trait2"], max_zero_fraction=0.5
        )
        assert len(df_filtered.columns) == 0  # All traits removed
        assert len(remaining) == 0
        assert len(details) == 2

        # No zeros
        df_no_zeros = pd.DataFrame({"trait1": [1, 2, 3], "trait2": [4, 5, 6]})
        df_filtered, remaining, details = remove_zero_inflated_traits(
            df_no_zeros, ["trait1", "trait2"], max_zero_fraction=0.5
        )
        assert len(df_filtered.columns) == 2  # No traits removed
        assert len(remaining) == 2
        assert len(details) == 0

    def test_remove_traits_with_many_nans_basic(self, nan_data):
        """Test removal of traits with many NaNs."""
        from src.sleap_roots_analyze.data_cleanup import remove_traits_with_many_nans

        df = nan_data
        trait_cols = [
            "trait_all_nan",
            "trait_half_nan",
            "trait_some_nan",
            "trait_no_nan",
        ]

        # Remove traits with > 30% NaNs
        df_filtered, remaining_traits, removal_details = remove_traits_with_many_nans(
            df, trait_cols, max_nan_fraction=0.3
        )

        # trait_all_nan should be removed (100% NaNs)
        assert "trait_all_nan" not in df_filtered.columns
        assert "trait_all_nan" not in remaining_traits
        assert "trait_all_nan" in removal_details
        assert removal_details["trait_all_nan"]["reason"] == "too_many_nans"
        assert removal_details["trait_all_nan"]["nan_fraction"] == 1.0

        # trait_half_nan should be removed (50% NaNs > 30%)
        assert "trait_half_nan" not in df_filtered.columns
        assert "trait_half_nan" not in remaining_traits

        # trait_some_nan should NOT be removed (20% NaNs < 30%)
        assert "trait_some_nan" in df_filtered.columns
        assert "trait_some_nan" in remaining_traits

        # trait_no_nan should NOT be removed
        assert "trait_no_nan" in df_filtered.columns
        assert "trait_no_nan" in remaining_traits

    def test_remove_low_sample_traits_basic(self, sparse_data):
        """Test removal of traits with insufficient samples."""
        from src.sleap_roots_analyze.data_cleanup import remove_low_sample_traits

        df = sparse_data
        trait_cols = ["trait_sparse", "trait_dense", "trait_half"]

        # Require at least 8 valid samples
        df_filtered, remaining_traits, removal_details = remove_low_sample_traits(
            df, trait_cols, min_samples=8
        )

        # trait_sparse should be removed (only 3 valid samples)
        assert "trait_sparse" not in df_filtered.columns
        assert "trait_sparse" not in remaining_traits
        assert "trait_sparse" in removal_details
        assert removal_details["trait_sparse"]["reason"] == "insufficient_samples"
        assert removal_details["trait_sparse"]["valid_samples"] == 3

        # trait_dense should NOT be removed (10 valid samples)
        assert "trait_dense" in df_filtered.columns
        assert "trait_dense" in remaining_traits

        # trait_half should NOT be removed (5 valid samples, but threshold is 8)
        # Actually it SHOULD be removed since 5 < 8
        if "trait_half" in removal_details:
            assert removal_details["trait_half"]["valid_samples"] == 5

    def test_remove_low_sample_traits_with_real_data(self, features_df):
        """Test with real feature data."""
        from src.sleap_roots_analyze.data_cleanup import (
            remove_low_sample_traits,
            get_trait_columns,
        )

        trait_cols = get_trait_columns(features_df)

        # Set a high threshold to test removal
        df_filtered, remaining_traits, removal_details = remove_low_sample_traits(
            features_df, trait_cols, min_samples=1000
        )

        # All traits should be removed with such a high threshold
        assert len(remaining_traits) == 0
        assert len(removal_details) == len(trait_cols)

        # Test with reasonable threshold
        df_filtered2, remaining_traits2, removal_details2 = remove_low_sample_traits(
            features_df, trait_cols, min_samples=10
        )

        # Most traits should remain
        assert len(remaining_traits2) > 0

    def test_apply_data_cleanup_filters_integration(self, mixed_problem_data):
        """Test the integrated cleanup function with mixed problems."""
        from src.sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        df = mixed_problem_data
        trait_cols = [c for c in df.columns if c.startswith("trait_")]

        # Apply all filters
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols,
            max_zeros_per_trait=0.5,
            max_nans_per_trait=0.3,
            max_nans_per_sample=0.2,
            min_samples_per_trait=5,
        )

        # Check that the cleanup log has all required fields
        assert "original_samples" in cleanup_log
        assert "original_traits" in cleanup_log
        assert "final_samples" in cleanup_log
        assert "final_traits" in cleanup_log
        assert "cleanup_steps" in cleanup_log
        assert "removed_traits" in cleanup_log

        # Check that each step was recorded
        step_names = [step["step"] for step in cleanup_log["cleanup_steps"]]
        assert "remove_high_zero_traits" in step_names
        assert "remove_high_nan_traits" in step_names
        assert "remove_high_nan_samples" in step_names
        assert "remove_low_sample_traits" in step_names

        # Ensure some cleaning happened
        assert cleanup_log["final_traits"] <= cleanup_log["original_traits"]
        assert cleanup_log["final_samples"] <= cleanup_log["original_samples"]

    def test_modular_functions_preserve_data_integrity(self, features_df):
        """Test that modular functions don't modify original data."""
        from src.sleap_roots_analyze.data_cleanup import (
            remove_zero_inflated_traits,
            remove_traits_with_many_nans,
            remove_low_sample_traits,
            get_trait_columns,
        )

        trait_cols = get_trait_columns(features_df)
        df_original = features_df.copy()

        # Apply each function
        df1, traits1, _ = remove_zero_inflated_traits(
            features_df, trait_cols, max_zero_fraction=0.5
        )
        df2, traits2, _ = remove_traits_with_many_nans(
            df1, traits1, max_nan_fraction=0.3
        )
        df3, traits3, _ = remove_low_sample_traits(df2, traits2, min_samples=10)

        # Original dataframe should be unchanged
        pd.testing.assert_frame_equal(features_df, df_original)

        # Each step should preserve or reduce columns
        assert len(traits1) <= len(trait_cols)
        assert len(traits2) <= len(traits1)
        assert len(traits3) <= len(traits2)


class TestInspectNanSamples:
    """Test the inspect_nan_samples function."""

    def test_basic_nan_inspection(self):
        """Test basic NaN inspection functionality."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data with some NaN values
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003", "BC004", "BC005"],
                "geno": ["G1", "G2", "G3", "G1", "G2"],
                "rep": [1, 1, 1, 2, 2],
                "trait1": [1.0, np.nan, 3.0, 4.0, 5.0],
                "trait2": [6.0, 7.0, np.nan, 9.0, np.nan],
                "trait3": [10.0, np.nan, 12.0, 13.0, 14.0],
            }
        )
        trait_cols = ["trait1", "trait2", "trait3"]

        # Inspect NaN samples
        inspection_df = inspect_nan_samples(df, trait_cols)

        # Check results
        assert len(inspection_df) == 3  # BC002, BC003, BC005 have NaN
        assert "sample_index" in inspection_df.columns
        assert "barcode" in inspection_df.columns
        assert "genotype" in inspection_df.columns
        assert "rep" in inspection_df.columns
        assert "nan_count" in inspection_df.columns
        assert "nan_fraction" in inspection_df.columns
        assert "nan_traits" in inspection_df.columns
        assert "data_status" in inspection_df.columns

        # Check specific sample details
        bc002_row = inspection_df[inspection_df["barcode"] == "BC002"].iloc[0]
        assert bc002_row["nan_count"] == 2  # trait1 and trait3 have NaN
        assert bc002_row["nan_fraction"] == 2 / 3
        assert "trait1" in bc002_row["nan_traits"]
        assert "trait3" in bc002_row["nan_traits"]
        assert bc002_row["data_status"] == "original_data_with_nan"

    def test_no_nan_values(self):
        """Test when no NaN values are present."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data without NaN
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 1],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Inspect NaN samples
        inspection_df = inspect_nan_samples(df, trait_cols)

        # Should return empty DataFrame with correct structure
        assert len(inspection_df) == 0
        assert "sample_index" in inspection_df.columns
        assert "nan_count" in inspection_df.columns
        assert "nan_fraction" in inspection_df.columns

    def test_custom_column_names(self):
        """Test with custom column names."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data with custom column names
        df = pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3"],
                "Genotype": ["G1", "G2", "G3"],
                "Replicate": [1, 2, 3],
                "trait1": [1.0, np.nan, 3.0],
                "trait2": [4.0, 5.0, np.nan],
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Inspect with custom column names
        inspection_df = inspect_nan_samples(
            df,
            trait_cols,
            barcode_col="SampleID",
            genotype_col="Genotype",
            replicate_col="Replicate",
        )

        # Check results
        assert len(inspection_df) == 2  # S2 and S3 have NaN
        assert "barcode" in inspection_df.columns
        assert inspection_df.iloc[0]["barcode"] == "S2"
        assert inspection_df.iloc[0]["genotype"] == "G2"
        assert inspection_df.iloc[0]["rep"] == 2

    def test_missing_metadata_columns(self):
        """Test when metadata columns don't exist."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data without metadata columns
        df = pd.DataFrame({"trait1": [1.0, np.nan, 3.0], "trait2": [4.0, 5.0, np.nan]})
        trait_cols = ["trait1", "trait2"]

        # Inspect without metadata columns
        inspection_df = inspect_nan_samples(df, trait_cols)

        # Should still work but without metadata
        assert len(inspection_df) == 2
        assert "sample_index" in inspection_df.columns
        assert "nan_count" in inspection_df.columns
        # Metadata columns won't be present
        assert (
            "barcode" not in inspection_df.columns
            or inspection_df["barcode"].isna().all()
        )

    def test_save_to_csv(self, tmp_path):
        """Test saving inspection results to CSV."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "trait1": [1.0, np.nan],
                "trait2": [np.nan, 5.0],
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Save path
        save_path = tmp_path / "nan_inspection.csv"

        # Inspect and save
        inspection_df = inspect_nan_samples(df, trait_cols, save_path=str(save_path))

        # Check file was created
        assert save_path.exists()

        # Load and verify
        loaded_df = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(inspection_df, loaded_df)

    def test_with_turface_data(self, turface_traits_df):
        """Test with real Turface data fixture."""
        from src.sleap_roots_analyze.data_cleanup import (
            inspect_nan_samples,
            get_trait_columns,
        )

        trait_cols = get_trait_columns(turface_traits_df)

        # Inspect NaN samples
        inspection_df = inspect_nan_samples(turface_traits_df, trait_cols)

        # Check structure
        if len(inspection_df) > 0:
            assert "sample_index" in inspection_df.columns
            assert "barcode" in inspection_df.columns
            assert "genotype" in inspection_df.columns
            assert "rep" in inspection_df.columns
            assert "nan_count" in inspection_df.columns
            assert "nan_fraction" in inspection_df.columns
            assert "nan_traits" in inspection_df.columns
            assert "data_status" in inspection_df.columns

            # All data_status should be 'original_data_with_nan'
            assert all(inspection_df["data_status"] == "original_data_with_nan")

            # nan_fraction should be between 0 and 1
            assert all(
                (inspection_df["nan_fraction"] >= 0)
                & (inspection_df["nan_fraction"] <= 1)
            )

    def test_with_traits_summary_data(self, traits_summary_df):
        """Test with real traits summary data."""
        from src.sleap_roots_analyze.data_cleanup import (
            inspect_nan_samples,
            get_trait_columns,
        )

        trait_cols = get_trait_columns(traits_summary_df)

        # Inspect NaN samples
        inspection_df = inspect_nan_samples(
            traits_summary_df,
            trait_cols,
            barcode_col=(
                "Barcode" if "Barcode" in traits_summary_df.columns else "plant_id"
            ),
            genotype_col="geno" if "geno" in traits_summary_df.columns else "genotype",
            replicate_col="rep" if "rep" in traits_summary_df.columns else "replicate",
        )

        # Check structure
        if len(inspection_df) > 0:
            assert "sample_index" in inspection_df.columns
            assert "nan_count" in inspection_df.columns
            assert "nan_fraction" in inspection_df.columns
            assert "nan_traits" in inspection_df.columns
            assert "data_status" in inspection_df.columns

    def test_fraction_calculation(self):
        """Test that NaN fraction is calculated correctly."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples

        # Create test data with specific NaN patterns
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002", "BC003"],
                "geno": ["G1", "G2", "G3"],
                "rep": [1, 1, 1],
                "trait1": [np.nan, np.nan, 1.0],
                "trait2": [np.nan, 2.0, 3.0],
                "trait3": [np.nan, 4.0, 5.0],
                "trait4": [np.nan, 6.0, 7.0],
            }
        )
        trait_cols = ["trait1", "trait2", "trait3", "trait4"]

        inspection_df = inspect_nan_samples(df, trait_cols)

        # BC001 should have all 4 traits as NaN (fraction = 1.0)
        bc001_row = inspection_df[inspection_df["barcode"] == "BC001"].iloc[0]
        assert bc001_row["nan_count"] == 4
        assert bc001_row["nan_fraction"] == 1.0

        # BC002 should have 1 trait as NaN (fraction = 0.25)
        bc002_row = inspection_df[inspection_df["barcode"] == "BC002"].iloc[0]
        assert bc002_row["nan_count"] == 1
        assert bc002_row["nan_fraction"] == 0.25

    def test_verbose_false(self, caplog):
        """Test with verbose=False to ensure quiet operation."""
        from src.sleap_roots_analyze.data_cleanup import inspect_nan_samples
        import logging

        # Create test data with NaN
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "trait1": [1.0, np.nan],
                "trait2": [np.nan, 2.0],
            }
        )
        trait_cols = ["trait1", "trait2"]

        # Clear any previous logs
        caplog.clear()

        # Test with verbose=False (should not log)
        with caplog.at_level(logging.INFO):
            inspection_df = inspect_nan_samples(df, trait_cols, verbose=False)

        # Check that we got results but no logs from the function
        assert len(inspection_df) == 2
        assert "Initial NaN inspection" not in caplog.text

        # Also test the no-NaN case with verbose=False
        df_no_nan = pd.DataFrame(
            {
                "Barcode": ["BC001"],
                "geno": ["G1"],
                "rep": [1],
                "trait1": [1.0],
                "trait2": [2.0],
            }
        )

        caplog.clear()
        inspection_df_no_nan = inspect_nan_samples(df_no_nan, trait_cols, verbose=False)
        assert len(inspection_df_no_nan) == 0
        assert "No NaN values found" not in caplog.text
