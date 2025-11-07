"""Tests for data_utils module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sleap_roots_analyze.data_utils import (
    sanitize_trait_names,
    convert_to_json_serializable,
    create_run_directory,
)


class TestSanitizeTraitNames:
    """Tests for sanitize_trait_names function."""

    def test_basic_dot_replacement(self):
        """Test that dots are replaced with spaces."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "barcode": ["A", "B", "C"],
        })
        trait_cols = ["Median.Number.of.Roots"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Median Number Roots" in result.columns
        assert "Median.Number.of.Roots" not in result.columns

    def test_unit_conversion_mm(self):
        """Test that .mm units are converted to (mm)."""
        df = pd.DataFrame({
            "Total.Root.Length.mm": [100, 200, 300],
            "Depth.mm": [50, 60, 70],
        })
        trait_cols = ["Total.Root.Length.mm", "Depth.mm"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Total Root Length (mm)" in result.columns
        assert "Depth (mm)" in result.columns

    def test_unit_conversion_mm2(self):
        """Test that .mm2 units are converted to (mm²)."""
        df = pd.DataFrame({
            "Average.Hole.Size.mm2": [10, 20, 30],
            "Network.Area.mm2": [100, 200, 300],
        })
        trait_cols = ["Average.Hole.Size.mm2", "Network.Area.mm2"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Average Hole Size (mm²)" in result.columns
        assert "Network Area (mm²)" in result.columns

    def test_unit_conversion_mm3(self):
        """Test that .mm3 units are converted to (mm³)."""
        df = pd.DataFrame({
            "Volume.mm3": [1000, 2000, 3000],
        })
        trait_cols = ["Volume.mm3"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Volume (mm³)" in result.columns

    def test_unit_conversion_deg(self):
        """Test that .deg units are converted to (°)."""
        df = pd.DataFrame({
            "Average.Root.Orientation.deg": [45, 90, 135],
        })
        trait_cols = ["Average.Root.Orientation.deg"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Average Root Orientation (°)" in result.columns


    def test_unit_conversion_g(self):
        """Test that _g units are converted to (g)."""
        df = pd.DataFrame({
            "root_g": [0.1, 0.2, 0.3],
            "shoot_g": [0.2, 0.3, 0.4],
        })
        trait_cols = ["root_g", "shoot_g"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Root (g)" in result.columns
        assert "Shoot (g)" in result.columns

    def test_unit_conversion_mg(self):
        """Test that _mg units are converted to (mg)."""
        df = pd.DataFrame({
            "biomass_mg": [100, 200, 300],
            "dry_weight_mg": [50, 60, 70],
        })
        trait_cols = ["biomass_mg", "dry_weight_mg"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Biomass (mg)" in result.columns
        assert "Dry Weight (mg)" in result.columns

    def test_abbreviations_enabled(self):
        """Test that abbreviations are applied when enabled."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "Maximum.Number.of.Roots": [5, 10, 15],
            "Average.Hole.Size.mm2": [10, 20, 30],
            "Minimum.Width.mm": [1, 2, 3],
        })
        trait_cols = [
            "Median.Number.of.Roots",
            "Maximum.Number.of.Roots",
            "Average.Hole.Size.mm2",
            "Minimum.Width.mm",
        ]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=True)
        
        assert "Med Num Roots" in result.columns
        assert "Max Num Roots" in result.columns
        assert "Avg Hole Size (mm²)" in result.columns
        assert "Min Width (mm)" in result.columns

    def test_abbreviations_disabled(self):
        """Test that abbreviations are NOT applied when disabled."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "Maximum.Number.of.Roots": [5, 10, 15],
        })
        trait_cols = ["Median.Number.of.Roots", "Maximum.Number.of.Roots"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Median Number Roots" in result.columns
        assert "Maximum Number Roots" in result.columns
        # Should NOT have abbreviations
        assert "Med Num Roots" not in result.columns
        assert "Max Num Roots" not in result.columns

    def test_hyphen_replacement(self):
        """Test that hyphens are replaced with spaces."""
        df = pd.DataFrame({
            "Width-to-Depth.Ratio": [1.5, 2.0, 2.5],
        })
        trait_cols = ["Width-to-Depth.Ratio"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Width To Depth Ratio" in result.columns

    def test_underscore_replacement(self):
        """Test that underscores are replaced with spaces."""
        df = pd.DataFrame({
            "root_shoot_ratio": [0.5, 0.6, 0.7],
        })
        trait_cols = ["root_shoot_ratio"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Root Shoot Ratio" in result.columns

    def test_filler_word_removal(self):
        """Test that filler words like 'of' and 'the' are removed."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "The.Total.Length": [100, 200, 300],
        })
        trait_cols = ["Median.Number.of.Roots", "The.Total.Length"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        # 'of' and 'the' should be removed
        assert "Median Number Roots" in result.columns
        assert "Total Length" in result.columns

    def test_return_mapping(self):
        """Test that mapping dictionary is returned when requested."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "Total.Root.Length.mm": [100, 200, 300],
        })
        trait_cols = ["Median.Number.of.Roots", "Total.Root.Length.mm"]
        
        result_df, mapping = sanitize_trait_names(
            df, trait_cols, abbreviate=True, return_mapping=True
        )
        
        # Check mapping exists and is correct
        assert isinstance(mapping, dict)
        assert "Median.Number.of.Roots" in mapping
        assert mapping["Median.Number.of.Roots"] == "Med Num Roots"
        assert "Total.Root.Length.mm" in mapping
        assert mapping["Total.Root.Length.mm"] == "Total Root Length (mm)"

    def test_return_mapping_false(self):
        """Test that only DataFrame is returned when mapping not requested."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
        })
        trait_cols = ["Median.Number.of.Roots"]
        
        result = sanitize_trait_names(df, trait_cols, return_mapping=False)
        
        # Should return only DataFrame, not tuple
        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, tuple)

    def test_data_preservation(self):
        """Test that data values are preserved during renaming."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "barcode": ["A", "B", "C"],
        })
        trait_cols = ["Median.Number.of.Roots"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False, barcode_col="barcode")
        
        # Data should be preserved (barcode is renamed to Barcode by default)
        assert result["Median Number Roots"].tolist() == [1, 2, 3]
        assert result["Barcode"].tolist() == ["A", "B", "C"]

    def test_non_trait_columns_unchanged(self):
        """Test that non-trait columns are sanitized appropriately."""
        df = pd.DataFrame({
            "Median.Number.of.Roots": [1, 2, 3],
            "barcode": ["A", "B", "C"],
            "geno": ["G1", "G2", "G3"],
        })
        trait_cols = ["Median.Number.of.Roots"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False, genotype_col="geno", barcode_col="barcode")
        
        # Metadata columns are sanitized by default
        assert "Barcode" in result.columns
        assert "Genotype" in result.columns
        # Original names should be gone
        assert "barcode" not in result.columns
        assert "geno" not in result.columns

    def test_empty_trait_list(self):
        """Test behavior with empty trait list."""
        df = pd.DataFrame({
            "col1": [1, 2, 3],
            "col2": [4, 5, 6],
        })
        trait_cols = []
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        # Should return unchanged dataframe
        assert result.equals(df)

    def test_already_clean_names(self):
        """Test that already clean names are handled gracefully."""
        df = pd.DataFrame({
            "Clean Name": [1, 2, 3],
            "Another Clean": [4, 5, 6],
        })
        trait_cols = ["Clean Name", "Another Clean"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        # Names should remain similar (may have minor changes due to processing)
        assert "Clean Name" in result.columns
        assert "Another Clean" in result.columns

    def test_multiple_units_in_dataset(self):
        """Test handling multiple different unit types."""
        df = pd.DataFrame({
            "Length.mm": [100, 200],
            "Area.mm2": [50, 60],
            "Volume.mm3": [1000, 2000],
            "Angle.deg": [45, 90],
        })
        trait_cols = ["Length.mm", "Area.mm2", "Volume.mm3", "Angle.deg"]
        
        result = sanitize_trait_names(df, trait_cols, abbreviate=False)
        
        assert "Length (mm)" in result.columns
        assert "Area (mm²)" in result.columns
        assert "Volume (mm³)" in result.columns
        assert "Angle (°)" in result.columns

    def test_real_world_trait_names(self):
        """Test with actual trait names from the project."""
        df = pd.DataFrame({
            "root_g": [0.1, 0.2],
            "shoot_g": [0.2, 0.3],
            "root_shoot_ratio": [0.5, 0.67],
            "Median.Number.of.Roots": [10, 12],
            "Maximum.Number.of.Roots": [30, 35],
            "Number.of.Root.Tips": [500, 600],
            "Total.Root.Length.mm": [5000, 6000],
            "Depth.mm": [200, 220],
            "Network.Area.mm2": [4000, 4500],
            "Lower.Root.Area.mm2": [1800, 2000],
            "Average.Hole.Size.mm2": [10, 12],
            "Average.Root.Orientation.deg": [45, 50],
            "Shallow.Angle.Frequency": [0.2, 0.3],
            "Medium.Angle.Frequency": [0.5, 0.4],
            "Steep.Angle.Frequency": [0.3, 0.3],
        })
        trait_cols = list(df.columns)
        
        result, mapping = sanitize_trait_names(
            df, trait_cols, abbreviate=True, return_mapping=True
        )
        
        # Check some key transformations
        assert "Root (g)" in result.columns
        assert "Shoot (g)" in result.columns
        assert "Root Shoot Ratio" in result.columns
        assert "Med Num Roots" in result.columns
        assert "Max Num Roots" in result.columns
        assert "Num Root Tips" in result.columns
        assert "Total Root Length (mm)" in result.columns
        assert "Depth (mm)" in result.columns
        assert "Network Area (mm²)" in result.columns
        assert "Lower Root Area (mm²)" in result.columns
        assert "Avg Hole Size (mm²)" in result.columns
        assert "Avg Root Orient (°)" in result.columns
        assert "Shallow Angle Freq" in result.columns
        assert "Medium Angle Freq" in result.columns
        assert "Steep Angle Freq" in result.columns
        
        # Check mapping contains all original names
        for col in trait_cols:
            assert col in mapping


    def test_metadata_sanitization_enabled(self):
        """Test that metadata columns are sanitized when enabled."""
        df = pd.DataFrame({
            "geno": ["G1", "G2", "G3"],
            "rep": [1, 2, 3],
            "barcode": ["A", "B", "C"],
            "root_g": [0.1, 0.2, 0.3],
        })
        trait_cols = ["root_g"]
        
        result, mapping = sanitize_trait_names(
            df, trait_cols, 
            sanitize_metadata=True, 
            return_mapping=True,
            genotype_col="geno",
            replicate_col="rep",
            barcode_col="barcode"
        )
        
        # Check metadata columns are renamed
        assert "Genotype" in result.columns
        assert "Replicate" in result.columns
        assert "Barcode" in result.columns
        assert "geno" not in result.columns
        assert "rep" not in result.columns
        
        # Check mapping includes metadata
        assert mapping["geno"] == "Genotype"
        assert mapping["rep"] == "Replicate"
        assert mapping["barcode"] == "Barcode"
        
        # Check data is preserved
        assert result["Genotype"].tolist() == ["G1", "G2", "G3"]
        assert result["Replicate"].tolist() == [1, 2, 3]

    def test_metadata_sanitization_disabled(self):
        """Test that metadata columns are NOT sanitized when disabled."""
        df = pd.DataFrame({
            "geno": ["G1", "G2", "G3"],
            "rep": [1, 2, 3],
            "root_g": [0.1, 0.2, 0.3],
        })
        trait_cols = ["root_g"]
        
        result = sanitize_trait_names(df, trait_cols, sanitize_metadata=False)
        
        # Check metadata columns are NOT renamed
        assert "geno" in result.columns
        assert "rep" in result.columns
        assert "Genotype" not in result.columns
        assert "Replicate" not in result.columns

    def test_metadata_genotype_variations(self):
        """Test that various genotype column names are handled."""
        # Test "geno"
        df1 = pd.DataFrame({"geno": ["G1", "G2"], "trait": [1, 2]})
        result1 = sanitize_trait_names(df1, ["trait"], sanitize_metadata=True, genotype_col="geno")
        assert "Genotype" in result1.columns
        
        # Test "genotype"  
        df2 = pd.DataFrame({"genotype": ["G1", "G2"], "trait": [1, 2]})
        result2 = sanitize_trait_names(df2, ["trait"], sanitize_metadata=True, genotype_col="genotype")
        assert "Genotype" in result2.columns
        
        # Test "Genotype" (already correct - no change)
        df3 = pd.DataFrame({"Genotype": ["G1", "G2"], "trait": [1, 2]})
        result3 = sanitize_trait_names(df3, ["trait"], sanitize_metadata=True, genotype_col="Genotype")
        assert "Genotype" in result3.columns

    def test_metadata_replicate_variations(self):
        """Test that various replicate column names are handled."""
        # Test "rep"
        df1 = pd.DataFrame({"rep": [1, 2], "trait": [1, 2]})
        result1 = sanitize_trait_names(df1, ["trait"], sanitize_metadata=True, replicate_col="rep")
        assert "Replicate" in result1.columns
        
        # Test "replicate"
        df2 = pd.DataFrame({"replicate": [1, 2], "trait": [1, 2]})
        result2 = sanitize_trait_names(df2, ["trait"], sanitize_metadata=True, replicate_col="replicate")
        assert "Replicate" in result2.columns

    def test_metadata_barcode_variations(self):
        """Test that barcode column names are handled."""
        # Test "barcode"
        df1 = pd.DataFrame({"barcode": ["A", "B"], "trait": [1, 2]})
        result1 = sanitize_trait_names(df1, ["trait"], sanitize_metadata=True, barcode_col="barcode")
        assert "Barcode" in result1.columns
        
        # Test "Barcode" (already correct)
        df2 = pd.DataFrame({"Barcode": ["A", "B"], "trait": [1, 2]})
        result2 = sanitize_trait_names(df2, ["trait"], sanitize_metadata=True, barcode_col="Barcode")
        assert "Barcode" in result2.columns

    def test_metadata_and_traits_together(self):
        """Test that both metadata and traits are sanitized together."""
        df = pd.DataFrame({
            "geno": ["G1", "G2"],
            "rep": [1, 2],
            "root_g": [0.1, 0.2],
            "Median.Number.of.Roots": [10, 12],
        })
        trait_cols = ["root_g", "Median.Number.of.Roots"]
        
        result, mapping = sanitize_trait_names(
            df, trait_cols, 
            abbreviate=True, 
            sanitize_metadata=True, 
            return_mapping=True,
            genotype_col="geno",
            replicate_col="rep"
        )
        
        # Check metadata
        assert "Genotype" in result.columns
        assert "Replicate" in result.columns
        
        # Check traits
        assert "Root (g)" in result.columns
        assert "Med Num Roots" in result.columns
        
        # Check mapping has both
        assert "geno" in mapping
        assert "rep" in mapping
        assert "root_g" in mapping
        assert "Median.Number.of.Roots" in mapping
        
        # Check all original columns are gone
        assert "geno" not in result.columns
        assert "rep" not in result.columns

    def test_metadata_no_interference_with_traits(self):
        """Test that metadata sanitization doesn't affect trait processing."""
        df = pd.DataFrame({
            "geno": ["G1", "G2"],
            "Total.Root.Length.mm": [100, 200],
        })
        trait_cols = ["Total.Root.Length.mm"]
        
        result = sanitize_trait_names(
            df, trait_cols, 
            abbreviate=False, 
            sanitize_metadata=True,
            genotype_col="geno"
        )
        
        # Both should be processed correctly
        assert "Genotype" in result.columns
        assert "Total Root Length (mm)" in result.columns

    def test_custom_replacement_basic(self):
        """Test basic custom replacement (crown -> seminal)."""
        df = pd.DataFrame({
            "crown_length_mm": [100, 200, 300],
            "crown_angle_deg": [45, 60, 75],
        })
        trait_cols = ["crown_length_mm", "crown_angle_deg"]

        result = sanitize_trait_names(
            df, trait_cols,
            custom_replacements={"crown": "seminal"}
        )

        assert "Seminal Length (mm)" in result.columns
        assert "Seminal Angle (°)" in result.columns
        assert "crown_length_mm" not in result.columns

    def test_custom_replacement_case_insensitive(self):
        """Test that custom replacements are case-insensitive."""
        df = pd.DataFrame({
            "Crown.Length": [100, 200, 300],
            "CROWN.Width": [10, 20, 30],
            "crown_angle": [45, 60, 75],
        })
        trait_cols = ["Crown.Length", "CROWN.Width", "crown_angle"]

        result = sanitize_trait_names(
            df, trait_cols,
            custom_replacements={"crown": "seminal"}
        )

        # All should have "crown" replaced with "seminal" regardless of case
        assert "Seminal Length" in result.columns
        assert "Seminal Width" in result.columns
        assert "Seminal Angle" in result.columns

    def test_custom_replacement_multiple(self):
        """Test multiple custom replacements in one call."""
        df = pd.DataFrame({
            "crown.length": [100, 200, 300],
            "primary.root.count": [5, 10, 15],
            "lateral.number": [20, 30, 40],
        })
        trait_cols = ["crown.length", "primary.root.count", "lateral.number"]

        result = sanitize_trait_names(
            df, trait_cols,
            custom_replacements={
                "crown": "seminal",
                "primary": "main",
                "lateral": "branch"
            },
            abbreviate=True
        )

        assert "Seminal Length" in result.columns
        assert "Main Root Count" in result.columns
        assert "Branch Num" in result.columns  # "Number" -> "Num" with abbreviate=True

    def test_custom_replacement_with_abbreviations(self):
        """Test that custom replacements work with abbreviations."""
        df = pd.DataFrame({
            "crown.maximum.length.mm": [100, 200, 300],
        })
        trait_cols = ["crown.maximum.length.mm"]

        result = sanitize_trait_names(
            df, trait_cols,
            custom_replacements={"crown": "seminal"},
            abbreviate=True
        )

        # Both custom replacement AND abbreviation should be applied
        assert "Seminal Max Length (mm)" in result.columns

    def test_custom_replacement_none_preserves_behavior(self):
        """Test that None custom_replacements preserves existing behavior."""
        df = pd.DataFrame({
            "crown.length.mm": [100, 200, 300],
        })
        trait_cols = ["crown.length.mm"]

        result_none = sanitize_trait_names(df, trait_cols, custom_replacements=None)
        result_empty = sanitize_trait_names(df, trait_cols, custom_replacements={})

        # Should use standard sanitization only
        assert "Crown Length (mm)" in result_none.columns
        assert "Crown Length (mm)" in result_empty.columns

    def test_custom_replacement_does_not_affect_metadata(self):
        """Test that custom replacements only apply to trait columns, not metadata."""
        df = pd.DataFrame({
            "geno": ["G1", "G2", "G3"],
            "crown.length": [100, 200, 300],
        })
        trait_cols = ["crown.length"]

        result = sanitize_trait_names(
            df, trait_cols,
            custom_replacements={"crown": "seminal"},
            sanitize_metadata=True,
            genotype_col="geno"
        )

        # Metadata should be sanitized normally
        assert "Genotype" in result.columns
        # Trait should have custom replacement
        assert "Seminal Length" in result.columns
        assert "geno" not in result.columns
        assert "crown.length" not in result.columns


class TestConvertToJsonSerializable:
    """Tests for convert_to_json_serializable function."""

    def test_numpy_integer(self):
        """Test conversion of numpy integers."""
        obj = {"value": np.int64(42)}
        result = convert_to_json_serializable(obj)
        assert isinstance(result["value"], int)
        assert result["value"] == 42

    def test_numpy_float(self):
        """Test conversion of numpy floats."""
        obj = {"value": np.float64(3.14)}
        result = convert_to_json_serializable(obj)
        assert isinstance(result["value"], float)
        assert abs(result["value"] - 3.14) < 0.001

    def test_numpy_bool(self):
        """Test conversion of numpy booleans."""
        obj = {"value": np.bool_(True)}
        result = convert_to_json_serializable(obj)
        assert isinstance(result["value"], bool)
        assert result["value"] is True

    def test_numpy_array(self):
        """Test conversion of numpy arrays."""
        obj = {"array": np.array([1, 2, 3])}
        result = convert_to_json_serializable(obj)
        assert isinstance(result["array"], list)
        assert result["array"] == [1, 2, 3]

    def test_nested_dict(self):
        """Test conversion of nested dictionaries."""
        obj = {
            "outer": {
                "inner": np.int64(42),
                "array": np.array([1.0, 2.0]),
            }
        }
        result = convert_to_json_serializable(obj)
        assert isinstance(result["outer"]["inner"], int)
        assert isinstance(result["outer"]["array"], list)

    def test_list_of_numpy(self):
        """Test conversion of lists containing numpy types."""
        obj = [np.int64(1), np.float64(2.5), np.bool_(True)]
        result = convert_to_json_serializable(obj)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], bool)


class TestCreateRunDirectory:
    """Tests for create_run_directory function."""

    def test_creates_directory(self, tmp_path):
        """Test that run directory is created."""
        base_dir = tmp_path / "runs"
        run_dir = create_run_directory(base_dir)
        
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == base_dir

    def test_creates_timestamped_name(self, tmp_path):
        """Test that directory has timestamped name."""
        base_dir = tmp_path / "runs"
        run_dir = create_run_directory(base_dir)
        
        # Should start with "run_" and have timestamp format
        assert run_dir.name.startswith("run_")
        assert len(run_dir.name) > 10  # "run_YYYYMMDD_HHMMSS"

    def test_creates_parent_directories(self, tmp_path):
        """Test that parent directories are created if needed."""
        base_dir = tmp_path / "nested" / "path" / "runs"
        run_dir = create_run_directory(base_dir)
        
        assert run_dir.exists()
        assert base_dir.exists()
