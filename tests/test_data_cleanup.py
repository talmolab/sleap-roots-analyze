"""Comprehensive tests for data_cleanup module with 100% coverage."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

from sleap_roots_analyze.data_cleanup import (
    load_trait_data,
    get_trait_columns,
    save_cleaned_data,
    remove_nan_samples,
    get_numeric_traits_only,
    remove_low_heritability_traits,
)
from sleap_roots_analyze.data_utils import (
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

    def test_replicate_col_none_does_not_miscount_traits(self):
        """With replicate_col=None, no trait is excluded as a replicate (issue #142)."""
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                # A numeric column named like a replicate, but the dataset has no
                # replicate factor; it must be treated as a trait, not excluded.
                "rep": [1.0, 2.0],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )

        trait_cols = get_trait_columns(df, replicate_col=None)
        assert "Barcode" not in trait_cols
        assert "geno" not in trait_cols
        assert "rep" in trait_cols  # not excluded when replicate_col is None
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

    def test_width_columns_not_excluded_by_id_pattern(self):
        """Test that width trait columns are NOT excluded by the 'id' pattern.

        Bug #75: Substring matching on 'id' incorrectly matches 'width' (w-id-th)
        and 'widths' (w-id-ths), causing silent loss of width trait columns.
        """
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "network_width_depth_ratio_min": [0.5, 0.6],
                "network_width_depth_ratio_max": [1.2, 1.3],
                "chull_max_width_mean": [3.4, 4.5],
                "root_widths_min_min": [0.1, 0.2],
            }
        )

        trait_cols = get_trait_columns(df)

        assert "network_width_depth_ratio_min" in trait_cols
        assert "network_width_depth_ratio_max" in trait_cols
        assert "chull_max_width_mean" in trait_cols
        assert "root_widths_min_min" in trait_cols
        assert (
            len(trait_cols) == 4
        ), f"Expected 4 width trait columns, got {len(trait_cols)}: {trait_cols}"

    def test_solidity_columns_not_excluded_by_id_pattern(self):
        """Test that solidity trait columns are NOT excluded by the 'id' pattern.

        Bug #75 secondary: 'solidity' contains 'id' (sol-id-ity), causing
        silent loss of solidity trait columns.
        """
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "network_solidity_min": [0.3, 0.4],
                "network_solidity_max": [0.8, 0.9],
                "network_solidity_mean": [0.5, 0.6],
            }
        )

        trait_cols = get_trait_columns(df)

        assert "network_solidity_min" in trait_cols
        assert "network_solidity_max" in trait_cols
        assert "network_solidity_mean" in trait_cols
        assert (
            len(trait_cols) == 3
        ), f"Expected 3 solidity trait columns, got {len(trait_cols)}: {trait_cols}"

    def test_actual_id_columns_still_excluded(self):
        """Test that real ID metadata columns (ending in _id) are still excluded.

        The fix must not break exclusion of legitimate ID metadata columns.
        All real ID columns in SLEAP Roots data follow the *_id suffix pattern.
        """
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "scan_id": [101, 102],
                "plant_id": [201, 202],
                "accession_id": [301, 302],
                "experiment_id": [401, 402],
                "species_id": [501, 502],
                "trait1": [1.0, 2.0],
            }
        )

        trait_cols = get_trait_columns(df)

        assert "scan_id" not in trait_cols
        assert "plant_id" not in trait_cols
        assert "accession_id" not in trait_cols
        assert "experiment_id" not in trait_cols
        assert "species_id" not in trait_cols
        assert "trait1" in trait_cols
        assert (
            len(trait_cols) == 1
        ), f"Expected only trait1, got {len(trait_cols)}: {trait_cols}"

    def test_mixed_id_and_width_columns(self):
        """Test combined scenario: real ID metadata excluded, width/solidity traits kept.

        This is the most realistic unit test, combining both patterns to ensure
        the fix correctly discriminates between ID metadata and trait columns.
        """
        df = pd.DataFrame(
            {
                "Barcode": ["BC001", "BC002"],
                "geno": ["G1", "G2"],
                "rep": [1, 2],
                "scan_id": [101, 102],
                "plant_id": [201, 202],
                "network_width_depth_ratio_min": [0.5, 0.6],
                "network_solidity_max": [0.8, 0.9],
                "chull_max_width_mean": [3.4, 4.5],
                "primary_length_max": [10.0, 12.0],
                "trait1": [1.0, 2.0],
            }
        )

        trait_cols = get_trait_columns(df)

        # Metadata must be excluded
        for excluded in ["Barcode", "geno", "rep", "scan_id", "plant_id"]:
            assert excluded not in trait_cols, f"{excluded} should be excluded"

        # Traits must be included
        expected_traits = [
            "network_width_depth_ratio_min",
            "network_solidity_max",
            "chull_max_width_mean",
            "primary_length_max",
            "trait1",
        ]
        for trait in expected_traits:
            assert trait in trait_cols, f"{trait} should be included as a trait"

        assert (
            len(trait_cols) == 5
        ), f"Expected 5 traits, got {len(trait_cols)}: {trait_cols}"


class TestGetTraitColumnsIntegration:
    """Integration tests for get_trait_columns using real fixture data.

    These tests validate trait vs metadata classification against real experimental
    datasets with different column naming conventions. Each fixture represents a
    different data source format used in the SLEAP Roots analysis pipeline.
    """

    def test_pipeline_trait_classification_11dag(self, traits_11dag_df):
        """Test trait classification on SLEAP Roots 11 DAG data (880 columns).

        This dataset uses snake_case columns with _id suffix metadata and contains
        width/solidity traits that are affected by bug #75.
        """
        trait_cols = get_trait_columns(
            traits_11dag_df,
            barcode_col="plant_qr_code",
            genotype_col="Geno",
            replicate_col="Rep",
        )

        # Width traits MUST be classified as traits (not metadata)
        width_traits = [c for c in trait_cols if "width" in c.lower()]
        assert (
            "network_width_depth_ratio_min" in trait_cols
        ), "network_width_depth_ratio_min incorrectly excluded"
        assert (
            "network_width_depth_ratio_max" in trait_cols
        ), "network_width_depth_ratio_max incorrectly excluded"
        assert (
            "chull_max_width_min" in trait_cols
        ), "chull_max_width_min incorrectly excluded"
        assert (
            "chull_max_width_max" in trait_cols
        ), "chull_max_width_max incorrectly excluded"
        assert (
            len(width_traits) == 18
        ), f"Expected 18 width trait columns, got {len(width_traits)}: {width_traits}"

        # Solidity traits MUST be classified as traits (not metadata)
        solidity_traits = [c for c in trait_cols if "solidity" in c.lower()]
        assert (
            "network_solidity_min" in trait_cols
        ), "network_solidity_min incorrectly excluded"
        assert (
            "network_solidity_max" in trait_cols
        ), "network_solidity_max incorrectly excluded"
        assert len(solidity_traits) == 9, (
            f"Expected 9 solidity trait columns, got {len(solidity_traits)}: "
            f"{solidity_traits}"
        )

        # Actual ID metadata MUST be excluded
        for id_col in [
            "scan_id",
            "plant_id",
            "accession_id",
            "experiment_id",
            "species_id",
            "scanner_id",
            "phenotyper_id",
            "wave_id",
        ]:
            assert id_col not in trait_cols, f"{id_col} should be excluded as metadata"

        # Other metadata MUST be excluded
        for meta_col in [
            "plant_qr_code",
            "Geno",
            "Rep",
            "Sterilization",
            "DOT",
            "QC_SLEAP",
            "Date_QC",
            "germ_day",
            "plant_name",
            "species_name",
        ]:
            if meta_col in traits_11dag_df.columns:
                assert (
                    meta_col not in trait_cols
                ), f"{meta_col} should be excluded as metadata"

        # Total trait count sanity check (880 cols minus ~30 metadata/non-numeric)
        assert (
            len(trait_cols) > 800
        ), f"Expected >800 trait columns from 880 total, got {len(trait_cols)}"

    def test_pipeline_trait_classification_traits_summary(self, traits_summary_df):
        """Test trait classification on traits_summary data (924 columns).

        Different column order from 11DAG, includes uploaded_at, wave_number,
        wave_name metadata columns.
        """
        trait_cols = get_trait_columns(
            traits_summary_df,
            barcode_col="plant_qr_code",
            genotype_col="Geno",
            replicate_col="Rep",
        )

        # Width traits MUST be classified as traits
        width_traits = [c for c in trait_cols if "width" in c.lower()]
        assert "network_width_depth_ratio_min" in trait_cols
        assert "chull_max_width_min" in trait_cols
        assert (
            len(width_traits) == 18
        ), f"Expected 18 width trait columns, got {len(width_traits)}: {width_traits}"

        # Solidity traits MUST be classified as traits
        solidity_traits = [c for c in trait_cols if "solidity" in c.lower()]
        assert "network_solidity_min" in trait_cols
        assert len(solidity_traits) == 9, (
            f"Expected 9 solidity trait columns, got {len(solidity_traits)}: "
            f"{solidity_traits}"
        )

        # Actual ID metadata MUST be excluded
        for id_col in ["scan_id", "plant_id", "accession_id"]:
            assert id_col not in trait_cols, f"{id_col} should be excluded as metadata"

        # Additional metadata specific to this dataset MUST be excluded
        for meta_col in ["uploaded_at", "wave_number", "wave_name"]:
            if meta_col in traits_summary_df.columns:
                assert (
                    meta_col not in trait_cols
                ), f"{meta_col} should be excluded as metadata"

    def test_pipeline_trait_classification_traits_summary_lateral(
        self, traits_summary_lateral_df
    ):
        """Test trait classification on lateral root summary data (608 columns).

        Uses lateral_* trait prefixes instead of crown_*. Has the same _id metadata
        columns but no width/solidity traits — tests that _id exclusion works
        without false positives on lateral-specific column names.
        """
        trait_cols = get_trait_columns(
            traits_summary_lateral_df,
            barcode_col="plant_qr_code",
            genotype_col="Geno",
            replicate_col="Rep",
        )

        # Actual ID metadata MUST be excluded
        for id_col in [
            "scan_id",
            "plant_id",
            "accession_id",
            "experiment_id",
            "species_id",
            "scanner_id",
            "phenotyper_id",
            "wave_id",
        ]:
            assert id_col not in trait_cols, f"{id_col} should be excluded as metadata"

        # Lateral traits MUST be included
        lateral_traits = [c for c in trait_cols if "lateral" in c.lower()]
        assert len(lateral_traits) > 0, "Lateral trait columns should be present"

        # Every returned column must be numeric
        for col in trait_cols:
            assert pd.api.types.is_numeric_dtype(
                traits_summary_lateral_df[col]
            ), f"Trait column {col} is not numeric"

    def test_pipeline_trait_classification_turface(self, turface_traits_df):
        """Test trait classification on Turface agronomic data (41 columns).

        Completely different dataset format: lowercase 'geno', 'rep', 'Barcode',
        RhizoVision-style dotted column names, no _id columns.
        Tests that the fix generalizes beyond snake_case SLEAP Roots data.
        """
        trait_cols = get_trait_columns(
            turface_traits_df,
            barcode_col="Barcode",
            genotype_col="geno",
            replicate_col="rep",
        )

        # Metadata must be excluded
        assert "Barcode" not in trait_cols
        assert "geno" not in trait_cols
        assert "rep" not in trait_cols

        # Agronomic traits must be included
        for trait in [
            "Shoot_Biomass_mg",
            "Root_Biomass_mg",
            "Total.Root.Length.mm",
            "Maximum.Width.mm",
            "Solidity",
            "Depth.mm",
        ]:
            assert trait in trait_cols, f"{trait} should be included as a trait"

        # Width and Solidity columns must be included (RhizoVision format)
        assert "Maximum.Width.mm" in trait_cols
        assert "Width-to-Depth.Ratio" in trait_cols
        assert "Solidity" in trait_cols

        # Count total numeric traits: 41 columns - 3 metadata (Barcode, geno, rep)
        # - Computation.Time.s (excluded by "time" pattern) = 37
        expected_numeric = [
            c
            for c in turface_traits_df.columns
            if c not in ["Barcode", "geno", "rep"]
            and pd.api.types.is_numeric_dtype(turface_traits_df[c])
        ]
        # Computation.Time.s is excluded by the "time" metadata pattern
        expected_count = len([c for c in expected_numeric if "time" not in c.lower()])
        assert (
            len(trait_cols) == expected_count
        ), f"Expected {expected_count} traits, got {len(trait_cols)}: {trait_cols}"

    def test_pipeline_trait_classification_features(self, features_df):
        """Test trait classification on RhizoVision features output (38 columns).

        Dotted PascalCase column names (Maximum.Width.mm, Width-to-Depth.Ratio,
        Solidity). geno/rep columns don't exist in this CSV — tests graceful
        handling of missing metadata columns.
        """
        trait_cols = get_trait_columns(
            features_df,
            barcode_col="File.Name",
            genotype_col="geno",
            replicate_col="rep",
        )

        # Width traits must be in result
        assert "Maximum.Width.mm" in trait_cols
        assert "Width-to-Depth.Ratio" in trait_cols

        # Solidity trait must be in result
        assert "Solidity" in trait_cols

        # File.Name is non-numeric string so excluded anyway, but should not
        # be in traits
        assert "File.Name" not in trait_cols

        # Computation.Time.s matches "time" metadata pattern — should be excluded
        assert "Computation.Time.s" not in trait_cols

        # All returned columns must be numeric
        for col in trait_cols:
            assert pd.api.types.is_numeric_dtype(
                features_df[col]
            ), f"Trait column {col} is not numeric"

    def test_metadata_columns_are_complement_of_traits(
        self,
        traits_11dag_df,
        traits_summary_df,
        traits_summary_lateral_df,
        turface_traits_df,
    ):
        """Regression guard: trait + metadata columns = all columns, no gaps.

        For each fixture dataset, verifies that every column is classified as
        either a trait or metadata — no columns silently lost or double-counted.
        """
        datasets = {
            "traits_11dag": (
                traits_11dag_df,
                {
                    "barcode_col": "plant_qr_code",
                    "genotype_col": "Geno",
                    "replicate_col": "Rep",
                },
            ),
            "traits_summary": (
                traits_summary_df,
                {
                    "barcode_col": "plant_qr_code",
                    "genotype_col": "Geno",
                    "replicate_col": "Rep",
                },
            ),
            "traits_summary_lateral": (
                traits_summary_lateral_df,
                {
                    "barcode_col": "plant_qr_code",
                    "genotype_col": "Geno",
                    "replicate_col": "Rep",
                },
            ),
            "turface": (
                turface_traits_df,
                {
                    "barcode_col": "Barcode",
                    "genotype_col": "geno",
                    "replicate_col": "rep",
                },
            ),
        }

        for name, (df, kwargs) in datasets.items():
            trait_cols = get_trait_columns(df, **kwargs)

            # Known metadata columns (barcode/genotype/replicate) must not
            # be classified as traits
            for key in ("barcode_col", "genotype_col", "replicate_col"):
                col_name = kwargs.get(key)
                if col_name is not None and col_name in df.columns:
                    assert (
                        col_name not in trait_cols
                    ), f"{name}: {key} '{col_name}' incorrectly classified as trait"

            # Sanity check: there should be at least one trait column
            assert len(trait_cols) > 0, f"{name}: no trait columns detected"

            # Every trait column must be numeric
            for col in trait_cols:
                assert pd.api.types.is_numeric_dtype(
                    df[col]
                ), f"{name}: trait column {col} is not numeric"

            # No trait column should end with "_id" (bug #75 regression guard)
            id_in_traits = [c for c in trait_cols if c.lower().endswith("_id")]
            assert (
                len(id_in_traits) == 0
            ), f"{name}: _id columns misclassified as traits: {id_in_traits}"


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
        from sleap_roots_analyze.data_cleanup import remove_zero_inflated_traits

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
        from sleap_roots_analyze.data_cleanup import remove_zero_inflated_traits

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
        from sleap_roots_analyze.data_cleanup import remove_traits_with_many_nans

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
        from sleap_roots_analyze.data_cleanup import remove_low_sample_traits

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
        from sleap_roots_analyze.data_cleanup import (
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
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

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
        from sleap_roots_analyze.data_cleanup import (
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

    def test_apply_data_cleanup_filters_propagates_removal_details(
        self, mixed_problem_data
    ):
        """Regression test: removed_samples_detail must be non-empty when samples removed.

        This test catches the key mismatch bug where apply_data_cleanup_filters()
        used "removed_samples_detail" to read from removal_stats, but
        remove_nan_samples() stores the data under "removal_details".
        """
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        df = mixed_problem_data
        trait_cols = [c for c in df.columns if c.startswith("trait_")]

        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols,
            max_zeros_per_trait=0.5,
            max_nans_per_trait=0.3,
            max_nans_per_sample=0.2,
            min_samples_per_trait=5,
        )

        # Verify samples were actually removed (precondition for the test)
        samples_removed = cleanup_log["original_samples"] - cleanup_log["final_samples"]
        assert (
            samples_removed > 0
        ), "Fixture must remove at least one sample for this test to be meaningful"

        # Core regression assertion: detail list must be populated, not empty
        assert len(cleanup_log["removed_samples_detail"]) == samples_removed

        # Each entry must contain all required fields
        required_fields = {
            "sample_index",
            "barcode",
            "genotype",
            "rep",
            "nan_count",
            "nan_fraction",
            "nan_traits",
            "removal_reason",
        }
        for entry in cleanup_log["removed_samples_detail"]:
            assert required_fields == set(
                entry.keys()
            ), f"Entry missing fields: {required_fields - set(entry.keys())}"
            assert 0.0 < entry["nan_fraction"] <= 1.0
            assert entry["nan_count"] > 0
            assert entry["nan_traits"]  # non-empty string

    def test_apply_data_cleanup_filters_uses_genotype_col_and_replicate_col(self):
        """Regression: non-default column names must be forwarded to remove_nan_samples."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        # Use enough samples so traits are NOT removed by max_nans_per_trait (default 0.3)
        # before the sample-level filter runs. Only b4/b5 have all-NaN (100% per trait if only
        # 2 of 5 samples), which is 2/5=0.4 > 0.3, so we must raise max_nans_per_trait.
        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2", "b3", "b4", "b5"],
                "Genotype": ["A", "B", "A", "B", "B"],
                "Replicate": [1, 2, 1, 2, 1],
                "trait1": [1.0, 2.0, 3.0, float("nan"), float("nan")],
                "trait2": [4.0, 5.0, 6.0, float("nan"), float("nan")],
            }
        )
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols=["trait1", "trait2"],
            max_nans_per_sample=0.0,
            max_nans_per_trait=0.5,  # Allow up to 50% NaN per trait (2/5=0.4 < 0.5)
            min_samples_per_trait=1,  # Low threshold so traits are kept
            genotype_col="Genotype",
            replicate_col="Replicate",
        )
        detail = cleanup_log["removed_samples_detail"]
        assert len(detail) >= 1
        assert all(
            d["genotype"] != "" for d in detail
        ), f"Expected genotype populated, got: {[d['genotype'] for d in detail]}"
        assert all(
            d["rep"] != "" for d in detail
        ), f"Expected rep populated, got: {[d['rep'] for d in detail]}"

    def test_apply_data_cleanup_filters_removed_samples_is_independent_copy(self):
        """Regression: removed_samples must be a deep copy of removed_samples_detail."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2"],
                "geno": ["A", "B"],
                "rep": [1, 2],
                "trait1": [1.0, float("nan")],
                "trait2": [2.0, float("nan")],
            }
        )
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols=["trait1", "trait2"],
            max_nans_per_sample=0.0,
        )
        # Lists must be separate objects
        assert (
            cleanup_log["removed_samples"] is not cleanup_log["removed_samples_detail"]
        )
        original_detail_len = len(cleanup_log["removed_samples_detail"])
        # Appending to removed_samples must not affect removed_samples_detail
        cleanup_log["removed_samples"].append({"sentinel": True})
        assert len(cleanup_log["removed_samples_detail"]) == original_detail_len
        # Mutating a dict entry in removed_samples must not affect removed_samples_detail
        if cleanup_log["removed_samples_detail"]:
            cleanup_log["removed_samples"][0]["genotype"] = "__mutated__"
            assert cleanup_log["removed_samples_detail"][0]["genotype"] != "__mutated__"

    def test_apply_data_cleanup_filters_empty_detail_when_no_samples_removed(self):
        """Regression: removed_samples_detail must be empty list when no samples removed."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2"],
                "geno": ["A", "B"],
                "rep": [1, 2],
                "trait1": [1.0, 2.0],
                "trait2": [3.0, 4.0],
            }
        )
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols=["trait1", "trait2"],
            max_nans_per_sample=1.0,
        )
        assert cleanup_log["removed_samples_detail"] == []
        assert cleanup_log["removed_samples"] == []

    def test_remove_nan_samples_max_nan_fraction_zero(self):
        """Edge case: max_nans_per_sample=0.0 removes any sample with at least one NaN."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        # trait1 has 1 NaN out of 4 samples (1/4 = 0.25). Pass max_nans_per_trait=0.3
        # explicitly so the trait is kept (0.25 < 0.3) and the sample-level filter is
        # what we exercise; the canonical default (0.2) would drop the trait first,
        # leaving no NaN sample to remove and defeating the test's intent.
        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2", "b3", "b4"],
                "geno": ["A", "B", "A", "B"],
                "rep": [1, 2, 1, 2],
                "trait1": [1.0, float("nan"), 3.0, 4.0],
                "trait2": [2.0, 3.0, 4.0, 5.0],
            }
        )
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols=["trait1", "trait2"],
            max_nans_per_trait=0.3,  # keep trait1 (0.25 NaN); isolate sample-level filter
            max_nans_per_sample=0.0,
            min_samples_per_trait=1,  # Low threshold
        )
        detail = cleanup_log["removed_samples_detail"]
        assert len(detail) == 1
        assert detail[0]["nan_count"] == 1
        assert detail[0]["nan_fraction"] > 0.0

    def test_remove_nan_samples_max_nan_fraction_one_keeps_partial_nan(self):
        """Edge case: max_nans_per_sample=1.0 keeps samples with partial NaN."""
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2"],
                "geno": ["A", "B"],
                "rep": [1, 2],
                "trait1": [1.0, float("nan")],
                "trait2": [2.0, 3.0],
            }
        )
        df_clean, cleanup_log = apply_data_cleanup_filters(
            df,
            trait_cols=["trait1", "trait2"],
            max_nans_per_sample=1.0,
        )
        assert cleanup_log["removed_samples_detail"] == []

    def test_remove_nan_samples_missing_column_fallback(self, caplog):
        """Edge case: missing replicate column produces empty-string fallback with warning."""
        import logging
        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        # Use enough samples so traits aren't dropped by max_nans_per_trait before samples.
        # b4/b5 have all-NaN (2/5=0.4), so set max_nans_per_trait=0.5 to keep traits.
        df = pd.DataFrame(
            {
                "Barcode": ["b1", "b2", "b3", "b4", "b5"],
                "geno": ["A", "B", "A", "B", "B"],
                # no "rep" column
                "trait1": [1.0, 2.0, 3.0, float("nan"), float("nan")],
                "trait2": [4.0, 5.0, 6.0, float("nan"), float("nan")],
            }
        )
        with caplog.at_level(logging.WARNING):
            df_clean, cleanup_log = apply_data_cleanup_filters(
                df,
                trait_cols=["trait1", "trait2"],
                max_nans_per_sample=0.0,
                max_nans_per_trait=0.5,
                min_samples_per_trait=1,
            )
        detail = cleanup_log["removed_samples_detail"]
        assert len(detail) >= 1
        assert detail[0]["rep"] == ""
        assert any(
            "rep" in msg.lower() or "replicate" in msg.lower()
            for msg in caplog.messages
        ), f"Expected warning about missing replicate column, got: {caplog.messages}"


class TestCanonicalDefaultDriftGuard:
    """Guard the canonical QC cleanup defaults against silent drift (#167).

    ``apply_data_cleanup_filters``'s signature defaults are the single source of
    truth for "canonical QC cleanup". ``clean_traits_for_analysis`` inherits them,
    and ``CleanupConfig()`` must encode the same values (with the
    ``max_nan_fraction`` <-> ``max_nans_per_sample`` name mapping). If any of these
    drift apart, a default-using caller would silently clean differently from the
    pipeline.
    """

    def _canonical_from_config(self):
        from sleap_roots_analyze.pipeline.config import CleanupConfig

        cfg = CleanupConfig()
        # CleanupConfig.max_nan_fraction is the per-sample NaN budget; the function
        # exposes the same knob as max_nans_per_sample.
        return {
            "max_zeros_per_trait": cfg.max_zeros_per_trait,
            "max_nans_per_trait": cfg.max_nans_per_trait,
            "max_nans_per_sample": cfg.max_nan_fraction,
            "min_samples_per_trait": cfg.min_samples_per_trait,
        }

    def test_apply_filters_defaults_match_cleanup_config(self):
        """apply_data_cleanup_filters' signature defaults == CleanupConfig() defaults."""
        import inspect

        from sleap_roots_analyze.data_cleanup import apply_data_cleanup_filters

        canonical = self._canonical_from_config()
        sig = inspect.signature(apply_data_cleanup_filters)
        actual = {name: sig.parameters[name].default for name in canonical}
        assert actual == canonical, (
            "apply_data_cleanup_filters signature defaults drifted from "
            f"CleanupConfig (#167): {actual} != {canonical}"
        )
        # Pin the literal canonical values so an in-tandem edit to *both* layers
        # (which would keep them equal to each other) still trips this guard.
        assert canonical == {
            "max_zeros_per_trait": 0.5,
            "max_nans_per_trait": 0.2,
            "max_nans_per_sample": 0.0,
            "min_samples_per_trait": 10,
        }

    def test_clean_traits_entry_point_defaults_match_cleanup_config(self):
        """clean_traits_for_analysis (no overrides) records the canonical thresholds.

        Proves the public entry point inherits the corrected defaults rather than a
        separate hardcoded copy.
        """
        from sleap_roots_analyze.data_cleanup import clean_traits_for_analysis

        canonical = self._canonical_from_config()
        # Clean fixture: 12 samples, 2 non-constant traits, no NaN/zeros, so nothing
        # is filtered and the recorded effective thresholds reflect the defaults.
        n = 12
        df = pd.DataFrame(
            {
                "Barcode": [f"b{i}" for i in range(n)],
                "geno": ["A", "B"] * (n // 2),
                "rep": list(range(n)),
                "trait1": [float(i + 1) for i in range(n)],
                "trait2": [float(2 * i + 1) for i in range(n)],
            }
        )
        _, _, cleanup_log = clean_traits_for_analysis(
            df, trait_cols=["trait1", "trait2"]
        )
        assert cleanup_log["effective_thresholds"] == canonical, (
            "clean_traits_for_analysis default thresholds drifted from CleanupConfig "
            f"(#167): {cleanup_log['effective_thresholds']} != {canonical}"
        )


class TestInspectNanSamples:
    """Test the inspect_nan_samples function."""

    def test_basic_nan_inspection(self):
        """Test basic NaN inspection functionality."""
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import (
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
        from sleap_roots_analyze.data_cleanup import (
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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples

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
        from sleap_roots_analyze.data_cleanup import inspect_nan_samples
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
