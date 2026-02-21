"""Tests for golden configuration templates.

This module validates that the golden templates in configs/templates/ are
complete and pass schema validation when placeholders are replaced with valid values.
"""

from pathlib import Path
import tempfile
import yaml

import pytest

from sleap_roots_analyze.pipeline.config.utils import (
    load_qc_config,
    load_viz_config,
    validate_qc_config,
    validate_viz_config,
)


@pytest.fixture
def templates_dir():
    """Return path to templates directory."""
    repo_root = Path(__file__).parent.parent
    return repo_root / "configs" / "templates"


@pytest.fixture
def temp_csv(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_path = tmp_path / "test_data.csv"
    csv_path.write_text(
        "Barcode,Genotype,Replicate,plant_age_days,trait1,trait2\n"
        "p1,A,1,0,1.0,2.0\n"
        "p2,A,2,0,1.1,2.1\n"
        "p3,B,1,0,1.2,2.2\n"
    )
    return str(csv_path)


@pytest.fixture
def temp_image_dir(tmp_path):
    """Create a temporary image directory for testing."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    return str(image_dir)


class TestQCTemplateGrouped:
    """Tests for qc_template_grouped.yaml."""

    def test_template_exists(self, templates_dir):
        """Template file exists."""
        template_path = templates_dir / "qc_template_grouped.yaml"
        assert template_path.exists()

    def test_template_validates_with_filled_placeholders(
        self, templates_dir, temp_csv, tmp_path
    ):
        """Template passes validation when placeholders are replaced."""
        template_path = templates_dir / "qc_template_grouped.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Replace placeholders
        template_dict["pipeline_name"] = "test_grouped_qc"
        template_dict["data"]["csv_path"] = temp_csv
        template_dict["data"]["group_by"] = "plant_age_days"

        # Write modified template to temp file
        temp_config_path = tmp_path / "test_qc_grouped.yaml"
        with open(temp_config_path, "w") as f:
            yaml.safe_dump(template_dict, f)

        # Load and validate
        config = load_qc_config(temp_config_path)
        validate_qc_config(config)  # Should not raise

        # Verify key fields
        assert config.pipeline_name == "test_grouped_qc"
        assert config.data.csv_path == temp_csv
        assert config.data.group_by == "plant_age_days"


class TestQCTemplateUngrouped:
    """Tests for qc_template_ungrouped.yaml."""

    def test_template_exists(self, templates_dir):
        """Template file exists."""
        template_path = templates_dir / "qc_template_ungrouped.yaml"
        assert template_path.exists()

    def test_template_validates_with_filled_placeholders(
        self, templates_dir, temp_csv, tmp_path
    ):
        """Template passes validation when placeholders are replaced."""
        template_path = templates_dir / "qc_template_ungrouped.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Replace placeholders
        template_dict["pipeline_name"] = "test_ungrouped_qc"
        template_dict["data"]["csv_path"] = temp_csv
        template_dict["columns"]["barcode"] = "Barcode"
        template_dict["columns"]["genotype"] = "Genotype"
        template_dict["columns"]["replicate"] = "Replicate"

        # Write modified template to temp file
        temp_config_path = tmp_path / "test_qc_ungrouped.yaml"
        with open(temp_config_path, "w") as f:
            yaml.safe_dump(template_dict, f)

        # Load and validate
        config = load_qc_config(temp_config_path)
        validate_qc_config(config)  # Should not raise

        # Verify key fields
        assert config.pipeline_name == "test_ungrouped_qc"
        assert config.data.csv_path == temp_csv
        assert config.data.group_by is None


class TestVizTemplateWithImages:
    """Tests for viz_template_with_images.yaml."""

    def test_template_exists(self, templates_dir):
        """Template file exists."""
        template_path = templates_dir / "viz_template_with_images.yaml"
        assert template_path.exists()

    def test_template_validates_with_filled_placeholders(
        self, templates_dir, temp_csv, temp_image_dir, tmp_path
    ):
        """Template passes validation when placeholders are replaced."""
        template_path = templates_dir / "viz_template_with_images.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Replace placeholders
        template_dict["pipeline_name"] = "test_viz_with_images"
        template_dict["data"]["csv_path"] = temp_csv
        template_dict["data"]["image_dir"] = temp_image_dir

        # Write modified template to temp file
        temp_config_path = tmp_path / "test_viz_with_images.yaml"
        with open(temp_config_path, "w") as f:
            yaml.safe_dump(template_dict, f)

        # Load and validate
        config = load_viz_config(temp_config_path)
        validate_viz_config(config)  # Should not raise

        # Verify key fields
        assert config.pipeline_name == "test_viz_with_images"
        assert config.data.csv_path == temp_csv
        assert config.data.image_dir == temp_image_dir
        assert config.interesting_genotypes.generate_image_grids is True
        assert config.interactive_viz.show_images_on_hover is True


class TestVizTemplateNoImages:
    """Tests for viz_template_no_images.yaml."""

    def test_template_exists(self, templates_dir):
        """Template file exists."""
        template_path = templates_dir / "viz_template_no_images.yaml"
        assert template_path.exists()

    def test_template_validates_with_filled_placeholders(
        self, templates_dir, temp_csv, tmp_path
    ):
        """Template passes validation when placeholders are replaced."""
        template_path = templates_dir / "viz_template_no_images.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Replace placeholders
        template_dict["pipeline_name"] = "test_viz_no_images"
        template_dict["data"]["csv_path"] = temp_csv

        # Write modified template to temp file
        temp_config_path = tmp_path / "test_viz_no_images.yaml"
        with open(temp_config_path, "w") as f:
            yaml.safe_dump(template_dict, f)

        # Load and validate
        config = load_viz_config(temp_config_path)
        validate_viz_config(config)  # Should not raise

        # Verify key fields
        assert config.pipeline_name == "test_viz_no_images"
        assert config.data.csv_path == temp_csv
        assert config.data.image_dir is None
        assert config.interesting_genotypes.generate_image_grids is False
        assert config.interactive_viz.show_images_on_hover is False


class TestRunManifestTemplate:
    """Tests for run_manifest_template.yaml."""

    def test_template_exists(self, templates_dir):
        """Template file exists."""
        template_path = templates_dir / "run_manifest_template.yaml"
        assert template_path.exists()

    def test_template_has_required_structure(self, templates_dir):
        """Template contains all required run manifest fields."""
        template_path = templates_dir / "run_manifest_template.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Check required top-level fields
        assert "run_name" in template_dict
        assert "description" in template_dict
        assert "qc_configs" in template_dict
        assert "viz_configs" in template_dict
        assert "qc_mapping" in template_dict

        # Check field types
        assert isinstance(template_dict["qc_configs"], list)
        assert isinstance(template_dict["viz_configs"], list)
        assert isinstance(template_dict["qc_mapping"], dict)

    def test_template_validates_with_filled_placeholders(self, templates_dir, tmp_path):
        """Template can be successfully filled with valid paths."""
        template_path = templates_dir / "run_manifest_template.yaml"

        # Load template as dict
        with open(template_path) as f:
            template_dict = yaml.safe_load(f)

        # Replace placeholders
        template_dict["run_name"] = "Test Run"
        template_dict["description"] = "Test run description"
        template_dict["qc_configs"] = ["qc/my_qc_config.yaml"]
        template_dict["viz_configs"] = ["viz/my_viz_config.yaml"]
        template_dict["qc_mapping"] = {"viz/my_viz_config.yaml": "qc/my_qc_config.yaml"}

        # Write modified template to temp file
        temp_manifest_path = tmp_path / "test_manifest.yaml"
        with open(temp_manifest_path, "w") as f:
            yaml.safe_dump(template_dict, f)

        # Verify it can be loaded back
        with open(temp_manifest_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["run_name"] == "Test Run"
        assert loaded["qc_configs"] == ["qc/my_qc_config.yaml"]
        assert loaded["viz_configs"] == ["viz/my_viz_config.yaml"]
        assert loaded["qc_mapping"] == {
            "viz/my_viz_config.yaml": "qc/my_qc_config.yaml"
        }
