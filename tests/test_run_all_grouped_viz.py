"""TDD tests for run-all grouped viz fan-out (GitHub #69).

Written BEFORE implementation per OpenSpec TDD workflow.
Covers tasks 5.1a–5.3j and integration tasks 6.2–6.3 from
openspec/changes/add-group-by-timepoint/tasks.md.

Root cause of the gap: these tests were defined in tasks.md but never
implemented, allowing the fan-out bug to ship undetected.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from sleap_roots_analyze.pipeline_runner import PipelineRunner


# =============================================================================
# Shared test helpers
# =============================================================================


def _write_manifest(directory: Path, qc_has_group_by: bool) -> Path:
    """Write a minimal run manifest and required config stubs."""
    directory.mkdir(parents=True, exist_ok=True)

    qc_dir = directory / "qc"
    qc_dir.mkdir(exist_ok=True)
    group_by_line = "  group_by: plant_age_days\n" if qc_has_group_by else ""
    (qc_dir / "test_qc.yaml").write_text(
        f"pipeline_name: test_qc\ndata:\n{group_by_line}  csv_path: data.csv\n"
    )

    viz_dir = directory / "viz"
    viz_dir.mkdir(exist_ok=True)
    (viz_dir / "test_viz.yaml").write_text(
        "pipeline_name: test_viz\ndata:\n  csv_path: old_data.csv\n"
    )

    manifest = {
        "run_name": "Test Run",
        "qc_configs": ["qc/test_qc.yaml"],
        "viz_configs": ["viz/test_viz.yaml"],
        "qc_mapping": {"viz/test_viz.yaml": "qc/test_qc.yaml"},
    }
    manifest_path = directory / "run_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f)
    return manifest_path


def _make_group_qc_dirs(base_dir: Path, group_by: str, values: list) -> list[Path]:
    """Create fake group QC output dirs, each with a 10_final_data.csv."""
    dirs = []
    for i, val in enumerate(values):
        ts = f"20260217_{110000 + i * 100:06d}"
        group_dir = base_dir / f"{group_by}_{val}_{ts}"
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "10_final_data.csv").write_text(
            "Barcode,Genotype,Replicate\np1,A,1\np2,B,1\n"
        )
        dirs.append(group_dir)
    return dirs


# =============================================================================
# Tasks 5.1a–5.1d: _find_grouped_qc_outputs
# =============================================================================


class TestFindGroupedQcOutputs:
    """_find_grouped_qc_outputs discovers all group dirs under a QC base dir."""

    def test_discovers_all_matching_dirs(self, tmp_path):
        """Task 5.1a: All subdirs whose names start with group_by column are returned."""
        qc_base = tmp_path / "qc"
        _make_group_qc_dirs(qc_base, "plant_age_days", [7, 9, 12])

        result = PipelineRunner._find_grouped_qc_outputs(qc_base, "plant_age_days")

        assert len(result) == 3
        names = [d.name for d in result]
        assert any("plant_age_days_7" in n for n in names)
        assert any("plant_age_days_9" in n for n in names)
        assert any("plant_age_days_12" in n for n in names)

    def test_ignores_non_matching_dirs(self, tmp_path):
        """Task 5.1b: Dirs not starting with the group_by prefix are excluded."""
        qc_base = tmp_path / "qc"
        _make_group_qc_dirs(qc_base, "plant_age_days", [7])
        (qc_base / "some_other_dir").mkdir()

        result = PipelineRunner._find_grouped_qc_outputs(qc_base, "plant_age_days")

        assert len(result) == 1
        assert "plant_age_days_7" in result[0].name

    def test_returns_empty_for_nonexistent_base(self, tmp_path):
        """Task 5.1c: Returns empty list when base dir does not exist."""
        result = PipelineRunner._find_grouped_qc_outputs(tmp_path / "gone", "col")
        assert result == []

    def test_returns_sorted_by_name(self, tmp_path):
        """Task 5.1d: Result is sorted by directory name (deterministic ordering)."""
        qc_base = tmp_path / "qc"
        # Create in scrambled order
        _make_group_qc_dirs(qc_base, "plant_age_days", [12, 3, 7])

        result = PipelineRunner._find_grouped_qc_outputs(qc_base, "plant_age_days")
        names = [d.name for d in result]

        assert names == sorted(names)


# =============================================================================
# Tasks 5.1e–5.1f: _get_qc_config_group_by
# =============================================================================


class TestGetQcConfigGroupBy:
    """_get_qc_config_group_by reads group_by from a QC config file."""

    def test_returns_column_name_when_set(self, tmp_path):
        """Task 5.1e: Returns the column name when group_by is present."""
        config = tmp_path / "qc.yaml"
        config.write_text("data:\n  group_by: plant_age_days\n  csv_path: x.csv\n")

        assert PipelineRunner._get_qc_config_group_by(config) == "plant_age_days"

    def test_returns_none_when_absent(self, tmp_path):
        """Task 5.1f (a): Returns None when group_by key is missing."""
        config = tmp_path / "qc.yaml"
        config.write_text("data:\n  csv_path: x.csv\n")

        assert PipelineRunner._get_qc_config_group_by(config) is None

    def test_returns_none_for_explicit_null(self, tmp_path):
        """Task 5.1f (b): Returns None when group_by is explicitly null."""
        config = tmp_path / "qc.yaml"
        config.write_text("data:\n  group_by: null\n  csv_path: x.csv\n")

        assert PipelineRunner._get_qc_config_group_by(config) is None

    def test_returns_none_for_missing_file(self, tmp_path):
        """Task 5.1f (c): Returns None gracefully for a nonexistent file."""
        assert PipelineRunner._get_qc_config_group_by(tmp_path / "gone.yaml") is None


# =============================================================================
# Tasks 5.3a–5.3c: _extract_group_label
# =============================================================================


class TestExtractGroupLabel:
    """_extract_group_label strips the YYYYMMDD_HHMMSS timestamp from dir names."""

    @pytest.fixture
    def runner(self, tmp_path):
        """PipelineRunner with grouped QC config (no run dirs created)."""
        manifest = _write_manifest(tmp_path / "cfg", qc_has_group_by=True)
        return PipelineRunner(manifest, output_dir=tmp_path / "out")

    def test_strips_timestamp_from_integer_group(self, runner):
        """Task 5.3a: Strips timestamp when group value is an integer."""
        assert (
            runner._extract_group_label("plant_age_days_7_20260217_114013")
            == "plant_age_days_7"
        )

    def test_strips_timestamp_from_two_digit_integer(self, runner):
        """Task 5.3a (two-digit): Handles two-digit integer group values."""
        assert (
            runner._extract_group_label("plant_age_days_12_20260217_115642")
            == "plant_age_days_12"
        )

    def test_strips_timestamp_from_string_group(self, runner):
        """Task 5.3b: Strips timestamp when group value is a string."""
        assert (
            runner._extract_group_label("experiment_id_exp1_20260217_114013")
            == "experiment_id_exp1"
        )

    def test_returns_name_unchanged_without_timestamp(self, runner):
        """Task 5.3c: Returns name unchanged when no YYYYMMDD_HHMMSS pattern found."""
        assert runner._extract_group_label("some_dir") == "some_dir"


# =============================================================================
# Tasks 5.3d–5.3j: _run_viz_pipelines fan-out behaviour
# =============================================================================


class TestRunVizPipelinesFanOut:
    """_run_viz_pipelines fans out to one viz run per QC group when grouped."""

    @pytest.fixture
    def grouped_runner(self, tmp_path):
        """PipelineRunner with grouped QC config and run dirs created."""
        manifest = _write_manifest(tmp_path / "cfg", qc_has_group_by=True)
        runner = PipelineRunner(manifest, output_dir=tmp_path / "out")
        runner._create_run_directories()
        return runner

    @pytest.fixture
    def ungrouped_runner(self, tmp_path):
        """PipelineRunner with ungrouped QC config and run dirs created."""
        manifest = _write_manifest(tmp_path / "cfg", qc_has_group_by=False)
        runner = PipelineRunner(manifest, output_dir=tmp_path / "out")
        runner._create_run_directories()
        return runner

    def test_calls_pipeline_n_times_for_n_groups(self, grouped_runner):
        """Task 5.3d: _run_pipeline_command called exactly N times for N groups."""
        group_dirs = _make_group_qc_dirs(
            grouped_runner.run_dir / "qc", "plant_age_days", [7, 9, 12]
        )
        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = group_dirs

        mock_cmd = MagicMock(return_value={"success": True, "output_path": None})
        with patch.object(grouped_runner, "_run_pipeline_command", mock_cmd):
            grouped_runner._run_viz_pipelines()

        assert mock_cmd.call_count == 3

    def test_result_keys_include_group_label(self, grouped_runner):
        """Task 5.3e: run_results['viz'] keys identify each group."""
        group_dirs = _make_group_qc_dirs(
            grouped_runner.run_dir / "qc", "plant_age_days", [7, 9, 12]
        )
        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = group_dirs

        def fake_run(pipeline_type, config_path, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(
            grouped_runner, "_run_pipeline_command", side_effect=fake_run
        ):
            grouped_runner._run_viz_pipelines()

        keys = list(grouped_runner.run_results["viz"].keys())
        assert len(keys) == 3
        assert any("plant_age_days_7" in k for k in keys)
        assert any("plant_age_days_9" in k for k in keys)
        assert any("plant_age_days_12" in k for k in keys)

    def test_each_group_gets_distinct_output_subdir(self, grouped_runner):
        """Task 5.3f: Each group is passed a distinct output directory."""
        group_dirs = _make_group_qc_dirs(
            grouped_runner.run_dir / "qc", "plant_age_days", [7, 9]
        )
        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = group_dirs

        captured_output_dirs: list[Path] = []

        def capture_run(pipeline_type, config_path, output_dir):
            captured_output_dirs.append(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(
            grouped_runner, "_run_pipeline_command", side_effect=capture_run
        ):
            grouped_runner._run_viz_pipelines()

        assert len(captured_output_dirs) == 2
        assert captured_output_dirs[0] != captured_output_dirs[1]
        names = [d.name for d in captured_output_dirs]
        assert any("plant_age_days_7" in n for n in names)
        assert any("plant_age_days_9" in n for n in names)

    def test_updated_config_written_under_group_subdir_with_updated_prefix(
        self, grouped_runner
    ):
        """Task 5.3g: _updated_* config is written inside the group's output subdir."""
        [group_dir] = _make_group_qc_dirs(
            grouped_runner.run_dir / "qc", "plant_age_days", [9]
        )
        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = [group_dir]

        captured_configs: list[Path] = []

        def capture_run(pipeline_type, config_path, output_dir):
            captured_configs.append(Path(config_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(
            grouped_runner, "_run_pipeline_command", side_effect=capture_run
        ):
            grouped_runner._run_viz_pipelines()

        assert len(captured_configs) == 1
        config_path = captured_configs[0]
        # Must live inside run_dir/viz/plant_age_days_9/
        assert "plant_age_days_9" in str(config_path)
        # Must follow the _updated_ naming convention
        assert config_path.name.startswith("_updated_")

    def test_updated_config_references_group_csv_path(self, grouped_runner):
        """Task 5.3h: Updated viz config csv_path points to the group's 10_final_data.csv."""
        [group_dir] = _make_group_qc_dirs(
            grouped_runner.run_dir / "qc", "plant_age_days", [9]
        )
        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = [group_dir]

        captured_configs: list[Path] = []

        def capture_run(pipeline_type, config_path, output_dir):
            captured_configs.append(Path(config_path))
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(
            grouped_runner, "_run_pipeline_command", side_effect=capture_run
        ):
            grouped_runner._run_viz_pipelines()

        content = captured_configs[0].read_text()
        expected_csv = (group_dir / "10_final_data.csv").as_posix()
        assert expected_csv in content

    def test_non_grouped_viz_runs_exactly_once(self, ungrouped_runner):
        """Task 5.3i: Without grouped QC, viz runs once (regression guard)."""
        qc_out = ungrouped_runner.run_dir / "qc" / "run_20260217_114013"
        qc_out.mkdir(parents=True)
        (qc_out / "10_final_data.csv").write_text("Barcode,Genotype\np1,A\n")
        ungrouped_runner.qc_outputs["qc/test_qc.yaml"] = qc_out

        mock_cmd = MagicMock(return_value={"success": True, "output_path": str(qc_out)})
        with patch.object(ungrouped_runner, "_run_pipeline_command", mock_cmd):
            ungrouped_runner._run_viz_pipelines()

        assert mock_cmd.call_count == 1

    def test_group_with_missing_csv_recorded_as_failure(self, grouped_runner):
        """Task 5.3j: Groups without 10_final_data.csv are failures, not exceptions."""
        qc_base = grouped_runner.run_dir / "qc"
        good_dir = qc_base / "plant_age_days_7_20260217_110000"
        bad_dir = qc_base / "plant_age_days_9_20260217_110100"
        good_dir.mkdir(parents=True)
        bad_dir.mkdir(parents=True)
        (good_dir / "10_final_data.csv").write_text("Barcode,Genotype\np1,A\n")
        # No CSV in bad_dir

        grouped_runner.qc_grouped_outputs["qc/test_qc.yaml"] = [good_dir, bad_dir]

        def fake_run(pipeline_type, config_path, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(
            grouped_runner, "_run_pipeline_command", side_effect=fake_run
        ):
            grouped_runner._run_viz_pipelines()  # Must not raise

        keys = list(grouped_runner.run_results["viz"].keys())
        bad_key = next((k for k in keys if "plant_age_days_9" in k), None)
        assert bad_key is not None
        assert grouped_runner.run_results["viz"][bad_key]["success"] is False

        # Good group still runs
        good_key = next((k for k in keys if "plant_age_days_7" in k), None)
        assert good_key is not None
        assert grouped_runner.run_results["viz"][good_key]["success"] is True


# =============================================================================
# Tasks 6.2–6.3: Integration tests — PipelineRunner level
# =============================================================================


class TestPipelineRunnerGroupedVizIntegration:
    """Integration tests: PipelineRunner fans out viz for grouped QC configs."""

    def test_run_results_contains_n_viz_entries_for_n_qc_groups(self, tmp_path):
        """Task 6.2: N grouped QC outputs → N entries in run_results['viz']."""
        manifest = _write_manifest(tmp_path / "cfg", qc_has_group_by=True)
        runner = PipelineRunner(manifest, output_dir=tmp_path / "out")
        runner._create_run_directories()

        group_dirs = _make_group_qc_dirs(
            runner.run_dir / "qc", "plant_age_days", [7, 9, 12]
        )
        runner.qc_grouped_outputs["qc/test_qc.yaml"] = group_dirs

        def fake_run(pipeline_type, config_path, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(runner, "_run_pipeline_command", side_effect=fake_run):
            runner._run_viz_pipelines()

        assert len(runner.run_results["viz"]) == 3

    def test_n_viz_output_subdirs_equals_n_qc_groups(self, tmp_path):
        """Task 6.3: N grouped QC outputs → N distinct viz output dirs under run_dir/viz."""
        manifest = _write_manifest(tmp_path / "cfg", qc_has_group_by=True)
        runner = PipelineRunner(manifest, output_dir=tmp_path / "out")
        runner._create_run_directories()

        group_dirs = _make_group_qc_dirs(
            runner.run_dir / "qc", "plant_age_days", [7, 9, 12]
        )
        runner.qc_grouped_outputs["qc/test_qc.yaml"] = group_dirs

        captured_output_dirs: list[Path] = []

        def capture_run(pipeline_type, config_path, output_dir):
            captured_output_dirs.append(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            return {"success": True, "output_path": str(output_dir)}

        with patch.object(runner, "_run_pipeline_command", side_effect=capture_run):
            runner._run_viz_pipelines()

        # Exactly 3 distinct output directories, all inside run_dir/viz/
        assert len(set(str(d) for d in captured_output_dirs)) == 3
        viz_base = runner.run_dir / "viz"
        for d in captured_output_dirs:
            assert str(d).startswith(str(viz_base))
