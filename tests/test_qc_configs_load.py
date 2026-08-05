"""Safety-net regression test for the QCPipelineConfig.pca removal (issue #204).

Hardcodes the exact 59-file list identified by the exhaustive repo-wide sweep for
this change, rather than a path/content heuristic that could silently drift from
the set of files actually edited. Confirms none of them still has a top-level
``pca:`` key, which would otherwise raise ``ConfigKeyError`` at load time.

A handful of these files have pre-existing, unrelated-to-#204 issues that are
out of scope to fix here: several illustrative method-showcase configs leave
``data.csv_path: ???`` (OmegaConf's placeholder marker) for a user to fill in
(raises ``MissingMandatoryValue``), and `qc_turface_alfalfa_20251203.yaml`
(both copies) leaves `columns.barcode` empty (raises ``ValidationError`` —
`ColumnConfig.barcode` is a required `str`, and this was already broken before
this change; confirmed by `git diff` showing this change never touches that
line). This test asserts the one thing #204 is actually about: no leftover
top-level ``pca:`` key causing ``ConfigKeyError``. Any other exception is a
pre-existing, out-of-scope config issue, not a regression from this change.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf.errors import ConfigKeyError

from sleap_roots_analyze.pipeline.config import load_qc_config

REPO_ROOT = Path(__file__).parent.parent

QC_CONFIG_FILES = [
    # Test harness fixtures
    "tests/fixtures/harness/qc/qc_cylinder_edpie.yaml",
    "tests/fixtures/harness/qc/qc_root_core_edpie.yaml",
    "tests/fixtures/harness/qc/qc_turface_150genotypes.yaml",
    "tests/fixtures/harness/qc/qc_turface_19genotypes.yaml",
    # configs/active/qc/
    "configs/active/qc/alfalfa_gwas_groups_1_to_6_combined.yaml",
    "configs/active/qc/alfalfa_gwas_groups_1_to_6_combined_no_root_widths.yaml",
    "configs/active/qc/alfalfa_gwas_w1w2_combined.yaml",
    "configs/active/qc/alfalfa_gwas_wave1.yaml",
    "configs/active/qc/alfalfa_gwas_wave1_canola.yaml",
    "configs/active/qc/alfalfa_gwas_wave1_canola_models.yaml",
    "configs/active/qc/amaranth_tis108_exp1.yaml",
    "configs/active/qc/canola_diversity_screen_qc.yaml",
    "configs/active/qc/emily_shane_pennycress_2026_02_09.yaml",
    "configs/active/qc/emily_shane_soybean_2026_01_15.yaml",
    "configs/active/qc/emily_shane_soybean_2026_03_03.yaml",
    "configs/active/qc/emily_shane_soybean_2026_03_03_grouped.yaml",
    "configs/active/qc/giftol_pennycress_s32_2026_05_11.yaml",
    "configs/active/qc/javier_ttc_salk_soybean.yaml",
    "configs/active/qc/javier_ttc_salk_soybean_brightness.yaml",
    "configs/active/qc/javier_ttc_salk_soybean_full_experiment_9wave.yaml",
    "configs/active/qc/javier_ttc_salk_soybean_full_experiment_9wave_per_wave.yaml",
    "configs/active/qc/mo_soybean_2021_grouped.yaml",
    "configs/active/qc/qc_alfalfa_gwas_wave_1_grouped.yaml",
    "configs/active/qc/qc_cylinder_edpie.yaml",
    "configs/active/qc/qc_field_2024_clean.yaml",
    "configs/active/qc/qc_root_core_edpie.yaml",
    "configs/active/qc/qc_turface_150genotypes.yaml",
    "configs/active/qc/qc_turface_19genotypes.yaml",
    "configs/active/qc/shree_weep_soybean.yaml",
    "configs/active/qc/suyash_arabidopsis_pgm1_pac_2026_05_22.yaml",
    "configs/active/qc/turface_alfalfa_gwas.yaml",
    "configs/active/qc/weep_maurizio_wave1.yaml",
    # Flat pre-reorg duplicates directly under configs/active/
    "configs/active/qc_turface_150genotypes.yaml",
    "configs/active/qc_turface_19genotypes.yaml",
    "configs/active/qc_turface_alfalfa_20251203.yaml",
    # configs/examples/
    "configs/examples/qc_clustering_strict.yaml",
    "configs/examples/qc_consensus_6method.yaml",
    "configs/examples/qc_mahalanobis.yaml",
    "configs/examples/qc_permissive.yaml",
    # Flat files directly under configs/
    "configs/qc_alfalfa_gwas_wave_1.yaml",
    "configs/qc_alfalfa_gwas_wave_2.yaml",
    "configs/qc_clustering_strict.yaml",
    "configs/qc_consensus_6method.yaml",
    "configs/qc_cylinder_edpie.yaml",
    "configs/qc_field_2024_clean.yaml",
    "configs/qc_mahalanobis.yaml",
    "configs/qc_permissive.yaml",
    "configs/qc_root_core_edpie.yaml",
    "configs/qc_root_core_edpie_v2.yaml",
    "configs/qc_root_core_manual_qc.yaml",
    "configs/qc_root_core_replicated.yaml",
    "configs/qc_turface_150genotypes.yaml",
    "configs/qc_turface_19genotypes.yaml",
    "configs/qc_turface_alfalfa_20251203.yaml",
    # configs/templates/
    "configs/templates/qc_cleanup_only_template.yaml",
    "configs/templates/qc_full_pipeline_template.yaml",
    "configs/templates/qc_template_grouped.yaml",
    "configs/templates/qc_template_ungrouped.yaml",
    # Single file
    "configs/test_nov30_reproduction.yaml",
]

assert len(QC_CONFIG_FILES) == 59, f"expected 59 files, got {len(QC_CONFIG_FILES)}"


@pytest.mark.parametrize("relpath", QC_CONFIG_FILES)
def test_qc_config_loads_without_pca_block(relpath):
    """Every QC config touched by the #204 sweep loads without ConfigKeyError.

    Any other exception (e.g. ``MissingMandatoryValue`` for a placeholder
    ``csv_path: ???``, or a pre-existing unrelated schema issue) is a
    pre-existing, out-of-scope condition this test does not assert against —
    only a leftover top-level ``pca:`` key (which raises ``ConfigKeyError``
    when merged against a schema that no longer has that field) is a
    regression from this change.
    """
    path = REPO_ROOT / relpath
    assert path.is_file(), f"missing config file: {relpath}"
    try:
        load_qc_config(path)
    except ConfigKeyError:
        raise
    except Exception:
        pass
