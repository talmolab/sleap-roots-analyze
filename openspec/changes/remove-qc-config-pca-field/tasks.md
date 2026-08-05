## 1. Baseline (do this first — gates everything else)

- [x] 1.1 Run `uv run pytest tests/test_pipeline_reproduction.py -q` before any change; confirm
      all tests pass (baseline: 45 passed).
- [x] 1.2 Run `uv run pytest tests/test_pipeline_config.py tests/test_replicate_optional.py
      tests/test_qc_pipeline.py tests/test_numerical_stability.py tests/test_golden_templates.py
      tests/test_grouped_pipeline_config_persistence.py -q` before any change; confirm all pass
      (baseline: 89 passed, 11 skipped).
- [x] 1.3 One-time dynamic equivalence check (not a permanently committed test — the field won't
      exist to set once removed): using
      `tests/test_qc_pipeline.py::TestQCPipelineIntegration::test_qc_pipeline_full_run`'s
      synthetic dataset (same 30-sample construction), ran `QCPipeline(...).run()` twice against
      the current, unmodified code — once with `config.pca` left at its default, once with
      `config.pca.feature_selection_strategy = "top_absolute"` and `config.pca.n_top_features =
      3` (both non-default) — and compared pipeline status, step names, and the full
      `10_final_data.csv` content byte-for-byte. **Result: identical.**
      (`{"identical": true}`, confirmed via a throwaway script, not committed). This is the
      dynamic proof that `config.pca` has zero effect on `QCPipeline` execution, complementing
      the static grep showing no QC step reads `config.pca`.

## 2. Sweep: strip `pca:` from every QC-only config file (do this *before* schema removal)

**Ordering note (reviewer-identified, corrects the original draft's order):** this section must
land before section 3's schema/validation removal. `OmegaConf`'s strict merge only breaks in one
direction — a YAML key the target dataclass lacks raises `ConfigKeyError`. If the schema loses
`pca` while any of these 59 files still has a `pca:` block, every test that loads one of them
directly breaks immediately, including three identified during review that section 1's baseline
commands don't cover: `tests/test_pipeline_reproduction.py` (harness configs, already in 1.1),
`tests/test_golden_templates.py` (`configs/templates/qc_template_{grouped,ungrouped}.yaml`), and
`tests/test_cli.py::test_qc_with_real_config_dry_run`
(`configs/qc_turface_150genotypes.yaml`). Doing the sweep first means the schema still has
`pca` while these files are edited, so `OmegaConf.merge` simply falls back to `PCAConfig()`
defaults for the (now pca-less) YAML — safe in this direction.

Verified exhaustive list (`grep -rl "^pca:" --include="*.yaml" configs/ tests/fixtures/harness/`,
excluding any path containing `/viz`, `configs/archive/`, or `configs/saved_backups/` —
cross-checked against a looser `feature_selection_strategy|n_top_features` grep with an
identical 59-file result set; independently re-verified during proposal review with the same
result, including confirming `configs/qc_consensus_6method.yaml`'s separate, untouched nested
`outlier_detection.pca:` block is a different field (`PCAOutlierConfig`) that the anchored
`^pca:` grep correctly does not match). For each file, delete the top-level `pca:` block and any
comment lines that exist solely to describe it; leave every other key untouched.

- [x] 2.1 Test harness fixtures (4 files) — do these first, then re-run
      `uv run pytest tests/test_pipeline_reproduction.py -q` for the fastest signal:
      `tests/fixtures/harness/qc/qc_cylinder_edpie.yaml`,
      `tests/fixtures/harness/qc/qc_root_core_edpie.yaml`,
      `tests/fixtures/harness/qc/qc_turface_150genotypes.yaml`,
      `tests/fixtures/harness/qc/qc_turface_19genotypes.yaml`.
- [x] 2.2 `configs/active/qc/*.yaml` (28 files) + the 3 flat pre-reorg duplicates directly under
      `configs/active/`: `alfalfa_gwas_groups_1_to_6_combined.yaml`,
      `alfalfa_gwas_groups_1_to_6_combined_no_root_widths.yaml`, `alfalfa_gwas_w1w2_combined.yaml`,
      `alfalfa_gwas_wave1.yaml`, `alfalfa_gwas_wave1_canola.yaml`,
      `alfalfa_gwas_wave1_canola_models.yaml`, `amaranth_tis108_exp1.yaml`,
      `canola_diversity_screen_qc.yaml`, `emily_shane_pennycress_2026_02_09.yaml`,
      `emily_shane_soybean_2026_01_15.yaml`, `emily_shane_soybean_2026_03_03.yaml`,
      `emily_shane_soybean_2026_03_03_grouped.yaml`, `giftol_pennycress_s32_2026_05_11.yaml`,
      `javier_ttc_salk_soybean.yaml`, `javier_ttc_salk_soybean_brightness.yaml`,
      `javier_ttc_salk_soybean_full_experiment_9wave.yaml`,
      `javier_ttc_salk_soybean_full_experiment_9wave_per_wave.yaml`,
      `mo_soybean_2021_grouped.yaml`, `qc_alfalfa_gwas_wave_1_grouped.yaml`,
      `qc_cylinder_edpie.yaml`, `qc_field_2024_clean.yaml`, `qc_root_core_edpie.yaml`,
      `qc_turface_150genotypes.yaml`, `qc_turface_19genotypes.yaml`, `shree_weep_soybean.yaml`,
      `suyash_arabidopsis_pgm1_pac_2026_05_22.yaml`, `turface_alfalfa_gwas.yaml`,
      `weep_maurizio_wave1.yaml` (all under `configs/active/qc/`); plus, directly under
      `configs/active/`: `qc_turface_150genotypes.yaml`, `qc_turface_19genotypes.yaml`,
      `qc_turface_alfalfa_20251203.yaml`. Re-run `uv run pytest tests/test_pipeline_reproduction.py
      tests/test_golden_templates.py -q` after this task.
- [x] 2.3 Remaining files (24): `configs/examples/qc_clustering_strict.yaml`,
      `qc_consensus_6method.yaml`, `qc_mahalanobis.yaml`, `qc_permissive.yaml`; flat files
      directly under `configs/`: `qc_alfalfa_gwas_wave_1.yaml`, `qc_alfalfa_gwas_wave_2.yaml`,
      `qc_clustering_strict.yaml`, `qc_consensus_6method.yaml`, `qc_cylinder_edpie.yaml`,
      `qc_field_2024_clean.yaml`, `qc_mahalanobis.yaml`, `qc_permissive.yaml`,
      `qc_root_core_edpie.yaml`, `qc_root_core_edpie_v2.yaml`, `qc_root_core_manual_qc.yaml`,
      `qc_root_core_replicated.yaml`, `qc_turface_150genotypes.yaml`,
      `qc_turface_19genotypes.yaml`, `qc_turface_alfalfa_20251203.yaml`;
      `configs/templates/qc_cleanup_only_template.yaml`,
      `qc_full_pipeline_template.yaml`, `qc_template_grouped.yaml`, `qc_template_ungrouped.yaml`;
      `configs/test_nov30_reproduction.yaml`.
- [x] 2.4 Confirm nothing was missed: re-run the exhaustive grep
      (`grep -rl "^pca:" --include="*.yaml" configs/ tests/ | grep -v "/viz" | grep -v
      "configs/archive/" | grep -v "configs/saved_backups/" | grep -v "expected/"`) and confirm
      zero results. **Confirmed: zero results, 59 files changed.**
- [x] 2.5 Run `uv run pytest tests/test_pipeline_reproduction.py tests/test_golden_templates.py
      tests/test_cli.py -k qc -q` and confirm green (schema still has `pca`, so this is just
      confirming the now-pca-less YAML files still load against the unchanged schema).
      **Confirmed green** (56 + 17 passed).

## 3. Remove `pca` from `QCPipelineConfig` + its validation (single atomic commit, after section 2)

Do not split this section across commits: removing the dataclass field without also removing
`validate_explicit_config()`/`validate_qc_config()`'s `config.pca.*` reads crashes every QC
config validation with `AttributeError`.

- [x] 3.1 In `tests/test_pipeline_config.py`, remove the `QCPipelineConfig.pca`-specific
      assertions/tests that no longer apply (16 total): `test_pipeline_config_creation`'s
      `assert isinstance(config.pca, PCAConfig)` line; `test_save_and_load_config`'s
      `original.pca.n_components = 0.9` / `loaded.pca.n_components == 0.9` lines;
      `test_merge_configs`'s `base.pca.n_components = 0.95` line, the `"pca": {"n_components":
      0.8}` key in its `overrides` dict literal (not just the `merged.pca.n_components ==
      0.8` assertion), and that assertion itself; and the full PCA-validation test block:
      `test_validate_config_invalid_pca_components`, `test_validate_config_invalid_pca_strategy`,
      `test_validate_config_valid_pca_strategies`,
      `test_validate_config_rejects_fractional_n_top_features_below_1`,
      `test_validate_config_rejects_non_whole_number_n_top_features_ge_1`,
      `test_validate_config_rejects_nan_n_top_features`,
      `test_validate_config_rejects_infinite_n_top_features`,
      `test_validate_config_accepts_nan_or_inf_n_top_features_for_extreme`,
      `test_validate_config_accepts_near_whole_number_from_float_accumulation`,
      `test_validate_config_accepts_top_variance_threshold_below_1`,
      `test_validate_config_accepts_any_n_top_features_for_extreme`,
      `test_validate_config_accepts_whole_number_n_top_features_for_any_strategy`,
      `test_validate_config_default_pca_values_pass`. Leave `test_pca_config_defaults` (tests
      `PCAConfig()` directly, not `QCPipelineConfig`) and `test_merge_configs_nested` (tests
      `outlier_detection.pca`, a different field) untouched.
- [x] 3.2 In `tests/test_replicate_optional.py::test_omit_replicate_in_yaml_disables_replicate`,
      remove the `"pca:\n  n_components: 2\n"` line from the inline YAML fixture string.
- [x] 3.3 Added `test_pipeline_config_has_no_pca_field` to `tests/test_pipeline_config.py`:
      confirms `not hasattr(config, "pca")` and that `QCPipelineConfig(pipeline_name="test",
      pca=PCAConfig())` raises `TypeError` (the field is genuinely gone, not just deprecated/
      ignored). Confirmed red (`AssertionError: assert not True`) against the pre-removal schema,
      then green after 3.4-3.6. (Scope note: a separate dedicated test for "`validate_qc_config`
      no longer requires `pca.n_components`" was not added as its own test — once the field is
      removed there is no way to construct the "unset n_components" scenario at all, and every
      remaining `validate_qc_config(config)` call across the suite already implicitly proves no
      `AttributeError` occurs; a standalone test would be redundant.)
- [x] 3.4 Remove `pca: PCAConfig = field(default_factory=PCAConfig)` and its docstring line from
      `QCPipelineConfig` (`pipeline/config/qc_config.py`); remove the now-unused `PCAConfig`
      import from that module's import list.
- [x] 3.5 Remove the `config.pca.n_components is None` check from `validate_explicit_config()`
      (`pipeline/config/utils.py`, "REQUIRED: PCA n_components" block).
- [x] 3.6 Remove the entire "Validate PCA config" block from `validate_qc_config()`
      (`pipeline/config/utils.py`) — `config.pca.n_components`, `feature_selection_strategy`,
      and `n_top_features` checks. Do **not** touch the near-identical block in
      `validate_viz_config()` (same file) — `VizPipelineConfig` keeps its `pca` field. Update the
      module-level `_WHOLE_NUMBER_TOLERANCE` comment (lines ~21-23), which currently says the
      tolerance applies to both `validate_qc_config()`/`validate_viz_config()` — narrow it to
      `validate_viz_config()` only.
- [x] 3.7 Run `uv run pytest tests/test_pipeline_config.py tests/test_replicate_optional.py -q`
      and confirm green (schema + validation + tests now consistent). **Confirmed: 42 passed.**

## 4. Safety-net regression test

- [x] 4.1 Wrote `tests/test_qc_configs_load.py`, hardcoding the exact 59-file list from section 2
      (not a path/content heuristic). **Design refinement found during implementation**: several
      of the 59 files are illustrative method-showcase configs with `data.csv_path: ???`
      (OmegaConf's placeholder marker, pre-existing and unrelated to #204) and one
      (`qc_turface_alfalfa_20251203.yaml`, both copies) has a pre-existing empty
      `columns.barcode` — both raise exceptions unrelated to a leftover `pca:` block. The test
      therefore asserts specifically "no `ConfigKeyError`" (tolerating any other exception as a
      pre-existing, out-of-scope condition), not "loads with zero exceptions." Confirmed red:
      manually re-appended a `pca:` block to `configs/qc_mahalanobis.yaml` and confirmed the test
      fails with `ConfigKeyError: Key 'pca' not in 'QCPipelineConfig'`, then restored it (via the
      strip script, since `git checkout --` reverted all the way to the pre-sweep HEAD).
- [x] 4.2 Run the new test against the post-sweep, post-schema-removal repo state and confirm it
      passes. **Confirmed: 59 passed.**
- [x] 4.3 **Post-self-review additions** (`/review-pr` pre-PR self-review found no BLOCKING issues
      across all 5 lenses, but two IMPORTANT testing-reviewer findings and one code-quality
      suggestion were worth taking): (a) tightened `test_qc_config_loads_without_pca_block`'s
      `except Exception: pass` to `except (MissingMandatoryValue, ValidationError): pass` — the
      two specific, already-documented pre-existing exception types, so an unrelated future
      regression in one of these 59 files no longer gets silently swallowed; (b) added
      `test_qc_config_with_pca_block_raises_config_key_error`, a committed regression test
      locking in the `ConfigKeyError` failure mode (previously only verified manually, per 4.1's
      red-check, and not preserved as a permanent test); (c) added
      `test_no_qc_config_has_a_pca_block_anywhere`, a drift tripwire that independently re-scans
      `configs/`/`tests/fixtures/harness/` for a top-level `pca:` key rather than trusting
      `QC_CONFIG_FILES` to stay in sync with the repo. Verified the drift tripwire actually
      catches a regression: manually reappended a `pca:` block to `configs/qc_mahalanobis.yaml`,
      confirmed the new test fails listing that exact file, then restored via `git checkout --`
      (safe now that the sweep is committed). Re-ran `tests/test_qc_configs_load.py`: **61
      passed.**

## 5. Docs

- [x] 5.1 Remove the `pca:` block from `pipeline/README.md`'s illustrative example config
      (`README.md:119-123`).
- [x] 5.2 `.claude/commands/configure-run-all.md`: remove the three now-invalid
      `pca.n_components`/`pca.feature_selection_strategy`/`pca.n_top_features` lines from the
      "6.1 — QC Config" customized-parameters checklist (they stay in "6.2 — Viz Config"'s
      checklist, which is correct and unaffected); add a "(for viz config)" qualifier to the
      "3.8 — PCA settings" question block's `n_components`/`feature_selection_strategy`/
      `n_top_features` questions, matching the existing "(for viz config)" annotation already on
      that same block's `pca_biplot_top_features` question.
- [x] 5.3 `.claude/commands/validate-config.md`: remove the
      `print(f"  PCA: {config.pca.n_components} components")` line from its QC-config validation
      example script.
- [x] 5.4 `docs/QC_PIPELINE_GUIDE.md`: remove the "**PCA Configuration**" subsection (currently
      listed under QC "Required Parameters").
- [x] 5.5 `configs/templates/README.md`: re-scope the "### PCA Settings" section to viz templates
      only — add "(viz template default)" to the `n_components`/`feature_selection_strategy`/
      `n_top_features` bullets, consistent with the existing annotation already on
      `pca_biplot_top_features`, since QC templates have no PCA settings left after this change.
- [x] 5.6 Add a `docs/CHANGELOG.md` `[Unreleased]` entry (`### Removed`, matching the file's
      existing section structure) describing the `QCPipelineConfig.pca` removal and pointing to
      #204.

## 6. Full verification and PR

- [x] 6.0 **Full-suite discovery (not in the original plan): `tests/test_step_pca_analysis.py`
      (4 fixtures) and `tests/fixtures_visualization.py` (8 fixtures, feeding
      `tests/test_step_generate_static_figures.py`) construct `QCPipelineConfig` — not
      `VizPipelineConfig` — as a generic test double for unit-testing the viz-only
      `PCAAnalysisStep`/`GenerateStaticFiguresStep` directly, relying on the now-removed `.pca`
      field (some via a `pca=PCAConfig(...)` constructor kwarg — now a `TypeError`, surfacing as
      a pytest `ERROR`; others via `config.pca.n_components` access inside the step itself — an
      `AttributeError`, surfacing as a `FAILED`). This was a pre-existing latent mismatch (these
      fixtures always used the wrong config class for what they test) that only my own full
      local test run surfaced — none of the five review subagents' targeted file lists included
      `test_step_pca_analysis.py` or `fixtures_visualization.py`, since their searches were
      scoped to QC-config-specific files. Fixed by switching all 12 constructions from
      `QCPipelineConfig` to `VizPipelineConfig` (the semantically-correct type — it declares
      `pca`, `static_viz`, `interactive_viz`, and `dashboard` as real fields, where
      `fixtures_visualization.py` had been assigning several of these dynamically as ad-hoc
      attributes onto a `QCPipelineConfig` instance that never formally declared them).
      Re-ran `tests/test_step_pca_analysis.py tests/test_step_generate_static_figures.py`:
      **107 passed.**
- [x] 6.1 Re-run section 1's baseline comparison (`test_pipeline_reproduction.py` + the broader
      QC config/pipeline test files) and confirm identical pass/skip counts to the section-1
      baseline. **Confirmed.**
- [x] 6.2 Run
      `uv run pytest --cov=src/sleap_roots_analyze --cov-report=xml --durations=20 -m "not integration" tests/`
      (exact CI invocation, `.github/workflows/ci.yml`) — full suite green. First run (before 6.0's
      fix) surfaced the regression above: 25 failed, 36 errors. Second run, after 6.0:
      **3016 passed, 37 skipped, 3 deselected, 0 failed.**
- [x] 6.3 Run `uv run black src/sleap_roots_analyze tests` and
      `uv run ruff check src/sleap_roots_analyze tests`. **Confirmed clean.**
- [x] 6.4 Run `openspec validate remove-qc-config-pca-field --strict`. **Confirmed valid.**
- [ ] 6.5 Open PR.

### Suggested commit sequence (per review)

1. `test(#204): remove QCPipelineConfig.pca assertions made obsolete` — section 3.1/3.2 (pure
   deletions against the *unchanged* schema; safe to land first, CI green).
2. `chore(#204): strip dead pca: block from harness and active QC configs` — section 2.1/2.2
   (35 files; schema still has `pca`, so `OmegaConf.merge` falls back to `PCAConfig()` defaults;
   CI green).
3. `chore(#204): strip dead pca: block from remaining QC configs` — section 2.3 (24 files; run
   the full suite here, not just the narrow section-1 commands, per the three hard dependents
   noted in section 2's ordering note; CI green).
4. `refactor(#204): remove pca field and validation from QCPipelineConfig` — section 3.3-3.6 (all
   59 YAML files already stripped, so this is now safe; run the full suite — this is the real
   no-op proof; CI green).
5. `test(#204): add QC config load safety-net regression test` — section 4.
6. `docs(#204): drop pca: from README/command docs/guide, update CHANGELOG` — section 5.
