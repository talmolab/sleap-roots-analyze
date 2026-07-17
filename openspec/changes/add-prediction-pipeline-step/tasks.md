> Full rationale for every design choice referenced below (nesting vs. standalone config, BLUP path
> fields, `platform_pairs` narrowing, validation gating, exception type, target-set construction)
> is in `design.md`'s Decisions section. Flag **Decision 6/12** (PC1-as-target uses `pca.fit_pca()`
> directly, not `fit_pca_on_fold`) to the user/reviewer explicitly before implementing Section 4 —
> it's the one place this tier's design diverges from a naive "just reuse Tier 3's PC1 utility
> everywhere" reading.
>
> **`/review-openspec` round 1 reconciled into this file** (full findings in design.md's
> "Adversarial Review Reconciliation (round 1)" section): `representative_selection_metric`
> restricted to `"variance"` only, `"heritability"` deferred (Decision 7); task 6's `depends_on` now
> includes `"01_load_cross_platform_data"` directly, not just `"05_visualize_cross_platform"`
> (Decision 8); the backward-compat oracle (Section 5) excludes `config.yaml` (Decision 9);
> `platform_pairs` cardinality is now explicitly tested (Decision 10); the PC1-as-target computation
> is pinned to `pca.fit_pca()` with `StandardScaler` pre-applied (Decision 12); several missing
> edge-case tests added throughout (zero/duplicate `platform_pairs` entries, `comparison_methods`
> duplicating `reduction_method`, zero target-representative-traits, zero/near-zero common-genotype
> overlap, a positive PC1 value-equality test); task 2.1's premature `CrossPlatformConfig` assertion
> removed (redundant with 3.1, and unpassable on Section 2 alone); Section 6's exact-equality
> assertion reworded to a tolerance; task 1.3 rewritten with an explicit scoping decision; task 8.2
> reworded to not foreclose a "reopen the design" outcome; Section 9's doc tasks corrected.
>
> **`/review-openspec` round 2 (fresh, no memory of round 1) reconciled into this file** (full
> findings in design.md's "Adversarial Review Reconciliation (round 2)" section): round 1's own
> Decision 8 fix (reading task 1's raw data directly) had reintroduced a crash/data-pollution risk —
> task 1's raw DataFrame isn't trait-only; fixed by selecting task 1's already-filtered trait-name
> metadata first (Decision 13, task 4.2). Genotype row-order alignment between `X`/`y`/`genotypes`
> was unenforced and untested — fixed via a canonical-index requirement plus a deliberate-shuffle
> regression test (Decision 14, task 4.5b). `blup_refit_per_fold`'s "inert" claim and
> `comparison_methods` self-duplication were both untested gaps, now covered (tasks 2.4a, 4.9a).
> Task 6.1's tolerance was actually wrong, not fixed — corrected to this codebase's real
> `rtol`/`atol` convention. spec.md's scenario coverage gained one more scenario (target-trait
> selection) that round 1 had claimed but not actually added. Section 9 expanded with a concrete
> YAML example and PC1-hyperparameter documentation. Tasks 1.3 and 8.4 gained enforcement/commit-
> mechanics specifics.
>
> **`/review-openspec` round 3 (fresh, no memory of rounds 1-2) reconciled into this file** (full
> findings in design.md's "Adversarial Review Reconciliation (round 3)" section — every prior
> round's code citation independently re-verified and confirmed accurate this round, including
> Decisions 12-15's exact mechanisms traced through the full `TaskResult`/`StepResult`/DAG-executor
> plumbing): real BLUP CSVs routinely contain NaN columns (failed model fits), previously
> unaddressed and untested — fixed via Decision 16 (task 4.1a). The BLUP CSV's genotype-column name
> was never specified anywhere — fixed via Decision 17 (task 4.1b). Decision 15's "ordering-only"
> claim had no enforcing test — fixed (task 5.1a). Task 1.3's suggested git tripwire command was
> empirically verified invalid and replaced. Section 10 gained a post-implementation
> `/pre-merge-check`/`/review-pr` task, matching Tier 3's own two-pass precedent. Several smaller
> test-naming/specification tightenings throughout (tasks 4.2b, 4.5b, 4.3a, 4.9a).

## 1. Fixtures (test-first)

- [x] 1.1 Add a 2-platform synthetic BLUP CSV fixture pair (`tests/fixtures/` or `tests/fixtures.py`
      — decide during implementation which convention this repo's existing fixture files follow for
      on-disk CSV fixtures vs. in-memory ones): two small genotype-indexed trait tables (~19
      genotypes, a handful of traits), shaped exactly like `extract_blup_table()`'s
      `08_blup_adjusted_means.csv` contract, with a known planted signal between them. A **single
      deterministic realization is sufficient** here (unlike Tier 3's N=20-seed-averaged fixtures,
      design.md's earlier Decision 6 in the archived Tier 3 proposal) — this fixture's job is a
      wiring-correctness equality check between the pipeline's output and a direct
      `logo_cv_predict()` call on the same data, not a statistical signal-recovery claim.
- [x] 1.2 Add a harness YAML (`tests/fixtures/harness/cross_platform/...`) wiring the two CSVs as
      `exp1_data_path`/`exp2_data_path` (steps 1-5, unchanged) **and** as `source_blup_path`/
      `target_blup_path` with `prediction.enabled: true`. Repo-root-relative paths only (Tier 3's
      round-1 pre-merge lesson: a hardcoded absolute path fails on every CI runner).
- [x] 1.3 **Hard gate — do this before any other change on this branch touches `src/`.** A curated
      `CrossPlatformPipeline` golden fixture already exists
      (`tests/fixtures/real/wheat_edpie/expected/cross_platform/<pairing>/`), but per Tier 3's own
      tasks.md (task 1.4, step 4) it is a deliberately curated subset — `config.yaml`,
      `cross_platform_alignment_summary.csv`, `cross_platform_correlations.csv`,
      `exp{1,2}_trait_clusters.csv`, `pipeline_summary.json` — excluding PNGs, logs, and step-1's
      loaded intermediate CSVs. It is **not** suitable as-is for a literal "byte-identical file list"
      backward-compat check (design.md Decision 9). Instead: capture a **fresh, full-file-list
      snapshot** by running `CrossPlatformPipeline` once, now, against the Section 1.1/1.2 small
      synthetic fixture (not real EDPIE data — keeps this regression test CI-fast and decoupled from
      Section 8's manual real-data validation), with `prediction:` absent from the config entirely.
      Commit this snapshot (excluding `config.yaml`, per Decision 9) as the "before" baseline Section
      5's oracle (5.3) diffs against, **as its own commit** — checkable at review time via
      `git merge-base --is-ancestor <snapshot-commit-sha> <first-src/-touching-commit-sha>` (found
      during `/review-openspec` round 2 that the gate was otherwise prose-only with no way to verify
      it was actually honored; found during round 3, by actually running it, that the originally
      suggested `git log --follow -- src/` check is invalid — `--follow` only applies to a single
      file and silently no-ops for a directory pathspec, behaving identically to plain
      `git log -- src/` and checking nothing about ordering).

## 2. `PredictionConfig` (test-first)

- [x] 2.1 Write failing test `test_prediction_config_defaults_to_disabled`: `PredictionConfig()` has
      `enabled=False`. (Scoped to `PredictionConfig` alone — the nested-on-`CrossPlatformConfig`
      assertion lives solely in task 3.1, since `CrossPlatformConfig.prediction` doesn't exist until
      Section 3; testing it here too made this task unpassable on Section 2's implementation alone,
      found during `/review-openspec` round 1.)
- [x] 2.2 Write failing test `test_prediction_config_validation_skipped_when_disabled`:
      `PredictionConfig(enabled=False, predictor_source="not_a_real_value",
      source_blup_path="/does/not/exist")` does not raise (Decision 4 — validation is a full no-op
      when disabled).
- [x] 2.3 Write failing test `test_prediction_config_rejects_invalid_enum_fields` (parametrized over
      `predictor_source` (`{blup, genotype_means}`), `reduction_method` (`{pls_latent,
      representatives, pc1}`), `representative_selection_metric` (`{variance}` only for this tier —
      including an explicit `"heritability"` case, which SHALL now be rejected as invalid, not
      accepted, per Decision 7), and an invalid entry inside `comparison_methods` (same 3-value set
      as `reduction_method`)), all with `enabled=True`: assert `ValueError` naming the invalid
      field/value.
- [x] 2.4 Write failing test `test_prediction_config_rejects_duplicate_method_in_comparison_methods`:
      `reduction_method="pls_latent"`, `comparison_methods=["pls_latent"]` (same method repeated) —
      assert `ValueError` at construction time (prevents a silent output-JSON overwrite in Section 4,
      found during `/review-openspec` round 1).
- [x] 2.4a Write failing test
      `test_prediction_config_rejects_duplicate_entries_within_comparison_methods`:
      `comparison_methods=["representatives", "representatives"]` (same method listed twice within
      the list itself, distinct from 2.4's cross-field case) — assert `ValueError` (same silent
      output-overwrite bug class, missed by 2.4 alone; found during `/review-openspec` round 2).
- [x] 2.5 Write failing test `test_prediction_config_blup_preflight_check_missing_path`:
      `enabled=True, predictor_source="blup"`, `source_blup_path` (or `target_blup_path`) pointing
      at a nonexistent file — assert `ValueError` raised at construction time, before any pipeline
      step exists to run.
- [x] 2.6 Write failing test
      `test_prediction_config_genotype_means_does_not_require_blup_paths`:
      `predictor_source="genotype_means"` with `source_blup_path=None`/`target_blup_path=None` does
      not raise (Decision 2 — this branch reuses task 1's already-loaded raw data instead, per
      Decision 8).
- [x] 2.7 Implement `PredictionConfig` in `pipeline/config/components.py` (plain, mutable
      `@dataclass`, per Decisions 2/4/5/7) with the field list from `proposal.md`.
      `blup_refit_per_fold: bool = False` stays in the schema (roadmap's settled field list) with no
      auto-force validation (Decision 7 — currently inert, no metric value triggers it). Use a
      `Counter`-based check for 2.4a (matching the duplicate-`representative_names` precedent in
      Tier 3's `logo_cv_predict`). Make 2.1-2.6 (incl. 2.4a) green.

## 3. `CrossPlatformConfig` wiring (test-first)

- [x] 3.1 Write failing test `test_cross_platform_config_gains_prediction_field`:
      `CrossPlatformConfig(...)` (existing required fields only, no `prediction:`) has
      `.prediction` as a `PredictionConfig` instance with all defaults.
- [x] 3.2 Write failing test
      `test_cross_platform_config_validates_platform_pairs_direction_against_exp_names`:
      `prediction.enabled=True` with `platform_pairs=[{"source": "not_exp1_or_exp2", "target":
      "also_not"}]` — assert `ValueError` naming the mismatch, raised from
      `CrossPlatformConfig.__post_init__` (Decision 3 — `PredictionConfig` alone can't see
      `exp1_name`/`exp2_name`).
- [x] 3.3 Write failing test `test_cross_platform_config_accepts_valid_platform_pairs_direction`:
      `platform_pairs=[{"source": exp1_name, "target": exp2_name}]` (or the reverse direction) does
      not raise.
- [x] 3.3a Write failing test `test_cross_platform_config_rejects_empty_platform_pairs_when_enabled`:
      `prediction.enabled=True` with `platform_pairs=[]` (the default) — assert `ValueError` stating
      exactly one entry is required, not an undocumented `IndexError` deep in validation code
      (Decision 10, found during `/review-openspec` round 1 by 3 of 5 reviewers independently).
- [x] 3.3b Write failing test
      `test_cross_platform_config_rejects_multiple_platform_pairs_entries`:
      `prediction.enabled=True` with `platform_pairs` holding 2 entries — assert `ValueError` (same
      Decision 10 cardinality check).
- [x] 3.4 Extend `CrossPlatformConfig.__post_init__` with the 3.2/3.3/3.3a/3.3b cross-checks —
      cardinality (Decision 10) validated before the direction-match check. Make 3.1-3.3b green.

## 4. `PredictCrossPlatformStep` (test-first)

- [x] 4.1 Write failing test
      `test_predict_step_builds_source_matrix_from_blup_when_predictor_source_blup`: loads
      `source_blup_path` as the predictor matrix `X`.
- [x] 4.1a Write failing test `test_predict_step_drops_trait_columns_containing_any_nan`
      (Decision 16, found during `/review-openspec` round 3 — real `08_blup_adjusted_means.csv`
      files routinely contain NaN columns for failed model fits, per Tier 1's own documented
      contract, and `logo_cv_predict` hard-rejects any NaN): a BLUP CSV fixture with one trait
      column entirely NaN (a failed-model trait) alongside otherwise-finite columns, on both source
      and target sides — assert the NaN column is dropped before `X`/target-candidate construction,
      not passed through to crash `logo_cv_predict`. Also write
      `test_predict_step_raises_clear_error_when_source_matrix_is_empty_after_nan_drop`: a source
      BLUP CSV where every trait column contains at least one NaN — assert a clear `ValueError`
      (distinct from, and stricter than, task 4.3a's zero-target-representative-traits case, which
      still has PC1 to fall back on; the source side has no equivalent fallback).
- [x] 4.1b Write failing test `test_predict_step_resolves_blup_genotype_column_name` (Decision 17,
      found during `/review-openspec` round 3 — no prior task specified how the BLUP CSV's genotype
      column is identified): a BLUP CSV fixture with its genotype column named `"Genotype"`
      (capitalized — the real, shipped convention from `extract_blup_table()`/
      `StatisticalAnalysisStep`) loads correctly; a second fixture with `"genotype"` (lowercase)
      also loads correctly (fallback); a third fixture with neither column present raises a clear
      `ValueError` naming both attempted column names, not a bare pandas `KeyError`. Explicitly NOT
      `exp1_genotype_col`/`exp2_genotype_col` (those govern the unrelated raw per-sample CSVs for
      steps 1-5).
- [x] 4.2 Write failing test
      `test_predict_step_builds_source_matrix_from_genotype_means_when_predictor_source_genotype_means`:
      reuses **task 1's own result** (`kwargs["01_load_cross_platform_data"]` — see Decision 8; the
      step's `depends_on` includes `"01_load_cross_platform_data"` directly, not just
      `"05_visualize_cross_platform"`, per Decision 15), selecting columns via task 1's
      already-computed `StepResult.metadata["exp1_trait_names"]`/`["exp2_trait_names"]` (Decision
      13 — these are already `exclude_cols`-filtered via `get_trait_columns()`) **before**
      aggregating via `.groupby(genotype_col).mean()` — not a bare groupby-mean over every column in
      task 1's raw `exp1_df`/`exp2_df`, which still contains `genotype`, `replicate`, and other
      metadata columns.
- [x] 4.2a Write failing test
      `test_predict_step_genotype_means_uses_full_raw_trait_set_even_when_trait_reduction_clustering_enabled`
      (regression test for the bug found during `/review-openspec` round 1, Decision 8, and
      corrected during round 2 per Decision 13):
      `predictor_source="genotype_means"` **and** `trait_reduction_method="clustering"` both set on
      the same `CrossPlatformConfig` — assert the predictor matrix `X`'s columns exactly equal task
      1's `exp1_trait_names`/`exp2_trait_names` metadata (the full, `exclude_cols`-filtered trait
      set), not the cluster-representative-reduced subset `ReduceTraitRedundancyStep` (task 2)
      would have produced, and not task 1's raw DataFrame's every column (`genotype`, `replicate`,
      etc. excluded). Confirms task 6 reads task 1's data directly and trait-filters it correctly,
      rather than being contaminated by either an intermediate step's own trait reduction or by
      skipping trait filtering entirely.
- [x] 4.2b Write failing test `test_predict_step_genotype_means_excludes_metadata_columns`
      (tightened during `/review-openspec` round 3 — the original name was disjunctive
      ("crashes_cleanly_or_excludes"), inviting an uncontrolled crash to count as a pass; only one
      outcome is actually accepted): task 1's raw `exp1_df`/`exp2_df` with a non-numeric metadata
      column present (e.g. a notes or date column) and `predictor_source="genotype_means"` — assert
      the resulting predictor matrix does NOT include that column (i.e. 4.2's trait-name-filtered
      selection actually excludes it; `get_trait_columns()` already substring-excludes
      notes/date-like columns), not that the step raises an unrelated `TypeError` from `.mean()` on
      a non-numeric column.
- [x] 4.3 Write failing test `test_predict_step_selects_target_representative_traits`: the
      **target** platform's cluster-representative traits (via `cluster_correlated_traits`/
      `select_cluster_representatives`, reused as-is, `representative_selection_metric="variance"`
      per Decision 7) become the primary prediction targets.
- [x] 4.3a Write failing test `test_predict_step_handles_zero_target_representative_traits`: when
      `select_cluster_representatives` returns an empty list for the target platform (a plausible
      degenerate case at small/CI-fixture scale), assert the step still runs successfully with only
      the PC1 target (`N=1`), not a crash.
- [x] 4.4 Write failing test `test_predict_step_computes_target_pc1_via_whole_dataset_pca_not_per_fold`
      (Decision 6/12): the `target_name="PC1"` value is computed via `pca.fit_pca()` (with
      `StandardScaler` applied first, `random_state=42`), fit once on the full common-genotype
      set — assert `fit_pca_on_fold` is **not** called for this purpose (spy/mock), reserving it for
      the source-side per-fold reduction when `reduction_method="pc1"` (4.6 below).
- [x] 4.4a Write failing test `test_predict_step_target_pc1_values_match_independent_whole_dataset_pca`
      (positive value check, added per `/review-openspec` round 1 — 4.4's mock alone only proves
      `fit_pca_on_fold` wasn't called, never that the computed values are actually correct): build a
      small synthetic target matrix with a known structure; independently compute
      `pca.fit_pca(StandardScaler().fit_transform(target_matrix), n_components=1, random_state=42)`
      in the test; assert the resulting `CrossPlatformPredictionResult`'s
      `TargetPrediction(target_name="PC1").y_true` matches that independent computation for every
      genotype within `pytest.approx`.
- [x] 4.5 Write failing test `test_predict_step_aligns_to_common_genotypes`: genotypes present in
      only one of source/target are excluded before calling `logo_cv_predict`.
- [x] 4.5a Write failing test
      `test_predict_step_raises_clear_error_when_common_genotypes_below_minimum`: source/target
      predictor matrices with fewer than 3 genotypes in common (including the zero-overlap case) —
      assert a clear, step-level `ValueError` naming the pair (source/target platform names) and the
      common-genotype count, not a bare pass-through of `logo_cv_predict`'s generic
      "fewer than 3 genotypes" message. (This is a stricter improvement over
      `LoadCrossPlatformDataStep`'s own zero-common-genotype precedent,
      `load_cross_platform_data.py:127-131`, not a literal mirror of it — that precedent's message
      doesn't itself name platform/pair identifiers either, corrected during `/review-openspec`
      round 2.)
- [x] 4.5b Write failing test `test_predict_step_derives_X_y_genotypes_from_one_canonical_index`
      (Decision 14, found during `/review-openspec` round 2 — `logo_cv_predict` treats row-order
      alignment between `X`/`y`/`genotypes` as an unenforced caller precondition; mechanism pinned
      during round 3, since no existing fixture pattern for this exists in this repo): build the
      target platform's BLUP/genotype-mean table with its rows reordered via `.iloc[::-1]` (reversed
      order) relative to the source platform's table (same genotype set) — assert the resulting
      `CrossPlatformPredictionResult` correctly pairs each genotype's source values with that same
      genotype's target value for **every target, including the PC1 target** (not only
      representative-trait targets — found during round 3 that PC1 is otherwise untested under
      shuffled input, since 4.4a's value-correctness check uses an already-well-ordered matrix),
      checked against an independently-computed, explicitly-`.loc[]`-indexed reference pairing —
      not a silently mis-paired result from incidental row-order assumptions.
- [x] 4.6 Write failing test `test_predict_step_calls_logo_cv_predict_once_per_target_per_method`:
      for `N` target traits (representatives + PC1) × `M` methods (`reduction_method` +
      `comparison_methods`, guaranteed distinct per task 2.4's config validation), `logo_cv_predict`
      is called `N * M` times, each with the expected `reduction_method` and (for
      `"representatives"`) `representative_names` drawn from the **source** platform's own
      cluster-representative selection (a separate application from 4.3's target-side selection).
- [x] 4.7 Write failing test `test_predict_step_builds_one_result_per_method`: one
      `CrossPlatformPredictionResult` per method, each holding all `N` targets' `TargetPrediction`
      entries (Tier 3's existing shape, reused unchanged).
- [x] 4.8 Write failing test `test_predict_step_saves_json_output_per_method`: one JSON file per
      method written to `run_dir` (naming convention decided during implementation, e.g.
      `06_prediction_<method>.json`) — safe from collision since 2.4 already rejects
      `comparison_methods` duplicating `reduction_method`.
- [x] 4.9 Implement `PredictCrossPlatformStep(BaseStep)` in new
      `src/sleap_roots_analyze/pipeline/steps/predict_cross_platform.py`, consuming
      `logo_cv_predict`/`fit_pca_on_fold`/`CrossPlatformPredictionResult` unchanged, plus `pca.fit_pca`
      directly for the PC1 target (Decision 12 — no new `PCAConfig`). Read `exp1_trait_names`/
      `exp2_trait_names` from task 1's metadata before any `genotype_means` aggregation (Decision
      13); drop any trait column containing any NaN, on both source and target sides (Decision 16);
      resolve the BLUP CSV's genotype column via `"Genotype"` then `"genotype"` (Decision 17); derive
      `X`/every `y`/`genotypes` from one canonical, sorted, explicitly-`.loc[]`-indexed common-genotype
      list (Decision 14); consume `kwargs["05_visualize_cross_platform"]` only to confirm ordering,
      never for data (Decision 15). Docstring documents the selection-bias note from Decision 11
      (target-trait selection uses full-outcome data — a distinct concern from fit-time leakage).
      Make 4.1-4.8 (incl. 4.1a/4.1b/4.2a/4.2b/4.3a/4.4a/4.5a/4.5b) green.
- [x] 4.9a Write failing test `test_predict_step_blup_refit_per_fold_is_inert`: an otherwise-identical
      config with `blup_refit_per_fold=True` vs. `blup_refit_per_fold=False` (both valid, since
      `representative_selection_metric="variance"` never triggers this field) — assert the resulting
      `CrossPlatformPredictionResult`s are identical (found during `/review-openspec` round 2 — the
      field's documented "inert" claim, Decision 7, had no regression test proving it). **This task
      is necessarily a post-implementation tripwire, not a pre-implementation red test** (clarified
      during round 3 — Section 4's "(test-first)" header applies to 4.1-4.9's design-driven tests;
      4.9a is different in kind, since a universally-quantified "this field has no effect" claim
      cannot be driven red without first injecting a bug): expect it to pass immediately upon
      writing, once 4.9 is implemented — if it ever fails, the implementation has accidentally wired
      the field to something and Decision 7 needs revisiting before proceeding.
- [x] 4.9b Write failing test `test_predict_step_never_reads_task5_data` (Decision 15, found during
      `/review-openspec` round 3 — the "ordering-only, not data" claim had no enforcing test): spy on
      `kwargs["05_visualize_cross_platform"]`, e.g. by passing a sentinel/corrupted `TaskResult` as
      that dependency's value while providing a normal, valid `kwargs["01_load_cross_platform_data"]`
      — assert the step still produces the correct `CrossPlatformPredictionResult` and never
      accesses `kwargs["05_visualize_cross_platform"].data`.

## 5. `CrossPlatformPipeline` task wiring (test-first)

- [x] 5.1 Write failing test `test_cross_platform_pipeline_appends_predict_task_when_enabled`:
      `CrossPlatformPipeline(config).create_tasks()` includes a 6th task,
      `depends_on=["01_load_cross_platform_data", "05_visualize_cross_platform"]` (both — Decision
      8), when `config.prediction.enabled=True`.
- [x] 5.2 Write failing test `test_cross_platform_pipeline_omits_predict_task_when_disabled`:
      `create_tasks()` returns exactly the existing 5 tasks when `config.prediction.enabled=False`
      (the default) — no 6th task constructed at all, not merely skipped at run time.
- [x] 5.3 Write failing test `test_cross_platform_pipeline_backward_compat_disabled_by_default`:
      running `CrossPlatformPipeline` against the Section 1.3 fresh golden-fixture snapshot's config
      (no `prediction:` block) produces byte-identical **analysis** output (file list + content,
      excluding `config.yaml` — Decision 9, since that file's content depends on the field's mere
      presence, not on `enabled`) to that snapshot — the CI backward-compat oracle from issue #196.
- [x] 5.4 Add `_run_predict_cross_platform` runner method (reading `kwargs["01_load_cross_platform_data"]`
      and `kwargs["05_visualize_cross_platform"]`) + the conditional `Task(...)` entry (both
      `depends_on` names) to `CrossPlatformPipeline.create_tasks()`. Make 5.1-5.3 green.

## 6. CI wiring-correctness oracle (test-first)

> **Requires Sections 1-5 fully implemented and green before this test can be written
> meaningfully** (found during `/review-openspec` round 1) — it exercises the complete pipeline
> path (config, validation, step, task wiring), not any single unit in isolation.

- [x] 6.1 Write failing test `test_predict_cross_platform_pipeline_matches_direct_logo_cv_predict_call`:
      run `CrossPlatformPipeline` against the Section 1.1/1.2 harness fixture
      (`prediction.enabled=True`); independently load the same two BLUP CSVs in the test and call
      `logo_cv_predict` directly with the same `reduction_method`/target trait; assert the
      pipeline's `CrossPlatformPredictionResult` R² for that target matches the direct call's R²
      via `numpy.testing.assert_allclose(..., rtol=1e-6, atol=1e-9)` — **not exact equality, and
      not an over-tight `rel=1e-9`** (round 1 reworded this away from exact equality but picked a
      tolerance ~1000x tighter than this codebase's own documented convention; corrected during
      `/review-openspec` round 2 to match `docs/reproducibility.md`'s actual
      `RTOL=1e-6, ATOL=1e-9` policy for cross-OS/BLAS-affected numerical-reproduction claims on the
      Ubuntu/Windows/macOS CI matrix) — the "wiring correctness, not just existence" oracle from
      issue #196.

## 7. CLI wiring (test-first)

- [x] 7.1 Write failing test `test_cli_cross_platform_dry_run_lists_prediction_step_when_enabled`:
      `sleap-roots-analyze cross-platform <config with prediction.enabled=True> --dry-run` output
      includes a 6th step line.
- [x] 7.2 Write failing test `test_cli_cross_platform_dry_run_omits_prediction_step_when_disabled`:
      same command with `prediction.enabled=False` (or the `prediction:` block absent entirely) —
      dry-run output has exactly the existing 5 steps, unchanged from today.
- [x] 7.3 Update `cli.py`'s `cross_platform()` dry-run steps list (conditional 6th tuple) and its
      docstring (mention prediction as an optional 6th step). Make 7.1-7.2 green.

## 8. Manual real-data validation (non-CI, pre-merge gate) — **run 2026-07-16, pending Elizabeth's sign-off**

> **Correction to task 8.1's original assumption, found before this run:** Tier 3 Section 8's
> BLUP tables are **not** reusable as-is. A dedicated vault investigation (`figures/README.md`'s
> "Pipeline Provenance" section, `scripts/figures/style.py`'s `PIPELINE_RUN`/`FIELD_PIPELINE_RUN`
> constants, and `scripts/figures/fig10_11_plan.md`, all in the external `wheat-edpie-paper`
> vault) established that Tier 3 used the **wrong data vintage for Turface19/Turface150** — the
> repo's already-committed Feb-12 fixture, chosen for convenience, not the paper's actual vintage.
> The explicit, written provenance record states the paper's real Turface19/Turface150/Cylinder
> figures all come from **`pipeline_run_20260319/2026-03-20_132531`** ("the one used for all paper
> figures," per `fig10_11_plan.md`); only the two Field/Root_Core-involving comparisons were
> rerun on Mar-30 (`pipeline_run_field_v2`/`cross_platform_field_v2`) — Cylinder's own QC data was
> never rerun and is byte-identical between the Mar-20 batch and the already-committed
> `cylinder_final_data_paper_vintage.csv` (confirmed: both are shape `(124, 880)`, identical
> columns). So this run rebuilt all 4 BLUP tables fresh from the *correct* vintage per platform:
> Turface19/Turface150/Cylinder from Mar-20 QC output (`07_data_outliers_removed.csv`), Field from
> the already-committed Mar-30 `root_core_final_data_paper_vintage.csv` — rather than reusing any
> of Tier 3's artifacts.

- [x] 8.1 Built real BLUP tables for all 4 EDPIE platforms via direct `extract_blup_table()` calls
      (script: `pipeline_runs/section8_manual_validation_20260716/build_blup_tables.py`, not
      committed — matches Tier 3's own precedent of non-committed manual-validation artifacts).
      `trait_cols` per platform = all columns minus that platform's real correlation-config
      exclude list (confirmed identical across every pairing that platform appears in) minus the
      `Genotype`/`Replicate`/`Barcode`/`Geno_rep` identifier columns. Result: Turface19 (19
      genotypes × 19 traits), Turface150 (156 genotypes × 16 traits), Cylinder (19 genotypes × 836
      traits), Field (20 genotypes × 24 traits) — zero all-NaN trait columns in any table. Built 4
      `CrossPlatformConfig` YAMLs (`pipeline_runs/section8_manual_validation_20260716/configs/`),
      each with `exp1`/`exp2_data_path`/`exclude_cols` copied verbatim from that pair's real
      correlation config (Mar-20 or Mar-30 vintage as established above) plus a `prediction:` block
      (`predictor_source=blup`, `reduction_method=pls_latent`,
      `comparison_methods=[representatives]`, `platform_pairs` naming the directed source/target).
      Ran `sleap-roots-analyze cross-platform <config>.yaml` for all 4 pairs — all completed
      successfully (6/6 tasks each, including the new `06_predict_cross_platform` task), 19 common
      genotypes found in every pair.
- [x] 8.2 Sanity-checked the resulting R²/RMSE/ρ (354 total target/method/pair rows,
      `pipeline_runs/section8_prediction_summary.csv`) against Tier 3 Section 8's pattern. No
      wiring bug found: **the qualitative pattern matches Tier 3's own finding independently, on
      corrected data.** Full-trait-matrix/PC1 predictions are noisy and mostly negative-R² across
      all 4 pairs (e.g. Cylinder→Field PC1, `pls_latent`: R²=-0.72, ρ=-0.59, p=0.0072 — same sign
      and comparable magnitude to Tier 3's own Cylinder→Field finding of ρ=-0.48, p=0.038, on data
      that didn't change vintage between the two runs). Turface19→Cylinder's PC1 result changed
      from Tier 3's ρ=-0.62 (p=0.005, wrong Turface vintage) to this run's ρ=-0.37 (p=0.12, correct
      vintage) — an expected, explained divergence (different underlying Turface19 data), not a
      wiring bug. A handful of individual representative-trait targets recover genuine positive
      signal (e.g. Cylinder→Field `representatives`, target `Root Count 20cm`: R²=0.25, ρ=+0.49,
      p=0.033; Turface19→Cylinder `representatives`, target `Seminal Angles Proximal Max Max`:
      R²=0.38, ρ=+0.62, p=0.004) — consistent with Tier 3's own qualitative conclusion that the
      full multi-trait predictor matrix is a curse-of-dimensionality problem at n≈19, while
      individual well-correlated traits still carry recoverable signal. No exceptions, crashes, or
      NaN-propagation across all 4 pairs × 2 methods × ~40-130 targets each.
- [x] 8.3 Findings recorded above. **Correction found during review (prompted by Elizabeth's own
      question about trait counts):** the BLUP-extraction script's exclude list missed
      `Computation.Time.s` — pipeline-generated processing-time metadata present in the raw
      Turface19/Turface150 QC CSVs, not a biological trait — so the first-pass Turface19/Turface150
      BLUP tables had 19/16 columns instead of the correct 18/15 (confirmed against the actual
      pipeline's own `ReduceTraitRedundancy` clustering-step input counts, which correctly excluded
      it throughout). Cylinder and Field's BLUP tables were unaffected (no such column present).
      Fixed the script, rebuilt both tables, and reran the 3 pairs where Turface19/Turface150 is
      the *source* platform (Turface19→Cylinder, Turface19→Field, Turface150→Turface19 — the only
      ones where this predictor-side contamination could reach `pls_latent`'s full-trait-matrix
      fit; Cylinder→Field is unaffected, no Turface involved). Effect: small, expected numeric
      shifts (e.g. Turface19→Cylinder PC1 R²: -0.480→-0.419; a handful of individual targets moved
      in/out of the p<0.05 threshold by noise-level margins) — **no change to the qualitative
      conclusion.** **Conclusion stands: no discrepancy inconsistent with a pure
      wiring-bug/expected-data-difference explanation — Decision 6/12 do not need reopening.**
      Final summary (post-fix): `pipeline_runs/section8_prediction_summary.csv`. Presented to
      Elizabeth for review 2026-07-16; **explicit sign-off still pending** (this checkbox tracks
      "findings recorded," not the sign-off itself — see task 10.5).
- [x] 8.5 **(New — statistical-validity caveat, added during Elizabeth's review of 8.1-8.3.)**
      Checked whether n≈18-20 genotypes and hundreds of candidate traits impose a hard limit on
      what this analysis can detect, independent of any implementation question:
      - **Power.** At n=18 (most Turface/Cylinder pairs), a two-sided Spearman/Pearson test at
        α=0.05 needs |r|≥0.62 for 80% power (Fisher-z calculation); at n=20 (Field), ≥0.59; at
        n=24, ≥0.55. Cross-checked against the real correlation output
        (`cross_platform_correlations.csv`, Turface19-vs-Cylinder, 1,290 pairs, n=18): median
        achieved power at the *observed* effect size is **8.8%**; only **4 of 1,290 pairs (0.3%)**
        reach |r|≥0.62; **zero** survive FDR correction. This is a property of sample size, not of
        any specific model choice — no reduction method or statistical test recovers detection
        power the sample size doesn't have.
      - **Multiple testing.** Applying BH-FDR correction to the 354 target/method/pair predictions
        from 8.1/8.2: raw uncorrected p<0.05 count = **63**; FDR-corrected count = **9**. The 9
        survivors all have *negative* Spearman ρ (predictions anti-correlated with truth, not
        correlated) — none of the positive-R² targets highlighted in 8.2's writeup survive
        correction, so they should be read as exploratory, not as confirmed findings.
      - **Single-realization CV-R² variance.** Independent of significance testing, this
        program's own Decision 6 (design.md, this tier's parent Tier 3 change) already
        empirically demonstrated that a single LOGO-CV run's R² at this sample size has very
        high variance even for a *known, planted* signal (single-seed R² ranged from -0.67 to
        +0.87 across seeds of the same synthetic signal strength). This means individual R²/ρ
        values from 8.1/8.2, taken alone, are not yet a validated statement about cross-platform
        predictability — that inferential step is exactly what Tier 4's (not-yet-built)
        permutation null is for, per Wolfgang's original ask (`roadmap.md` Goal section), which
        explicitly required a permutation null alongside R²/RMSE/ρ.
      - **Conclusion:** none of this invalidates Section 8's actual purpose — confirming the
        `PredictCrossPlatformStep`/config-driven wiring runs correctly and produces statistically
        sane small-sample behavior (matching Tier 3's own findings independently). But Section 8's
        R²/RMSE/ρ numbers should not be read as validated claims about real cross-platform
        predictability until Tier 4 exists; that limitation is inherent to n≈19, not a defect in
        this tier's implementation.
- [ ] 8.4 **Not triggered** — 8.2 found no discrepancy inconsistent with a wiring-bug/expected-
      data-difference explanation; no `design.md` Decision reopened, no additional `fix:` commit
      needed as a result of this section.

## 9. Docs

- [x] 9.1 **No `docs/API.md` entry.** Corrected during `/review-openspec` round 1: `API.md`'s `##`
      sections mirror `sleap_roots_analyze.__all__` (confirmed neither `CrossPlatformConfig` nor any
      existing pipeline Step class is in `__all__` — Configs/Steps are documented in guide docs, not
      the `__all__`-driven API reference). `PredictionConfig`/`PredictCrossPlatformStep` follow that
      same precedent — no API.md changes, no TOC entry. (The original task's TOC-invariant citation
      was also a broadened paraphrase of Tier 3's actual finding — the real invariant is "every `##`
      *module* heading," not "every `##` heading"; `API.md` already has 3 non-module `##` headings
      with no TOC entries, e.g. `Error Handling`.)
- [x] 9.2 Add a `docs/CHANGELOG.md` `[Unreleased]` `### Added` entry.
- [x] 9.3 **Extend the existing section, don't add a new one.** `docs/CROSS_PLATFORM_ANALYSIS.md`
      already has a `## Cross-Platform Genotype-Effect Prediction` section (shipped by Tier 3),
      closing with: *"This tier ships the statistical machinery only — `PredictionConfig` and a
      `PredictCrossPlatformStep` pipeline wiring (Tier 3.5)... are separate, later changes."* Extend
      that section — add `###` subheadings for the new content rather than appending bare prose
      (matching this doc's own convention elsewhere, e.g. `### Configuration`/`### Required
      Parameters`; found during `/review-openspec` round 3 that the section currently has none and
      round 1/2's additions risked reading as bolted-on) — and **correct its now-stale closing
      sentence** (found during round 1) rather than authoring a duplicate parallel section
      elsewhere. Split per round 3's finding that this task had become an unsplit compound of 5
      distinct asks, inconsistent with this file's own one-assertion-per-checkbox convention used
      everywhere else (2.4/2.4a, 3.3/3.3a/3.3b, etc.):
- [x] 9.3a Correct the stale closing sentence; add a `### Configuration` subheading with a concrete
      YAML example showing a `prediction:` block (`enabled`, `predictor_source`, `reduction_method`,
      `comparison_methods`, `representative_selection_metric`, `platform_pairs`,
      `source_blup_path`/`target_blup_path`) added to an existing `CrossPlatformConfig` YAML — the
      only way a user can actually learn this feature's YAML syntax, since `proposal.md`/`spec.md`
      are internal planning artifacts, not shipped docs (found insufficient in round 1, added round
      2).
- [x] 9.3b Add a `### Current Limitations` subheading (or similar) noting: PC1-as-target's
      computation (`pca.fit_pca()` with `StandardScaler` pre-applied, `random_state=42` fixed) is
      **not user-configurable** in this tier; `representative_selection_metric="heritability"` is
      not yet supported (only `"variance"`); `blup_refit_per_fold` is present in the schema but
      currently inert (found round 2 for the first two notes; the third added round 3, since it had
      the same "internal-doc-only" gap as the other two but was missed).
- [x] 9.3c Add a one-line note (same subheading) that only genotypes common to **both** source and
      target BLUP/genotype-mean tables are used for prediction — genotypes present in only one side
      are silently excluded from the result, not merely from an error path (found during
      `/review-openspec` round 3: above the hard-error minimum-genotype-count threshold, this
      set-intersection behavior was previously undocumented anywhere a user would see it).
- [x] 9.4 Add a docstring cross-reference (not a behavior change) on `PredictCrossPlatformStep`
      itself, not on `CrossPlatformSummaryGenerator`/`cross_platform_summary.py` (pinned during
      `/review-openspec` round 1 to stay clear of follow-up #197's territory even for a
      comment-only edit) — noting that `CrossPlatformSummaryGenerator` does not yet surface
      prediction results — tracked as follow-up
      [#197](https://github.com/talmolab/sleap-roots-analyze/issues/197), not fixed here. **Also**
      add the same one-line note under task 9.3's new doc subheadings (found during round 2: a
      docstring alone isn't where a user who runs `/cross-platform-summary` and doesn't see
      predictions would look) — this doc-only mention does not touch `cross_platform_summary.py`
      itself, so #197's scope boundary stays intact.

## 10. Validation

- [x] 10.1 `openspec validate add-prediction-pipeline-step --strict` — resolve every reported issue.
- [x] 10.2 `/lint` (black + ruff) on all changed files.
- [x] 10.3 Full `uv run pytest --cov --cov-branch` — no regressions, all new tests (Sections 2-7)
      green.
- [x] 10.4 `/review-openspec` — adversarial proposal review, ≥1 round, reconcile literally into
      `design.md`. **Round 1 complete** (5 parallel reviewers; 5 BLOCKING + ~14 IMPORTANT findings,
      reconciled into `design.md`'s "Adversarial Review Reconciliation (round 1)" section, this
      file, and `proposal.md` — see Decisions 7-12). **Round 2 complete** (a second, independent
      5-agent pass, run fresh with no memory of round 1, specifically to catch anything round 1
      missed): found 2 new HIGH-severity issues — one a direct consequence of round 1's own fix
      (Decision 8's `depends_on` change had reintroduced a crash/data-pollution risk, fixed by
      Decision 13; genotype row-order alignment was unenforced and untested, fixed by Decision 14)
      — plus corroborated/new gaps (`blup_refit_per_fold` inertness untested, `comparison_methods`
      self-duplication unrejected, Section 6's tolerance was actually wrong not fixed, one spec.md
      scenario round 1 claimed but hadn't actually added, Section 9's doc plan insufficient for an
      end user), all reconciled into `design.md`'s "Adversarial Review Reconciliation (round 2)"
      section, this file, and `proposal.md` — see Decisions 13-15. **Round 3 complete** (a third,
      independent 5-agent pass, run fresh with no memory of rounds 1-2 — every prior round's code
      citation independently re-verified and confirmed accurate, including Decisions 12-15's exact
      mechanisms traced through the full `TaskResult`/`StepResult`/DAG-executor plumbing): found 1
      new HIGH (real BLUP CSVs routinely contain NaN columns for failed model fits, previously
      unaddressed — fixed by Decision 16) + 1 new MEDIUM-HIGH (the BLUP CSV's genotype-column name
      was never specified, a real implementer landmine — fixed by Decision 17), plus several
      smaller process/documentation/test-specification gaps, all reconciled into `design.md`'s
      "Adversarial Review Reconciliation (round 3)" section, this file, and `proposal.md` — see
      Decisions 16-17. Findings diminished in severity and volume across the three rounds (5
      BLOCKING → 2 new HIGH → 1 new HIGH + 1 new MEDIUM-HIGH), a converging signal; a fourth round is
      available on request but not proactively recommended. This task is not satisfied until the
      user has reviewed and approved the fully-reconciled proposal — required before implementation
      (Sections 1-9) begins, per the roadmap's per-tier loop.
- [ ] 10.5 Complete Section 8's manual EDPIE validation and get Elizabeth's explicit sign-off before
      opening the PR.
- [ ] 10.6 **Post-implementation code review** (found missing during `/review-openspec` round 3 —
      Tier 3's own two `/review-pr` passes, pre-PR and on-the-open-PR, caught its real implementation
      bugs: a CI path bug, input validation gaps, a mypy baseline violation): run `/pre-merge-check`
      and/or `/review-pr` against the complete implementation diff before opening the PR, and again
      after CI runs on the open PR — this is the review stage Tier 3's own task 8.4-equivalent
      precedent (trailing `fix:` commits) actually depends on existing.
