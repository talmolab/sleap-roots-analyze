## 1. `select_n_features_by_variance()` helper (`pca.py`)

Independently committable — this function is uncalled by production code
until section 2, so it carries zero blast radius on its own.

- [ ] 1.1 Write failing tests in `tests/test_pca.py` for
      `select_n_features_by_variance(feature_contributions_df, threshold)`:
      threshold met exactly at a row boundary, threshold requiring all
      features, threshold met by fewer than all features (verify cumulative
      `fractional_contribution` meets but doesn't wildly overshoot),
      `threshold <= 0` (resolves to exactly 1 feature, no exception), and a
      single-feature DataFrame.
- [ ] 1.2 Implement `select_n_features_by_variance()` in `pca.py`,
      mirroring `select_n_components()`'s
      `np.argmax(cumulative >= threshold) + 1` pattern over
      `fractional_contribution`, with the explicit `threshold <= 0` branch
      per `design.md` Decision 2.
- [ ] 1.3 Run `uv run pytest tests/test_pca.py -k variance` and confirm green.

## 2. `PCAConfig` schema + `PCAAnalysisStep` call site (single atomic commit)

**Do not split this section across commits.** Per `design.md`'s Risks
section: changing `PCAConfig.n_top_features`'s default to `10.0` without
also updating the call site to cast to `int(...)` for count branches
crashes every `feature_selection_strategy` (not just `"extreme"`) with
`TypeError: slice indices must be integers...` inside
`select_top_features_from_pca()`'s `np.argsort(...)[::-1][:n_features_to_select]`
slicing. The schema change, the call-site rewrite, and the test updates
below must land together.

- [ ] 2.1 Write failing tests in `tests/test_step_pca_analysis.py`:
      - a run with `n_components` selecting >=3 PCs and
        `feature_selection_strategy` set to `"extreme"`, `"top_absolute"`,
        and `"top_contribution"` (all three, not just `"extreme"`) selects
        features from PCs beyond PC1/PC2 (fixes #203 for these methods).
      - `feature_selection_strategy="extreme"` always yields exactly
        1-most-positive + 1-most-negative per retained PC, regardless of
        `n_top_features`'s value (including old configs' `1`/`5`).
      - `feature_selection_strategy="extreme"` with `n_components=1`
        (single retained PC) yields at most 2 features, deduplicated to 1
        if only a single feature is available.
      - `feature_selection_strategy="top_variance"` with `n_top_features=0.8`
        selects a feature count whose cumulative `fractional_contribution`
        meets but doesn't wildly overshoot `0.8`.
      - `feature_selection_strategy="top_variance"` with `n_top_features=0`
        (or negative) selects exactly 1 feature, no exception.
      - `feature_selection_strategy="top_variance"` with
        `n_top_features=1.0` (exact boundary) selects exactly 1 feature,
        not "enough features for 100% variance."
      - `feature_selection_strategy="top_variance"` with `n_top_features=5`
        (unchanged, `>= 1`) still selects exactly 5 features.
      - `feature_selection_strategy="extreme"` emits a `logger.info` record
        (assert via `caplog`) stating `n_top_features` is not read for this
        method, regardless of `n_top_features`'s value; no such record is
        emitted for any other strategy.
      - update the existing `test_pca_different_feature_selection_strategies`
        assertion (`len(top_features) >= config.pca.n_top_features`), which
        no longer holds for `"extreme"` under the new contract — in this
        same commit, not left broken across a commit boundary.
- [ ] 2.2 Update `PCAConfig.n_top_features` in `components.py`: change
      `n_top_features: int = 10` to `n_top_features: float = 10.0`, and
      rewrite its docstring to document all three cases: ignored for
      `"extreme"`; `< 1` = variance-fraction threshold / `>= 1` = count
      (including the `1.0`-is-a-count-not-100%-threshold footgun) for
      `"top_variance"`; plain whole-number count (validated `>= 1`, see
      section 3) for `"top_absolute"`/`"top_contribution"`.
- [ ] 2.3 Update `PCAAnalysisStep.execute()`: always pass
      `pc_indices=list(range(n_components))`; branch on
      `feature_selection_strategy`/`n_top_features` per `design.md`
      Decisions 1–2 to resolve `n_features_to_select`, casting to
      `int(...)` for every count branch; emit the `logger.info()` message
      from Decision 1 whenever `feature_selection_strategy == "extreme"`.
- [ ] 2.4 Run `uv run pytest tests/test_step_pca_analysis.py` and confirm green.

## 3. Config validation (`pipeline/config/utils.py`)

- [ ] 3.1 Write failing tests in `tests/test_pipeline_config.py` (for
      `validate_qc_config()`) and `tests/test_viz_pipeline_config.py` (for
      `validate_viz_config()`) — these are the two files where each
      function's existing PCA-validation tests already live
      (`test_validate_config_invalid_pca_components`,
      `test_validate_config_invalid_pca_strategy`,
      `test_validate_config_valid_pca_strategies` in the former;
      `test_invalid_pca_n_components_raises`, `test_invalid_pca_strategy_raises`
      in the latter) — covering every scenario in the `config-management`
      spec delta:
      - `n_top_features < 1` rejected for `"top_absolute"`/
        `"top_contribution"`, asserting via `pytest.raises(ValueError,
        match=...)` that the message names both config fields, the
        offending strategy, and requires an integer `>= 1` (per the
        spec's own message-content requirement — don't just assert
        `ValueError` is raised).
      - a non-whole-number `n_top_features >= 1` (e.g. `5.7`) rejected for
        every strategy except `"extreme"` (`"top_variance"`,
        `"top_absolute"`, `"top_contribution"`), asserting the message
        states the fractional part would be truncated.
      - a whole-number `n_top_features` (e.g. `5.0` or `5`) `>= 1` is
        **accepted** for every one of the four strategies
        (`"extreme"`, `"top_absolute"`, `"top_contribution"`,
        `"top_variance"`) — not just `"top_variance"`.
      - a threshold `n_top_features < 1` is accepted for `"top_variance"`.
      - any `n_top_features` value, including `< 1` or fractional, is
        accepted for `"extreme"`.
      - default config values (`feature_selection_strategy="top_variance"`,
        `n_top_features=10.0`) pass.
- [ ] 3.2 Add the two `design.md` Decision 3 checks to both
      `validate_qc_config()` and `validate_viz_config()`, using a tolerance
      comparison (`abs(n_top - round(n_top)) > 1e-9`), not exact `!=`, for
      the whole-number check.
- [ ] 3.3 Run both test files and confirm green.

## 4. Config and docs cleanup

- [ ] 4.1 Remove the now-meaningless `n_top_features` line from every
      active config pairing `feature_selection_strategy: "extreme"` with an
      explicit `n_top_features` — verified list of **28 files**: the 27
      under `configs/active/viz/` (`alfalfa_gwas_groups_1_to_6_combined.yaml`,
      `alfalfa_gwas_groups_1_to_6_combined_no_root_widths.yaml`,
      `alfalfa_gwas_w1w2_combined.yaml`, `alfalfa_gwas_wave1.yaml`,
      `alfalfa_gwas_wave1_canola.yaml`, `alfalfa_gwas_wave1_canola_models.yaml`,
      `amaranth_tis108_exp1.yaml`, `canola_diversity_screen_qc.yaml`,
      `emily_shane_pennycress_2026_02_09.yaml`, `emily_shane_soybean_2026_01_15.yaml`,
      `emily_shane_soybean_2026_03_03.yaml`, `emily_shane_soybean_2026_03_03_grouped.yaml`,
      `giftol_pennycress_s32_2026_05_11.yaml`, `javier_ttc_salk_soybean.yaml`,
      `javier_ttc_salk_soybean_brightness.yaml`,
      `javier_ttc_salk_soybean_full_experiment_9wave.yaml`,
      `javier_ttc_salk_soybean_full_experiment_9wave_per_wave.yaml`,
      `shree_weep_soybean.yaml`, `suyash_arabidopsis_pgm1_pac_2026_05_22.yaml`,
      `turface_alfalfa_gwas.yaml`, `viz_alfalfa_gwas_wave_1_grouped.yaml`,
      `viz_cylinder_edpie.yaml`, `viz_field_2024_clean.yaml`,
      `viz_root_coring.yaml`, `viz_turface_150genotypes.yaml`,
      `viz_turface_19genotypes.yaml`, `weep_maurizio_wave1.yaml`) plus the
      flat pre-reorg duplicate `configs/active/viz_turface_150genotypes.yaml`.
      **`configs/active/qc/*.yaml` needs no edits for this task** — verified
      no QC config pairs `"extreme"` with an explicit `n_top_features`.
- [ ] 4.2 Rewrite stale `n_top_features`-related comments (not just the
      key itself) in every one of these locations:
      - `configs/active/viz/viz_alfalfa_gwas_wave_1_grouped.yaml` (`pca:` block comment)
      - `configs/active/viz/alfalfa_gwas_wave1_canola_models.yaml` (`pca:` block
        paragraph explaining "5 means 5 per direction per PC")
      - `configs/active/viz/viz_cylinder_edpie.yaml` — both the `pca:` block
        comment AND the separate `static_viz:` comment ("PCA biplot feature
        control - matches pca.n_top_features for extreme strategy")
      - `configs/active/viz/viz_field_2024_clean.yaml` (`# Top features for
        biplot` inline comment, which conflates this field with the
        separate `static_viz.pca_biplot_top_features`)
      - `configs/active/viz/viz_root_coring.yaml` (`# Matches
        N_TOP_FEATURES_BIPLOT` inline comment)
      - `configs/active/viz/viz_turface_19genotypes.yaml` — both the `pca:`
        block comment AND the `static_viz:` comment
      - `configs/examples/viz_comprehensive.yaml`, `viz_publication.yaml`,
        `viz_standard.yaml` — both the `pca:` block comment ("controls how
        many features are selected for ANALYSIS purposes... interesting
        genotype identification" — also fixing the unrelated pre-existing
        wrong claim per #206's original investigation) AND the
        `static_viz:` biplot comment ("For "extreme" selection strategies,
        set this to match the desired per-extreme count...")
      - `configs/examples/viz_minimal.yaml` (`pca:` block comment only —
        its biplot comment doesn't reference `"extreme"` and needs no change)
      All rewrites describe the new semantics: ignored under `"extreme"`;
      threshold-vs-count under `"top_variance"`; whole-number count,
      validated, for `"top_absolute"`/`"top_contribution"`.
- [ ] 4.3 Add a parametrized regression test (e.g. in
      `tests/test_pipeline_config.py` or a new `tests/test_viz_configs_load.py`)
      that calls `load_viz_config()` then `validate_viz_config()` on all 28
      files edited in 4.1 (the 27 under `configs/active/viz/` plus the flat
      `configs/active/viz_turface_150genotypes.yaml`), asserting no
      exception — a concrete, re-runnable guard against a YAML syntax error
      or indentation break introduced by removing the `n_top_features`
      line, replacing the earlier vague "spot-check a sample" plan.
- [ ] 4.4 Add a `docs/CHANGELOG.md` `[Unreleased]` entry: a `### Fixed`
      entry for #203 (PC scoping) and a `### Changed` entry for #206
      (variance-driven selection), matching the file's existing
      `[Unreleased]` section structure (`### Added` / `### Fixed` /
      `### Changed`, in that order).

## 5. Full verification and PR

- [ ] 5.1 Run
      `uv run pytest --cov=src/sleap_roots_analyze --cov-report=xml --durations=20 -m "not integration" tests/`
      (exact CI invocation, `.github/workflows/ci.yml`) — full suite green.
- [ ] 5.2 Run `uv run black src/sleap_roots_analyze tests` and
      `uv run ruff check src/sleap_roots_analyze tests`.
- [ ] 5.3 Run `openspec validate fix-pca-top-feature-selection --strict`.
- [ ] 5.4 Open PR (commit plan below); after merge, comment on #203
      cross-linking to the PR and close it as resolved (same pattern as
      #64/#68).

## 6. Follow-up refactors from PR review (design.md Decisions 2 & 5)

Both are behavior-preserving refactors (no observable output change for any
existing caller) — TDD here means: add a direct unit test for the new
helper, refactor callers to use it, then confirm the *existing* regression
suites for every affected function still pass unchanged (proving byte-for-byte
behavior preservation), plus one new equivalence test proving the specific
divergence this closes is actually closed.

- [ ] 6.1 Write a unit test for a new `_first_index_crossing_threshold(cumulative,
      threshold, total) -> int` helper in `tests/test_pca.py` (exact
      boundary, threshold never reached → `total`, single-element input).
- [ ] 6.2 Implement `_first_index_crossing_threshold()` in `pca.py`; refactor
      `select_n_components()` and `select_n_features_by_variance()` to call
      it (keeping `select_n_features_by_variance()`'s own `threshold <= 0`
      special-case local to that function, since it has no
      `select_n_components()` analog).
- [ ] 6.3 Run the full existing test suites for both functions
      (`TestSelectNComponents`, `TestSelectNFeaturesByVariance` in
      `tests/test_pca.py`) unchanged and confirm all still pass — this is
      the behavior-preservation proof for 6.2.
- [ ] 6.4 Write a failing equivalence test in `tests/test_pca.py`: run
      `perform_pca_analysis()` on real (seeded) data, then assert
      `select_top_features_from_pca(method="top_variance", ...)`'s
      internal per-feature contribution values are *exactly* equal
      (`np.array_equal`, not `np.allclose`) to
      `pca_results["feature_contributions"]["total_contribution"]` — this
      should fail before the fix (different summation order) and pass
      after.
- [ ] 6.5 Implement a new `_total_variance_contribution(loadings,
      eigenvalues, n_features=None)` helper in `pca.py` per design.md
      Decision 5; refactor `perform_pca_analysis()`'s `total_contributions`
      computation and `select_top_features_from_pca()`'s `"top_variance"`
      branch to both call it. Confirm 6.4's test now passes, and the full
      existing `test_pca.py`/`test_visualization.py` suites (covering
      `create_pca_biplot`/`create_umap_colored_by_top_traits`'s
      `"top_variance"` behavior) still pass unchanged.
- [ ] 6.6 Add a named `_WHOLE_NUMBER_TOLERANCE = 1e-9` module-level constant
      in `pipeline/config/utils.py`, replacing the bare `1e-9` literal
      duplicated in both `validate_qc_config()` and `validate_viz_config()`.
- [ ] 6.7 Fix the leftover double-blank-line in
      `configs/active/viz/suyash_arabidopsis_pgm1_pac_2026_05_22.yaml` left
      by the section-4 `n_top_features` line removal (cosmetic; direct fix,
      no test needed).
- [ ] 6.8 Run the full suite, black, ruff, and `openspec validate --strict`
      once more before pushing.

### Suggested commit sequence

1. `feat: add select_n_features_by_variance() PCA helper` — section 1 files.
2. `fix(#203, #206): scope pc_indices to retained PCs and resolve n_top_features per strategy` — all of section 2 (schema + call site + test updates), as one commit per the atomicity note above.
3. `fix: reject fractional/non-integer n_top_features for count-based PCA selection strategies` — section 3.
4. `chore: remove stale n_top_features config lines and comments for extreme PCA selection` — section 4.1–4.3 (line removal, comment rewrites, and the accompanying config-load regression test land together, since the test is what verifies the edits).
5. `docs: changelog entries for PCA feature-selection fixes (#203, #206)` — section 4.4.
6. `refactor: extract shared crossing-threshold and variance-contribution helpers` — section 6.1–6.7, as one commit (the two extractions are independent of each other but each individually needs its helper + both refactored callers + passing regression suite to land atomically).
