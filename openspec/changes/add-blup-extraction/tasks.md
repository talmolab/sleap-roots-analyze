> Test fixtures: reuse `heritability_data_known_h2` (`tests/fixtures.py:182`, balanced —
> 20 genotypes × 5 reps) for the balanced-BLUP≈raw-mean oracle. Add one new fixture,
> `heritability_data_unbalanced_reps`, for the shrinkage oracle (task 1.1). Section 1/2
> tests import `from sleap_roots_analyze.statistics import calculate_heritability_estimates,
> extract_blup_table` and `from sleap_roots_analyze.result_types import BLUPResult`
> directly (the package-root export doesn't land until §5).

## 1. Fixtures + `calculate_heritability_estimates` BLUP extraction (test-first)

- [x] 1.1 Add `heritability_data_unbalanced_reps` fixture to `tests/fixtures.py`
      (mirror `heritability_data_known_h2`'s structure and column names —
      `geno`/`rep`/`Barcode`, seed 42): a single trait `trait_unbalanced` with a
      known genetic variance σ²_G=4.0 and residual variance σ²_E=1.0
      (genotype effect `np.random.normal(0, 2.0)`, per-rep noise
      `np.random.normal(0, 1.0)`, base value 50). 20 genotypes total: 10
      "low-rep" genotypes (`G01`-`G10`) with **n=2** reps each, 10 "high-rep"
      genotypes (`G11`-`G20`) with **n=20** reps each. Return `(df, meta)` where
      `meta = {"low_rep_genotypes": [...10 ids...], "high_rep_genotypes":
      [...10 ids...], "trait": "trait_unbalanced"}` — the test computes raw
      means and the grand mean itself from `df` rather than the fixture
      pre-computing them, so there's one source of truth.
- [x] 1.2 Write failing test `test_blup_extracted_for_successful_trait`
      (`tests/test_statistics.py`, new `TestBLUPExtraction` class): call
      `calculate_heritability_estimates` on `heritability_data_known_h2`; assert
      each successful (`model_type == "mixed_model"`) trait's per-trait dict
      contains `blup` (a `dict` keyed by every genotype label present in the
      input) and `intercept` (a `float`); assert `type(intercept) is float` and
      every `blup` value `is float` (guard against a leaked
      `np.float64`/pandas scalar).
- [x] 1.3 Write failing test `test_existing_return_shape_unchanged`: with
      `remove_low_h2=False` (default), the return value is still a plain `dict`
      containing the pre-existing per-trait keys (`heritability`, `var_genetic`,
      `var_residual`, `mean_n_reps`, `n_genotypes`, `n_observations`,
      `model_type`, `reps_per_geno_stats`) unchanged, with `blup`/`intercept`
      added alongside; with `remove_low_h2=True`, the return value is still the
      4-tuple `(heritability_results, df_filtered, removed_traits,
      removal_details)` and `heritability_results`'s per-trait dicts carry the
      same additive keys — including for a trait present in `removed_traits`
      (heritability-based removal filters `df_filtered`/`removed_traits`, it
      does not strip keys from `heritability_results`).
- [x] 1.4a Write failing test `test_single_genotype_trait_has_no_blup_keys`:
      a trait with fewer than 2 genotypes (`len(reps_per_geno) < 2`, the
      existing `"Insufficient genotypes..."` error path) has no `blup`/
      `intercept` key.
- [x] 1.4b Write failing test `test_mixed_model_fit_failure_has_no_blup_keys`:
      mock `statsmodels.formula.api.mixedlm` (or construct data that reliably
      fails to converge) so the mixed-model `try` block raises; assert that
      trait's dict has `model_type == "mixed_model_failed"` and no `blup`/
      `intercept` key. (Distinct code path from 1.4a — do not conflate the two
      under one test.)
- [x] 1.4c Write failing test
      `test_anova_based_and_no_variance_traits_have_no_blup_keys_no_crash`:
      call `calculate_heritability_estimates(..., force_method="anova_based")`
      on `heritability_data_known_h2` and separately on a fixture with
      `nunique() == 1` (reuse or adapt `heritability_zero_data`-style constant
      values); assert neither call raises, and both traits' dicts have
      `model_type` in `{"anova_based", "no_variance"}` with no `blup`/
      `intercept` key. This guards the real bug risk that the shared per-trait
      dict literal (`statistics.py:414-428`) is reached by the ANOVA-based
      branch too, which never fits a mixedlm model and has no `result` object.
- [x] 1.5 Write failing test `test_adjusted_mean_matches_independent_raw_mean`
      (an independent oracle, not tautological — do NOT compare
      `intercept + blup[g]` against itself): for a known-H2 balanced trait,
      compute `intercept + blup[g]` from `calculate_heritability_estimates`'s
      own returned dict and assert it is within a documented tolerance of
      `df.groupby("geno")[trait].mean()` computed directly from the input
      DataFrame — i.e. the same comparison 2.5 will later run against
      `extract_blup_table`'s output, but exercised here directly against the
      dict's `blup`/`intercept` fields so this section's tests don't depend on
      §2 existing yet.
- [x] 1.6 Implement the extraction in `calculate_heritability_estimates`
      (`statistics.py:338-372`, the `if use_mixed_model:` branch only): inside
      the inner `try` block, immediately after
      `result = model.fit(reml=True)`, compute
      `blup = {str(g): float(v.iloc[0]) for g, v in result.random_effects.items()}`
      and `intercept = float(result.fe_params["Intercept"])` as local
      variables (access `result.random_effects` exactly once). Initialize
      `blup = None` and `intercept = None` before the `if use_mixed_model:`/
      `else:` branch (so the ANOVA-based branch leaves them `None`), and in the
      shared dict literal (~line 414), add the keys conditionally: build the
      dict without `blup`/`intercept` first, then
      `if blup is not None: heritability_results[trait]["blup"] = blup;
      heritability_results[trait]["intercept"] = intercept`. This ensures the
      ANOVA-based branch (no `result` object) cannot reference an unbound
      variable. Make 1.2–1.5 green (1.4a/1.4b/1.4c should already be green from
      existing behavior — confirm, don't newly break them). Update the
      function's `Returns:` docstring to document the new `blup`/`intercept`
      keys and the mixed-model-only condition.
- [x] 1.7 Write a failing backward-compatibility regression test in
      `tests/test_heritability_result.py` (the existing
      `TestHeritabilityResultNonBreaking` class): build a hand-crafted
      per-trait dict containing the pre-existing keys **plus** the new
      `blup`/`intercept` keys, pass it through
      `HeritabilityResult.from_heritability_dict`, and assert the resulting
      `TraitHeritability` fields are unaffected by the extra keys — confirms
      the "purely additive" claim from this same commit rather than assuming
      it from the adapter's use of `.get()`. No implementation change needed
      (`from_heritability_dict` already reads via `.get(...)` on a fixed key
      set); this should already be green once 1.6 lands — if it isn't,
      that's a real regression to fix before moving on.

## 2. `extract_blup_table()` (test-first)

- [x] 2.1 Write failing test `test_extract_blup_table_success_values`
      (`tests/test_statistics.py`): build a minimal `heritability_results` dict
      by hand (2 traits succeeded with known `blup`/`intercept`, one trait
      failed — no `blup` key), call `extract_blup_table(heritability_results)`,
      assert the returned `pd.DataFrame` has `adjusted_mean = intercept +
      blup[g]` for every genotype/succeeded-trait cell (`pytest.approx`).
- [x] 2.2 Write failing test `test_extract_blup_table_failed_trait_is_nan_column`:
      the failed trait's column is present, every value is `NaN`
      (`df["trait_failed"].isna().all()`), and no value is `0.0`.
- [x] 2.3 Write failing test `test_extract_blup_table_shape`: rows = union of
      genotypes across traits, columns = `trait_cols` order (excluding
      `__calculation_metadata__`); row/column counts match a hand-built fixture.
- [x] 2.4 Write failing test `test_extract_blup_table_does_not_mutate_input`: deep-
      copy `heritability_results` before the call, assert equality after (mirror
      the `_assert_dict_unchanged` pattern from `tests/test_umap_result.py` /
      `tests/test_cluster_result.py`).
- [x] 2.4a Write failing test `test_extract_blup_table_run_level_error_dict`:
      call `extract_blup_table({"error": "Missing required columns: ['geno']"})`
      (the run-level short-circuit form with no per-trait entries); assert no
      exception is raised and the returned `pd.DataFrame` is empty (0 rows, 0
      columns).
- [x] 2.4b Write failing test `test_extract_blup_table_all_traits_failed`: build
      a `heritability_results` dict where every trait has no `blup` key (e.g.
      all `{"error": ...}` or `model_type in {"anova_based", "no_variance",
      "mixed_model_failed"}`); assert no exception is raised, the returned
      `pd.DataFrame` has zero rows and one column per input trait, and every
      column is entirely `NaN` (this is the empty-genotype-universe case —
      there is no `blup` dict anywhere to source a genotype index from).
- [x] 2.4c Write failing test `test_extract_blup_table_cell_level_nan_for_partial_genotype_coverage`:
      build a `heritability_results` dict with two succeeded traits whose
      `blup` dicts cover different, overlapping-but-not-identical genotype
      sets (e.g. `"trait_a"`'s `blup` has `{"G01", "G02"}`, `"trait_b"`'s has
      `{"G01", "G02", "G03"}`); assert the returned DataFrame has a `"G03"`
      row, `df.loc["G03", "trait_a"]` is `NaN`, and `df.loc["G03", "trait_b"]`
      is a finite value — a cell-level gap, distinct from 2.2's whole-column
      failure.
- [x] 2.5 Write failing integration test `test_blup_table_balanced_matches_raw_mean`
      using `heritability_data_known_h2`: run `calculate_heritability_estimates`
      then `extract_blup_table`; for each genotype/trait, assert the
      BLUP-adjusted mean is within tolerance of that genotype's raw trait mean
      (`df.groupby("geno")[trait].mean()`) — the balanced-design oracle. Use a
      **per-trait** tolerance, not one shared constant: empirically (verified
      against this exact fixture, seed 42) `max|adjusted_mean - raw_mean|` is
      ≈0.13 for `trait_high_h2`, ≈0.31 for `trait_moderate_h2`, and ≈0.37 for
      `trait_low_h2` — so use `atol=0.3` for `trait_high_h2` and `atol=0.5`
      for the other two, not a single flat `atol=0.5` shared across all three
      traits with no margin check. (A noise-scaled formula such as
      `2 * sqrt(var_residual / mean_n_reps)` was considered as a general
      alternative to hardcoded per-trait constants, but rejected: on this
      fixture it evaluates to ≈0.89 for every trait — looser than the
      empirical maxima above by 2-7x and insensitive to `var_genetic`, so it
      would mask exactly the kind of shrinkage-formula regression this test
      exists to catch. Use the concrete per-trait values.)
- [x] 2.6 Write failing integration test
      `test_blup_table_unbalanced_shrinks_low_rep_genotypes` using
      `heritability_data_unbalanced_reps` (task 1.1): the "grand mean" is the
      model's own `intercept` (not the naive `df[trait].mean()` — those
      differ slightly under an unbalanced design), and the comparison is the
      per-genotype **shrinkage ratio** `|adjusted_mean - intercept| /
      |raw_mean - intercept|`, not raw gap magnitude — both genotype groups
      draw their true effect from the same distribution, so a high-rep
      genotype can legitimately land a larger raw/adjusted gap than any
      low-rep genotype by chance; the ratio isolates the shrinkage factor
      itself (theory.md: `lambda = var_genetic / (var_genetic +
      var_residual / n_reps)`), which is smaller for n=2 than n=20
      regardless of which genotype drew the larger true effect. Assert (a)
      every genotype's adjusted gap is smaller than its raw gap, and (b) the
      mean shrinkage ratio for low-rep genotypes is smaller than for
      high-rep genotypes — the unbalanced-design oracle.
- [x] 2.7 Implement `extract_blup_table(heritability_results: dict) -> pd.DataFrame`
      in `statistics.py` (near `calculate_heritability_estimates`):
      - If `heritability_results` is the run-level short-circuit form (has an
        `"error"` key with a string value and no per-trait entries), return an
        empty `pd.DataFrame()` immediately.
      - Otherwise, collect the genotype universe as the union of every
        succeeded trait's `blup.keys()` (an empty union — zero succeeded
        traits — yields zero rows).
      - Build one column per trait (excluding `__calculation_metadata__`):
        `intercept + blup[g]` where the trait succeeded **and** `g` is a key
        in that trait's `blup` dict; `np.nan` otherwise (covers failed
        traits, ANOVA-based/no-variance traits, and genotypes absent from a
        succeeded trait's own `blup` dict).
      - Return a DataFrame indexed by genotype (the collected union, possibly
        empty), columns in trait-cols order.
      Full Google-style docstring (Args/Returns), documenting the run-level
      short-circuit, all-failed, and cell-level-NaN behaviors. Make 2.1–2.4c
      and 2.5–2.6 green.

## 3. `BLUPResult` + adapter (test-first)

- [x] 3.1 Create `tests/test_blup_result.py` (mirror `tests/test_heritability_result.py`
      / `tests/test_umap_result.py`). Write failing test
      `test_json_roundtrip_native_types`: build a `BLUPResult` via
      `BLUPResult.from_blup_table(df, intercepts=...)` from a hand-built
      `extract_blup_table()`-shaped DataFrame with one all-NaN (failed) column;
      `json.dumps(dataclasses.asdict(result))` succeeds; parsed values are native
      `str`/`float`.
- [x] 3.2 Write failing test `test_fields_are_native_types_pre_serialization`
      (assert on dataclass fields directly, pre-JSON, mirroring the sibling
      comment that JSON round-trips hide an `np.float64` leak): every
      `genotype_names`/`trait_names`/`failed_traits` element `is str`; every
      `adjusted_means` element `is float`; every `intercepts` value `is float`.
- [x] 3.3 Write failing test `test_failed_trait_excluded_from_matrix_not_nan`:
      the failed (all-NaN) column's name is in `failed_traits`, NOT in
      `trait_names`; no element of `adjusted_means` is `NaN`/`Infinity`;
      `to_json()` succeeds without raising.
- [x] 3.3a Write failing test `test_cell_level_nan_column_classified_as_failed`:
      from a DataFrame built like 2.4c's (one trait with a single `NaN` cell,
      otherwise finite), assert that trait's name is in `failed_traits`, not
      `trait_names` — a partially-finite column is not eligible for the
      always-finite matrix.
- [x] 3.3b Write failing test `test_zero_succeeded_traits_not_misclassified`:
      from a zero-row DataFrame (all columns entirely `NaN`, per 2.4b's
      shape), assert `genotype_names == []`, `trait_names == []`,
      `failed_traits` contains every input column name, and `to_json()`
      succeeds. This specifically guards against a naive `df[col].notna().all()`
      partition, which is vacuously `True` on a zero-row column in pandas.
- [x] 3.4 Write failing test `test_to_json_rejects_non_finite_adjusted_mean`
      (mirror `test_to_json_rejects_non_finite_h2`): construct a `BLUPResult`
      directly (bypassing the adapter) with a `NaN`/`Infinity` in
      `adjusted_means`; `to_json()` raises `ValueError`; `to_dict()` does not.
- [x] 3.5 Write failing adapter tests: `genotype_names == [str(g) for g in
      df.index]` (row order preserved); `trait_names`/`failed_traits` partition
      `df.columns` by column-finiteness, in original column order;
      `adjusted_means` shape is `(len(genotype_names), len(trait_names))`;
      `intercepts` has exactly one entry per `trait_names` name and none for
      `failed_traits` names; the input `df` is unchanged after the call
      (`_assert_dict_unchanged`-style deep-copy compare, adapted for a DataFrame).
- [x] 3.6 Add `BLUPResult(frozen=True)` to `result_types.py` with fields
      `genotype_names: list[str]`, `trait_names: list[str]`,
      `adjusted_means: list[list[float]]`, `failed_traits: list[str] =
      field(default_factory=list)`, `intercepts: dict[str, float] =
      field(default_factory=dict)`; `to_dict()`/`to_json()` (the
      `allow_nan=False` finite-floats contract, copied from `HeritabilityResult`).
      Google-style docstring with a complete `Attributes:` block (the
      `check_public_api_docs` audit requires every field name to appear) and the
      shallow-`frozen=True` read-only caveat. Append `"BLUPResult"` to
      `result_types.__all__`.
- [x] 3.7 Add `BLUPResult.from_blup_table(df, *, intercepts=None)` classmethod.
      Partition `df.columns`: a column is **succeeded** only if it has at
      least one row AND `df[col].notna().all()` — explicitly treat a
      zero-row DataFrame's columns as **failed**, not succeeded (do not rely
      on `.notna().all()` alone, since it is vacuously `True` on an empty
      column: `pd.Series([], dtype=float).notna().all() is True`). Build
      `adjusted_means` from `df[trait_names].to_numpy().tolist()`;
      `intercepts` defaults to `{}` if not supplied, else filtered to
      `trait_names` only. Non-mutating. Make 3.1–3.5 (incl. 3.3a/3.3b) green.

## 4. Config + pipeline output (test-first)

- [x] 4.1 Write failing test `test_generate_blup_table_default_true`
      (`tests/test_step_statistical_analysis.py` — no dedicated config-defaults
      test file exists in this repo; a small standalone assertion here is the
      right home): `StatisticsConfig().generate_blup_table is True`.
- [x] 4.2 Add `generate_blup_table: bool = True` to `StatisticsConfig`
      (`pipeline/config/components.py:524`), with a docstring `Attributes:` entry
      explaining the `calculate_heritability` gating (mirror the
      `HeritabilityConfig.generate_diagnostics` docstring style), and noting
      that setting it `True` while `calculate_heritability=False` is inert —
      no exception, no warning, just no BLUP output, since there's no model
      fit to extract from. Make 4.1 green.
- [x] 4.3 Write failing test `test_blup_csv_written_when_both_enabled`
      (`tests/test_step_statistical_analysis.py`, mirror
      `test_heritability_file_generated`): run `StatisticalAnalysisStep.execute()`
      with default `VizPipelineConfig`-style config (`calculate_heritability=True`,
      `generate_blup_table=True`); assert `data/08_blup_adjusted_means.csv` exists
      under `tmp_path`, and its row count equals `sample_data["Genotype"].nunique()`.
- [x] 4.4 Write failing test `test_blup_csv_absent_when_generate_blup_table_false`:
      same step, `generate_blup_table=False`; assert the CSV is absent while
      `08_heritability_results.csv` is still present.
- [x] 4.5 Write failing test `test_blup_csv_absent_when_heritability_disabled`:
      `calculate_heritability=False`, `generate_blup_table=True` (the
      default); assert the CSV is absent and no exception is raised. This is
      an ordinary, legitimate configuration (heritability turned off
      entirely) — assert no warning is raised either (e.g. via
      `warnings.catch_warnings(record=True)` and an empty list), since a
      warning here would be noise on a common config, not a useful signal
      (see design.md Decision 5 for why this reverses an earlier warn-based
      draft).
- [x] 4.5a Write failing test `test_blup_table_works_with_qc_config_no_statistics`
      (mirror the existing `test_heritability_works_with_qc_config_no_statistics`
      at `tests/test_step_statistical_analysis.py:469`): run
      `StatisticalAnalysisStep.execute()` with a bare `QCPipelineConfig` (no
      `statistics=` field set at all); assert no `AttributeError` is raised and
      `08_blup_adjusted_means.csv` is written (both flags resolve to their
      defaults via the same `getattr(config, "statistics", None)` fallback
      `calculate_heritability` already uses). This is the regression guard for
      the QC-pipeline crash risk found in review.
- [x] 4.6 Implement the write path in `StatisticalAnalysisStep.execute()`
      (`pipeline/steps/statistical_analysis.py`):
      - Resolve `generate_blup_table` with the same `getattr(config,
        "statistics", None)` guard already used for `calculate_heritability`
        (`statistical_analysis.py:154-159`), defaulting to `True` when
        `config.statistics` is absent (the QC-pipeline case).
      - If `calculate_heritability` is `True` and `generate_blup_table` is
        `True`: call `extract_blup_table(heritability_results)` and
        `self.save_dataframe(blup_df, "08_blup_adjusted_means.csv", data_dir)`,
        appending to `files`. Add this in the **second** `if
        calculate_heritability:` block (~line 303-308, the one that writes
        `08_heritability_results.csv`) — not the earlier block (~line 161)
        that only computes `heritability_results`, since `data_dir`/`files`
        don't exist yet at that point.
      - If `calculate_heritability` is `False`, do nothing further regardless
        of `generate_blup_table` — no exception, no warning. (An earlier
        draft added a warning here mirroring the `umap.enabled` precedent;
        reversed after round-2 review found it would fire on the ordinary,
        common configuration of disabling heritability — see design.md
        Decision 5.) Do **not** touch `validate_viz_config()` for this
        change.
      Make 4.3–4.5a green.

## 5. Public exports + docs

- [x] 5.1 Write failing test extending the statistics-api export test in
      `tests/test_public_api.py` (confirmed existing eight-function import/
      `__all__` check, `STATISTICS_FUNCTIONS` list): the import list grows to
      nine functions including `extract_blup_table`; `__all__` contains it with
      no duplicates; `typing.get_type_hints` and the docstring-completeness
      check (`TestDocsInSync`) both cover it like the other eight.
- [x] 5.2 Write failing test `test_blupresult_importable_from_root` in
      `tests/test_blup_result.py` (class `TestBLUPResultExport`, mirroring
      `TestHeritabilityResultExport`/`TestUMAPResultExport`): `from
      sleap_roots_analyze import BLUPResult` succeeds; `"BLUPResult"` in
      `sleap_roots_analyze.__all__` with no duplicates; importable from
      `sleap_roots_analyze.result_types` and listed in `result_types.__all__`.
- [x] 5.3 Add `extract_blup_table` to the statistics import/`__all__` block in
      `__init__.py` (alongside the existing eight); add `BLUPResult` to the
      `result_types` import block and `__all__` (the "Serializable result types
      (#130)" group). Make 5.1–5.2 green. (This is the commit where the public-
      API docstring audit fires — 3.6's complete `Attributes:` block and the
      `extract_blup_table` docstring must already be in place.)
- [x] 5.4 Update the `result_types.py` module docstring enumeration (the
      "`PCAResult` … `HeritabilityResult` (#128), `ClusterResult` (#129), and
      `UMAPResult` (#180) follow" sentence) to include `BLUPResult (#109)` —
      citing its issue number, matching every other entry's per-type citation.
- [x] 5.5 Update `docs/result-types.md`: add the BLUP row to the types table
      (Built from `extract_blup_table` DataFrame; adapter
      `BLUPResult.from_blup_table(df, *, intercepts=None)`).
- [x] 5.6 Update `docs/API.md` `## statistics Module` section: add an entry for
      `extract_blup_table` matching its final signature/defaults.
- [x] 5.6a Update `docs/QC_PIPELINE_GUIDE.md`'s "Analysis Files" list (~line
      146, alongside the existing `08_heritability_results.csv` entry): add
      `` `08_blup_adjusted_means.csv` - BLUP-adjusted genotype means per trait
      (when `generate_blup_table` and `calculate_heritability` are both
      enabled) ``.
- [x] 5.7 Add a `docs/CHANGELOG.md` `[Unreleased]` entry, split like the
      `UMAPResult` entry it mirrors (which separates a `### Changed` bullet
      for additive keys on an *existing* function from `### Added` for
      brand-new symbols — do not lump both under `### Added`):
      - `### Added`: `extract_blup_table`, `BLUPResult`,
        `BLUPResult.from_blup_table`, `StatisticsConfig.generate_blup_table`,
        and the new `08_blup_adjusted_means.csv` pipeline output.
      - `### Changed`: `calculate_heritability_estimates` additively returns
        `blup`/`intercept` keys per trait when its mixed model succeeds.
      Add a one-line rationale (mirroring the `UMAPResult` entry's style):
      these are Tier 1 of the cross-platform genotype-prediction program
      (#109). Do **not** bump `pyproject.toml` or add a dated version
      heading — per the established convention (the `UMAPResult` entry is
      still sitting in `[Unreleased]` with no version cut since), this
      bundles into a future release PR, not this one.

## 6. Validation

- [x] 6.1 `openspec validate add-blup-extraction --strict` — resolve every
      reported issue before requesting review.
- [x] 6.2 `/lint` (black + ruff) on all changed files.
- [x] 6.3 Full `uv run pytest --cov --cov-branch` — confirm no regressions in
      existing `test_statistics.py`, `test_step_statistical_analysis.py`
      (including the pre-existing `test_heritability_works_with_qc_config_no_statistics`
      and `test_heritability_calculated_when_enabled`, both of which now
      exercise the new BLUP code paths for the first time),
      `test_result_serialization.py` (should still skip
      `calculate_heritability_estimates` — its return type is still a dict),
      and confirm the new `tests/test_blup_result.py` and extended
      `tests/test_public_api.py` suites pass.
- [x] 6.4 `/review-openspec` — adversarial proposal review (≥1 round) before
      requesting user approval, per the roadmap's per-tier loop.
