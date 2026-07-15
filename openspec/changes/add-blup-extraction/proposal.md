## Why

The QC pipeline fits a linear mixed model (`statsmodels MixedLM` with REML) in
`calculate_heritability_estimates` (`src/sleap_roots_analyze/statistics.py:195-448`)
to estimate broad-sense heritability (H²), but discards the fitted model's random
effects. These random effects are BLUPs (Best Linear Unbiased Predictions) —
shrinkage-adjusted genotype estimates that are more reliable than raw genotype
means, especially for unbalanced designs (the wheat EDPIE field platform has
n=2-3 reps per genotype). `result.random_effects` is a lazy property that is
never accessed today even though the model is already fit.

This is Tier 1 (`add-blup-extraction`, tracking issue
[talmolab/sleap-roots-analyze#109](https://github.com/talmolab/sleap-roots-analyze/issues/109))
of a larger cross-platform genotype prediction program: BLUP-adjusted genotype
means are the predictor substrate that later tiers (ridge/PLS regression with
leave-one-genotype-out cross-validation, not in scope here) will use to test
whether one phenotyping platform's genotype effects predict another's. See the
program roadmap and statistical grounding at
`c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md`
(external to this repo; referenced here for provenance only).

## What Changes

- **Extract BLUPs during the existing model fit.** In
  `calculate_heritability_estimates`, when a trait's mixed model succeeds
  (`model_type == "mixed_model"`), add two keys to that trait's per-trait
  result dict: `blup` (a `dict[genotype, float]` from `result.random_effects`,
  accessed exactly once) and `intercept`
  (`float(result.fe_params["Intercept"])`). Purely additive — both existing
  return shapes (plain dict when `remove_low_h2=False`; 4-tuple when
  `remove_low_h2=True`) are unchanged, no new parameter, no signature change.
  The function has two other success paths that reach the same shared
  per-trait dict literal without ever fitting a mixed model — the ANOVA-based
  path (`force_method="anova_based"`) and the no-variance short-circuit — and
  those traits correctly get no `blup`/`intercept` keys, since there is no
  `result.random_effects` to extract for them.
- **Add `extract_blup_table()`** — a new function in `statistics.py` that
  builds a genotype × trait adjusted-means `pd.DataFrame` from a
  `heritability_results` dict: `adjusted_mean[g, trait] = intercept + blup[g]`
  for succeeded traits; a genuine `NaN` column (not dropped, not zero-filled)
  for any trait whose model failed, used the ANOVA-based/no-variance path, or
  was skipped. Handles the run-level short-circuit form (`{"error": ...}`,
  no per-trait entries) and the zero-succeeded-traits case without raising.
- **Add `BLUPResult`** — a new frozen `@dataclass` in `result_types.py`,
  following the `PCAResult`/`HeritabilityResult`/`UMAPResult` pattern exactly
  (`to_dict()`/`to_json(allow_nan=False)` finite-float contract). Fields:
  `genotype_names`, `trait_names` (succeeded traits only), `adjusted_means`
  (always-finite `list[list[float]]`), `failed_traits` (names only, mirrors
  `HeritabilityResult.failed_traits`), `intercepts`. Built via the classmethod
  `BLUPResult.from_blup_table(df, intercepts=...)`, which drops `NaN` columns
  into `failed_traits` so the dataclass's numeric matrix is always finite even
  though the source CSV/DataFrame carries real `NaN`s. A column needs to be
  *entirely* finite to count as succeeded — even a single cell-level gap (a
  genotype covered by one succeeded trait's model but not another's, since
  each trait's genotype set is computed independently) reclassifies that
  whole trait into `failed_traits`, identically to a trait whose model failed
  outright.
- **Add config flag** `StatisticsConfig.generate_blup_table: bool = True`
  (`pipeline/config/components.py`), gated on `calculate_heritability` also
  being `True`. Setting `generate_blup_table=True` while
  `calculate_heritability=False` does not raise and does not warn — it simply
  produces no BLUP output, since there is no model fit to extract from. (A
  warning mirroring the `umap.enabled` precedent was considered and rejected
  after round-2 review — see `design.md` Decision 5: that precedent defaults
  `False` and gates a rare, deliberate opt-in, whereas `generate_blup_table`
  defaults `True` and the inert combination arises from the ordinary act of
  disabling heritability.)
- **Pipeline output**: `StatisticalAnalysisStep`
  (`pipeline/steps/statistical_analysis.py`) writes `08_blup_adjusted_means.csv`
  to `run_dir/data/` alongside `08_heritability_results.csv` when both config
  flags are enabled. `StatisticalAnalysisStep` runs in **both** the QC
  pipeline (`qc_pipeline.py`) and the Viz pipeline (`viz_pipeline.py`), but
  only `VizPipelineConfig` composes `StatisticsConfig` — `QCPipelineConfig` has
  no `statistics` field at all. The step already resolves this for
  `calculate_heritability` via `getattr(config, "statistics", None)`
  (defaulting to `True` when absent, at `statistical_analysis.py:154-159`);
  `generate_blup_table` must be resolved with the same guard, so a QC-pipeline
  run always gets the default for both flags and cannot configure
  `generate_blup_table` any other way. The CSV write itself belongs in the
  step's *second* `if calculate_heritability:` block (the one that already
  writes `08_heritability_results.csv`, after `data_dir`/`files` exist), not
  the earlier block that only computes `heritability_results`.
- **Public exports**: `extract_blup_table` and `BLUPResult` are exported from
  the package root and `__all__`, per the existing statistics-api /
  serializable-result-types conventions.

No breaking changes — every change is additive (new dict keys, a new
function, a new dataclass, a new config field defaulting to the
already-enabled behavior, a new output file). Existing callers of
`calculate_heritability_estimates`, `StatisticalAnalysisStep`, and
`StatisticsConfig` continue to work unchanged.

## Design decisions (resolved via brainstorming this session — full rationale and alternatives considered in `design.md`)

- `BLUPResult` dataclass, not BLUPs folded into the existing heritability
  dict — the roadmap's flagged blocking question for Tier 1, resolved by
  reading `result_types.py` first.
- `blup`/`intercept` are added unconditionally (no new opt-in parameter) —
  the extraction is free and purely additive, but only for traits whose fit
  is `model_type == "mixed_model"`.
- Failed-trait handling splits by artifact: the CSV keeps a genuine `NaN`
  column; `BLUPResult` excludes failed columns and lists their names in
  `failed_traits`.
- Module placement matches the existing analytical/result-type split:
  `extract_blup_table()` in `statistics.py`, `BLUPResult` in
  `result_types.py`.
- Config flag on `StatisticsConfig` (not a new `BLUPConfig`, not on
  `HeritabilityConfig`), default `True`, silent no-op (documented in its
  docstring) when it's inert — a warning was tried and reversed, see
  `design.md` Decision 5.
- **Tier 2 continuity note (non-blocking for Tier 1)**: the roadmap's open
  question about `R_j` (field block/replicate) will be answered in Tier 2
  (`add-heritability-fixed-effects`, #114) as a **fixed effect** via a new
  `fixed_effects` parameter (`value ~ <fixed_effects> + (1|genotype)`), not a
  second random effect — no re-architecture of Tier 1's BLUP extraction is
  needed.

## Impact

### Affected specs

- `statistics-api` (MODIFIED) — `extract_blup_table` added to the public
  surface (9 functions total); `calculate_heritability_estimates`'s additive
  `blup`/`intercept` keys documented as non-breaking.
- `serializable-result-types` (ADDED + MODIFIED) — `BLUPResult` type, adapter,
  and public export (ADDED); `calculate_heritability_estimates`'s additive
  `blup`/`intercept` keys folded into the existing "Non-Breaking Heritability
  Return Shape" requirement (MODIFIED).
- `config-management` (ADDED) — `StatisticsConfig.generate_blup_table` gating
  behavior.

### Affected code

- `src/sleap_roots_analyze/statistics.py` — extract BLUPs in
  `calculate_heritability_estimates`; add `extract_blup_table()`
- `src/sleap_roots_analyze/result_types.py` — add `BLUPResult`
- `src/sleap_roots_analyze/pipeline/config/components.py` — add
  `StatisticsConfig.generate_blup_table`
- `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` — write
  `08_blup_adjusted_means.csv`
- `src/sleap_roots_analyze/__init__.py` — root exports + `__all__`
- `docs/API.md`, `docs/result-types.md`, `docs/CHANGELOG.md`,
  `docs/QC_PIPELINE_GUIDE.md` (Analysis Files output list)
- `tests/test_statistics.py`, `tests/test_blup_result.py` (new, mirrors
  `tests/test_heritability_result.py`), `tests/fixtures.py` (new unbalanced-reps
  fixture), `tests/test_step_statistical_analysis.py`, `tests/test_public_api.py`

### Explicitly out of scope

- Tier 2 (`add-heritability-fixed-effects`, #114): fixed-effects param, R_j
  handling.
- Tier 3+ (LOGO-CV, prediction pipeline, permutation nulls, figures): separate
  future changes.
- Any change to `calculate_heritability_estimates`'s H² formula or existing
  variance-component behavior.
