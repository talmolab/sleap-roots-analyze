## Why

`calculate_heritability_estimates` (`src/sleap_roots_analyze/statistics.py:195-479`)
fits a genotype-only mixed model, `smf.mixedlm("value ~ 1", model_data,
groups=genotype)`. There is no way to add fixed-effect covariates (experiment,
wave, batch, scanner). For multi-experiment GWAS panels where genotypes are not
balanced across experiments, any systematic per-experiment shift (scanner
calibration drift, cylinder placement, image pre-processing batch differences)
gets absorbed into the genotype term and inflates apparent H². Mauricio
Chiurazzi's combined alfalfa GWAS (35 accessions × 6 Bloom experiments) shows
this concretely: `Scanline First Ind P25` reports H² ≈ 0.83 despite
within-genotype variance dominating (naive ICC 0.087–0.318), because genotypes
are not balanced across the 6 experiments and there is no way to add a fixed
covariate to absorb the batch shift.

This is Tier 2 (`add-heritability-fixed-effects`, tracking issue
[talmolab/sleap-roots-analyze#114](https://github.com/talmolab/sleap-roots-analyze/issues/114))
of the wheat EDPIE cross-platform genotype-prediction program, unblocked by
Tier 1 (`add-blup-extraction`, merged as #189/#190). See the program roadmap
and statistical grounding at
`c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md`
(external to this repo; referenced here for provenance only).

## What Changes

- **Add `fixed_effects` parameter to `calculate_heritability_estimates`.**
  `fixed_effects: Optional[List[str]] = None`. When provided, the mixed-model
  formula becomes `"value ~ " + " + ".join(f"C({fe})" for fe in
  fixed_effects)` instead of `"value ~ 1"` — every fixed effect is wrapped in
  `C()` unconditionally (always treated as categorical, no dtype inference),
  since fixed effects in this context are metadata-style confounders
  (experiment, wave, batch, scanner), not continuous covariates. Missing
  `fixed_effects` columns extend the existing top-level `required_cols`
  validation (`{"error": "Missing required columns: [...]"}`), alongside
  `genotype_col`/`replicate_col`. The per-trait model subset becomes `df[[trait,
  genotype_col] + fixed_effects].dropna()` (today: `df[[trait,
  genotype_col]].dropna()`) — only changes behavior when `fixed_effects` is
  set; `None` (the default) is byte-for-byte identical to current behavior.
  Model-fit failures raised by `statsmodels` (non-convergence, a fixed effect
  fully confounded with genotype, etc.) are caught by the existing per-trait
  `try/except`, recorded as `{"error": "Mixed model failed: ...",
  "model_type": "mixed_model_failed"}`. Additionally, since `MixedLM.fit()`
  does not reliably *raise* on a fixed effect that is (near-)fully confounded
  with genotype — it can instead emit a `ConvergenceWarning` and still return
  a plausible-looking `result` — the fit is wrapped in
  `warnings.catch_warnings(record=True)` with `warnings.simplefilter("always")`
  (required so a repeat identical warning for a later trait isn't silently
  dropped by Python's default once-per-location filter), and a warning
  matching `ConvergenceWarning`'s category (checked via `issubclass`, not by
  matching message text — several real convergence-related messages don't
  contain the word "convergence") is treated as a fit failure for that trait,
  surfaced the same way as a raised exception. Without this, the exact
  batch-confounded scenario this tier
  targets could silently produce a plausible-but-degenerate fit with no error
  signal — reproducing, via over-fitting/aliasing, the same class of
  silently-wrong-H² problem this tier exists to fix.
- **Empirical frequency-weighted intercept when fixed effects are present.**
  With fixed effects in the formula, `result.fe_params` holds `Intercept` plus
  one treatment-coded offset per non-reference level per fixed effect —
  `Intercept` alone represents "value at patsy's reference level" (the first
  level in sorted order for a plain column, or the first declared category
  for a pandas `Categorical` — not reliably "alphabetical"), not a
  panel-typical value. `intercept` is instead computed as a **sample
  frequency-weighted average** across each fixed effect's levels *as observed
  in that trait's own post-`dropna()` fitted rows* (weighting each level's
  fitted contribution by its share of `model_data`), summed with the base
  `Intercept`. This is deliberately described as an empirical/sample-margin
  quantity, not a population-typical or EMM/lsmeans-style value: it is
  sensitive to each trait's own missing-data pattern (two traits sharing the
  same `fixed_effects` columns can get different weights if their `dropna()`
  drops different rows) and to incidental sample-size imbalance across levels
  (e.g. an experiment with more scans purely for logistical reasons pulls the
  intercept toward it, regardless of biological representativeness). The
  coefficient lookup for each level is done by parsing `result.fe_params`'s
  actual fitted parameter names (regex `^C\({fe}\)\[T\.(.*)\]$`), not by
  reconstructing the expected key string forward from the observed level
  value — the latter risks a silent dtype/formatting mismatch (e.g. a
  `float64` fixed-effect column) being misattributed to the reference level's
  implicit `0.0` contribution with no error. When `fixed_effects` is `None`,
  this collapses to today's plain `result.fe_params["Intercept"]` — no
  behavior change for existing callers. This value flows unchanged into
  `extract_blup_table()`'s existing `adjusted_mean = intercept + blup[g]`
  formula and into `BLUPResult.intercepts` (no code changes needed in either —
  both already just consume whatever `intercept` float
  `calculate_heritability_estimates` produces; `BLUPResult`'s docstring is
  updated to describe the new semantics of a value it already stores
  verbatim).
- **`replicate_col` is untouched, and independent of `fixed_effects`.** A
  block/replicate fixed effect (R_j, confirming Tier 1's tentative design
  note) is expressed by passing that column's name into `fixed_effects` (e.g.
  `fixed_effects=["block"]`) — no coupling between the two parameters, no new
  validation linking them.
- **Add `StatisticsConfig.fixed_effects: Optional[List[str]] = None`**
  (`pipeline/config/components.py`) — not `HeritabilityConfig`, which gates
  the later low-H² filtering step (`FilterHeritabilityStep`) and has no
  relationship to how the model itself is fit. This deviates from issue
  #114's illustrative YAML sketch (`heritability.fixed_effects`), which
  predates inspection of how the config is actually wired;
  `StatisticalAnalysisStep` is the only caller of
  `calculate_heritability_estimates`, and it already reads
  `config.statistics`. `StatisticalAnalysisStep` resolves `fixed_effects` via
  the same `getattr(config, "statistics", None)` fallback already used for
  `calculate_heritability`/`generate_blup_table`, defaulting to `None` when
  `config.statistics` is absent (the QC-pipeline case).
- **Documentation note** (per issue #114's acceptance criteria): `fixed_effects`
  should only contain metadata-style covariates that confound with genotype
  (experiment, wave, batch, scanner), not biological/phenotypic traits — stated
  in the docstring, not runtime-enforced (unenforceable semantically).

No breaking changes — `fixed_effects=None` (default) reproduces today's
formula, subset, and intercept computation exactly. Existing callers of
`calculate_heritability_estimates`, `StatisticalAnalysisStep`, and
`StatisticsConfig` continue to work unchanged.

## Design decisions (resolved via brainstorming this session — full rationale and alternatives in `design.md`)

- Config lives on `StatisticsConfig`, not `HeritabilityConfig` — matches where
  the calculation actually happens, deviating deliberately from issue #114's
  YAML sketch.
- Every `fixed_effects` column is wrapped in `C()` unconditionally — no dtype
  inference, since a numeric-looking metadata column (e.g. `wave_number`)
  must still be treated as categorical, not a continuous covariate.
- The BLUP-adjusted-mean intercept becomes an empirical, sample
  frequency-weighted average across each trait's own observed fixed-effect
  levels, rather than pinned to patsy's arbitrary reference level — cheap to
  compute (a weighted sum over coefficients `fe_params` already contains),
  and removes the "value under an arbitrary baseline category" caveat for
  the EDPIE paper's supplementary tables. Explicitly documented as a
  sample-margin quantity (sensitive to each trait's own missing-data pattern
  and to incidental level-frequency imbalance), not a population-typical or
  EMM/lsmeans-style value — the distinction matters if anyone later reads
  `adjusted_mean` as "this genotype's typical phenotype."
- The per-level coefficient lookup for the marginal intercept parses
  `result.fe_params`'s actual fitted parameter names rather than
  reconstructing the expected key string forward from raw level values — the
  latter risks silently misattributing a real, non-reference level to the
  reference level's implicit `0.0` on a dtype/formatting mismatch (revised
  after review found the forward-reconstruction approach could fail silently
  on a `float64`-dtype fixed-effect column).
- `fixed_effects` and `replicate_col` are fully independent parameters — `R_j`
  (field block/replicate as a fixed effect) is expressed by naming that column
  in `fixed_effects`; no special-casing added.
- Model-fit failures from fixed effects (non-convergence, confounding with
  genotype) reuse the existing per-trait `try/except`, extended to also treat
  a captured `ConvergenceWarning` as a fit failure — no new upfront
  identifiability validation layer, but no longer relying solely on
  `statsmodels` raising (revised after review found `MixedLM.fit()` does not
  reliably raise on a fixed effect confounded with genotype).

## Impact

### Affected specs

- `statistics-api` (MODIFIED) — new `fixed_effects` parameter and formula
  behavior on `calculate_heritability_estimates`; `intercept`'s marginal-average
  semantics when fixed effects are present, threading into the existing "BLUP
  Adjusted-Means Table Extraction" requirement's `intercept + blup[g]` formula
  (the formula text is unchanged; what `intercept` represents is clarified).
- `serializable-result-types` (MODIFIED) — `BLUPResult.intercepts`'s field
  description updated to note the empirical frequency-weighted semantics when
  the underlying `calculate_heritability_estimates` call used `fixed_effects`
  (corrected after review: the previous draft of this section incorrectly
  referenced a nonexistent `HeritabilityResult` per-trait `intercept` field —
  `TraitHeritability` has no such field; only `BLUPResult.intercepts` carries
  intercept values). No field-shape or adapter-logic changes — `BLUPResult`
  and `from_blup_table()` already pass through whatever `intercept` float
  each trait's source dict carries, unchanged.
- `config-management` (ADDED) — `StatisticsConfig.fixed_effects` and its
  threading through `StatisticalAnalysisStep`.

### Affected code

- `src/sleap_roots_analyze/statistics.py` — `fixed_effects` parameter,
  formula/subset changes, marginal-intercept helper (with regex-based
  coefficient lookup and `ConvergenceWarning` capture) in
  `calculate_heritability_estimates`; update the function's own `Returns:`
  docstring section, whose current `intercept` description
  (`result.fe_params["Intercept"]`, unconditionally) becomes inaccurate once
  the marginal-intercept branch lands.
- `src/sleap_roots_analyze/result_types.py` — update `BLUPResult.intercepts`'s
  docstring (`Attributes:` block, currently at line ~832) to describe the
  empirical frequency-weighted semantics when the source used
  `fixed_effects`. No code change — the dataclass and its adapter already
  store/pass through whatever `intercept` float they're given.
- `src/sleap_roots_analyze/pipeline/config/components.py` — add
  `StatisticsConfig.fixed_effects`.
- `src/sleap_roots_analyze/pipeline/steps/statistical_analysis.py` — thread
  `fixed_effects` into the `calculate_heritability_estimates(...)` call.
- `docs/API.md`, `docs/CHANGELOG.md`, `docs/result-types.md` (added after
  review: the `BLUPResult` row needs the same intercepts caveat, mirroring
  existing per-type caveat bullets already in that file) — signature/behavior
  updates.
- `tests/test_statistics.py`, `tests/fixtures.py` (new batch-confounded and
  field-block fixtures), `tests/test_step_statistical_analysis.py`,
  `tests/test_blup_result.py` (pass-through test for fixed-effects-derived
  intercepts).

### Explicitly out of scope

- Tier 3+ (LOGO-CV, prediction pipeline, permutation nulls, figures): separate
  future changes.
- Any change to the H² formula itself, or to variance-component extraction
  for the genotype random effect.
- Coupling or validation between `fixed_effects` and `replicate_col` (e.g.
  auto-inclusion, duplicate-name guards) — fully independent by design.
- Identifiability/collinearity pre-validation for `fixed_effects` beyond
  `ConvergenceWarning` capture — no upfront check that a fixed effect is a
  deterministic function of genotype before fitting; the fit is allowed to
  attempt and its warning/failure is what's caught.
- Continuous/ordinal fixed-effect covariates — `fixed_effects` is `List[str]`
  and every entry is always `C()`-wrapped; supporting a genuinely continuous
  covariate (e.g. a linear drift term) would require a future, likely
  signature-breaking change (e.g. a typed mapping of column → coding scheme)
  and is not addressed here.
- Equal-weighted (EMM/lsmeans-style) marginal intercepts — the shipped
  intercept is sample-frequency-weighted over each trait's own fitted rows,
  not equally weighted across levels; a future tier could add an
  equally-weighted variant if a genuinely design-invariant "population
  typical" value is needed.
- Follow-up issue [#192](https://github.com/talmolab/sleap-roots-analyze/issues/192)
  (whether `replicate_col`'s presence-validation still earns its keep in
  `calculate_heritability_estimates`/`analyze_trait_variance`) — surfaced
  during this brainstorm, filed separately, not part of this change.
