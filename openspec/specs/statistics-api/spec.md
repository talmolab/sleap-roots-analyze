# statistics-api Specification

## Purpose
TBD - created by archiving change expose-statistics-functions. Update Purpose after archive.
## Requirements
### Requirement: Public Statistics API Surface

The package SHALL export all heritability, ANOVA, trait-variance, and BLUP
functions defined in `statistics.py` from the top-level `sleap_roots_analyze`
namespace so downstream code can import them without reaching into internal
modules.

#### Scenario: All nine statistics functions are importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import (calculate_trait_statistics, perform_anova_by_genotype, calculate_heritability_estimates, identify_high_heritability_traits, analyze_heritability_thresholds, analyze_trait_variance, diagnose_heritability_issues, compare_trait_heritabilities, extract_blup_table)`
- **THEN** the import SHALL succeed
- **AND** each imported object SHALL be identity-equal (`is`) to the function
  defined in `sleap_roots_analyze.statistics`

#### Scenario: Each function is listed in `__all__` and bound by star import

- **WHEN** `sleap_roots_analyze.__all__` is inspected
- **THEN** it SHALL contain the name of each of the nine statistics functions
- **AND** SHALL contain no duplicate entries
- **AND** `from sleap_roots_analyze import *` SHALL bind each of those names

### Requirement: Resolvable Statistics Type Hints

Each public statistics function SHALL have type hints that resolve at runtime, so
downstream tool-schema generators that call `typing.get_type_hints()` do not fail.

#### Scenario: get_type_hints succeeds on every public function

- **WHEN** `typing.get_type_hints(fn)` is called on each of the nine functions
- **THEN** it SHALL return without raising `NameError`
- **AND** every parameter and the return value SHALL carry a type annotation

### Requirement: Documented Statistics Functions

Each public statistics function SHALL have a complete Google-style docstring, and
the module SHALL describe its scope, so consumers and tooling can understand the
contract.

#### Scenario: Every public function has Args and Returns sections

- **WHEN** any of the nine public statistics functions is inspected
- **THEN** its docstring SHALL include populated Args and Returns sections in
  Google style (and a Raises section where the function raises)
- **AND** functions returning structured dictionaries SHALL enumerate the returned
  keys in the Returns section

#### Scenario: Module docstring distinguishes statistics from cross-experiment analysis

- **WHEN** the `statistics.py` module docstring is read
- **THEN** it SHALL describe the module's scope (single-experiment heritability,
  ANOVA, and trait-variance analysis)
- **AND** it SHALL name `cross_experiment_analysis` to clarify how that scope differs

### Requirement: Statistics API Reference Documentation

The project documentation SHALL keep its hand-maintained API reference and
changelog in sync with the newly-public statistics functions.

#### Scenario: API reference lists all nine statistics functions

- **WHEN** `docs/API.md` is viewed
- **THEN** the `## statistics Module` section SHALL include a reference entry for
  each of the nine statistics functions, including `extract_blup_table`
- **AND** each entry's documented signature and defaults SHALL match the code

#### Scenario: Changelog records the newly-public API

- **WHEN** `docs/CHANGELOG.md` `[Unreleased]` section is viewed
- **THEN** it SHALL include an `### Added` entry noting `extract_blup_table` is
  now importable from `sleap_roots_analyze`

### Requirement: BLUP Adjusted-Means Table Extraction

The package SHALL provide `extract_blup_table(heritability_results)` in
`statistics.py` that builds a genotype × trait adjusted-means table from the
dict returned by `calculate_heritability_estimates()`. For each trait whose
per-trait entry carries a `blup` dict and an `intercept` float (i.e. the mixed
model succeeded), the adjusted mean for genotype `g` SHALL be
`intercept + blup[g]`. `intercept` is supplied by
`calculate_heritability_estimates()` and MAY be a frequency-weighted marginal
value (see "Heritability Model Fixed Effects") when that call used
`fixed_effects`; `extract_blup_table()` itself is unaware of `fixed_effects`
and applies the same `intercept + blup[g]` formula regardless of how
`intercept` was computed. For a trait with no `blup`/`intercept` (the model
failed, errored, used the ANOVA-based or no-variance path, or was skipped),
the entire column SHALL be `NaN` — not omitted from the table and not
zero-filled. A genotype absent from one succeeded trait's `blup` dict but
present in another's (a real possibility, since
`calculate_heritability_estimates` computes each trait's genotype set
independently via a per-trait `dropna()`) SHALL produce a cell-level `NaN` for
that genotype/trait combination, not a row drop or a column-level failure. The
function SHALL NOT mutate its input, and SHALL NOT raise when
`heritability_results` is a run-level short-circuit (`{"error": "..."}`, no
per-trait entries) or when every trait failed (zero succeeded traits).

#### Scenario: Successful traits populate adjusted means

- **GIVEN** a `heritability_results` dict where trait `"trait_a"` has
  `heritability_results["trait_a"]["blup"] == {"G01": 0.5, "G02": -0.5}` and
  `heritability_results["trait_a"]["intercept"] == 10.0`
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the returned `pd.DataFrame` SHALL have a row per genotype (`"G01"`,
  `"G02"`) and a `"trait_a"` column
- **AND** `df.loc["G01", "trait_a"] == pytest.approx(10.5)` and
  `df.loc["G02", "trait_a"] == pytest.approx(9.5)`

#### Scenario: Failed-model trait produces a NaN column, not a dropped or zero column

- **GIVEN** a `heritability_results` dict where trait `"trait_failed"` has no
  `blup` key (e.g. `{"error": "Mixed model failed: ..."}`) while at least one
  other trait succeeded
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the returned `pd.DataFrame` SHALL still contain a `"trait_failed"`
  column
- **AND** every value in that column SHALL be `NaN` (`pd.isna(...).all()`)
- **AND** no value in that column SHALL be `0.0`

#### Scenario: A trait solved via the ANOVA-based or no-variance path has no BLUP column, without raising

- **GIVEN** a `heritability_results` dict where trait `"trait_anova"` succeeded
  with `model_type == "anova_based"` (no fitted mixed model exists for this
  trait, since `force_method="anova_based"` never calls `smf.mixedlm`) or
  `model_type == "no_variance"` (the trait short-circuited before any model
  fit), and therefore carries no `blup`/`intercept` keys
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the call SHALL NOT raise
- **AND** the `"trait_anova"` column SHALL be entirely `NaN`, identically to a
  failed trait's column

#### Scenario: Row and column shape matches genotypes and traits

- **GIVEN** a `heritability_results` dict where the union of every succeeded
  trait's `blup.keys()` contains `N` distinct genotypes, and `T` total traits
  are present (including failed, ANOVA-based, no-variance, and skipped ones)
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the returned `pd.DataFrame` SHALL have exactly `N` rows (indexed by
  that genotype union, not any single trait's own genotype set) and `T`
  columns (one per trait, in `trait_cols` order, excluding the
  `__calculation_metadata__` key)

#### Scenario: A genotype missing from one succeeded trait's blup dict gets a cell-level NaN

- **GIVEN** two traits that both succeed (`model_type == "mixed_model"`), where
  trait `"trait_a"`'s `blup` dict covers genotypes `{"G01", "G02"}` and trait
  `"trait_b"`'s `blup` dict covers `{"G01", "G02", "G03"}` (e.g. `"trait_a"`
  dropped `"G03"`'s only observation as NaN before the model was fit)
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the returned `pd.DataFrame` SHALL have a `"G03"` row
- **AND** `df.loc["G03", "trait_a"]` SHALL be `NaN` while `df.loc["G03",
  "trait_b"]` SHALL be a finite adjusted mean — a cell-level gap, distinct from
  the whole-column `NaN` of a failed trait

#### Scenario: A run-level short-circuit dict produces an empty table without raising

- **GIVEN** `heritability_results == {"error": "Missing required columns: ['geno']"}`
  (the run-level short-circuit form `calculate_heritability_estimates` returns
  when required columns are absent, carrying no per-trait entries)
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the call SHALL NOT raise
- **AND** the returned `pd.DataFrame` SHALL be empty (zero rows, zero columns)

#### Scenario: Zero succeeded traits produce an all-NaN table, not a misclassified one

- **GIVEN** a `heritability_results` dict where every trait failed (no trait
  carries a `blup` key), so the genotype universe collected from succeeded
  traits is empty
- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** the returned `pd.DataFrame` SHALL have zero rows (no genotype
  universe to index by) and one column per input trait
- **AND** every column SHALL be treated as failed, not as vacuously "all
  finite" by virtue of having zero rows

#### Scenario: Input dict is not mutated

- **WHEN** `extract_blup_table(heritability_results)` is called
- **THEN** `heritability_results`'s keys and values SHALL be unchanged after the
  call

### Requirement: BLUP Shrinkage and Balanced-Design Properties

The BLUP-adjusted means produced by `extract_blup_table()` SHALL exhibit the
mixed-model shrinkage property described in `theory.md`: under a balanced
design (equal replication per genotype), the adjusted mean SHALL approximate
the raw genotype mean; under an unbalanced design, genotypes with fewer
replicates SHALL be shrunk toward the grand mean more than genotypes with more
replicates.

#### Scenario: Balanced design — adjusted mean approximates the raw genotype mean

- **GIVEN** a balanced dataset (equal replicates per genotype) with a known,
  non-trivial genetic variance component
- **WHEN** `calculate_heritability_estimates()` is run and its result passed to
  `extract_blup_table()`
- **THEN** for every genotype and trait, the adjusted mean SHALL be within a
  documented, noise-scaled tolerance of that genotype's raw trait mean
  (`df.groupby(genotype_col)[trait].mean()`)

#### Scenario: Unbalanced design — low-replicate genotypes shrink further than high-replicate genotypes

- **GIVEN** an unbalanced dataset where some genotypes have substantially fewer
  replicates than others, with a known genetic variance component
- **WHEN** `calculate_heritability_estimates()` is run and its result passed to
  `extract_blup_table()`
- **THEN** for a low-replicate genotype, `|adjusted_mean - grand_mean|` SHALL be
  smaller than `|raw_mean - grand_mean|` (shrinkage toward the grand mean)
- **AND** this shrinkage gap SHALL be larger for low-replicate genotypes than
  for high-replicate genotypes (shrinkage scales inversely with replication)

### Requirement: Heritability Model Fixed Effects

`calculate_heritability_estimates()` SHALL accept an optional
`fixed_effects: Optional[List[str]] = None` parameter. When `None` (the
default), behavior SHALL be byte-for-byte identical to a call without this
parameter — the model formula remains `"value ~ 1"` and `intercept` remains
`float(result.fe_params["Intercept"])`.

When `fixed_effects` is a non-empty list, the mixed-model formula for the
`mixed_model` path SHALL become `"value ~ " + " + ".join(f"C({fe})" for fe in
fixed_effects)`, wrapping every named column in `C(...)` unconditionally —
every fixed effect SHALL be treated as categorical regardless of its pandas
dtype, with no dtype-based inference. Every name in `fixed_effects` SHALL be
validated for presence in `df`, extending the existing top-level
`required_cols` check (alongside `genotype_col` and, when truthy,
`replicate_col`): a missing fixed-effect column SHALL produce the same
run-level `{"error": "Missing required columns: [...]"}` short-circuit as a
missing `genotype_col`, listing every missing column. Every name in
`fixed_effects` SHALL also be validated with `isinstance(fe, str) and
fe.isidentifier()` before being interpolated into the formula string — the
`isinstance` check SHALL be evaluated first (short-circuiting), so a
non-`str` element (e.g. an int-labeled column, plausible for a CSV-derived
batch/wave/scanner code) produces the same run-level structural error rather
than an uncaught `AttributeError` from calling `.isidentifier()` on a
non-string. A name that is not a valid Python identifier (e.g. one
containing a patsy formula operator such as `*` or `:`) SHALL produce a
run-level structural error rather than being interpolated — without this, a
column name containing an operator character could silently misparse as a
patsy expression over other, differently-named columns rather than a
literal reference to itself. A `fixed_effects` name that duplicates
`genotype_col` or `replicate_col` SHALL also produce a run-level structural
error rather than being interpolated — without this, the duplicate column
selection surfaces as a confusing pandas-internal error deep inside the
per-trait loop (e.g. `"Grouper for 'geno' not 1-dimensional"`) rather than a
clear structural error. The per-trait model
subset SHALL become `df[[trait, genotype_col] + fixed_effects].dropna()`
(dropping rows with a `NaN` in any fixed-effect column, in addition to the
existing trait/genotype `NaN` handling) — this subset change SHALL only take
effect when `fixed_effects` is non-empty.

A mixed-model fit failure introduced by `fixed_effects` (non-convergence, a
fixed effect confounded with genotype, or any other `statsmodels` exception)
SHALL be caught by the existing per-trait `try/except` around the model fit,
recorded as `{"error": "Mixed model failed: ...", "model_type":
"mixed_model_failed"}` for that trait — identical handling to a
non-`fixed_effects`-related fit failure (this reuse of the existing
`try/except` applies regardless of whether `fixed_effects` is set, since it
requires no new code). **The additional warning-capture behavior below,
however, SHALL apply only when `fixed_effects` is non-empty** — this is
required for the "byte-for-byte identical when `fixed_effects=None`"
guarantee earlier in this requirement: an unconditional warning-to-failure
check would be a behavior change for existing, non-`fixed_effects` callers
(a convergence warning on a plain `"value ~ 1"` fit would newly become a
failure where it previously succeeded), which this requirement's opening
paragraph explicitly rules out. When `fixed_effects` is non-empty, the fit
call SHALL be wrapped in `warnings.catch_warnings(record=True)` with
`warnings.simplefilter("always")` called immediately inside that block (NOT
`record=True` alone — Python's default once-per-source-location filter can
otherwise silently drop a repeat occurrence of the same warning for a later
trait in the same process). A captured warning SHALL be treated as a fit
failure for that trait (same error dict shape) only when
`issubclass(warning.category, statsmodels.tools.sm_exceptions.ConvergenceWarning)`
— checked by category, NOT by matching the warning's message text (several
real `statsmodels` convergence-related warning messages do not contain the
word "convergence" at all). A captured warning of any other category SHALL
NOT be treated as a fit failure. This is required even though `statsmodels`
did not raise — `MixedLM.fit()` does not reliably raise on a fixed effect
that is (near-)fully confounded with genotype, so relying on raised
exceptions alone would let such a fit silently succeed with degenerate
parameters. No new upfront identifiability or collinearity pre-validation
(checked before fitting) SHALL be added; only the fit's own exception/warning
signals are observed.

`fixed_effects` and `replicate_col` SHALL be fully independent: `replicate_col`
SHALL NOT be automatically included in `fixed_effects`, and no validation
SHALL link the two parameters. A block/replicate fixed effect SHALL be
expressed by naming that column directly in `fixed_effects`.

When `fixed_effects` is non-empty and a trait's mixed model succeeds,
`intercept` SHALL be computed as an empirical, sample frequency-weighted
value rather than the raw `result.fe_params["Intercept"]`: for each fixed
effect, each level's fitted contribution (`0.0` for the reference level
dropped by patsy's treatment coding; its own coefficient in
`result.fe_params` for every other level) SHALL be weighted by that level's
share of the fitted `model_data` rows (that trait's own post-`dropna()`
subset), summed across levels within that fixed effect, then summed across
all fixed effects and added to the base `Intercept` coefficient. This value
is a sample-margin quantity, not a population-typical or EMM/lsmeans-style
equally-weighted marginal mean — it is sensitive to that trait's own
missing-data pattern and to incidental level-frequency imbalance, and two
traits sharing the same `fixed_effects` columns MAY receive different
per-level weights. The per-level coefficient SHALL be recovered by parsing
`result.fe_params`'s actual fitted parameter names (matching
`^C\({fe}\)\[T\.(.*)\]$` for each fixed effect `fe`), not by reconstructing
the expected key string forward from each observed level's raw value; each
recovered level string SHALL be matched back to `model_data[fe]`'s values by
equality, not by positional pairing against a separately-sorted list of
levels (positional pairing silently mispairs frequencies when a fixed effect
is a `pandas.Categorical` with a non-default `categories=` order, since
patsy's reference level is then the first *declared* category, not the first
in sorted order). The implementation SHALL assert the number of
non-reference coefficients recovered for a fixed effect equals
`model_data[fe].nunique() - 1` — computed on that trait's own
post-`dropna()` fitted subset, not the raw input `df` (a level present in
`df` can be entirely absent from a specific trait's `model_data` due to that
trait's own missingness pattern) — raising rather than silently defaulting a
mismatched level's contribution to `0.0`.

The docstring SHALL document that `fixed_effects` is intended for
metadata-style covariates that confound with genotype (experiment, wave,
batch, scanner), not biological/phenotypic traits — a documentation
convention, not a runtime-enforced check. It SHALL also document that
`intercept` is an empirical frequency-weighted (not population-typical)
value when `fixed_effects` is set, and that the row-filtering subset change
applies identically regardless of `force_method` (the ANOVA-based path
still ignores `fixed_effects` in its own variance-component computation).

#### Scenario: fixed_effects=None reproduces current behavior exactly

- **WHEN** `calculate_heritability_estimates(df, trait_cols, ...)` is called
  without `fixed_effects` (or with `fixed_effects=None`)
- **THEN** the returned dict SHALL be identical, key-for-key and value-for-value,
  to a call made before this parameter existed — including the `"value ~ 1"`
  formula and `intercept == float(result.fe_params["Intercept"])`

#### Scenario: A missing fixed-effect column produces a structural error

- **GIVEN** `fixed_effects=["experiment"]` where `"experiment"` is not a column
  in `df`
- **WHEN** `calculate_heritability_estimates(df, trait_cols, fixed_effects=["experiment"])`
  is called
- **THEN** the return value SHALL be `{"error": "Missing required columns:
  [...]"}` listing `"experiment"`, with no per-trait entries — the same
  run-level short-circuit shape as a missing `genotype_col`

#### Scenario: A fixed-effect column name containing a patsy operator is rejected, not silently misparsed

- **GIVEN** `fixed_effects=["rep*block"]` on a `df` that also has separate
  `"rep"` and `"block"` columns
- **WHEN** `calculate_heritability_estimates(df, trait_cols,
  fixed_effects=["rep*block"])` is called
- **THEN** the call SHALL produce a loud, run-level structural error (the
  name fails `fe.isidentifier()`)
- **AND** the formula SHALL NOT be constructed with `"rep*block"`
  interpolated as if it were a literal column reference — it SHALL NOT
  silently evaluate as elementwise multiplication of the separate `"rep"`
  and `"block"` columns

#### Scenario: A non-string fixed_effects element is rejected, not crashed

- **GIVEN** `fixed_effects=[5]` where `5` is a valid, int-labeled column in
  `df` (plausible for a CSV-derived batch/wave/scanner code)
- **WHEN** `calculate_heritability_estimates(df, trait_cols,
  fixed_effects=[5])` is called
- **THEN** the call SHALL produce the same run-level `{"error": "Invalid
  fixed_effects column name(s): [...]"}` structural error as an
  invalid-identifier string name
- **AND** no exception (e.g. `AttributeError` from calling `.isidentifier()`
  on a non-`str`) SHALL propagate to the caller

#### Scenario: A fixed_effects name duplicating genotype_col or replicate_col is rejected

- **GIVEN** `fixed_effects=["geno"]` where `"geno"` is also `genotype_col`
  (or, symmetrically, `fixed_effects=[replicate_col]`)
- **WHEN** `calculate_heritability_estimates` is called
- **THEN** the call SHALL produce a run-level structural error naming the
  duplicated column(s)
- **AND** the per-trait loop SHALL NOT be reached — without this check, the
  duplicate-column selection surfaces as a confusing pandas-internal error
  (e.g. `"Grouper for 'geno' not 1-dimensional"`) rather than a clear
  structural error

#### Scenario: Fixed-effect columns are always treated as categorical

- **GIVEN** a fixed-effect column whose values are numeric-looking (e.g.
  `wave_number` with values `1`, `2`, `3`) but represent discrete metadata
  groups
- **WHEN** the mixed model is fit with that column in `fixed_effects`
- **THEN** the fitted formula SHALL wrap the column in `C(...)`, producing one
  coefficient per non-reference level in `result.fe_params` (treatment
  coding), NOT a single continuous-slope coefficient

#### Scenario: Rows with a NaN fixed-effect value are dropped from the model fit

- **GIVEN** a row with valid `trait` and `genotype_col` values but a `NaN` in a
  named `fixed_effects` column
- **WHEN** `calculate_heritability_estimates(df, trait_cols,
  fixed_effects=[...])` is called with that column included
- **THEN** that row SHALL be excluded from the per-trait model subset
- **AND** the same row SHALL NOT be excluded when the same call is made with
  `fixed_effects=None`

#### Scenario: A batch-confounded synthetic fixture shows corrected H² below uncorrected H²

- **GIVEN** a synthetic dataset where genotypes are unevenly distributed
  across two or more "experiment" batches, with a systematic per-batch shift
  baked into the trait values (mirroring issue #114's Bloom-experiment
  scenario)
- **WHEN** `calculate_heritability_estimates` is called once with
  `fixed_effects=None` and once with `fixed_effects=["experiment"]`
- **THEN** the `fixed_effects=None` run's heritability estimate SHALL be
  greater than the `fixed_effects=["experiment"]` run's estimate for the
  batch-confounded trait

#### Scenario: A model-fit failure from fixed effects is recorded like any other failure

- **GIVEN** a trait whose mixed-model fit raises when `fixed_effects` is set
  (e.g. a `statsmodels` convergence failure)
- **WHEN** `calculate_heritability_estimates` processes that trait
- **THEN** that trait's per-trait dict SHALL be `{"error": "Mixed model
  failed: ...", "model_type": "mixed_model_failed"}`
- **AND** no exception SHALL propagate out of `calculate_heritability_estimates`
- **AND** processing SHALL continue for the remaining traits

#### Scenario: A convergence warning during fit is treated as a failure, not a silent success

- **GIVEN** a trait whose mixed-model fit, with `fixed_effects` set, emits a
  convergence warning (e.g. `statsmodels`' `ConvergenceWarning`) but does not
  raise
- **WHEN** `calculate_heritability_estimates` processes that trait
- **THEN** that trait's per-trait dict SHALL be classified as failed (the
  same `{"error": ..., "model_type": "mixed_model_failed"}` shape as a raised
  exception), NOT returned as a successful `mixed_model` result with
  `blup`/`intercept`/`heritability` values
- **AND** processing SHALL continue for the remaining traits

#### Scenario: A warning of an unrelated category does not fail the trait

- **GIVEN** a trait whose mixed-model fit, with `fixed_effects` set, emits a
  warning of a category other than `ConvergenceWarning` (e.g. a plain
  `UserWarning`) but does not raise, on otherwise-normal data
- **WHEN** `calculate_heritability_estimates` processes that trait
- **THEN** that trait's per-trait dict SHALL be a normal successful
  `mixed_model` result (`blup`/`intercept`/`heritability` present, no
  `error` key) — an implementation that treats any captured warning as a
  failure, regardless of category, violates this scenario

#### Scenario: A convergence warning is not caught when fixed_effects is unset

- **GIVEN** a trait whose mixed-model fit emits a `ConvergenceWarning` but
  does not raise, called with `fixed_effects=None` (or omitted)
- **WHEN** `calculate_heritability_estimates` processes that trait
- **THEN** that trait's per-trait dict SHALL be a normal successful
  `mixed_model` result, exactly as it would be without this tier's changes —
  the warning-capture behavior added by this tier SHALL apply only when
  `fixed_effects` is non-empty, preserving the byte-for-byte compatibility
  guarantee for `fixed_effects=None` callers

#### Scenario: Empirical frequency-weighted intercept matches a hand-computed average

- **GIVEN** a fixture with a single fixed effect having two levels observed
  at known, unequal frequencies in the fitted data
- **WHEN** `calculate_heritability_estimates(df, trait_cols,
  fixed_effects=["experiment"])` is called and the trait's mixed model
  succeeds
- **THEN** the returned `intercept` SHALL equal
  `fe_params["Intercept"] + level_frequency[level] * offset[level]` summed
  over the fixed effect's non-reference levels (`offset[level]` from
  `result.fe_params`, `0.0` implicitly for the reference level), within
  floating-point tolerance — an independent, hand-computed oracle, not a
  tautological re-derivation of the implementation

#### Scenario: Multiple fixed effects contribute independently to the intercept

- **GIVEN** `fixed_effects=["experiment", "block"]`, each with its own
  observed level frequencies
- **WHEN** the mixed model succeeds
- **THEN** the returned `intercept` SHALL equal the base `Intercept` plus the
  independent frequency-weighted contribution of `experiment`'s levels plus
  the independent frequency-weighted contribution of `block`'s levels (patsy's
  `+` composes fixed effects additively, not as an interaction) — the two
  effects' contributions SHALL NOT be conflated or double-counted

#### Scenario: A float-dtype fixed-effect column does not corrupt the coefficient lookup

- **GIVEN** a fixed-effect column stored as `float64` (e.g. `wave_number` with
  values `1.0`, `2.0`, `3.0`, a realistic case when the source column had a
  `NaN` elsewhere in the original data before this trait's `dropna()`)
- **WHEN** the mixed model is fit with that column in `fixed_effects` and the
  intercept is computed
- **THEN** every non-reference level's coefficient SHALL be correctly
  attributed (recovered by parsing `result.fe_params`'s actual fitted
  parameter names, not by reconstructing a key from the raw `float64` value)
- **AND** the recovered non-reference coefficient count SHALL equal
  `model_data[fe].nunique() - 1` — a mismatch SHALL raise rather than
  silently attribute a real level's contribution as `0.0`

#### Scenario: A non-default categorical level order does not mispair frequencies with coefficients

- **GIVEN** a fixed-effect column declared as `pandas.Categorical` with an
  explicit, non-alphabetical/non-numeric `categories=[...]` order (so
  patsy's reference level is the first *declared* category, not the first in
  sorted order)
- **WHEN** the mixed model is fit with that column in `fixed_effects` and the
  intercept is computed
- **THEN** each recovered level's frequency SHALL be matched to its correct
  coefficient by equality against `model_data[fe]`'s actual values, NOT by
  positional pairing against a separately-sorted list of unique levels —
  the returned `intercept` SHALL match an independent hand-computation using
  the fixture's known level frequencies and offsets

#### Scenario: A repeated identical convergence warning fails every affected trait, not just the first

- **GIVEN** two different traits in the same
  `calculate_heritability_estimates` call whose mixed-model fits (with
  `fixed_effects` set) both emit the same `ConvergenceWarning` (same message,
  same source location) without raising
- **WHEN** `calculate_heritability_estimates` processes both traits in the
  same call
- **THEN** both traits' per-trait dicts SHALL be classified as failed — the
  fit-wrapping SHALL force `warnings.simplefilter("always")` so that
  Python's default once-per-location warning filter does not silently drop
  the second trait's identical warning

#### Scenario: fixed_effects does not affect the ANOVA-based path's model, only its row filtering

- **GIVEN** `fixed_effects=["experiment"]` and
  `force_method="anova_based"`
- **WHEN** `calculate_heritability_estimates` processes a trait
- **THEN** that trait's per-trait model subset SHALL still exclude rows with
  a `NaN` in the `"experiment"` column (the same row-filtering as the
  mixed-model path)
- **AND** the ANOVA-based variance-component computation SHALL NOT use
  `fixed_effects` in any way — `model_type` SHALL be `"anova_based"`, with no
  `C(...)`-wrapped formula ever constructed for this path

#### Scenario: Field-block fixed effect changes BLUP-adjusted means relative to genotype-only

- **GIVEN** a field-block-style fixture with a systematic per-block shift in
  trait values
- **WHEN** `calculate_heritability_estimates` is run once with
  `fixed_effects=None` and once with `fixed_effects=["block"]`, and each
  result is passed to `extract_blup_table()`
- **THEN** the two runs' adjusted-means tables SHALL differ for at least one
  genotype/trait pair beyond floating-point tolerance

#### Scenario: Shrinkage still scales inversely with replication when fixed_effects is set

- **GIVEN** a field-block-style fixture unbalanced across fixed-effect
  levels, with some genotypes having substantially fewer replicates than
  others (mirroring Tier 1's unbalanced-design shrinkage oracle), and block
  composition skew applied orthogonally to the replicate-count grouping (not
  correlated with which genotypes are low- vs high-replicate)
- **WHEN** `calculate_heritability_estimates(..., fixed_effects=["block"])`
  is run
- **THEN** for every genotype, `abs(blup[genotype])` SHALL be smaller than
  `|raw_mean_detrended[genotype] - reference_level_intercept|`, where
  `raw_mean_detrended` is computed by subtracting the fitted `C(block)`
  coefficient for each observation's (non-reference) block level before
  averaging within genotype (NOT the naive
  `df.groupby(genotype)[trait].mean()`, which is itself contaminated by each
  genotype's own block composition — the exact effect being corrected for),
  and `reference_level_intercept` is the fitted model's raw
  `fe_params["Intercept"]` — NOT `calculate_heritability_estimates`'s own
  returned `intercept`, which for a `fixed_effects` run is the empirical
  frequency-weighted value (see the "Heritability Model Fixed Effects"
  requirement), not the reference-level value this comparison needs. Using
  the empirical frequency-weighted `intercept` here (or comparing against
  `extract_blup_table()`'s already-summed `adjusted_mean` instead of
  `blup[genotype]` directly) introduces a constant offset that does not
  cancel under the absolute-value comparison and breaks this property for a
  subset of genotypes — confirmed empirically during implementation
- **AND** this shrinkage gap SHALL be larger for low-replicate genotypes than
  for high-replicate genotypes, matching Tier 1's existing guarantee

