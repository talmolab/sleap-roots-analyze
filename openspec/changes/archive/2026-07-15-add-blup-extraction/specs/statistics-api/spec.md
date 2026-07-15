## MODIFIED Requirements

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

## ADDED Requirements

### Requirement: BLUP Adjusted-Means Table Extraction

The package SHALL provide `extract_blup_table(heritability_results)` in
`statistics.py` that builds a genotype × trait adjusted-means table from the
dict returned by `calculate_heritability_estimates()`. For each trait whose
per-trait entry carries a `blup` dict and an `intercept` float (i.e. the mixed
model succeeded), the adjusted mean for genotype `g` SHALL be
`intercept + blup[g]`. For a trait with no `blup`/`intercept` (the model
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
