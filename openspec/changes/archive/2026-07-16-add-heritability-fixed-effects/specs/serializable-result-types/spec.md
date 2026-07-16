## MODIFIED Requirements

### Requirement: Non-Breaking Heritability Return Shape

`calculate_heritability_estimates()` SHALL keep its return shape unchanged by `HeritabilityResult`, BLUP extraction, or the `fixed_effects` parameter.
The function SHALL keep returning its
existing dict / tuple (a plain `dict` when `remove_low_h2=False`, or a 4-tuple
`(heritability_results, df_filtered, removed_traits, removal_details)` when
`remove_low_h2=True`) so all current callers continue to work unchanged. BLUP
extraction MAY add `blup` (`dict[str, float]`) and `intercept` (`float`) keys
to a trait's own per-trait dict, but only when that trait's mixed model fit
succeeds (`model_type == "mixed_model"`); a trait solved via the ANOVA-based
path (`model_type == "anova_based"`), the no-variance short-circuit
(`model_type == "no_variance"`), or any error path SHALL NOT carry
`blup`/`intercept` keys, since no fitted mixed-model `result` object exists
for those paths. `result.random_effects` SHALL be accessed exactly once per
successful trait fit. When the call used a non-empty `fixed_effects`,
`intercept` SHALL be the empirical frequency-weighted value described under
`statistics-api`'s "Heritability Model Fixed Effects" requirement rather than
the raw `result.fe_params["Intercept"]` — this SHALL NOT change the key's
type (`float`) or its presence condition (mixed-model success only).

#### Scenario: Existing heritability return is preserved and not mutated

- **WHEN** `calculate_heritability_estimates()` is called with
  `remove_low_h2=False`
- **THEN** it SHALL return the existing trait-keyed dict (including the
  `__calculation_metadata__` entry)
- **AND** calling `HeritabilityResult.from_heritability_dict(d, threshold)` SHALL
  leave `d`'s keys and values unchanged
- **AND** the `HeritabilityResult` view SHALL be opt-in, built only via
  `HeritabilityResult.from_heritability_dict()`

#### Scenario: Existing dict return is preserved and only additively extended with BLUPs

- **WHEN** `calculate_heritability_estimates(df, trait_cols, ...)` is called
  with `remove_low_h2=False` (the default) and a trait's mixed model succeeds
- **THEN** the returned `dict`'s per-trait entry SHALL still contain the
  existing keys (`heritability`, `var_genetic`, `var_residual`, `mean_n_reps`,
  `n_genotypes`, `n_observations`, `model_type`, `reps_per_geno_stats`)
- **AND** it SHALL additionally contain `blup` (a `dict[str, float]` keyed by
  genotype label) and `intercept` (a `float`)
- **AND** the top-level return type SHALL remain a plain `dict`

#### Scenario: remove_low_h2=True keeps its 4-tuple shape with BLUP keys

- **WHEN** `calculate_heritability_estimates(df, trait_cols, ..., remove_low_h2=True)`
  is called
- **THEN** the return value SHALL still be a 4-tuple
  `(heritability_results, df_filtered, removed_traits, removal_details)`
- **AND** `heritability_results`'s per-trait entries SHALL carry the same
  additive `blup`/`intercept` keys as the `remove_low_h2=False` case, even for
  traits later dropped from `df_filtered`/`removed_traits` (heritability-based
  trait removal filters the DataFrame; it does not strip keys from
  `heritability_results`)

#### Scenario: A trait whose mixed model fit fails carries no blup/intercept keys

- **WHEN** a trait's mixed model fit raises (caught internally, recorded as
  `{"error": ..., "model_type": "mixed_model_failed"}`)
- **THEN** that trait's per-trait dict SHALL NOT contain a `blup` or
  `intercept` key

#### Scenario: A trait solved via the ANOVA-based or no-variance path carries no blup/intercept keys

- **WHEN** a trait succeeds via `force_method="anova_based"` (`model_type ==
  "anova_based"`) or via the no-variance short-circuit (`model_type ==
  "no_variance"`, all values identical)
- **THEN** that trait's per-trait dict SHALL NOT contain a `blup` or
  `intercept` key
- **AND** no exception SHALL be raised while producing that trait's result,
  even though no fitted mixed-model `result` object exists for it

#### Scenario: fixed_effects does not change which traits carry blup/intercept keys

- **WHEN** `calculate_heritability_estimates(df, trait_cols,
  fixed_effects=[...])` is called
- **THEN** the same condition governs `blup`/`intercept` presence as without
  `fixed_effects` — `model_type == "mixed_model"` for that trait — with
  `intercept`'s value computed per the marginal-intercept rule instead of the
  raw `fe_params["Intercept"]`

### Requirement: Serializable BLUP Result Type

The package SHALL provide a frozen stdlib `@dataclass` `BLUPResult` in
`result_types.py` that captures only the JSON-serializable science of an
`extract_blup_table()` run: the genotype × trait adjusted-means matrix for
traits whose mixed model succeeded, plus the names of traits that failed.
Every scalar field SHALL be a native Python type (`int`, `float`, `str`) —
not a numpy or pandas scalar — and every array field a (nested) list thereof,
so that `json.dumps(dataclasses.asdict(result))` succeeds without a custom
serializer. `BLUPResult` SHALL provide `to_dict()` and `to_json()` (the
`allow_nan=False` finite-floats contract, matching `PCAResult`/
`HeritabilityResult`/`UMAPResult`).

`BLUPResult`'s fields are: `genotype_names: list[str]`, `trait_names: list[str]`
(only traits whose model succeeded — failed traits are excluded from this list
and from `adjusted_means`), `adjusted_means: list[list[float]]` (shape
`(n_genotypes, n_traits)`, aligned to `genotype_names` × `trait_names`, always
finite), `failed_traits: list[str]` (names only, no values — mirrors
`HeritabilityResult.failed_traits`), and `intercepts: dict[str, float]` (one
entry per succeeded trait in `trait_names`). When the source
`heritability_results` was produced with a non-empty `fixed_effects`,
`intercepts`' values are the empirical frequency-weighted intercepts described
under `statistics-api`'s "Heritability Model Fixed Effects" requirement,
rather than a raw reference-level coefficient — `BLUPResult` and its adapter
are unaware of `fixed_effects` and store whatever `intercept` float each
trait's source dict carries, unchanged.

#### Scenario: BLUPResult round-trips through JSON as native types

- **WHEN** a `BLUPResult` is built from an `extract_blup_table()` DataFrame via
  `BLUPResult.from_blup_table()` and passed to
  `json.dumps(dataclasses.asdict(result))`
- **THEN** the call SHALL succeed without raising
- **AND** parsing the string back with `json.loads` SHALL yield every element of
  `genotype_names` and `trait_names` and `failed_traits` as a Python `str`,
  every element of `adjusted_means` as a Python `float`, and every value in
  `intercepts` as a Python `float`
- **AND** the round-tripped `adjusted_means` values SHALL equal the input
  values within floating-point tolerance

#### Scenario: Fields are native types before serialization

- **WHEN** the `BLUPResult` dataclass fields are inspected directly, before any
  JSON serialization
- **THEN** every element of `genotype_names`, `trait_names`, and
  `failed_traits` SHALL be a native `str`, every element of `adjusted_means` a
  native `float`, and every value of `intercepts` a native `float` — not a
  numpy or pandas scalar (a JSON round-trip would silently cast a leaked
  `np.float64` to `float`, so this check MUST be pre-serialization)

#### Scenario: A failed trait never appears as NaN in the dataclass

- **GIVEN** an `extract_blup_table()` DataFrame with a genuine `NaN` column for
  a failed trait
- **WHEN** `BLUPResult.from_blup_table()` builds the result
- **THEN** that trait's name SHALL appear in `failed_traits`
- **AND** that trait SHALL NOT appear in `trait_names`
- **AND** `adjusted_means` SHALL contain no `NaN`/`Infinity` value contributed
  by that trait
- **AND** `to_json()` SHALL succeed without raising (the finite-floats
  contract is satisfied even though the source table has a NaN column)

#### Scenario: A cell-level NaN (partial genotype coverage) also excludes that trait as failed

- **GIVEN** an `extract_blup_table()` DataFrame where a trait's column has
  finite values for most genotypes but `NaN` for one genotype absent from that
  trait's `blup` dict (see the cell-level-NaN scenario under "BLUP
  Adjusted-Means Table Extraction")
- **WHEN** `BLUPResult.from_blup_table()` builds the result
- **THEN** that trait SHALL be classified as failed (its name in
  `failed_traits`, excluded from `trait_names` and `adjusted_means`) — a
  partially-finite column is not eligible to appear in the always-finite
  matrix

#### Scenario: Zero succeeded traits produce an empty (not misclassified) result

- **GIVEN** an `extract_blup_table()` DataFrame with zero rows (no genotype
  universe, because every trait failed) and one column per input trait
- **WHEN** `BLUPResult.from_blup_table()` builds the result
- **THEN** every column SHALL be classified as failed (`failed_traits`
  contains all trait names, `trait_names` and `adjusted_means` are empty) —
  a zero-row column SHALL NOT be treated as vacuously finite
- **AND** `genotype_names` SHALL be an empty list
- **AND** `to_json()` SHALL succeed without raising

#### Scenario: to_json rejects a non-finite adjusted mean

- **WHEN** `to_json()` is called on a `BLUPResult` constructed directly (not via
  the adapter) with a non-finite value (`NaN` or `Infinity`) in
  `adjusted_means`
- **THEN** a `ValueError` SHALL be raised (under the default `allow_nan=False`)
  rather than emitting the non-standard `NaN`/`Infinity` tokens a strict JSON
  consumer rejects

#### Scenario: intercepts values pass through unchanged when the source used fixed_effects

- **GIVEN** an `extract_blup_table()`-shaped DataFrame and an `intercepts`
  mapping produced from a `heritability_results` dict whose
  `calculate_heritability_estimates` call used a non-empty `fixed_effects`
- **WHEN** `BLUPResult.from_blup_table(df, intercepts=...)` is called
- **THEN** `result.intercepts` SHALL contain exactly those marginal-intercept
  values, unchanged — `BLUPResult`/`from_blup_table` SHALL NOT recompute or
  reinterpret them
