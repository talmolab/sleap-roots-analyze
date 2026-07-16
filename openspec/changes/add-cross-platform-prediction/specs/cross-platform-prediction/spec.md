## ADDED Requirements

### Requirement: Per-Fold PCA Utility

The package SHALL provide `fit_pca_on_fold(X_train, X_test, n_components=1) -> np.ndarray` in
`src/sleap_roots_analyze/cross_platform_prediction.py` that fits a fresh
`sklearn.decomposition.PCA` on `X_train` only and returns `X_test` projected onto the resulting
components. This utility SHALL NOT reuse, reference, or depend on the pipeline-level `PCA` step
in `pca.py`, and SHALL NOT retain any state between calls.

#### Scenario: Projection depends only on X_train, not X_test

- **WHEN** `fit_pca_on_fold(X_train, X_test_a)` and `fit_pca_on_fold(X_train, X_test_b)` are
  called with the same `X_train` but different `X_test_a`/`X_test_b`
- **THEN** both calls SHALL produce projections computed from identical PCA components (i.e. an
  independently-fit `PCA(n_components=n_components).fit(X_train)` applied via `.transform()`
  SHALL reproduce both results exactly)

#### Scenario: Output shape matches n_test and n_components

- **WHEN** `X_train` has shape `(n_train, n_traits)` and `X_test` has shape `(n_test, n_traits)`
- **THEN** the returned array SHALL have shape `(n_test, n_components)`

#### Scenario: Raises before calling sklearn when n_traits < n_components

- **WHEN** `X_train.shape[1] < n_components`
- **THEN** the function SHALL raise `ValueError` without constructing a `PCA` object

#### Scenario: Deterministic for a full-rank X_train

- **WHEN** `fit_pca_on_fold` is called twice with identical `X_train`, `X_test`, and
  `n_components=1`, and `X_train` is full rank
- **THEN** both calls SHALL return identical output

#### Scenario: Inputs are not mutated

- **WHEN** `fit_pca_on_fold(X_train, X_test)` is called
- **THEN** neither `X_train` nor `X_test` SHALL be modified by the call

### Requirement: Leave-One-Genotype-Out Cross-Validated Prediction

The package SHALL provide `logo_cv_predict(X, y, genotypes, reduction_method="pls_latent",
representative_names=None)` in `cross_platform_prediction.py` that predicts each genotype's
target value using a model fit on every other genotype (leave-one-genotype-out cross-validation),
and reports aggregate R², Root Mean Squared Error (RMSE), and Spearman rank correlation (ρ, with
p-value) computed over the concatenated set of leave-one-out predictions across all folds. A
fresh `sklearn.pipeline.Pipeline` SHALL be instantiated and fit inside each fold; no step SHALL be
fit on data that includes the held-out genotype.

`X` SHALL be a `pandas.DataFrame` of shape `(n_genotypes, n_traits)`, with columns named by trait
and index by genotype label — not a bare `numpy.ndarray` — so that `representative_names` (a list
of trait names, taken directly from `select_cluster_representatives()`'s own return type) can be
used to select columns (`X[representative_names]`) without a separate name-to-index resolution
step. Callers SHALL ensure `X`'s columns never include the target trait's own values (an
unenforceable-from-`X`-alone precondition, since `logo_cv_predict` cannot itself distinguish a
predictor column from an accidentally-included target column).

`reduction_method` SHALL support three values:
- `"pls_latent"` (default): a `StandardScaler` + `PLSRegression(n_components=1)` pipeline fit
  directly on the full trait matrix — no separate dimensionality-reduction step.
- `"representatives"`: `X` reduced to the columns named by `representative_names` (selected
  once, before the fold loop, since this is an unsupervised, non-data-leaking selection) before a
  `StandardScaler` + `Ridge()` pipeline is fit. `representative_names` SHALL be required
  (non-`None`) when this method is selected.
- `"pc1"`: `X` reduced to a single principal-component score computed **per fold** via
  `fit_pca_on_fold`, before a `StandardScaler` + `Ridge()` pipeline is fit.

Any other value of `reduction_method` SHALL raise `ValueError`.

#### Scenario: A fresh Pipeline is fit inside each fold, not before the loop

- **WHEN** `logo_cv_predict` runs LOGO-CV over `n` genotypes
- **THEN** a fresh model SHALL be constructed and fit exactly `n` times, each time using only the
  `n - 1` training genotypes for that fold
- **AND** no single fitted model instance SHALL be reused, `set_params()`-mutated, or refit across
  more than one fold

#### Scenario: StandardScaler never sees the held-out genotype

- **WHEN** any fold's `StandardScaler` step is fit
- **THEN** its fit data SHALL exclude that fold's held-out genotype's row

#### Scenario: PLSRegression never sees the held-out genotype's target value

- **WHEN** `reduction_method="pls_latent"` and any fold's `PLSRegression` step is fit
- **THEN** its `y` argument SHALL exclude that fold's held-out genotype's target value

#### Scenario: pls_latent uses a fixed n_components=1

- **WHEN** `reduction_method="pls_latent"`
- **THEN** every fold's `PLSRegression` instance SHALL be constructed with `n_components=1`
- **AND** no inner cross-validation loop SHALL search over alternative component counts

#### Scenario: representative_names are fixed before the fold loop

- **WHEN** `reduction_method="representatives"` with `representative_names` provided
- **THEN** the same `representative_names` SHALL be used to reduce `X` in every fold — no
  per-fold re-selection

#### Scenario: pc1 reduction calls fit_pca_on_fold per fold with that fold's data only

- **WHEN** `reduction_method="pc1"`
- **THEN** `fit_pca_on_fold` SHALL be called with that fold's `X_train`/`X_test` only — never
  with data spanning more than one fold's training set. Per theory.md Section 3.1's documented
  two-call pattern, `fit_pca_on_fold` is called twice per fold: once as
  `fit_pca_on_fold(X_train, X_train, ...)` to reduce the training matrix, once as
  `fit_pca_on_fold(X_train, X_test, ...)` to reduce the held-out genotype — both calls fit a
  fresh `PCA` on the same `X_train` (deterministic, so this is a correctness-neutral, if
  avoidable, double fit — not a leakage risk)

#### Scenario: Output has one prediction per genotype, in input order

- **WHEN** `logo_cv_predict(X, y, genotypes, ...)` completes
- **THEN** the result SHALL contain exactly `len(genotypes)` predictions, ordered identically to
  the input `genotypes` sequence

#### Scenario: Planted-signal fixture recovers a comfortably positive mean R² across repeated realizations

- **WHEN** `logo_cv_predict` runs once per realization of a fixed set of N independent fixture
  realizations, each where `y` is a known linear combination of `X` plus calibrated noise (signal
  strength `s`)
- **THEN** the **mean** aggregate R² over concatenated leave-one-out predictions, averaged across
  all N realizations, SHALL be comfortably positive and within an empirically-established range,
  for both `pls_latent` and `representatives` (or `ridge`-equivalent) reduction methods
- **AND** a single realization's R² is NOT required to individually be close to `s` — LOGO-CV R²
  at this program's sample size (n≈19) has high per-realization variance; only the mean across
  repeated realizations is asserted

#### Scenario: Pure-noise fixture produces a mean R² clearly separated from the signal fixture

- **WHEN** `logo_cv_predict` runs once per realization of a fixed set of N independent fixture
  realizations where `X` and `y` are independently drawn with no planted relationship
- **THEN** the **mean** aggregate R² across all N realizations SHALL be comfortably separated
  (lower, by an empirically-justified margin) from the planted-signal fixture's mean R²
- **AND** this mean is NOT required to equal approximately 0 — LOGO-CV R² on pure noise at small
  `n` can be, and is expected to be, negative (a known, correct property of `r2_score`, not a
  defect)

#### Scenario: Synthetic non-EDPIE fixture generalizes

- **WHEN** `logo_cv_predict` runs on a fixture with a genotype count, trait count, and column
  naming scheme unrelated to the wheat EDPIE dataset, with a known planted signal
- **THEN** the aggregate R² SHALL recover that planted signal similarly to the EDPIE-shaped
  planted-signal fixture, confirming no hidden coupling to EDPIE-specific shapes or names

#### Scenario: RMSE and Spearman ρ are reported alongside R²

- **WHEN** `logo_cv_predict` completes
- **THEN** the result SHALL include RMSE and Spearman ρ (with its p-value), each computed over
  the same concatenated leave-one-out predictions used for R²

### Requirement: Input Validation

`logo_cv_predict` SHALL validate its inputs before entering the fold loop and raise `ValueError`
with a clear message on any of the following, rather than allowing an unhandled or unrelated
exception to surface partway through cross-validation.

#### Scenario: Mismatched array lengths are rejected

- **WHEN** `len(X) != len(y)` or `len(X) != len(genotypes)`
- **THEN** `logo_cv_predict` SHALL raise `ValueError`

#### Scenario: Invalid reduction_method is rejected

- **WHEN** `reduction_method` is not one of `"pls_latent"`, `"representatives"`, `"pc1"`
- **THEN** `logo_cv_predict` SHALL raise `ValueError` naming the valid values

#### Scenario: representatives method without representative_names is rejected

- **WHEN** `reduction_method="representatives"` and `representative_names` is `None`
- **THEN** `logo_cv_predict` SHALL raise `ValueError` upfront, before entering the fold loop

#### Scenario: Too few genotypes for LOGO-CV is rejected

- **WHEN** `len(genotypes) < 2`
- **THEN** `logo_cv_predict` SHALL raise `ValueError`

#### Scenario: NaN in X is rejected

- **WHEN** `X` contains any `NaN` value
- **THEN** `logo_cv_predict` SHALL raise `ValueError` rather than silently fitting on or
  propagating the `NaN` — this is a realistic input, not a hypothetical one, since a
  `08_blup_adjusted_means.csv` failed-trait column (per `extract_blup_table`'s documented
  behavior in the `statistics-api` spec) is entirely `NaN`

#### Scenario: Constant y does not raise

- **WHEN** `y` has zero variance (all identical values)
- **THEN** `logo_cv_predict` SHALL NOT raise — R²/Spearman ρ MAY be `NaN` or otherwise degenerate,
  matching whatever `sklearn`/`scipy`'s own documented behavior is for this input, rather than
  `logo_cv_predict` inventing a new contract for this case

### Requirement: Explicit Leakage Regression Test

The package's test suite SHALL include a standalone test proving that a deliberately-leaked LOGO-CV
implementation (scaler and model fit on the full dataset — including the held-out genotype —
before the fold loop) produces a detectably inflated mean R² relative to the correctly-hygienic
implementation, averaged across the same set of N independent planted-signal fixture realizations
used by the Leave-One-Genotype-Out Cross-Validated Prediction requirement's planted-signal
scenario above.

#### Scenario: Outside-fold-fit mean R² is inflated relative to inside-fold-fit mean R²

- **WHEN** LOGO-CV R² is computed for each of N independent planted-signal fixture realizations,
  twice per realization — once with the scaler and model fit inside each fold
  (`fit_inside_fold=True`), once with both fit on the full dataset before the loop
  (`fit_inside_fold=False`) — and each side's R² is averaged across all N realizations
- **THEN** the ratio `mean(r2_outside_fold) / max(mean(r2_inside_fold), 1e-6)` SHALL be at least
  1.10

#### Scenario: Production code path matches only the inside-fold-fit behavior

- **WHEN** `logo_cv_predict` (the requirement above) is inspected
- **THEN** it SHALL contain no code path equivalent to fitting a scaler or model on data spanning
  more than one fold's training set

### Requirement: Trait-Set Identity Oracle

Cluster-representative trait selection for cross-platform prediction SHALL reuse the existing
`cluster_correlated_traits`/`select_cluster_representatives` functions
(`cross_experiment_analysis.py`) unchanged, applied independently to the **field** and
**cylinder** genotype-mean trait matrices (one row per genotype, raw arithmetic per-genotype
mean — matching `ReduceTraitRedundancyStep`'s `.groupby().mean()`, not a BLUP-adjusted mean; see
design.md Decision 2), at the existing default `threshold=0.8`. Every resulting field
representative trait SHALL then be correlated (Spearman) against every resulting cylinder
representative trait, on the real EDPIE `root_core_vs_cylinder` genotype-mean data. Among the
pairs with `|ρ| >= 0.55`, the count of **distinct** field traits and the count of **distinct**
cylinder traits appearing in those pairs SHALL deterministically reproduce the trait counts
reported in the wheat EDPIE paper's Section 3.4 (14 field traits, 28 cylinder traits) — a
trait-set **identity** check on the correlation-filtered, downstream trait set, not a numeric
correlation/R² threshold and not the raw per-platform representative counts (which are an
intermediate quantity, not the oracle's target).

> **Resolved 2026-07-16 — see design.md Decision 2's resolution and tasks.md task 1.4.** A
> handoff investigation confirmed against the real Mar-30 paper-run artifacts
> (`cross_platform_correlations.csv` in `wheat-edpie-paper/data/cross_platform_field_v2/
> cross_platform_Root_Core_EDPIE_vs_Cylinder_EDPIE_20260330_213908/`) that clustering alone
> produces 22 field / 129 cylinder representatives (not the raw counts of 28/14 the earlier draft
> of this requirement asserted directly), and that correlating all 22×129=2,838 representative
> pairs, filtering to `|ρ|>=0.55` (36 pairs survive), and counting distinct traits per side among
> those 36 pairs reproduces 14 field / 28 cylinder exactly. The investigation also found the
> fixture this repo had already committed for `root_core_vs_cylinder` (28 field / 121 cylinder
> representatives) comes from an unrelated, older 2026-02-12 data vintage — not the paper's own
> run — and must be regenerated from the Mar-30 vintage (task 1.4) before the scenarios below can
> pass.

#### Scenario: Clustering each platform independently reproduces the intermediate representative counts

- **WHEN** `cluster_correlated_traits`/`select_cluster_representatives` is run on the real EDPIE
  field genotype-mean matrix at `threshold=0.8`
- **THEN** the resulting representative-trait count SHALL be 22
- **WHEN** the same is run on the real EDPIE cylinder genotype-mean matrix
- **THEN** the resulting representative-trait count SHALL be 129

#### Scenario: Correlation filtering at |ρ|≥0.55 reproduces the Section 3.4 trait counts

- **WHEN** every field representative trait (22, from the scenario above) is correlated (Spearman)
  against every cylinder representative trait (129), on the real EDPIE `root_core_vs_cylinder`
  genotype-mean data
- **THEN** exactly 2,838 pairs SHALL be tested, of which exactly 36 SHALL have `|ρ| >= 0.55`
- **AND** among those 36 pairs, the count of distinct field traits SHALL be 14 and the count of
  distinct cylinder traits SHALL be 28

#### Scenario: Selection is deterministic given the same input

- **WHEN** `cluster_correlated_traits`/`select_cluster_representatives` is run twice on the same
  genotype-mean matrix
- **THEN** both runs SHALL produce identical cluster assignments and identical representative
  selections
