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
representative_indices=None)` in `cross_platform_prediction.py` that predicts each genotype's
target value using a model fit on every other genotype (leave-one-genotype-out cross-validation),
and reports aggregate R², Root Mean Squared Error (RMSE), and Spearman rank correlation (ρ, with
p-value) computed over the concatenated set of leave-one-out predictions across all folds. A
fresh `sklearn.pipeline.Pipeline` SHALL be instantiated and fit inside each fold; no step SHALL be
fit on data that includes the held-out genotype.

`reduction_method` SHALL support three values:
- `"pls_latent"` (default): a `StandardScaler` + `PLSRegression(n_components=1)` pipeline fit
  directly on the full trait matrix — no separate dimensionality-reduction step.
- `"representatives"`: `X` reduced to the columns named by `representative_indices` (selected
  once, before the fold loop, since this is an unsupervised, non-data-leaking selection) before a
  `StandardScaler` + `Ridge()` pipeline is fit.
- `"pc1"`: `X` reduced to a single principal-component score computed **per fold** via
  `fit_pca_on_fold`, before a `StandardScaler` + `Ridge()` pipeline is fit.

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

#### Scenario: representatives indices are fixed before the fold loop

- **WHEN** `reduction_method="representatives"` with `representative_indices` provided
- **THEN** the same `representative_indices` SHALL be used to reduce `X` in every fold — no
  per-fold re-selection

#### Scenario: pc1 reduction calls fit_pca_on_fold once per fold with that fold's data only

- **WHEN** `reduction_method="pc1"`
- **THEN** `fit_pca_on_fold` SHALL be called once per fold with that fold's `X_train`/`X_test`
  only — never with data spanning more than one fold's training set

#### Scenario: Output has one prediction per genotype, in input order

- **WHEN** `logo_cv_predict(X, y, genotypes, ...)` completes
- **THEN** the result SHALL contain exactly `len(genotypes)` predictions, ordered identically to
  the input `genotypes` sequence

#### Scenario: Planted-signal fixture recovers R² near the planted signal strength

- **WHEN** `logo_cv_predict` runs on a fixture where `y` is a known linear combination of `X`
  plus calibrated noise (signal strength `s`)
- **THEN** the aggregate R² over concatenated leave-one-out predictions SHALL be within an
  empirically-established tolerance of `s`, for both `pls_latent` and `representatives` (or
  `ridge`-equivalent) reduction methods

#### Scenario: Pure-noise fixture produces R² near zero

- **WHEN** `logo_cv_predict` runs on a fixture where `X` and `y` are independently drawn with no
  planted relationship
- **THEN** the aggregate R² SHALL be approximately 0, within tolerance

#### Scenario: Synthetic non-EDPIE fixture generalizes

- **WHEN** `logo_cv_predict` runs on a fixture with a genotype count, trait count, and column
  naming scheme unrelated to the wheat EDPIE dataset, with a known planted signal
- **THEN** the aggregate R² SHALL recover that planted signal similarly to the EDPIE-shaped
  planted-signal fixture, confirming no hidden coupling to EDPIE-specific shapes or names

#### Scenario: RMSE and Spearman ρ are reported alongside R²

- **WHEN** `logo_cv_predict` completes
- **THEN** the result SHALL include RMSE and Spearman ρ (with its p-value), each computed over
  the same concatenated leave-one-out predictions used for R²

### Requirement: Explicit Leakage Regression Test

The package's test suite SHALL include a standalone test proving that a deliberately-leaked LOGO-CV
implementation (scaler and model fit on the full dataset — including the held-out genotype —
before the fold loop) produces a detectably inflated R² relative to the correctly-hygienic
implementation, on a planted-signal fixture.

#### Scenario: Outside-fold-fit R² is inflated relative to inside-fold-fit R²

- **WHEN** LOGO-CV R² is computed twice on the same planted-signal fixture — once with the
  scaler and model fit inside each fold (`fit_inside_fold=True`), once with both fit on the full
  dataset before the loop (`fit_inside_fold=False`)
- **THEN** the ratio `r2_outside_fold / max(r2_inside_fold, 1e-6)` SHALL be at least 1.10

#### Scenario: Production code path matches only the inside-fold-fit behavior

- **WHEN** `logo_cv_predict` (the requirement above) is inspected
- **THEN** it SHALL contain no code path equivalent to fitting a scaler or model on data spanning
  more than one fold's training set

### Requirement: Trait-Set Identity Oracle

Cluster-representative trait selection for cross-platform prediction SHALL reuse the existing
`cluster_correlated_traits`/`select_cluster_representatives` functions
(`cross_experiment_analysis.py`) unchanged, applied to a **genotype-mean/BLUP-level** trait
matrix (one row per genotype), at the existing default `threshold=0.8`. On the real EDPIE
cylinder and field genotype-mean matrices, this selection SHALL deterministically reproduce the
same representative trait sets reported in the wheat EDPIE paper's Section 3.4 (28 cylinder + 14
field traits) — a trait-set **identity** check, not a numeric correlation/R² threshold.

#### Scenario: Real EDPIE data reproduces the Section 3.4 trait counts

- **WHEN** `cluster_correlated_traits`/`select_cluster_representatives` is run on the verified
  real EDPIE cylinder genotype-mean matrix at `threshold=0.8`
- **THEN** the resulting representative-trait count SHALL be 28
- **WHEN** the same is run on the verified real EDPIE field genotype-mean matrix
- **THEN** the resulting representative-trait count SHALL be 14

#### Scenario: Selection is deterministic given the same input

- **WHEN** `cluster_correlated_traits`/`select_cluster_representatives` is run twice on the same
  genotype-mean matrix
- **THEN** both runs SHALL produce identical cluster assignments and identical representative
  selections
