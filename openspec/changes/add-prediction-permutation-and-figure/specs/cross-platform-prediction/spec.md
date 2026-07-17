## ADDED Requirements

### Requirement: Permutation Test

The package SHALL provide `permutation_test(X, y, genotypes, reduction_method="pls_latent",
representative_names=None, n_permutations=1000, random_state=42)` in
`src/sleap_roots_analyze/cross_platform_prediction.py` that computes a shuffled-genotype-label
permutation-null significance test for `logo_cv_predict()`'s R², RMSE, and Spearman ρ, plus the
top-quartile recovery metric (see the Top-Quartile Recovery Metric requirement).

The function SHALL first call `logo_cv_predict(X, y, genotypes, reduction_method,
representative_names)` once, on the real (unshuffled) `y`, to obtain the observed R², RMSE,
Spearman ρ, and top-quartile recovery. It SHALL then draw `n_permutations` independent
permutations of `y` relative to `genotypes` (using a single `numpy.random.Generator` seeded with
`random_state`), calling `logo_cv_predict()` once per permutation with the same `X`,
`reduction_method`, and `representative_names`, and computing top-quartile recovery for that
permutation using its own shuffled `y` as ground truth (not the original, unshuffled `y`) against
that same call's leave-one-genotype-out predictions.

The function SHALL return a `PermutationResult` (see the `serializable-result-types` capability)
holding the observed values, the four null distributions (each of length `n_permutations`), and
one-sided p-values for R², RMSE, and Spearman ρ, computed as
`(count(null >= observed) + 1) / (n_permutations + 1)`.

#### Scenario: Observed value matches a direct logo_cv_predict call on the unshuffled y

- **WHEN** `permutation_test(X, y, genotypes, reduction_method)` is called
- **THEN** its returned `observed_r2`/`observed_rmse`/`observed_spearman_rho` SHALL exactly match
  the result of an independent `logo_cv_predict(X, y, genotypes, reduction_method)` call made
  with the same inputs

#### Scenario: Null distributions have length n_permutations

- **WHEN** `permutation_test(..., n_permutations=N)` completes
- **THEN** `null_r2`, `null_rmse`, `null_spearman_rho`, and `null_top_quartile_recovery` SHALL
  each have exactly `N` elements

#### Scenario: Each permutation shuffles y relative to genotypes, not X

- **WHEN** any single permutation iteration runs
- **THEN** `X` and `genotypes` SHALL be passed to that iteration's `logo_cv_predict()` call
  unmodified, and only `y`'s association with `genotypes`/`X`'s rows SHALL be shuffled

#### Scenario: Same random_state produces identical null distributions

- **WHEN** `permutation_test(X, y, genotypes, reduction_method, n_permutations=N,
  random_state=S)` is called twice with identical arguments
- **THEN** both calls SHALL produce bit-identical `null_r2`/`null_rmse`/`null_spearman_rho`/
  `null_top_quartile_recovery` arrays

#### Scenario: Different random_state produces different null distributions

- **WHEN** `permutation_test(X, y, genotypes, reduction_method, n_permutations=N,
  random_state=S1)` and the same call with `random_state=S2` (`S1 != S2`) are both run
- **THEN** the two calls' `null_r2` arrays SHALL NOT be element-wise identical

#### Scenario: A permutation's top-quartile recovery uses that permutation's shuffled y as truth

- **WHEN** one permutation iteration's shuffled `y` is `y_shuffled` and its LOGO-CV predictions
  are `y_pred_shuffled`
- **THEN** that iteration's `null_top_quartile_recovery` entry SHALL equal
  `top_quartile_recovery(y_shuffled, y_pred_shuffled)`, not
  `top_quartile_recovery(y, y_pred_shuffled)` (the original, unshuffled `y`)

#### Scenario: One-sided p-values follow the standard permutation-test formula

- **WHEN** `permutation_test()` completes with observed value `obs` and null distribution `null`
  of length `N` for a given metric
- **THEN** that metric's p-value SHALL equal `(count(v >= obs for v in null) + 1) / (N + 1)`

### Requirement: Top-Quartile Recovery Metric

The package SHALL provide `top_quartile_recovery(y_true, y_pred, q=None)` in
`src/sleap_roots_analyze/cross_platform_prediction.py` that computes the fraction of the true
top-`q` genotypes (ranked by `y_true`, descending) that appear among the predicted top-`2q`
genotypes (ranked by `y_pred`, descending). `q` SHALL default to `round(len(y_true) / 4)` when not
supplied.

#### Scenario: Perfect prediction recovers all top-q genotypes

- **WHEN** `y_pred` is a strictly monotonic function of `y_true` (e.g. `y_pred == y_true`)
- **THEN** `top_quartile_recovery(y_true, y_pred)` SHALL equal `1.0`

#### Scenario: Recovery is computed against the top-2q predicted set, not the top-q predicted set

- **WHEN** `top_quartile_recovery(y_true, y_pred, q=Q)` is called
- **THEN** the denominator of the recovery fraction SHALL be `Q` (the count of true top-`Q`
  genotypes), and the predicted set checked for membership SHALL contain `2 * Q` genotypes
  (the predicted top-`2Q`), not `Q`

#### Scenario: Default q is one quarter of the genotype count, rounded

- **WHEN** `top_quartile_recovery(y_true, y_pred)` is called with `q` omitted and
  `len(y_true) == 19`
- **THEN** the effective `q` used SHALL equal `round(19 / 4)` (`5`)

### Requirement: Permutation Test Input Validation

`permutation_test` SHALL validate its own arguments (`n_permutations`, `random_state`) before
entering the permutation loop, and SHALL surface any `logo_cv_predict` input-validation failure
(invalid `X`/`y`/`genotypes`/`reduction_method`/`representative_names`, per the
`Leave-One-Genotype-Out Cross-Validated Prediction`/`Input Validation` requirements) from its
initial observed-value call, before running any permutation.

#### Scenario: Non-positive n_permutations is rejected

- **WHEN** `n_permutations <= 0`
- **THEN** `permutation_test` SHALL raise `ValueError`, before calling `logo_cv_predict` at all

#### Scenario: Invalid logo_cv_predict inputs surface from the observed-value call

- **WHEN** `X`, `y`, `genotypes`, or `reduction_method` would cause `logo_cv_predict` to raise
  `ValueError` (e.g. mismatched lengths, an invalid `reduction_method`, duplicate genotype labels)
- **THEN** `permutation_test` SHALL raise the same `ValueError`, from its initial observed-value
  call, before entering the permutation loop
