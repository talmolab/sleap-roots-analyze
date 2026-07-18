## ADDED Requirements

### Requirement: Permutation Test

The package SHALL provide `permutation_test(X, y, genotypes, reduction_method="pls_latent",
representative_names=None, n_permutations=1000, random_state=42)` in
`src/sleap_roots_analyze/cross_platform_prediction.py` that computes a shuffled-genotype-label
permutation-null significance test for `logo_cv_predict()`'s R², RMSE, and Spearman ρ, plus the
top-quartile recovery metric (see the Top-Quartile Recovery Metric requirement). `random_state`
SHALL accept anything `numpy.random.default_rng()` itself accepts — an `int`, a
`numpy.random.SeedSequence`, or a `numpy.random.Generator` — since the caller (see the
`cross-platform-analysis` capability's Visualize Cross-Platform Prediction Pipeline Step
requirement) derives per-`(target, method)` seeds as `numpy.random.SeedSequence` children via
`SeedSequence(...).spawn(N)`, not as plain integers, and passes each child through unchanged
(`numpy.random.default_rng` accepts a `SeedSequence` directly — no int-extraction step exists or
is needed). **Reproducibility caveat, `int`/`SeedSequence` vs. `Generator`:** `int` and
`SeedSequence` are stateless value-seeds — reusing the same value across two calls always
reproduces the same null draws (the "Same random_state produces identical null distributions"
scenario below). A `Generator` instance is **mutable and stateful**
(`numpy.random.default_rng(existing_generator)` returns that exact instance unaltered, per numpy's
own documented behavior) — passing the *same* `Generator` instance to two separate
`permutation_test()` calls advances its internal state between calls and will **not** reproduce
the same null draws the second time. This function accepts a `Generator` for flexibility, but only
`int`/`SeedSequence` inputs carry a reproducibility guarantee; `VisualizePredictionStep` never
passes a bare `Generator` for exactly this reason (only `SeedSequence` children).

The function SHALL first call `logo_cv_predict(X, y, genotypes, reduction_method,
representative_names)` once, on the real (unshuffled) `y`, to obtain the observed R², RMSE,
Spearman ρ, and top-quartile recovery. It SHALL then construct one `numpy.random.Generator` via
`numpy.random.default_rng(random_state)` (not `numpy.random.Generator(random_state)` directly,
which requires a `BitGenerator` instance and rejects a bare `int`) and draw `n_permutations`
independent permutations of `y` relative to `genotypes` from that one generator, calling
`logo_cv_predict()` once per permutation with the same `X`, `reduction_method`, and
`representative_names`, and computing top-quartile recovery for that permutation using its own
shuffled `y` as ground truth (not the original, unshuffled `y`) against that same call's
leave-one-genotype-out predictions.

The function SHALL return a `PermutationResult` (see the `serializable-result-types` capability)
holding the observed values, the four null distributions (each of length `n_permutations`), and
one-sided p-values for R², RMSE, and Spearman ρ — **not the same formula for all three**, since
"more extreme" points in opposite directions for RMSE than for R²/ρ:
- **R² and Spearman ρ** (higher is better prediction): `p = (count(null >= observed) + 1) /
  (n_permutations + 1)` — an observed value in the right tail of the null is evidence of real
  signal.
- **RMSE** (lower is better prediction): `p = (count(null <= observed) + 1) / (n_permutations +
  1)` — an observed value in the *left* tail of the null is evidence of real signal. Using the
  R²/ρ-style formula for RMSE would invert the result: a genuinely good (low-RMSE) model would
  read as non-significant (`p ≈ 1`), and a genuinely bad (high-RMSE) model would read as highly
  significant (`p ≈ 0`).

A permutation's LOGO-CV fold structure can independently produce a non-finite `spearman_rho` (e.g.
a degenerate fold whose predictions are constant) even when that permutation's shuffled `y` is
itself non-constant — this is a model-degeneracy event distinct from, and not ruled out by, the
data-degeneracy check `PredictCrossPlatformStep` already performs on *observed* values **before
that pipeline step calls this function**. That upstream guard does not exist for a direct
Python-API caller (Decision 1's own explicit design goal — this function is Python-API-only, not
pipeline-only, and is self-contained precisely so a caller doesn't need a separate upstream check):
a caller passing data with a constant `y` or a model that degenerates to constant predictions on
the real, unshuffled call gets exactly the same non-finite-`spearman_rho`/`r2` risk, on the
*observed* side, that this requirement already addresses for the null side. The function SHALL
therefore scan **both** the observed values (`observed_r2`/`observed_rmse`/`observed_spearman_rho`)
**and** every null distribution (`null_r2`/`null_rmse`/`null_spearman_rho`/
`null_top_quartile_recovery`) for non-finite entries before returning, and SHALL raise `ValueError`
naming the offending metric and, for a null-side failure, the 0-indexed permutation number(s) at
which it occurred (for an observed-side failure, naming that it is the observed value, not a
permutation index) — rather than allowing a non-finite value to propagate into
`PermutationResult`/`CrossPlatformPermutationResult.to_json()`'s `allow_nan=False` contract as an
unnamed failure. The observed-side check SHALL run first (cheap, one call, fails immediately
rather than after `n_permutations` further calls that would be wasted on unusable input); the
null-side scan runs **after** all `n_permutations` calls complete, not fail-fast on the first
occurrence — chosen for implementation simplicity (one deterministic pass reporting every
offending index at once, rather than a per-iteration check that changes the function's control
flow) and because a genuinely non-finite-producing bug is expected to affect many permutations
within one target, not a rare one-off, so failing fast saves little wall-clock time in the case
that matters, while complicating the accounting of *which* permutations were and weren't run.

#### Scenario: Observed value matches a direct logo_cv_predict call on the unshuffled y

- **WHEN** `permutation_test(X, y, genotypes, reduction_method)` is called
- **THEN** its returned `observed_r2`/`observed_rmse`/`observed_spearman_rho` SHALL exactly match
  the result of an independent `logo_cv_predict(X, y, genotypes, reduction_method)` call made
  with the same inputs
- **AND** `observed_top_quartile_recovery` SHALL exactly equal
  `top_quartile_recovery(y, that_call.y_pred)` — derived from the same observed call's predictions,
  not a separately-shuffled or stale value

#### Scenario: Null distributions have length n_permutations

- **WHEN** `permutation_test(..., n_permutations=N)` completes
- **THEN** `null_r2`, `null_rmse`, `null_spearman_rho`, and `null_top_quartile_recovery` SHALL
  each have exactly `N` elements

#### Scenario: Each permutation shuffles y relative to genotypes, not X

- **WHEN** any single permutation iteration runs
- **THEN** `X` and `genotypes` SHALL be passed to that iteration's `logo_cv_predict()` call
  unmodified, and only `y`'s association with `genotypes`/`X`'s rows SHALL be shuffled

#### Scenario: Same int or SeedSequence random_state produces identical null distributions

- **WHEN** `permutation_test(X, y, genotypes, reduction_method, n_permutations=N,
  random_state=S)` is called twice, in the same process, with identical arguments, where `S` is
  either a plain `int` or a `numpy.random.SeedSequence` value (parametrize both)
- **THEN** both calls SHALL produce bit-identical `null_r2`/`null_rmse`/`null_spearman_rho`/
  `null_top_quartile_recovery` arrays — this in-process determinism guarantee is distinct from the
  `cross-platform-analysis` capability's cross-`joblib`-worker-process tolerance requirement
  (`assert_allclose(rtol=1e-6, atol=1e-9)`, not bit-identical), which concerns a different variable
  (worker-process BLAS thread count), not this function's own determinism. **Not tested for a bare
  `Generator` input**, which is stateful and does not carry this guarantee (see the requirement
  body's reproducibility caveat above) — passing the same `Generator` instance twice is expected to
  produce *different* results, not identical ones.

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

#### Scenario: One-sided p-values follow the standard permutation-test formula, per metric direction

- **WHEN** `permutation_test()` completes with observed value `obs` and null distribution `null`
  of length `N`, for R² or Spearman ρ (higher is better)
- **THEN** that metric's p-value SHALL equal `(count(v >= obs for v in null) + 1) / (N + 1)`
- **WHEN** the same completes for RMSE (lower is better)
- **THEN** `p_value_rmse` SHALL equal `(count(v <= obs for v in null) + 1) / (N + 1)` — the
  opposite-tail formula, not the R²/ρ formula

#### Scenario: A good (low-RMSE) result does not read as non-significant

- **GIVEN** a planted-signal fixture where the observed RMSE is comfortably below the bulk of the
  null RMSE distribution (a genuinely good model)
- **WHEN** `permutation_test()` computes `p_value_rmse`
- **THEN** `p_value_rmse` SHALL be small (comfortably below `0.5`), not large — proving the
  left-tail formula is actually in effect, not the right-tail formula silently inverted

#### Scenario: Non-finite null values are rejected with a named, actionable error

- **GIVEN** a permutation iteration whose LOGO-CV fold structure produces a non-finite
  `spearman_rho` (e.g. a degenerate fold with constant predictions), independent of whether that
  permutation's shuffled `y` was itself constant
- **WHEN** `permutation_test()` finishes drawing all `n_permutations` null values
- **THEN** it SHALL raise `ValueError` naming the offending metric (`r2`/`rmse`/`spearman_rho`/
  `top_quartile_recovery`) and the permutation index/indices at which a non-finite value occurred,
  rather than returning a `PermutationResult` that would later fail
  `CrossPlatformPermutationResult.to_json()`'s finite-floats contract with no indication of which
  permutation caused it

#### Scenario: Non-finite observed values are rejected before any permutation runs

- **GIVEN** `y` is constant (zero variance), or the observed `logo_cv_predict` call otherwise
  produces a non-finite `spearman_rho` — a legal `logo_cv_predict` input/output (per that
  function's own "Constant y does not raise" contract), reachable by a direct Python-API caller
  with no upstream pipeline guard
- **WHEN** `permutation_test()` runs
- **THEN** it SHALL raise `ValueError` naming the offending observed metric, immediately after the
  observed-value call and before drawing any of the `n_permutations` null values (not after
  wastefully completing the full permutation loop on data already known to be unusable)

### Requirement: Top-Quartile Recovery Metric

The package SHALL provide `top_quartile_recovery(y_true, y_pred, q=None)` in
`src/sleap_roots_analyze/cross_platform_prediction.py` that computes the fraction of the true
top-`q` genotypes (ranked by `y_true`, descending) that appear among the predicted top-`2q`
genotypes (ranked by `y_pred`, descending). `q` SHALL default to `round(len(y_true) / 4)` when not
supplied.

Under a random (uninformative) `y_pred`, the expected recovery is `2q / len(y_true)` by linearity
of expectation (a fixed top-`q` set has, in expectation, a `2q / len(y_true)` fraction of its
members appear in a randomly-drawn top-`2q` set) — at this program's real scale
(`len(y_true) ≈ 15-24`, `q = round(len(y_true) / 4)`), this ranges **≈44-55%** depending on `n` and
Python's banker's rounding (e.g. `n=18 → q=4 → 44.4%`; `n=19 → q=5 → 52.6%`; `n=22 → q=6 → 54.5%`
— computed directly for every `n` in this range, not a single illustrative figure), not the
roadmap's originally-estimated "≈25%". The permutation null's actual mean recovery under
`permutation_test()` (see the Permutation Test requirement's oracle tests) SHALL be verified
empirically against this `2q / n` expectation during implementation, not assumed from the
roadmap's earlier, unverified estimate.

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

#### Scenario: Small n does not produce a zero or degenerate q

- **WHEN** `top_quartile_recovery(y_true, y_pred)` is called with `q` omitted and `len(y_true)` at
  this program's smallest real scale (`n=3`, `logo_cv_predict`'s own minimum)
- **THEN** the effective `q` used SHALL be at least `1` (not `0`, which would make the metric
  vacuously `0/0`-undefined), and `2q` SHALL NOT exceed `len(y_true)`

#### Scenario: An explicitly-supplied invalid q is rejected

- **WHEN** a caller explicitly passes `q=0`, a negative `q`, or a `q` such that `2 * q >
  len(y_true)`
- **THEN** `top_quartile_recovery` SHALL raise `ValueError` naming the invalid `q` and
  `len(y_true)`, rather than silently computing a vacuous or out-of-range recovery fraction — this
  is a stricter contract than the *default*-`q` case above, which the function itself computes and
  guarantees valid

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
