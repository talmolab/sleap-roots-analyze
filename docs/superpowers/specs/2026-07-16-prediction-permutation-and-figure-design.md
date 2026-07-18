# Design: Prediction Permutation Null & Figure Delivery (Tier 4)

**Status:** Approved for OpenSpec proposal drafting.
**Program:** Wheat EDPIE cross-platform genotype-prediction program (Tier 4 of 0-4).
**External context:** `c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md`.
**Tracking issue:** not yet filed — draft during this tier's OpenSpec proposal, get Elizabeth's
go-ahead, then file cross-referencing epic #49 (no new epic), matching Tier 3/3.5's precedent.

## Why

Tier 3 (`add-cross-platform-prediction`, merged #195) shipped `logo_cv_predict()`,
`fit_pca_on_fold()`, and `CrossPlatformPredictionResult` as stateless Python-API functions,
deliberately written to support a future permutation loop without refactoring (Tier 3's own
Decision 3). Tier 3.5 (`add-prediction-pipeline-step`, merged #199) wired the observed
(non-permuted) prediction into `CrossPlatformPipeline` as an optional `PredictCrossPlatformStep`.

Neither tier delivers statistical significance (is a given R² distinguishable from chance at
n≈19?) or a paper-ready figure. This tier closes both gaps: a permutation-null significance test
(Python API, hard-depends on Tier 3 only) and `VisualizePredictionStep` (a new optional pipeline
step, soft-depends on Tier 3.5) that renders a 3-panel figure per directed pair.

## Goals / Non-Goals

- **Goals:** `permutation_test()` and `top_quartile_recovery()` (Python API, `cross_platform_
  prediction.py`); `PermutationResult`/`CrossPlatformPermutationResult` (`result_types.py`);
  a `joblib.Parallel`-based runtime strategy verified to fit the roadmap's 30-minute feasibility
  gate at real EDPIE scale; `VisualizePredictionStep` as an optional 7th pipeline task; a
  3-panel `prediction_figure.png` per pair; `theory.md`'s permutation-null pseudo-code addendum
  (Tier 0 erratum, this tier's contribution).
- **Non-Goals:** any change to `logo_cv_predict`, `fit_pca_on_fold`, `CrossPlatformPredictionResult`,
  or `PredictCrossPlatformStep`'s existing (Tier 3.5) behavior/outputs — only additive extensions;
  any change to `CrossPlatformSummaryGenerator` (still #197's territory); any change to
  `/configure-run-all`/`/dry-run`/`/validate-config` (still #198's territory, pre-existing).

## Section 1: Python-API additions (Tier 3 module, no pipeline dependency)

`cross_platform_prediction.py` (Tier 3's existing, stateless module) gains:

- **`permutation_test(X, y, genotypes, reduction_method="pls_latent", representative_names=None,
  n_permutations=1000, random_state=42) -> PermutationResult`.** Shuffles `y` relative to
  `genotypes` `n_permutations` times (a fixed, seeded `numpy.random.Generator`, one independent
  shuffle per iteration), calling `logo_cv_predict()` once per shuffle (each call already runs the
  internal 19-fold LOGO-CV loop — "N=1000 permutations" means 1000 `logo_cv_predict` calls, not
  1000×19). Returns the full null distributions for R², RMSE, and Spearman ρ, plus one-sided
  p-values for each (`p = (#null ≥ observed + 1) / (n_permutations + 1)`). Each permutation's
  `top_quartile_recovery` null value (see below) is computed with that permutation's *shuffled* `y`
  as ground truth against that same call's LOGO-CV predictions — not the original, unshuffled `y`
  — so the null reflects chance-level recovery under the shuffled labeling, not recovery of the
  real ranking.
- **`top_quartile_recovery(y_true, y_pred, q=None) -> float`.** The roadmap's settled metric:
  fraction of the true top-`q` genotypes (by `y_true`) that appear in the predicted top-`2q` (by
  `y_pred`). `q` defaults to `round(n / 4)`. Used for both the observed value (on real LOGO-CV
  predictions) and, per-permutation, the null distribution.

`permutation_test()` is self-contained: given the *real* (unshuffled) `y`, it first calls
`logo_cv_predict(X, y, genotypes, reduction_method, representative_names)` once to populate
`observed_r2`/`rmse`/`spearman_rho`/`top_quartile_recovery`, then runs the `n_permutations` shuffled
calls for the null distributions — a caller gets a complete result from one call, without needing
a separate `logo_cv_predict()` call first for convenience. In `VisualizePredictionStep`, this
observed value is expected to exactly reproduce task 6's already-reported `TargetPrediction`
(both call `logo_cv_predict` with identical inputs) — the wiring test in the Testing Plan below
cross-checks this, the same "wiring correctness, not just existence" pattern as Tier 3.5's own
Section 6 oracle.

`result_types.py` gains:

- **`PermutationResult`** (frozen dataclass, same finite-floats/`to_json` convention as
  `LOGOCVResult`/`TargetPrediction`): `target_name`, `observed_r2`/`rmse`/`spearman_rho`/
  `top_quartile_recovery`, `null_r2`/`null_rmse`/`null_spearman_rho`/`null_top_quartile_recovery`
  (each length `n_permutations`), `p_value_r2`/`p_value_rmse`/`p_value_spearman_rho`,
  `n_permutations`.
- **`CrossPlatformPermutationResult`** (mirrors `CrossPlatformPredictionResult`):
  `source_platform`, `target_platform`, `reduction_method`, `predictions: list[PermutationResult]`.
  Kept **separate** from (not nested inside) `CrossPlatformPredictionResult` — permutation is
  optional and heavier than the observed-result concern; a caller who only wants observed values
  should not need to know permutation exists.

## Section 2: Parallelization strategy & runtime

Empirically measured this session, using the real `logo_cv_predict` on a `(19, 15)` synthetic
matrix:

- Individual `logo_cv_predict` calls cost ~16ms (`pls_latent`/`representatives`) to ~27ms (`pc1`).
- Parallelizing **individual permutation calls** via `joblib.Parallel` (`loky` backend, batched or
  unbatched) is **slower than serial** at every tested `n_jobs` (4/8/16) — worker spawn/pickling
  overhead dwarfs a ~16-20ms task.
- Parallelizing **across independent targets** (each worker runs one target's own full,
  serial 1000-permutation loop — a ~16s unit of work per target) measured **3.85× speedup at
  `n_jobs=8`** on this 16-core machine; more workers (`n_jobs=16`) get worse, not better
  (oversubscription).

Applying the `n_jobs=8`/per-target strategy to the worst real case (Cylinder-as-target pair has
~129 representative traits per Tier 3.5's Section 8 manual validation; both the primary
`reduction_method` and one `comparison_methods` entry; N=1000; all 4 directed pairs):

| | Serial | Parallel (n_jobs=8, per-target) |
|---|---|---|
| Total estimated wall time | ~105.5 min | **~27.4 min** |

This is under the roadmap's 30-minute feasibility gate, but not by a wide margin — flagged as a
risk (see Risks) to re-verify against real Cylinder-scale data (hundreds of raw traits before
clustering, not this session's 15-trait synthetic benchmark) during this tier's manual validation.

**Design:** `PredictionConfig.permutation_n_jobs: int = 8` (not `-1`/all-cores — matches the
measured optimum for this workload shape, not a generic "use all cores" default).
`VisualizePredictionStep` calls `joblib.Parallel(n_jobs=config.prediction.permutation_n_jobs)`
over the full list of `(target_name, method)` units, each invoking `permutation_test()` once (that
function's own internal loop stays serial). `joblib` becomes a direct import (previously only a
transitive `scikit-learn` dependency) — add it explicitly to `pyproject.toml`'s dependency list
rather than relying on the transitive pin.

`theory.md` (Tier 0) gets a new section: permutation-null pseudo-code (matching its existing
LOGO-CV/per-fold-PCA pseudo-code style), documenting the shuffle-and-refit procedure and the
per-target (not per-permutation) parallelization strategy above, since this session's benchmark
result is a genuinely counter-intuitive lesson (the "obvious" parallelization axis is the wrong
one) worth carrying forward for any future tier that loops `logo_cv_predict`.

## Section 3: Pipeline wiring

**Extend `PredictCrossPlatformStep`** (Tier 3.5, already merged) additively: `StepResult.data`
gains a new key, `predictor_matrices`, holding the already-computed `source_clean`/`target_clean`
DataFrames and `source_representative_names`/`target_representatives` (all computed during task
6's own execution, previously discarded after use). No existing `data`/`metadata`/
`files_generated` key changes shape — fully backward-compatible with every Tier 3.5 test.

**New `VisualizePredictionStep`** (`pipeline/steps/visualize_prediction.py`), optional 7th task on
`CrossPlatformPipeline`. `depends_on=["06_predict_cross_platform"]` — for both data *and*
ordering this time (unlike Tier 3.5 Decision 15's ordering-only second dependency; this step
genuinely reads task 6's `predictor_matrices` and observed results). Entirely absent from
`create_tasks()` unless `config.prediction.visualize` is `True` (not merely skipped — matching
Tier 3.5's `enabled`-gating precedent).

Step behavior:
1. Read `predictor_matrices` and the observed `CrossPlatformPredictionResult`(s) from task 6.
2. For every `(target_name, method)` combination (both `reduction_method` and every
   `comparison_methods` entry), run `permutation_test()` via the `joblib.Parallel`-over-targets
   strategy from Section 2.
3. Save one `CrossPlatformPermutationResult` JSON per method: `07_permutation_<method>.json`.
4. Build the composite 3-panel figure (Section 4) using only the **primary** `reduction_method`'s
   results (not every `comparison_methods` entry), save `07_prediction_figure.png`.

**Config** — new fields nested on `PredictionConfig` (not a new sibling config — `visualize` only
ever makes sense when `enabled=True`, so a separate config would just recreate the same gate):
`visualize: bool = False`, `n_permutations: int = 1000`, `permutation_random_state: int = 42`,
`permutation_n_jobs: int = 8`. `CrossPlatformConfig.__post_init__` gains one more cross-check:
`visualize=True` with `enabled=False` raises `ValueError` at construction time — same plain-
`ValueError` convention as every other Tier 3.5 validation (no new exception type).

## Section 4: Figure content & oracle strategy

**Figure content.** New module `sleap_roots_analyze/visualize_prediction.py` (parallel to
`cross_experiment_analysis.py`'s home for `create_correlation_summary_plot` etc. — plotting logic
as plain, independently-testable functions; the Step only orchestrates + saves), with
`create_prediction_figure(...) -> matplotlib.Figure`, 3 subplot panels:

1. **Obs-vs-pred scatter** — PC1 target only (single scalar per genotype, comparable across all 4
   pairs), one point per genotype, from the primary method's observed `CrossPlatformPredictionResult`.
2. **CV-R²-vs-null strip/violin** — every representative-trait target's observed R² plotted as
   points/strip, overlaid on a violin of the *pooled* null R² distribution (every target's
   permutation null combined into one set) for the primary method — gives the reader the whole
   real-vs-null picture in one glance, motivated by the "for the paper" full per-trait scope.
3. **Top-quartile recovery bar chart** — 2 bars: mean observed recovery rate across all targets vs.
   the corresponding mean null expectation.

Saved as `run_dir / "07_prediction_figure.png"` (`dpi=300, bbox_inches="tight"`, matching
`visualize_cross_platform.py`'s existing convention exactly). One figure per pair (not one per
method) — `comparison_methods` still get full permutation JSON data (Section 3, step 3) for the
paper's numeric tables/appendix, just not their own PNG.

**Oracles** (concretizing the roadmap's Tier 4 row):

- **Permutation p-value uniformity.** A K-S calibration test: run `permutation_test()` on ~30-50
  independent pure-noise fixtures (real `X`, unrelated `y`), collect the resulting p-values,
  K-S-test them against `Uniform(0,1)`, assert `p > 0.05`. Uses a reduced `n_permutations` (e.g.
  200) to stay CI-fast — distinct from the `n_permutations=1000` production default, mirroring
  Tier 3.5's CI-fast-synthetic-fixture-vs.-manual-validation split.
- **Signal separation.** Reuses Tier 3's N=20-seed-averaged planted-signal/noise fixtures (Tier 3
  Decision 6's precedent — a single realization's LOGO-CV R² is too noisy at n≈19 to pin a
  threshold against). Assert mean signal R² is comfortably above that signal fixture's own
  permutation-null median; assert mean noise-fixture R² falls within its own null's
  median ± 1σ.
- **Top-quartile recovery.** Same planted-signal/noise fixtures: assert ≥80% (signal) vs. an
  **empirically-verified** null expectation — the roadmap's "≈25%" is a starting estimate, not a
  value to pin blind; verify the real number via actual computation during implementation, per
  this program's established "don't assume, verify" convention (Tier 3 Decision 6's own precedent
  of catching a wrong assumed R² this same way).
- **Figure provenance.** PNG exists, non-empty, and a hash/timestamp check confirms it was
  regenerated from the current run's input CSVs (not stale from a prior run) — a CI-fast
  synthetic-fixture check, following Tier 3.5's golden-fixture-snapshot pattern.

## Risks / Trade-offs

- **27.4-minute estimate is close to the 30-minute gate and extrapolated from a small synthetic
  fixture (`n_traits=15`), not real Cylinder-scale data (hundreds of raw traits pre-clustering).**
  Must be re-measured against real EDPIE data during this tier's manual validation (Section-8-
  equivalent) before merge; if real timing exceeds 30 minutes, the fallback is documented here:
  reduce `n_jobs` search further, or scope the pipeline-computed permutation (not the Python-API
  function itself) to primary-method-only (drop the `comparison_methods` sweep from the
  per-pipeline-run permutation JSON, keeping it available via direct Python-API calls for anyone
  who wants it), roughly halving the estimate.
- **`joblib` becomes a direct dependency**, not merely transitive via `scikit-learn`. Low risk
  (already vendored, already imported transitively everywhere `scikit-learn` is used) but is a
  `pyproject.toml` change worth calling out explicitly, since the roadmap's Goal section states
  "no new dependencies" for the overall program.
- **Full per-trait permutation scope (chosen for the paper) produces large JSON files** for
  Cylinder-target pairs (~129 targets × 1000 null values × 3 metrics ≈ 387,000 floats per
  method-pair). Acceptable for a research artifact, but worth noting these JSONs will be
  meaningfully larger than Tier 3.5's `06_prediction_<method>.json` files.
- **Pooling all targets' null R² into one violin (figure panel 2) discards per-target null
  identity** — a reader cannot tell from the figure alone which specific traits' nulls contribute
  to which part of the pooled distribution. Acceptable since the full per-target breakdown remains
  available in the `07_permutation_<method>.json` files for anyone who needs it; the figure is a
  summary view, not the complete record.

## Section 5: Manual real-data validation gate (non-CI, pre-merge, sign-off required)

Following Tier 3's Section 8 and Tier 3.5's Section 8 precedent exactly: a non-CI validation task
against real EDPIE data, required before merge, with Elizabeth's explicit sign-off — not merely a
CI-green signal. This is also where this design's biggest open risk (the 27.4-minute estimate,
Section 2/Risks) gets resolved with real numbers instead of a synthetic-fixture extrapolation.

1. **Build/reuse real inputs.** Reuse Tier 3.5's Section 8 real BLUP tables and 4 directed-pair
   `CrossPlatformConfig` YAMLs (`pipeline_runs/section8_manual_validation_20260716/`) if still
   valid; rebuild via the same non-committed script pattern if the underlying QC vintage has moved.
   Extend each YAML's `prediction:` block with `visualize: true`, `n_permutations: 1000`,
   `permutation_n_jobs: 8` (the design defaults — no per-run override needed unless the timing
   check below says otherwise).
2. **Run all 4 directed pairs** through the full 7-task pipeline (`sleap-roots-analyze
   cross-platform <config>.yaml`), including `Turface19→Cylinder` — the worst-case pair (Cylinder
   as target, ~129 representative traits per Tier 3.5's own Section 8 finding).
3. **Re-measure real wall time**, per pair and total, and record it here (superseding the
   synthetic-fixture-derived 27.4-minute estimate). If any pair — or the total — exceeds the
   roadmap's 30-minute gate, apply one of the documented fallbacks (Risks section) *before*
   proceeding to sign-off, not after.
4. **Sanity-check permutation-based p-values against Tier 3.5's Section 8.2/8.3 findings**, which
   used `scipy.stats.spearmanr`'s asymptotic p-value (documented, per Tier 3 Decision 9, as
   imprecise below n≈20-30 — exactly the gap Tier 4's permutation null exists to close). In
   particular: do the handful of nominally-significant real hits Tier 3.5 found (e.g.
   Cylinder→Field `Root Count 20cm`, asymptotic R²=0.25/ρ=+0.49/p=0.033; Turface19→Cylinder
   `Seminal Angles Proximal Max Max`, asymptotic R²=0.38/ρ=+0.62/p=0.004) still look significant
   under the permutation-based p-value? A large divergence between asymptotic and permutation
   p-values on these specific, already-flagged targets would be a meaningful finding for the paper,
   not just a sanity check.
5. **Cross-check against Tier 3.5's Section 8.5 multiple-testing/power caveat** (raw p<0.05 count
   63/354, FDR-corrected count 9/354, all 9 negative ρ). Does permutation-based inference reinforce
   or change that caveat's conclusion? Record the answer — this is the kind of finding this
   program's own Section 8 process has surfaced before (Tier 3.5's own vintage-correction and
   `Computation.Time.s`-exclusion fixes both came from exactly this kind of close manual look).
6. **Visually inspect all 4 `07_prediction_figure.png` outputs** for legibility and correctness
   (readable axis labels/titles, PC1 scatter shows genotype-level points, violin/strip panel shows
   a visible real-vs-null gap or lack thereof, bar chart's two bars are distinguishable).
7. **Sign-off requirement:** findings presented to Elizabeth; this task is not complete until she
   has reviewed and explicitly signed off, mirroring Tier 3/3.5's `/pre-merge-check` gate — a green
   CI run is necessary but not sufficient.

## Testing Plan (outline — full breakdown belongs in tasks.md)

1. `permutation_test()`/`top_quartile_recovery()` unit tests (input validation, determinism given
   a fixed seed, shape/length contracts) — Python API, no pipeline dependency.
2. `PermutationResult`/`CrossPlatformPermutationResult` serialization round-trip tests, matching
   the `to_json`/finite-floats-contract pattern of every sibling result type.
3. The four oracle tests from Section 4 above (K-S calibration, signal separation, top-quartile
   recovery, figure provenance).
4. `PredictCrossPlatformStep`'s additive `predictor_matrices` extension — a backward-compat
   regression test confirming existing `data`/`metadata`/`files_generated` keys are unchanged.
5. `VisualizePredictionStep` wiring tests (task presence/absence gated by `visualize`,
   `visualize=True` + `enabled=False` rejected, joblib parallelization produces identical results
   to a serial reference run for a small fixture, JSON + PNG both produced, and each
   `PermutationResult`'s `observed_*` fields exactly reproduce task 6's already-reported
   `TargetPrediction` values for the same target/method).
6. `CrossPlatformPipeline` task-list tests (7th task present/absent).
7. Section 5's manual real-data validation gate, non-CI, before merge — see above.
