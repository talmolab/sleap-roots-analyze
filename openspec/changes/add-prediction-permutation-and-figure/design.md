## Context

This is Tier 4 (`add-prediction-permutation-and-figure`) of the wheat EDPIE cross-platform
genotype-prediction program. See the program roadmap and statistical grounding at
`c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md` (external
to this repo; referenced here for provenance only). The full brainstormed design — including this
session's empirical runtime benchmarks — was written and approved by Elizabeth before this
proposal, at
`docs/superpowers/specs/2026-07-16-prediction-permutation-and-figure-design.md`; this file
restates it in this program's established Decisions-with-rationale format for adversarial review.

Tier 3 (`add-cross-platform-prediction`, merged #195, archived
`2026-07-16-add-cross-platform-prediction`) shipped `logo_cv_predict()`, `fit_pca_on_fold()`, and
`CrossPlatformPredictionResult` as stateless Python-API functions, explicitly designed (Tier 3
Decision 3) to support a future permutation loop without refactoring. Tier 3.5
(`add-prediction-pipeline-step`, merged #199, archived `2026-07-17-add-prediction-pipeline-step`)
wired the observed prediction into `CrossPlatformPipeline` as `PredictCrossPlatformStep` (task 6).
Neither tier is in scope to change here except by strictly additive extension.

Prior investigation this session (recorded in the design doc) established:

- Individual `logo_cv_predict()` calls cost ~16ms (`pls_latent`/`representatives`) to ~27ms
  (`pc1`) on a `(19, 15)` synthetic matrix.
- `joblib.Parallel` parallelizing individual permutation calls is measurably **slower** than
  serial at every tested `n_jobs` (4/8/16, both `loky` and `threading` backends, with and without
  BLAS-thread-count pinning) — worker/thread overhead dwarfs a ~16-20ms task.
- Parallelizing across independent **targets** (each worker runs one target's own full, serial
  1000-permutation loop — a ~16s unit of work) measured 3.85× speedup at `n_jobs=8` on a 16-core
  machine; `n_jobs=16` measured worse (oversubscription).
- Real EDPIE target-trait counts per pair range from ~15-24 (Turface19/Field as target) up to
  ~129 (Cylinder as target), per Tier 3.5's own Section 8 manual validation.

## Goals / Non-Goals

- **Goals:** `permutation_test()`/`top_quartile_recovery()` (Python API, `cross_platform_
  prediction.py`, hard-depends on Tier 3 only); `PermutationResult`/
  `CrossPlatformPermutationResult` (`result_types.py`); a `joblib.Parallel`-across-targets runtime
  strategy verified (this session, and again against real data in Section 5's manual gate) to fit
  the roadmap's 30-minute feasibility gate; an additive extension to `PredictCrossPlatformStep`
  exposing its already-computed predictor matrices; `VisualizePredictionStep` as an optional 7th
  pipeline task producing one permutation-results JSON per method and one composite figure per
  pair; a `theory.md` permutation-null pseudo-code addendum.
- **Non-Goals:** any change to `logo_cv_predict`, `fit_pca_on_fold`, `CrossPlatformPredictionResult`,
  `cluster_correlated_traits`, `select_cluster_representatives`, or any of `PredictCrossPlatformStep`'s
  existing (Tier 3.5) behavior/outputs — this tier only adds; `CrossPlatformSummaryGenerator`
  surfacing permutation/prediction results (#197); `/configure-run-all`/`/dry-run`/`/validate-config`
  coverage (#198); heritability-based `representative_selection_metric` (Tier 3.5 Decision 7,
  still deferred — this tier does not revisit it).

## Decisions

### Decision 1: `permutation_test()` is self-contained — computes the observed value itself, not just the null

**What:** `permutation_test(X, y, genotypes, reduction_method="pls_latent",
representative_names=None, n_permutations=1000, random_state=42) -> PermutationResult` first calls
`logo_cv_predict(X, y, genotypes, reduction_method, representative_names)` once, on the real
(unshuffled) `y`, to populate `observed_r2`/`observed_rmse`/`observed_spearman_rho`/
`observed_top_quartile_recovery`. It then draws `n_permutations` independent shuffles of `y`
relative to `genotypes` (a single seeded `numpy.random.Generator(random_state)`, one shuffle per
iteration, no shared mutable state across iterations) and calls `logo_cv_predict()` once per
shuffle, collecting the four metrics' null distributions and computing one-sided p-values
(`p = (#null >= observed + 1) / (n_permutations + 1)`) for R², RMSE, and Spearman ρ.

**Why:** A caller using the Python API directly (this tier's explicit design goal — permutation
testing is Python-API-only, not pipeline-only) should get a complete, self-describing result from
one call, without first calling `logo_cv_predict()` separately purely to obtain the observed value
for comparison. The extra call is cheap (one `logo_cv_predict()` invocation, ~16-27ms) relative to
the `n_permutations` calls that follow. In `VisualizePredictionStep`, this observed value is
expected to exactly reproduce Tier 3.5's already-reported `TargetPrediction` for the same
target/method (both call `logo_cv_predict` with identical inputs, and the function is
deterministic) — enforced as a wiring-correctness regression test (see tasks.md), the same
"wiring correctness, not just existence" pattern as Tier 3.5's own Section 6 oracle.

**Alternatives considered:**
- **`permutation_test()` takes the observed `LOGOCVResult` as a parameter instead of recomputing
  it.** Rejected: saves one ~16-27ms call per invocation at the cost of a less self-contained
  function signature and an extra caller-side responsibility (computing the observed result
  correctly with matching inputs) that has no enforcement if done wrong.

### Decision 2: `top_quartile_recovery()` is a standalone, reusable function

**What:** `top_quartile_recovery(y_true, y_pred, q=None) -> float` computes the fraction of the
true top-`q` genotypes (ranked by `y_true`) that appear in the predicted top-`2q` (ranked by
`y_pred`). `q` defaults to `round(len(y_true) / 4)`. Used both for the observed value (real `y`,
real LOGO-CV predictions) and, once per permutation inside `permutation_test()`, for the null
distribution — where that permutation's *shuffled* `y` is the ground truth for that call, not the
original unshuffled `y` (the null must reflect chance-level recovery under the shuffled labeling,
not recovery of the real ranking against a scrambled prediction).

**Why:** This metric is new in this tier (absent from Tier 3's `LOGOCVResult`/`TargetPrediction`)
and needed identically in two places (observed computation, per-permutation null computation) —
a standalone function avoids duplicating the ranking/recovery logic.

**Alternatives considered:**
- **Inline the recovery calculation separately in the observed-value and null-loop code paths.**
  Rejected: duplicates non-trivial ranking logic for no benefit.

### Decision 3: `CrossPlatformPermutationResult` is a separate result type, not nested inside `CrossPlatformPredictionResult`

**What:** `PermutationResult` (per-target) and `CrossPlatformPermutationResult` (per pair/method,
holding a `predictions: list[PermutationResult]`) mirror `TargetPrediction`/
`CrossPlatformPredictionResult`'s shape and `to_dict()`/`to_json()` convention exactly, but are
new, independent dataclasses in `result_types.py` — not new fields grafted onto
`CrossPlatformPredictionResult`/`TargetPrediction`.

**Why:** Permutation is optional (gated by `PredictionConfig.visualize`) and substantially more
expensive than the observed-result computation it wraps. A caller who only wants the observed
prediction (Tier 3.5's existing, unchanged use case) should not need `CrossPlatformPredictionResult`
to carry permutation fields that are frequently absent/irrelevant. Keeping the types separate also
means Tier 3.5's already-shipped `CrossPlatformPredictionResult`/`TargetPrediction` contract is
untouched by this tier, satisfying the Non-Goals above literally.

**Alternatives considered:**
- **Add `null_r2`/`p_value_r2`/etc. as optional fields directly on `TargetPrediction`.** Rejected:
  would change an already-shipped, already-tested dataclass's shape for every existing consumer
  (even if the new fields default to `None`), and conflates two orthogonal concerns (an observed
  LOGO-CV result vs. a permutation-null significance test) in one type.

### Decision 4: `joblib.Parallel` parallelizes across targets, not across individual permutation calls — empirically verified, not assumed

**What:** `VisualizePredictionStep` (not `permutation_test()` itself, which stays internally
serial) calls `joblib.Parallel(n_jobs=config.prediction.permutation_n_jobs)` over the full list of
`(target_name, method)` units for a pair, each invoking one complete `permutation_test()` call
(its own internal `n_permutations`-length loop stays serial, single-process).
`PredictionConfig.permutation_n_jobs: int = 8` (not `-1`/all-cores).

**Why:** Measured this session (design doc Section 2, reproduced above in Context): parallelizing
individual `logo_cv_predict` calls (~16-27ms each) via `joblib.Parallel` is **slower than serial**
at every tested `n_jobs`, on both `loky` and `threading` backends — process/thread overhead
dwarfs the task. Parallelizing whole per-target permutation loops (~16s units of work) instead
measured 3.85× speedup at `n_jobs=8`; `n_jobs=16` measured *worse* (oversubscription) — so `8`,
not the number of logical cores, is the chosen default. Applying this to the worst real case
(Cylinder-as-target pair, ~129 targets, both `reduction_method` + one `comparison_methods` entry,
N=1000, all 4 pairs): serial ≈105.5 min, parallel (n_jobs=8) ≈27.4 min — under the roadmap's
30-minute gate.

The 27.4-minute figure is a synthetic-fixture extrapolation (`n_traits=15`), not a measurement
against real Cylinder-scale data (hundreds of raw traits pre-clustering) — re-verified against
real EDPIE data in Section 5's manual gate before merge (see Risks).

**Alternatives considered:**
- **Parallelize individual permutation calls (the "obvious" axis).** Empirically measured
  slower than serial at this workload's task size — rejected on direct evidence, not intuition.
- **`n_jobs=-1` (use all logical cores).** Rejected: measured worse than `n_jobs=8` on the same
  16-core machine (oversubscription) — "more cores" is not "faster" for this specific workload
  shape.
- **Reduce the permutation scope (fewer targets, primary-method-only) instead of parallelizing.**
  Considered and rejected by Elizabeth during brainstorming — full per-representative-trait
  permutation nulls were explicitly requested for the paper; documented as the fallback if the
  Section 5 real-data re-measurement exceeds 30 minutes (see Risks), not the primary plan.

### Decision 5: `joblib` becomes an explicit direct dependency

**What:** `joblib` is added to `pyproject.toml`'s dependency list, not left as an implicit
transitive dependency of `scikit-learn`.

**Why:** This tier imports `joblib.Parallel`/`joblib.delayed` directly in
`visualize_prediction.py`. Relying on a transitive pin (which `scikit-learn` could change or drop
in a future version, outside this package's control) for a direct import is fragile.

**Alternatives considered:**
- **Rely on the transitive pin, no `pyproject.toml` change.** Rejected: a direct import of a
  package not declared as a direct dependency is a latent break waiting for an unrelated
  `scikit-learn` version bump.

### Decision 6: `PredictCrossPlatformStep` is extended additively to expose its predictor matrices

**What:** `PredictCrossPlatformStep.execute()`'s returned `StepResult.data` gains a new key,
`predictor_matrices`, holding the already-computed `source_clean`/`target_clean` DataFrames and
`source_representative_names`/`target_representatives` (all computed during the step's own
execution, previously discarded after use). Every existing `data`/`metadata`/`files_generated` key
is unchanged in shape and content.

**Why:** `VisualizePredictionStep` needs the same source/target predictor matrices
`PredictCrossPlatformStep` already builds (BLUP loading, NaN-column dropping, canonical-genotype
alignment, representative-name selection — Tier 3.5 Decisions 2/13/14/16/17) to run permutations.
Duplicating that logic in a second step would create two places to keep in sync if it ever
changes — a real drift risk this program has hit before (Tier 3.5's own Decision 8 bug, where a
"fix" reintroduced a data-pollution risk because the fix wasn't traced through the one place that
mattered). Reusing task 6's already-computed matrices via an additive `StepResult.data` extension
keeps Decisions 2/13/14/16/17's logic in exactly one place.

**Alternatives considered:**
- **`VisualizePredictionStep` independently rebuilds the matrices from `source_blup_path`/
  `target_blup_path`/task 1's raw data.** Rejected: duplicates BLUP-loading/NaN-dropping/alignment
  logic in a second step, the drift risk described above.

### Decision 7: New `PredictionConfig` fields nest on the existing config, not a new sibling

**What:** `PredictionConfig` gains `visualize: bool = False`, `n_permutations: int = 1000`,
`permutation_random_state: int = 42`, `permutation_n_jobs: int = 8`. No new top-level or sibling
config dataclass. `CrossPlatformConfig.__post_init__` gains one more cross-check: `visualize=True`
with `enabled=False` raises `ValueError` at construction time (same plain-`ValueError` convention
as every other Tier 3.5 validation — no new exception type).

**Why:** `VisualizePredictionStep` is only ever meaningful when `PredictCrossPlatformStep` has
actually run (`enabled=True`) — a separate sibling config (e.g. a `VisualizationConfig`) would
just recreate the same enabled-gate coupling `PredictionConfig` already provides, per Tier 3.5
Decision 1's reasoning about nesting matching the actual per-pair workflow (one command, one run
directory, one config file per directed pair).

**Alternatives considered:**
- **A new sibling `VisualizationConfig` field on `CrossPlatformConfig`.** Rejected: introduces a
  second `enabled`-equivalent flag to keep in sync with `prediction.enabled`, for no benefit over
  nesting on the config that already gates the step it depends on.

### Decision 8: `VisualizePredictionStep`'s dependency on task 6 is for both data and ordering

**What:** `VisualizePredictionStep`'s `depends_on=["06_predict_cross_platform"]` supplies both the
`predictor_matrices` and observed `CrossPlatformPredictionResult`(s) data (Decision 6) *and*
guarantees task 6 completes before task 7 runs.

**Why:** Unlike Tier 3.5 Decision 15 (where task 6's second dependency, on task 5, was ordering-
only because task 6 never needed task 5's data), task 7 genuinely needs task 6's data — there is
no analogous ordering-only-dependency ambiguity to resolve here.

**Alternatives considered:** None — this is the direct, un-ambiguous case Tier 3.5 Decision 15 had
to disambiguate away from.

### Decision 9: Figure scope — PC1 scatter + pooled all-targets violin, primary method only, one PNG per pair

**What:** `create_prediction_figure()` (new module, `visualize_prediction.py`) builds one
3-panel `matplotlib.Figure` per directed pair, using only the primary `reduction_method`'s
results (not each `comparison_methods` entry):
1. **Obs-vs-pred scatter** — PC1 target only, one point per genotype.
2. **CV-R²-vs-null strip/violin** — every representative-trait target's observed R² as points,
   overlaid on a violin of the *pooled* null R² distribution (every target's permutation null
   combined into one set).
3. **Top-quartile recovery bar chart** — 2 bars: mean observed recovery rate across all targets
   vs. the corresponding mean null expectation.

Saved as `run_dir / "07_prediction_figure.png"` (`dpi=300, bbox_inches="tight"`, matching
`visualize_cross_platform.py`'s existing convention). `comparison_methods` still get full
permutation JSON output (Decision 10) for the paper's numeric tables/appendix, just not their own
PNG.

**Why:** Elizabeth confirmed this scope directly during brainstorming (design doc Section 4).
PC1 is the one target that is always exactly 1-per-pair and comparable across all 4 directed
pairs, making it the natural "headline" scatter; pooling all targets' nulls into one violin
surfaces the full per-trait breadth the "for the paper" full-scope permutation run was built to
support, in one glance, without needing dozens of per-trait panels.

**Alternatives considered:**
- **PC1 only, for all three panels.** Considered; rejected as the primary design — doesn't
  surface the per-representative-trait breadth the full-scope permutation run exists to support.
- **One figure per (pair, method).** Considered; rejected in favor of one clean headline PNG per
  pair, with `comparison_methods` numeric data still available via the JSON outputs.

### Decision 10: Permutation output naming

**What:** One `CrossPlatformPermutationResult` JSON per method: `07_permutation_<method>.json`
(mirroring Tier 3.5's `06_prediction_<method>.json` naming exactly). One figure per pair:
`07_prediction_figure.png`.

**Why:** Consistent with this pipeline's existing numbered-step-file convention; the method-in-
filename pattern is a direct, unmodified precedent from Tier 3.5.

**Alternatives considered:** None — straightforward continuation of an existing convention.

### Decision 11: Oracle fixtures reuse Tier 3's precedent; the top-quartile-recovery null value is verified, not assumed

**What:**
- **Uniformity oracle:** a K-S calibration test running `permutation_test()` on ~30-50
  independent pure-noise fixtures (real `X`, unrelated `y`), K-S-testing the resulting p-values
  against `Uniform(0,1)`, asserting `p > 0.05`. Uses a reduced `n_permutations` (e.g. 200) to stay
  CI-fast, distinct from the `n_permutations=1000` production default.
- **Signal-separation oracle:** reuses Tier 3's N=20-seed-averaged planted-signal/noise fixtures
  (Tier 3 Decision 6) — asserts mean signal R² is comfortably above that fixture's own
  permutation-null median, and mean noise-fixture R² falls within its own null's median ± 1σ.
- **Top-quartile-recovery oracle:** same fixtures; asserts ≥80% (signal) vs. an **empirically
  verified** null expectation — the roadmap's "≈25%" is a starting estimate, not a value to pin
  blind; the real number is computed and recorded during implementation, the same discipline that
  caught Tier 3 Decision 6's wrong assumed single-seed R².
- **Figure provenance oracle:** PNG exists, non-empty, and a hash/timestamp check confirms it was
  regenerated from the current run's input CSVs, following Tier 3.5's golden-fixture-snapshot
  pattern.

**Why:** This program has twice caught a wrong assumed statistical value by verifying it
empirically instead of trusting a plausible-sounding number (Tier 3 Decision 6's single-seed
fixture; Tier 3.5's data-vintage correction) — the same discipline applies here to a metric with
no prior empirical basis in this codebase.

**Alternatives considered:** None — this is a direct continuation of an established program
convention, not a new design choice.

## Risks / Trade-offs

- **27.4-minute estimate is close to the 30-minute gate and extrapolated from a small synthetic
  fixture (`n_traits=15`), not real Cylinder-scale data.** Must be re-measured during Section 5's
  manual validation before merge. If real timing exceeds 30 minutes, documented fallback: drop
  `comparison_methods` from the *pipeline-computed* permutation JSON (primary method only),
  keeping the full sweep available via direct Python-API calls for anyone who wants it — roughly
  halves the estimate. A second fallback, not preferred: reduce `n_permutations` below 1000 for
  the pipeline-computed run only (Python-API callers keep the 1000 default).
- **`joblib` becomes a direct dependency**, contradicting the program roadmap's Goal-section
  language ("no new dependencies"). Low practical risk (already vendored transitively via
  `scikit-learn`, already imported transitively everywhere `scikit-learn` runs) but disclosed here
  explicitly as a deviation from that stated intent, since it is a real `pyproject.toml` change.
- **Full per-trait permutation scope produces large JSON files** for Cylinder-target pairs
  (~129 targets × 1000 null values × 3 metrics ≈ 387,000 floats per method-pair). Acceptable for a
  research artifact; noted as meaningfully larger than Tier 3.5's `06_prediction_<method>.json`
  files.
- **Pooling all targets' null R² into one violin (Decision 9, panel 2) discards per-target null
  identity in the figure** — a reader cannot tell from the PNG alone which specific traits'
  nulls contribute to which part of the pooled distribution. Mitigated: the full per-target
  breakdown remains in the `07_permutation_<method>.json` files; the figure is a summary view, not
  the complete record.

## Migration Plan

Purely additive when `visualize=False` (the default) — every existing `CrossPlatformConfig` YAML
and every existing Tier 3.5 test is unaffected. `PredictCrossPlatformStep`'s extended
`StepResult.data` is additive (new key only); `CrossPlatformPipeline` gains one new,
conditionally-included task. No existing function, CLI flag, or config field changes shape.

## Open Questions

None blocking prior to `/review-openspec`. Carried forward, not this tier's job: `#197`
(`CrossPlatformSummaryGenerator` not surfacing prediction/permutation results — this tier's new
figures/JSON outputs increase the pressure to fix this, worth flagging again, not fixing here) and
`#198` (`/configure-run-all`/`/dry-run`/`/validate-config` coverage gaps, pre-existing).
