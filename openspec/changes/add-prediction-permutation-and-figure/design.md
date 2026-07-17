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
relative to `genotypes` (a single generator built via `numpy.random.default_rng(random_state)` —
**not** `numpy.random.Generator(random_state)` directly, which requires a `BitGenerator` instance
and rejects a bare `int`; `default_rng` accepts `int`, `SeedSequence`, or `Generator` uniformly,
which is what lets `VisualizePredictionStep` pass per-target `SeedSequence` children through with
no int-extraction step — see round 2's reconciliation below, this replaces an invalid API call an
earlier draft of this decision specified — one shuffle per iteration, no shared mutable state
across iterations) and calls `logo_cv_predict()` once per shuffle, collecting the four metrics'
null distributions and computing one-sided p-values
(`p = (#null >= observed + 1) / (n_permutations + 1)`) for R², RMSE, and Spearman ρ.

**Added during `/review-openspec` round 1:** a permutation's LOGO-CV fold structure can
independently produce a non-finite `spearman_rho` (a model-degeneracy event — e.g. a degenerate
fold with constant predictions) even when that permutation's shuffled `y` is itself non-constant;
this is distinct from, and not ruled out by, the data-degeneracy check `PredictCrossPlatformStep`
already performs on *observed* values before `permutation_test()` is ever called. The function
therefore scans every null distribution for non-finite entries before returning, raising
`ValueError` naming both the offending metric and the permutation index — otherwise a single
degenerate fold, buried inside a ~27-minute parallel run, would surface as an unnamed crash deep
inside `CrossPlatformPermutationResult.to_json()`'s `allow_nan=False` contract.

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
serial) calls `joblib.Parallel(n_jobs=config.prediction.permutation_n_jobs, backend="loky")` over
the full list of `(target_name, method)` units for a pair, each invoking one complete
`permutation_test()` call (its own internal `n_permutations`-length loop stays serial,
single-process). `PredictionConfig.permutation_n_jobs: int = 8` (not `-1`/all-cores).

**Added during `/review-openspec` round 1:** each `(target_name, method)` combination is given an
independently-derived `random_state`, via `numpy.random.SeedSequence(permutation_random_state)
.spawn(N)` (`N` = the total combination count), rather than reusing the single configured seed
identically for every combination. Results across different `permutation_n_jobs` values are
verified via `numpy.testing.assert_allclose(rtol=1e-6, atol=1e-9)`, not bit-identical — see the
round-1 reconciliation below for why both of these were originally wrong.

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

**Added during `/review-openspec` round 1:** `StepResult.data` is, concretely, a plain dict keyed
by method name (`results_by_method`, one entry per `reduction_method`/`comparison_methods` value).
Adding a sibling `"predictor_matrices"` key is safe *only* because no valid `reduction_method`/
`comparison_methods` value can itself equal `"predictor_matrices"` — both are constrained to
`{"pls_latent", "representatives", "pc1"}`. This is a load-bearing invariant on the reduction-method
enum, not an incidental fact; it is now stated explicitly in the `cross-platform-analysis` spec
delta and covered by a test, so a future change extending that enum can't silently collide with
this key.

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
  caught Tier 3 Decision 6's wrong assumed single-seed R². **Added during `/review-openspec` round
  1:** the roadmap's "≈25%" is not just unverified but arithmetically wrong at this program's real
  scale. Under a random (uninformative) `y_pred`, the expected recovery of a fixed top-`q` set in a
  randomly-drawn top-`2q` set is `2q / n` by linearity of expectation — at `n=19, q=5`, that's
  `10/19 ≈ 52.6%`, not `25%`. This must be stated explicitly (now in the `cross-platform-prediction`
  spec's Top-Quartile Recovery Metric requirement) so the empirical measurement in tasks.md 9.4a is
  correctly interpreted against the right theoretical baseline, not "sanity-checked" against a
  wrong number that happens to sound plausible.
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

None blocking after `/review-openspec` round 1's reconciliation (see below). Carried forward, not
this tier's job: `#197` (`CrossPlatformSummaryGenerator` not surfacing prediction/permutation
results — this tier's new figures/JSON outputs increase the pressure to fix this, worth flagging
again, not fixing here) and `#198` (`/configure-run-all`/`/dry-run`/`/validate-config` coverage
gaps, pre-existing).

## Adversarial Review Reconciliation (round 1)

`/review-openspec` ran 5 parallel reviewers (spec quality, TDD/testing, pipeline architecture &
statistical correctness, documentation, git workflow) against this proposal before any
implementation began. 5 BLOCKING and 11 IMPORTANT findings, reconciled as follows:

- **BLOCKING — the MODIFIED "Predict Cross-Platform Genotype Values Pipeline Step" requirement
  silently dropped 7 of the current spec's 19 scenarios.** My own initial read of the current spec
  (to build the MODIFIED delta) cut off partway through the requirement (line 1560 of 1615) without
  my noticing — a mechanical transcription error, not a judgment call. Per `openspec/AGENTS.md`'s
  own explicit warning ("the archiver will replace the entire requirement with what you provide
  here; partial deltas will drop previous details"), archiving as-is would have permanently deleted
  7 real acceptance criteria, including the backward-compatibility byte-identical guarantee and
  both dry-run scenarios — directly contradicting this proposal's own "all extensions are additive"
  claim. Fixed: the full original 19 scenarios restored verbatim, in original order, plus this
  tier's 1 new scenario.
- **BLOCKING — no finiteness guard on the permutation null distributions before
  `to_json(allow_nan=False)`.** A permutation's LOGO-CV fold structure can independently produce a
  non-finite `spearman_rho` even when that permutation's shuffled `y` is non-constant (a
  model-degeneracy event, distinct from the data-degeneracy check `PredictCrossPlatformStep`
  already performs on observed values) — previously unaddressed, would have surfaced as an unnamed
  crash deep inside a ~27-minute parallel run. Fixed: Decision 1 and the `cross-platform-prediction`
  spec's Permutation Test requirement now specify a proactive non-finite scan naming both the
  offending metric and the permutation index, with a corresponding test (tasks.md 2.13a).
- **BLOCKING — `top_quartile_recovery`'s null baseline is `2q/n` ≈ 52.6% at real scale
  (`n=19, q=5`), not the roadmap's unverified "≈25%".** Arithmetic (expected recovery of a random
  top-`2q` draw), not opinion. Fixed: Decision 11 and the `cross-platform-prediction` spec's
  Top-Quartile Recovery Metric requirement now state this explicitly, so tasks.md 9.4a's empirical
  measurement is checked against the correct theoretical baseline, not a plausible-sounding wrong
  number.
- **BLOCKING — task 12.3's premise was factually backwards.** Verified directly:
  `CrossPlatformPredictionResult`/`TargetPrediction` are in `__all__` but have **no** `docs/API.md`
  entries. Fixed: task 12.3 rewritten to state the verified truth (no `API.md` change for the new
  dataclasses either; `permutation_test`/`top_quartile_recovery` do get entries, matching
  `logo_cv_predict`/`fit_pca_on_fold`'s precedent).
- **BLOCKING — task 12.2 missed correcting a stale forward-reference** in
  `docs/CROSS_PLATFORM_ANALYSIS.md` calling this tier "a separate, later change" — exactly the trap
  Tier 3.5's own review caught and fixed once already for a different sentence. Fixed: task 12.2
  now explicitly includes this correction, plus the missing `top_quartile_recovery` doc mention,
  the new output-file-naming documentation, and a `#197`-pressure note.
- **IMPORTANT — every target/method reused one `permutation_random_state`**, correlating (not
  independently sampling) the null draws Decision 9's pooled violin panel combines across targets.
  Fixed: Decision 4 and the `cross-platform-analysis` spec now derive one independent seed per
  `(target, method)` combination via `numpy.random.SeedSequence(...).spawn(N)`.
- **IMPORTANT — the K-S calibration test (tasks.md 9.1) was an intrinsically flaky CI gate** unless
  every fixture seed and the oracle's own `random_state` are pinned to committed literals (a K-S
  test against `Uniform(0,1)` on ~30-50 samples has a real, non-negligible false-failure rate by
  chance). Fixed: tasks.md 1.2 now requires fixed, committed literal seeds throughout.
- **IMPORTANT — tasks.md 6.5's "bit-identical parallel vs. serial" claim conflicted with real
  `joblib`/BLAS behavior.** `loky` worker processes may resolve a different default BLAS thread
  count than the main process — a documented source of ULP-level floating-point differences,
  independent of this step's own correctness (this codebase already has an established convention
  for exactly this class of cross-BLAS difference, `rtol=1e-6, atol=1e-9`,
  `docs/reproducibility.md`). Fixed: the `cross-platform-analysis` spec and tasks.md 6.5 now assert
  `numpy.testing.assert_allclose` at that established tolerance, not bit-identical equality; the
  `joblib` backend (`loky`) is now stated explicitly rather than left unspecified.
- **IMPORTANT — the additive `predictor_matrices` key relies on an unstated invariant** (no valid
  `reduction_method`/`comparison_methods` value can collide with the string
  `"predictor_matrices"`). Fixed: Decision 6 and the `cross-platform-analysis` spec now state this
  explicitly, with a corresponding scenario.
- **IMPORTANT — `joblib`'s dependency addition (originally task 10.1) was numbered *after* Section
  6, which imports it.** Fixed: moved to a new Section 5a, explicitly ordered before Section 6.
- **IMPORTANT — task 9.4 conflated an empirical spike with "write failing test".** You cannot
  write a meaningful assertion against a value that hasn't been computed yet. Fixed: split into
  9.4a (a spike, explicitly not a red/green TDD step) and 9.4b (the actual test, written against
  9.4a's now-known value).
- **IMPORTANT — missing edge-case tests**: `n_permutations=1` boundary, an explicit
  `comparison_methods=[]` (`K=0`) case, `VisualizePredictionStep` with zero representative-trait
  (PC1-only) targets under `joblib.Parallel`, and 5.2's regression test not explicitly asserting
  `predictor_matrices` is the *only* new key. Fixed: tasks.md 2.12a, 6.2a, 6.1a, and an explicit
  key-set assertion added to 5.2, respectively.
- **IMPORTANT — Section 11's timing relative to the PR lifecycle was ambiguous.** Tier 3.5's own
  real commit history (PR #199) shows manual validation running *between* two rounds of PR-review
  fix commits, not strictly pre-PR-open. Fixed: tasks.md Section 11 now states this explicitly.
- **IMPORTANT — task 10.2 (now 10.1) didn't state `theory.md` lives in a separate external git
  repository**, so a PR reviewer might expect to find it in this repo's diff. Fixed: stated
  explicitly.
- **IMPORTANT — `CrossPlatformPermutationResult`'s requirement was missing the "no sklearn/numpy
  object" parity scenario** its sibling `CrossPlatformPredictionResult` requirement has. Fixed:
  added to the `serializable-result-types` spec delta, with a corresponding test (tasks.md 3.3a).
- **SUGGESTION — Section 2 (14 tasks) and Section 6 (8 tasks) were too coarse for this program's
  established per-commit granularity.** Fixed: both split, with explicit commit-boundary notes
  (Section 2: `top_quartile_recovery` vs. `permutation_test`; Section 6: wiring/reuse →
  parallelization → JSON/figure I/O).
- **SUGGESTION — no boundary scenario for `top_quartile_recovery` when `q` rounds to 0 or exceeds
  `n/2`.** Fixed: added to the `cross-platform-prediction` spec, with a corresponding test
  (tasks.md 2.3a).
- **SUGGESTION — doc tasks risked restating `design.md`'s benchmark/rationale prose (DRY).**
  Fixed: task 12.2 now explicitly scopes to usage documentation, cross-referencing `design.md`'s
  Decisions for rationale detail.

Full re-validation (`openspec validate add-prediction-permutation-and-figure --strict`) passes
after all round-1 fixes.

## Adversarial Review Reconciliation (round 2)

A second, independent round of the same 5-agent review (run fresh, with no memory of round 1,
specifically to catch anything round 1 missed) found round 1's investigative work mostly held up
under direct re-verification (the restored 19-scenario MODIFIED requirement, the `2q/n` math, the
`predictor_matrices` collision invariant, and the `SeedSequence`/`loky`/tolerance language were all
independently confirmed against the current spec/tasks text, not just design.md's own narrative).
But it also found 2 new BLOCKING issues — both real implementation-blocking bugs, not polish — plus
a further batch of IMPORTANT gaps:

- **BLOCKING (new) — round 1's own RNG fix was itself invalid.** `numpy.random.Generator(random_state)`
  requires a `BitGenerator` instance and raises `TypeError` on a bare `int` — the exact API round 1's
  fix specified. Separately, the `SeedSequence.spawn(N)` → `permutation_test(random_state=...)`
  handoff (also introduced in round 1) was left completely unspecified: `spawn()` returns
  `SeedSequence` objects, not ints, and `permutation_test`'s signature was still typed as a plain
  `int`. Fixed: `permutation_test` now builds its RNG via `numpy.random.default_rng(random_state)`,
  which uniformly accepts `int`, `SeedSequence`, or `Generator` — resolving both problems at once,
  with no int-extraction step needed anywhere. Decision 1, Decision 4, and the
  `cross-platform-prediction` spec's Permutation Test requirement all now state this explicitly;
  task 2.8 gained a parametrized case covering both input types, and task 2.14 states the exact
  constructor to use and why (`default_rng`, not `Generator` directly).
- **BLOCKING (new) — Section 6 (`VisualizePredictionStep`) depended on `create_prediction_figure()`
  (the old Section 8) without that section being sequenced first.** Task 6.8's own test mocked "the
  figure-building function," which didn't exist yet at that point in the task ordering — a real
  sequencing bug missed by every round-1 reviewer (including the git-workflow lens, which checked
  ordering but not this specific cross-section dependency). Fixed: the figure-content module
  (`visualize_prediction.py`'s `create_prediction_figure()`) is now Section 6, sequenced *before*
  the step (now Section 7) that consumes it — a pure renumbering, not a behavior change.
- **IMPORTANT (new) — Section 6's (now Section 7's) "3 test-commits, then 1 implementation commit
  at the end" pattern didn't achieve real commit atomicity.** Verified against Tier 3.5's actual
  commit history (its own analogous section landed tests and implementation together in one
  commit, never split): committing round 1's 3 test-only groups in sequence would leave
  `uv run pytest` red at each of those 3 commit boundaries, only turning green at the final,
  separate implementation commit — worse than Tier 3.5's own precedent, not better. Fixed:
  restructured into 3 genuine red→green pairs (7a: wiring/reuse, 7b: `joblib` parallelization, 7c:
  JSON/figure I/O), each landing its own working implementation increment.
- **IMPORTANT (new) — oracle tests (9.2/9.3/9.4b) never stated a reduced `n_permutations` for CI**,
  unlike 9.1, which did. At the production default (`n_permutations=1000`) across the existing
  N=20-seed fixture, these could individually cost minutes and collectively threaten the shared
  30-minute CI job timeout across 3 OSes. Fixed: 9.2/9.3/9.4b now explicitly state
  `n_permutations=200`, matching 9.1's own CI-fast convention.
- **IMPORTANT (new) — round 1's Section 11 timing claim was factually wrong, not just imprecise.**
  Verified directly against Tier 3.5's real commit timestamps (branch `add-prediction-pipeline-step`,
  still available locally): manual validation (`5d5b8e6`, `b5e5cf0`) landed *after* **both**
  5-subagent review rounds (`718260b`/`afb4b91`, then `e5294e0`), with a further CI-driven fix
  (`d401cf0`) landing even after Elizabeth's sign-off — not "between" two rounds, as round 1's text
  claimed. Fixed: Section 11's timing note corrected to state validation is a late-stage gate that
  can trail every review round, not one reliably sandwiched between two.
- **IMPORTANT (new) — 2 previously-untested `PredictionConfig` fields had no validation at all**:
  `permutation_n_jobs` (a non-positive value would surface `joblib.Parallel`'s own raw error deep
  inside the step) and `permutation_random_state` (an invalid seed would surface
  `numpy.random.SeedSequence`'s own raw error). Fixed: both gain the same
  validated-only-when-`visualize=True` treatment as `n_permutations`, with named `ValueError`s and
  corresponding tests (tasks.md 4.4a/4.4b).
- **IMPORTANT (new) — the seed-enumeration order underlying `SeedSequence.spawn(N)` was never
  pinned down.** Tests could pass regardless of which `(target, method)` enumeration order was
  chosen (a rerun of the same code reproduces the same order trivially), silently leaving the
  actual order — dict-iteration order of `results_by_method`, easy to perturb in a future refactor
  — unspecified as a real contract. Fixed: the `cross-platform-analysis` spec now states the exact
  canonical order (methods first, then target names in `CrossPlatformPredictionResult.predictions`
  order), and task 7b.3 tests against it explicitly.
- **IMPORTANT (new) — partial-failure/atomic-write semantics for `VisualizePredictionStep`'s JSON
  output were unspecified.** If one `(target, method)` combination failed while others for the same
  pair would have succeeded, it was unstated whether already-computed methods' JSON files should
  still be written. Fixed: the `cross-platform-analysis` spec now states all-or-nothing per pair
  (no partial JSON output on any failure), with a corresponding test (tasks.md 7c.6).
- **IMPORTANT (new) — the non-finite-null guard's fail-fast-vs-complete-then-report choice was
  undiscussed**, despite being cost-relevant in a ~27-minute parallel run. Fixed: the
  `cross-platform-prediction` spec now states and justifies the choice (complete-then-report,
  since a genuine bug is expected to affect many permutations, not one rare occurrence — failing
  fast saves little wall-clock time while complicating which-permutations-ran accounting).
- **IMPORTANT (new) — `assert_allclose(rtol=1e-6, atol=1e-9)` reused a tolerance convention
  established for smaller arrays** (PCA loadings/eigenvalues/cluster centers), applied here
  element-wise across null distributions of up to 1000 values across many targets/methods, without
  stating what fixture size bounds the resulting false-failure-rate risk. Fixed: task 7b.5 now
  states an explicit, small `n_permutations` (50) for this specific test, bounding the comparison
  surface.
- **IMPORTANT (new) — no wiring-correctness oracle existed for `observed_top_quartile_recovery`
  specifically** (unlike the other 3 observed metrics, which have an explicit direct-`logo_cv_predict`-
  call cross-check). Fixed: task 2.5 now asserts this metric against an independent
  `top_quartile_recovery(y, y_pred)` computation from the same observed call.
- **IMPORTANT (new) — the pooled-null violin panel's implicit invariant (every target within a pair
  shares the same genotype count) was unstated**, and the independent per-target seeding fix (round
  1) raised a legitimate question about whether pooling nulls from targets with wildly different
  counts still makes sense. Verified: task 6's `dropna(axis=1, how="any")` trait-column filtering
  never changes the row/genotype count, so this concern doesn't actually arise — but the invariant
  was worth stating explicitly rather than leaving a future reviewer to re-derive it. Fixed: added
  to the Visualize step's requirement body (step 5).
- **IMPORTANT (new, documentation) — the round-1 fix of "cross-reference `design.md`'s Decisions"
  for shipped-doc content was itself a documentation anti-pattern.** `design.md` moves to
  `openspec/changes/archive/<change-id>/design.md` on archival — a shipped-doc pointer to it goes
  stale/dangling the moment this change archives. Fixed: task 12.2 now states the two key numbers
  (the parallelization headline, the `2q/n` chance-level baseline) directly in
  `docs/CROSS_PLATFORM_ANALYSIS.md`, rather than deferring to a soon-to-be-archived file.
- **IMPORTANT (new, documentation) — `top_quartile_recovery` and the full 4-field YAML example
  were under-specified in task 12.2.** Fixed: both named explicitly.
- **SUGGESTION (new) — missing an explicit-invalid-`q` scenario for `top_quartile_recovery`**
  (only the *default*-`q` small-`n` case was covered). Fixed: added, with a corresponding test
  (tasks.md 2.3b).
- **SUGGESTION (new) — the bit-identical (single-process, task 2.8) and tolerance-based
  (cross-`joblib`-worker, task 7b.5) determinism claims could read as contradictory** without an
  explicit note that they concern different variables. Fixed: one clarifying sentence added to the
  "Same random_state" scenario.
- **SUGGESTION (new, documentation) — `theory.md`'s external-vault addendum has no repo-side
  pointer anywhere**, making it discoverable only via archived OpenSpec history or vault access.
  Documented as a known, accepted gap (task 10.1) — not fixed in this tier, no obvious low-cost fix
  identified.
- **Verified clean, no action needed:** the `2q/n` arithmetic was independently re-derived via the
  hypergeometric mean and confirmed correct; the `predictor_matrices` key-collision invariant was
  re-checked against the current `_VALID_REDUCTION_METHODS` tuple and still holds; the K-S
  calibration test's fixture-seed-pinning (round 1's fix) was confirmed unambiguous.

Full re-validation (`openspec validate add-prediction-permutation-and-figure --strict`) passes
after all round-2 fixes (94 tasks, up from 88 — 2 new BLOCKING fixes and the Section 6/7
restructuring account for most of the growth).
