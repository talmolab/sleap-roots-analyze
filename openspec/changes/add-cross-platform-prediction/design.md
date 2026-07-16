## Context

This is Tier 3 (`add-cross-platform-prediction`, tracking issue
[talmolab/sleap-roots-analyze#194](https://github.com/talmolab/sleap-roots-analyze/issues/194))
of the wheat EDPIE cross-platform genotype-prediction program, which reframes the paper's
cross-platform result from *correlation* to *predictability* per Wolfgang's 2026-04-03 ask. See
the program roadmap and statistical grounding at
`c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md` (external
to this repo; referenced here for provenance only — `theory.md`'s CV-hygiene contract and
LOGO-CV/per-fold-PCA pseudo-code are the normative reference this tier implements against).

Tier 1 (`add-blup-extraction`, merged #189/#190) produced `extract_blup_table()` /
`BLUPResult`, giving a `(n_genotypes, n_traits)` BLUP-adjusted-means matrix per platform. Tier 2
(`add-heritability-fixed-effects`, merged #193) improves field-BLUP quality via
`fixed_effects` but is not a hard dependency — this tier works with genotype-only BLUPs too.

Neither `sklearn.model_selection` (`LeaveOneOut`), `Ridge`, `PLSRegression`, nor `Pipeline` are
used anywhere in the codebase today (confirmed by repo-wide grep) — this tier introduces all
four with no existing in-repo LOGO-CV pattern to extend. The closest analog is `pca.py`'s
`StandardScaler` fit/transform pattern.

The existing `cluster_correlated_traits`/`select_cluster_representatives` functions
(`cross_experiment_analysis.py:1832-2014`, consumed today by `ReduceTraitRedundancyStep`) already
implement the correlation-threshold clustering + highest-variance-representative selection this
tier's trait-set identity oracle needs — no new clustering code is required, only a new test that
calls the existing functions on genotype-mean/BLUP-level input and asserts the reproduced trait
set.

## Goals / Non-Goals

- **Goals:** `fit_pca_on_fold()` utility (theory.md §5 contract, verbatim); `logo_cv_predict()`
  implementing the CV-hygiene contract (whole `Pipeline` fit inside each LOGO fold) across three
  `reduction_method` values (`pls_latent` default, `representatives`, `pc1`); aggregate CV R²,
  RMSE, and Spearman ρ computed over concatenated leave-one-out predictions; `predictor_source`
  runtime guard (`{blup, genotype_means}`); a new `CrossPlatformPredictionResult` frozen
  dataclass; the five acceptance-criteria oracles from issue #194 (planted-signal recovery,
  leakage regression test, PC1 per-fold oracle, trait-set identity oracle, synthetic non-EDPIE
  generalizability fixture); a non-CI manual validation task against real EDPIE platform data
  before merge.
- **Non-Goals:** the `PredictionConfig` dataclass, `PredictCrossPlatformStep`, and any CLI/pipeline
  wiring (Tier 3.5); the permutation null and its figures (Tier 4); PLS component-count search
  (fixed at 1, see Decision 1); any change to `cluster_correlated_traits`/
  `select_cluster_representatives` themselves (reused as-is); any change to
  `calculate_heritability_estimates`, `extract_blup_table`, or `BLUPResult` (Tier 1/2, unchanged).

## Decisions

### Decision 1: PLS component count fixed at `n_components=1`, not inner-CV

**What:** `reduction_method="pls_latent"` always uses `PLSRegression(n_components=1)`. No nested
inner-CV search over candidate component counts.

**Why:** Resolved with Elizabeth during this tier's brainstorm. Two independent reasons converge
on the same answer:

1. **Statistical:** at n≈18 training genotypes per fold, PLS with more components has more
   learned X→y-covarying axes available to fit noise with (each component is a learned linear
   combination of traits chosen to maximize covariance with y — see the brainstorm's PLS
   explanation). `n_components=1` is the most constrained, least overfit-prone configuration, and
   is exactly what theory.md §3's pseudo-code hardcodes.
2. **Runtime:** Tier 4's permutation null reruns the entire LOGO-CV loop N=1000 times across 4
   directed pairs × 2 reduction methods — already ≈19×4×2×1000 ≈ 152,000 model fits (see Decision
   3). Nested inner-CV would multiply that by the inner search's fold count × candidate count
   (≈15-50×), pushing an estimated low-minutes runtime into multiple hours and forcing a
   `joblib.Parallel` strategy just to make Tier 4 usable. It would also add a second nested
   CV-hygiene surface (leakage inside the inner loop) for a hyperparameter theory.md already
   treats as fixed.

**Alternatives considered:**
- **Fixed n=1 primary + n=2/3 as extra `comparison_methods` entries.** Considered; deferred rather
  than shipped now — no immediate evidentiary need for it, and it multiplies Tier 4's permutation
  cost per extra method for a robustness check nobody has asked for yet. If `n_components=1`
  underperforms on real EDPIE data (see the manual validation task), this is the natural follow-up
  to propose then, with a concrete motivating result in hand rather than speculatively.
- **Inner-CV search over 1-3.** Rejected for the reasons above.

### Decision 2: Cluster-representative-selection input is genotype-mean/BLUP-level, not raw sample-level

**What:** The trait-set identity oracle (and any future representative-selection call in this
module) computes the Spearman correlation matrix that feeds `cluster_correlated_traits` on the
**genotype × trait BLUP-adjusted-means matrix** (one row per genotype), not on raw replicate-level
rows.

**Why:** Resolved with Elizabeth during this tier's brainstorm. BLUP values are already
genotype-indexed (Tier 1's substrate); the existing `ReduceTraitRedundancyStep`
(`pipeline/steps/reduce_trait_redundancy.py`) already groups by genotype before computing
correlations — this is consistent with existing pipeline behavior, not a new convention. It is
also the only choice that keeps the trait-set identity oracle deterministic given the same BLUP
substrate the rest of this tier's prediction machinery consumes (raw sample-level correlation
would use a different, differently-sized input and risks not reproducing Section 3.4's exact
28+14 trait set for reasons unrelated to clustering correctness). Per the roadmap's own framing:
"BLUP values are order-independent; only representative *choice* is affected" — i.e. this decision
affects which traits get selected as representatives, not the shrinkage/CV-hygiene properties of
the BLUP substrate itself.

**Alternatives considered:**
- **Raw sample-level correlation.** Rejected: more input rows (better-powered correlation
  estimates in principle), but diverges from the BLUP substrate the rest of the tier operates on
  and from `ReduceTraitRedundancyStep`'s existing convention.

### Decision 3: Permutation runtime is documented, not designed around, in Tier 3

**What:** This tier's proposal records the estimated Tier 4 permutation-null runtime
(19 folds × 4 directed pairs × 2 reduction methods × N=1000 permutations ≈ 152,000 total
Ridge/PLS fits on ~18×10 matrices — almost certainly low minutes on a single core, well under the
roadmap's 30-minute feasibility gate) and states no parallelization is needed at that estimate.
`logo_cv_predict()` and its scoring helpers are written as plain, stateless functions with no
global state and no per-call I/O, so Tier 4 can wrap them in a permutation loop — and later in
`joblib.Parallel` if the *measured* runtime differs from this estimate — without refactoring.

**Why:** Resolved with Elizabeth during this tier's brainstorm. Tier 3 does not implement
permutation itself (Tier 4 does), so building parallelization scaffolding now would be
speculative. But *not* considering it at all risks a Tier 3 API shape (e.g. hidden mutable state,
a module-level cache) that would force a refactor once Tier 4 needs to call it in a tight loop
1000+ times. Writing the core as stateless functions costs nothing now and avoids that later
rework.

**Alternatives considered:**
- **No design constraint at all, revisit at Tier 4.** Rejected: costs nothing extra to write
  stateless functions from the start, and doing so removes a class of future refactor risk for
  free.
- **Build `joblib.Parallel` support into Tier 3 now.** Rejected: the estimate suggests it isn't
  needed; building it now is speculative effort against an unmeasured problem.

### Decision 4: Reported CV metrics are aggregate (pooled), not per-single-held-out-genotype

**What:** `logo_cv_predict()` produces one prediction per genotype (the standard LOGO-CV output —
`y_pred[i]` is the prediction for genotype `i` when genotype `i` was held out). R², RMSE, and
Spearman ρ are each computed **once**, over the full set of concatenated leave-one-out
predictions (`r2_score(y_true_all, y_pred_all)`, matching theory.md §4.3's `logo_cv_r2` exactly),
not as a separate score per single held-out genotype.

**Why:** R² and Spearman ρ are undefined (or degenerate) for a single data point — LOGO-CV's test
fold has exactly one genotype, so "R² per fold" cannot mean "R² computed from that fold's one
prediction" without a well-defined statistical meaning. Issue #194's and the roadmap's phrase
"CV R², RMSE, and Spearman ρ per fold" is read here as "these three metrics, computed from the
per-fold (leave-one-genotype-out) predictions" — i.e. *how* the predictions were generated
(fold-by-fold), not that each fold individually produces its own scalar R². This matches
theory.md's only worked reference implementation (`logo_cv_r2`), which pools all folds' held-out
predictions before scoring. The per-genotype residuals (`y_true[i] - y_pred[i]`) remain available
in `CrossPlatformPredictionResult` for anyone who wants a per-genotype breakdown (e.g. RMSE is
still meaningful per-genotype as an absolute error, unlike R²/ρ) — this decision only concerns
what the three headline metrics mean.

**Flagging this explicitly for review:** this is an interpretation of ambiguous roadmap/issue
language, not a fact confirmed elsewhere in the program's grounding documents. If this
interpretation is wrong, it should be caught and corrected during `/review-openspec` or by
Elizabeth directly, before implementation.

**Alternatives considered:**
- **Report only a single pooled R²/RMSE/ρ, drop "per fold" from the naming.** Considered;
  rejected as a naming-only change with no behavioral difference from the chosen design — kept
  the roadmap's own vocabulary intact instead of introducing new terminology for the same
  quantity.

### Decision 5: One `CrossPlatformPredictionResult` per (platform pair, reduction method); nested `TargetPrediction` list covers representative traits + PC1

**What:** `CrossPlatformPredictionResult` (new frozen dataclass, `result_types.py`) holds
`source_platform`, `target_platform`, `predictor_source`, `reduction_method`, and a
`predictions: list[TargetPrediction]` — one `TargetPrediction` per prediction target: each
cluster-representative trait in the target platform (the primary prediction target) plus one
additional `TargetPrediction` with `target_name="PC1"` (computed via `fit_pca_on_fold`, R²
reported separately per the roadmap's settled decisions). `comparison_methods` (e.g.
`representatives` alongside a `pls_latent` primary) produce additional
`CrossPlatformPredictionResult` instances, one per method — not a third nesting level.

**Why:** Mirrors the existing `HeritabilityResult`/`TraitHeritability` nesting pattern
(`result_types.py`) exactly: one result object per "run," a list of per-target entries inside.
Keeping `reduction_method` at the top level (not per-target) matches how the roadmap frames it —
a single primary method per run, with `comparison_methods` producing separate, parallel runs
rather than a single result object trying to hold every method's numbers for every target
simultaneously (which would require a two-level nested structure with no existing precedent in
`result_types.py` to follow).

**Alternatives considered:**
- **A single flat result per (pair, target, method) triple, no nesting.** Rejected: loses the
  `HeritabilityResult`-style "one run, many entries" grouping that makes a full 4-pair ×
  (28 or 14 representative traits + PC1) × 2-method sweep easy to serialize as one JSON blob per
  (pair, method) rather than dozens of scattered small files.

## Risks / Trade-offs

- **Decision 4's metric-interpretation risk.** If "per fold" was meant literally (a distinct
  metric for every one of the 19 held-out genotypes), this tier's `CrossPlatformPredictionResult`
  shape would need a list of 19 per-genotype scalar entries instead of one pooled scalar per
  target. Flagged explicitly above and in the proposal for review before implementation begins.
- **`n_components=1` may underperform on real EDPIE data.** Mitigated by the manual real-data
  validation task (non-CI, pre-merge gate) — if the fixed-1 PLS model's R²/RMSE/ρ on the 4 real
  directed pairs looks implausible against the known correlation numbers already in the roadmap
  (e.g. Turface19→Cylinder ρ fold 0.67, p 0.002), that is a concrete signal to revisit Decision 1
  via a follow-up issue, not a reason to change this tier's design speculatively now.
- **`fit_pca_on_fold` is deliberately duplicated in intent from the pipeline-level `PCA` step in
  `pca.py`.** This is intentional (theory.md §3.1's explicit "WRONG" example shows why reusing the
  pipeline-level step leaks the held-out genotype into loadings) but means there are now two
  PCA-fitting code paths in the codebase with different contracts — worth a clear docstring
  cross-reference in both directions so a future reader doesn't try to consolidate them.
- **Trait-set identity oracle depends on real EDPIE data matching Section 3.4's clustering input
  exactly.** Confirmed during this tier's proposal drafting: `tests/data/` already has real EDPIE
  trait CSVs (`Turface_all_traits_2024.csv`, `Field_2024_clean.csv`-equivalent,
  `Wheat_EDPIE_cylinder_master_data.xlsx`), and the external vault has an already-computed
  `exp1_trait_clusters.csv`/`exp2_trait_clusters.csv` pair (columns: `trait`, `cluster_id`,
  `is_representative`, `variance`) from the actual pipeline run behind Section 3.4
  (`cross_platform_field_v2`, dated 2026-03-30). Whether `tests/data/`'s existing files are the
  same processed (genotype-mean/BLUP-level) form used in that run, and the exact 28/14 trait
  names, are **not yet verified** — this is implementation-time work (tasks.md), not resolved
  here. Following this repo's established precedent (Tier 1/2's fixture parameters were pinned
  only after direct simulation against the real library, not derived analytically), the trait-set
  identity oracle's exact assertion (a pinned list of 28/14 trait names vs. a length-only check)
  is decided once `cluster_correlated_traits`/`select_cluster_representatives` is actually run
  against verified real data during implementation.

## Migration Plan

Purely additive — no existing function, config, or pipeline behavior changes. New module
(`cross_platform_prediction.py`), new result type (`CrossPlatformPredictionResult`), new tests and
fixtures only. No existing caller is affected.

## Open Questions

None blocking Tier 3, beyond Decision 4's explicit flag above (to be resolved during
`/review-openspec` or directly with Elizabeth, not silently assumed). Tier 3.5's config-wiring
question (where `PredictionConfig` nests) and Tier 4's permutation/figure design remain open in
their own tiers.
