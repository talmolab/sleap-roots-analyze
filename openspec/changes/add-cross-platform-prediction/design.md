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
tier's trait-set identity oracle needs — no new clustering code is required. **However, see
Decision 2 (revised after `/review-openspec` round 1): what exactly the trait-set identity oracle
should assert is currently BLOCKED pending an investigation into the real Section 3.4 pipeline —
do not start Section 5 of tasks.md until that investigation returns.**

## Goals / Non-Goals

- **Goals:** `fit_pca_on_fold()` utility (theory.md §5 contract, verbatim); `logo_cv_predict()`
  implementing the CV-hygiene contract (whole `Pipeline` fit inside each LOGO fold) across three
  `reduction_method` values (`pls_latent` default, `representatives`, `pc1`); aggregate CV R²,
  RMSE, and Spearman ρ computed over concatenated leave-one-out predictions; `predictor_source` as
  provenance metadata (`{blup, genotype_means}` — see Decision 8; the actual runtime *guard*
  belongs to Tier 3.5's `PredictionConfig.__post_init__`, not this tier); a new
  `CrossPlatformPredictionResult` frozen dataclass; the five acceptance-criteria oracles from issue
  #194 (planted-signal recovery, leakage regression test, PC1 per-fold oracle, trait-set identity
  oracle — **blocked, see Decision 2** — and synthetic non-EDPIE generalizability fixture); a
  non-CI manual validation task against real EDPIE platform data before merge.
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

### Decision 2: Cluster-representative-selection input is genotype-mean/BLUP-level, not raw sample-level — **REVISED after `/review-openspec` round 1; trait-set identity oracle BLOCKED**

**What (original, pre-review):** The trait-set identity oracle computes the Spearman correlation
matrix that feeds `cluster_correlated_traits` on the **genotype × trait BLUP-adjusted-means
matrix**, treating "genotype-mean" and "BLUP-adjusted-mean" as interchangeable.

**Why this was wrong (found during `/review-openspec` round 1, confirmed by direct code read and
by counting a real committed fixture):**

1. **"Genotype-mean" and "BLUP-adjusted-mean" are NOT the same matrix.**
   `ReduceTraitRedundancyStep` (`pipeline/steps/reduce_trait_redundancy.py:208`) computes
   `df.groupby("genotype")[trait_names].mean()` — a **plain arithmetic per-genotype mean of raw
   sample-level measurements**. There is no mixed model, no `random_effects`, no shrinkage
   anywhere in that path. This is the actual code path that produced the real Section 3.4 result:
   the vault's `cross_platform_field_v2` run (dated 2026-03-30) **predates Tier 1's
   `extract_blup_table()`/BLUP extraction entirely** (Tier 1 merged 2026-07-15) — BLUP-adjusted
   means could not have been the substrate for the historical 28+14 result, because they didn't
   exist yet when that result was produced. The original Decision 2 conflated two genuinely
   different matrices (raw genotype-mean vs. BLUP-shrunk genotype-mean) under one label.
2. **BLUP shrinkage is not a uniform rescaling — it can change which traits cluster together and
   which is highest-variance.** theory.md §1.4: $\hat g_i \approx \lambda_i(\bar y_i - \hat\mu)$,
   where $\lambda_i$ depends per-genotype on $n_i$, $\sigma^2_G$, $\sigma^2_E$. Low-heritability,
   low-replicate traits get pulled toward the grand mean much more than high-heritability traits.
   Since `cluster_correlated_traits` clusters on Spearman correlation and
   `select_cluster_representatives` picks the highest-**variance** trait per cluster, differential
   shrinkage across traits can change both the correlation structure and the variance ranking —
   so running the "same" clustering on BLUP-adjusted means is not a neutral substitution.
3. **The 28/14 target itself is very likely not "raw per-platform representative count" at all.**
   A real, already-committed fixture in this repo
   (`tests/fixtures/real/wheat_edpie/expected/cross_platform/root_core_vs_cylinder/exp{1,2}_trait_clusters.csv`)
   — a genuine run of `cluster_correlated_traits`/`select_cluster_representatives` on real EDPIE
   data — shows **28 representative traits for Field (exp1) and 121 for Cylinder (exp2)**, counted
   directly from the `is_representative` column. Neither number matches "28 cylinder + 14 field."
   Re-reading `results_3.4_draft_20260330.md` (external vault) more carefully: *"Of 2,838 trait
   pairs tested between field and cylinder experiments, 36 had |ρ| ≥ 0.55, spanning 14 field
   traits and 28 cylinder traits (cluster representatives)."* This reads as: 14/28 is the count of
   **distinct representative traits that happen to appear in the subset of cross-platform
   correlation pairs exceeding |ρ|≥0.55** — a downstream artifact of clustering *plus*
   cross-platform correlation filtering, not a pure per-platform clustering output. The oracle as
   originally scoped (reproduce 28/14 directly from `select_cluster_representatives` alone) tests
   the wrong quantity.

**Current status: BLOCKED.** A handoff investigation has been requested (external vault access
needed to find the exact script/run behind `results_3.4_draft_20260330.md`'s numbers and confirm
the reconstructed cluster→correlate→filter→count-distinct pipeline above, or correct it). **Do not
begin tasks.md Section 5 (trait-set identity oracle) until this returns.** Everything else in this
tier (Sections 1-4, 6-10) does not depend on the answer and can proceed. Once resolved, this
Decision and the `cross-platform-prediction` spec's "Trait-Set Identity Oracle" requirement need a
follow-up revision reflecting the actual mechanism.

**What is NOT blocked:** `cluster_correlated_traits`/`select_cluster_representatives` reuse for the
`reduction_method="representatives"` *prediction* path (Section 3) does not depend on this
resolution — that path only needs *some* deterministic, unsupervised representative-selection
mechanism (variance-based, at any consistent aggregation level), not a specific reproduction of
28/14. Only the standalone "trait-set identity oracle" (Section 5) is blocked.

**Alternatives considered (for the eventual resolution):**
- **Raw genotype-mean, matching `ReduceTraitRedundancyStep` exactly.** Likely correct for
  reproducing the historical clustering step itself, but per finding 3 above, clustering alone may
  not be the full mechanism the 14/28 figure measures.
- **BLUP-adjusted mean.** Rejected as the oracle's substrate (though still valid as a *separate*,
  clearly-labeled robustness/ablation report, per finding 2 above) — it cannot be what produced a
  pre-Tier-1 historical result.

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

### Decision 6: Planted-signal and pure-noise fixtures use an N-seed averaged R², not a single-realization pinned tolerance — **added after `/review-openspec` round 1**

**What:** theory.md §4.2's `make_planted_signal_fixture` (single realization, seed=42) is not used
as-is for the planted-signal-recovery oracle (tasks.md 3.8) or the leakage regression test's
absolute R² values. Instead, the fixture is redesigned as: `n_genotypes=19` (fixed, matches real
EDPIE constraint), **`n_traits=3`** (reduced from theory.md's illustrative 10),
`signal_strength=0.8`, run across a **fixed, deterministic set of N=20 seeds**; the oracle asserts
on the **mean** LOGO-CV R² across those 20 realizations, not a single seed's value.

**Why:** Independently verified during `/review-openspec` round 1 (both by a reviewing subagent
and reproduced directly): running theory.md's literal single-seed recipe
(`n_genotypes=19, n_traits=10, signal_strength=0.8, seed=42`) through real LOGO-CV with default
`Ridge()`/`PLSRegression(n_components=1)` gives **R² = 0.469 (ridge) / 0.448 (pls)** at seed=42 —
not "close to 0.8." Sweeping seeds 1-7 gives Ridge R² ranging **-0.67 to 0.87**, PLS **-0.17 to
0.58** — sign-flipping, not stably near the planted `signal_strength`. This is a fundamental
small-sample property (at n=19 training genotypes with p≥8-10 traits, `LeaveOneOut` R² has very
high variance — Marchenko-Pastur-scale eigenvalue spread in $X^TX$ at this aspect ratio, per one
reviewer's derivation), not an implementation bug: no single realization at this scale is expected
to land tightly on a target R².

A parameter sweep (`n_traits` ∈ {3, 5, 8, 10}, Ridge `alpha` ∈ {1.0, 0.1}, N=20-50 seeds) found:
reducing `n_traits` to 3 shrinks per-seed variance somewhat (Ridge signal mean=0.72, sd=0.15 at
n_traits=3 vs. mean=0.39, sd=0.40 at n_traits=10) but does **not** eliminate it enough for a
single-seed tolerance to be reliable — even at n_traits=3, single-seed Ridge R² still ranges 0.28
to 0.89. What **does** reliably and stably separate signal from noise is the **mean across many
independent fixture realizations**: at n_traits=3, N=50 seeds, Ridge signal mean=0.717 (sd=0.149)
vs. noise mean=-0.279 (sd=0.186) — a gap of ~1.0, several SDs wide, for both `ridge` and
`pls_latent` methods. This matches standard practice for evaluating a stochastic small-sample CV
estimator (report averages over repeated draws, not one draw) and gives a stable, reproducible
number to pin, unlike the single-seed approach.

**Concrete parameters to lock in tasks.md (empirically verified during this review round; final
implementation MUST re-verify with the pinned RNG API/seeds and record the actual numbers, per
this repo's established fixture-verification precedent — do not assume these exact figures
transfer byte-for-byte to a different RNG call order):**
- `n_genotypes=19`, `n_traits=3`, `signal_strength=0.8`, seeds `1..20` (fixed, deterministic list,
  not re-derived per run).
- Planted-signal fixture: mean LOGO-CV R² (both `ridge` and `pls_latent`) expected in the
  **0.5-0.8** range, comfortably positive and comfortably separated from the noise fixture.
- Pure-noise fixture (independently-drawn `X`/`y`, no planted relationship): mean LOGO-CV R²
  expected **negative** (LOGO-CV R² can and should go negative when a model does worse than
  predicting the training mean — this is a known, correct property of `r2_score` on small-`n`
  CV, not a bug) — assert it is comfortably separated from (well below) the signal fixture's mean,
  not that it is "≈0."
- Oracle assertion: `mean(signal_r2_across_20_seeds) - mean(noise_r2_across_20_seeds) >` some
  empirically-justified margin (e.g. 0.5), rather than either quantity matching an absolute target
  in isolation.

**This is a correction to `theory.md` (Tier 0), which already marked this fixture design "done."**
Elizabeth should be made aware that Tier 0's grounding document's illustrative fixture recipe does
not hold up at implementation scale, independent of this specific tier — worth a `theory.md`
erratum/update for future tiers (e.g. Tier 4's permutation null) that might otherwise reuse the
same single-realization assumption.

**Alternatives considered:**
- **Reframe as a purely relative, single-seed comparison ("signal R² > noise R² this run").**
  Considered; rejected as the sole fix because even this weaker property does not hold for every
  single seed at every `n_traits` tested (e.g. `pls_latent` at n_traits=3 showed a signal-seed
  minimum of 0.028 and a noise-seed maximum of 0.234 across 50 seeds — an overlap that a
  single-seed relative comparison could occasionally get backwards). The N-seed-averaged design
  above does not have this problem.
- **Increase `signal_strength` toward 1.0 to force a bigger gap.** Rejected: theory.md §4.4
  explicitly warns the leakage regression test's fixture must stay below `signal_strength≈0.95` or
  leakage becomes undetectable (ceiling effect) — don't want to trade one fragility for another.

### Decision 7: `logo_cv_predict` takes a labeled `X` (DataFrame) and `representative_names` (trait name list), not a bare `np.ndarray` + integer indices — **added after `/review-openspec` round 1**

**What:** `logo_cv_predict(X, y, genotypes, reduction_method="pls_latent",
representative_names=None)`, where `X` is a `pandas.DataFrame` (`(n_genotypes, n_traits)`, columns
named by trait, index by genotype label — matching how `BLUPResult`/`extract_blup_table()` already
shape their output) rather than a bare `numpy.ndarray`. `representative_names` (renamed from the
original `representative_indices`) is a list of **trait names** (column labels), taken directly
from `select_cluster_representatives()`'s own return type.

**Why:** Found during `/review-openspec` round 1: `select_cluster_representatives`
(`cross_experiment_analysis.py:1961-2014`) returns `List[str]` — **trait names**, never positional
integer indices; its one existing consumer (`reduce_trait_redundancy.py:221,280-281`) uses the
names directly as DataFrame column labels. The original proposal's `representative_indices:
Optional[Sequence[int]]` parameter, combined with `X: np.ndarray` (a bare matrix with no column
metadata), left completely unspecified how a caller converts `select_cluster_representatives`'s
string output into whatever `representative_indices` was supposed to contain — a real integration
gap an implementer would hit as a bug, not a documentation nicety. Requiring `X` to be a labeled
`DataFrame` (matching the rest of this codebase's convention of using pandas throughout
`cross_experiment_analysis.py` and `statistics.py`, rather than dropping to bare `ndarray`
mid-pipeline) removes the ambiguity entirely: `representative_names` is used as
`X[representative_names]`, no index bookkeeping required anywhere.

`fit_pca_on_fold` keeps its `np.ndarray` signature (theory.md §5's contract, unchanged) — this
decision only affects `logo_cv_predict`'s public signature; internally, `logo_cv_predict` extracts
`.values`/`.to_numpy()` as needed before handing arrays to sklearn.

**Alternatives considered:**
- **Keep `np.ndarray` + require the caller to separately pass a `trait_names: Sequence[str]`
  parallel array, resolving names to positions internally.** Rejected: adds a parameter and an
  internal name→index resolution step for no benefit over just accepting the DataFrame directly,
  given the rest of the codebase already works this way.

### Decision 8: `predictor_source` is provenance metadata in Tier 3, not a validated runtime guard — **revised after `/review-openspec` round 1**

**What:** `predictor_source: str` (`{"blup", "genotype_means"}`) is stored as a plain metadata
field on `CrossPlatformPredictionResult`, describing which substrate a given run's `X` came from.
Tier 3's functions (`logo_cv_predict`, `fit_pca_on_fold`) do **not** validate or branch on this
value — they accept whatever `X` DataFrame they're given regardless of provenance.

**Why:** The original proposal called this a "runtime guard" in multiple places, but no function
signature, spec scenario, or task ever specified what the guard would check or reject — found
during `/review-openspec` round 1 as a real proposal/implementation mismatch, not just loose
wording. The *actual* pre-flight guard described in the roadmap's settled decisions
("`PredictionConfig.__post_init__` raises `ConfigValidationError` if `blup_adjusted_means.csv`
path is not resolvable") is explicitly Tier 3.5's `PredictionConfig` scope (config-load-time
validation before any pipeline step runs) — it has no equivalent meaning for Tier 3's plain
Python-API functions, which have no config object and no file-path resolution step at all.
Downgrading the language here to match what Tier 3 actually builds avoids promising validation
behavior this tier doesn't (and shouldn't yet) implement.

**Alternatives considered:**
- **Build a real `predictor_source` validation into `logo_cv_predict` now.** Rejected: there is
  nothing meaningful to validate at the plain-function level — `X` is just a DataFrame; whether it
  came from a BLUP table or a raw genotype-means table is a caller-side fact, not something
  `logo_cv_predict` can independently verify from the matrix alone.

### Decision 9: Documented, not solved, statistical limitations — Ridge `alpha`, RMSE cross-trait scale, Spearman p at small n, X/y-exclusion contract

**What:** Four small items surfaced during `/review-openspec` round 1, each resolved by
documentation rather than new code:

1. **`Ridge()` uses sklearn's default `alpha=1.0`**, undiscussed in the original proposal (unlike
   PLS's `n_components`, which got a full Decision). This is an accepted default for this tier, not
   a tuned value — documented explicitly in `logo_cv_predict`'s docstring as a known, unexamined
   choice, not silently glossed over. A future tier could revisit this the same way Decision 1
   handles PLS components, if real-data validation (Section 8) suggests it matters.
2. **RMSE is not comparable across traits on different scales** (representative traits, PC1, and
   different platform pairs all have different native units/scales) — `CrossPlatformPredictionResult`
   does not add a normalized-RMSE field this tier; this is documented as a known limitation in the
   result type's docstring, deferred to whichever future consumer (Tier 4's figures, or a paper
   table) actually needs cross-trait RMSE comparison.
3. **`spearman_p` (from `scipy.stats.spearmanr`) uses an asymptotic approximation** that is known
   to be imprecise below n≈20-30 — documented in `TargetPrediction`'s docstring as a descriptive,
   not hypothesis-test-grade, statistic at this program's n≈18-19. Tier 4's permutation null is the
   rigorous inference layer for this program; `spearman_p` here should not be over-read in the
   interim.
4. **Explicit precondition: `X`'s columns must never include the target trait's own values.**
   `select_cluster_representatives`'s unsupervised selection is only safe to fix pre-loop because
   nothing in Tier 3's scope lets it see `y` — but this safety depends on a caller-side invariant
   (predictor columns exclude the target) that was never written down. Documented as an explicit
   docstring/spec precondition on `logo_cv_predict`, since Tier 3.5's future pipeline wiring is the
   eventual caller responsible for upholding it.

**Why:** All four are real, correctly-identified gaps from `/review-openspec` round 1 that don't
require new code or a design change to resolve — just an explicit, honest statement in the right
docstring/spec so a future reader (or Tier 3.5/4 implementer) isn't misled into treating an
unexamined default or an unstated precondition as a settled, validated design choice.

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
- **Trait-set identity oracle mechanism is BLOCKED, not just under-specified.** Superseded by
  Decision 2's round-1 revision above: this is no longer a "verify exact numbers at implementation
  time" risk (like a Tier 1/2 fixture-parameter deferral) — it's a confirmed mismatch between what
  the oracle was designed to test (raw `select_cluster_representatives` output) and what the real
  Section 3.4 result appears to actually be (a downstream artifact of clustering plus
  cross-platform correlation filtering). Blocks tasks.md Section 5 specifically; does not block
  Sections 1-4, 6-10.

## Migration Plan

Purely additive — no existing function, config, or pipeline behavior changes. New module
(`cross_platform_prediction.py`), new result type (`CrossPlatformPredictionResult`), new tests and
fixtures only. No existing caller is affected.

## Open Questions

**Blocking:** Decision 2's trait-set identity oracle mechanism — a handoff investigation has been
requested (external vault access needed); tasks.md Section 5 is on hold pending its return.

**Not blocking:** Decision 4's metric-interpretation flag remains open for final confirmation but
does not block implementation (Sections 1-4 already implement it consistently, and it's cheap to
revise if wrong before Section 6 consumes it). Tier 3.5's config-wiring question (where
`PredictionConfig` nests) and Tier 4's permutation/figure design remain open in their own tiers.

## Adversarial Review Reconciliation (round 1)

`/review-openspec` ran 5 parallel reviewers (spec quality, TDD/testing, statistical correctness,
documentation, git workflow) against this proposal before any implementation began. 2 BLOCKING and
9 IMPORTANT findings, reconciled as follows:

- **BLOCKING — trait-set identity oracle tests the wrong substrate/quantity.** Two reviewers
  independently found this; confirmed by direct code read (`reduce_trait_redundancy.py:208`'s
  plain `.groupby().mean()`, predating Tier 1's BLUP extraction) and by counting a real committed
  fixture (`tests/fixtures/real/wheat_edpie/...root_core_vs_cylinder/exp{1,2}_trait_clusters.csv`:
  28 field / 121 cylinder representatives, not 14/28). Fixed: Decision 2 rewritten; Section 5
  blocked pending a handoff investigation into the real Section 3.4 pipeline (cluster → correlate
  representative pairs → filter |ρ|≥0.55 → count distinct traits per side, reconstructed from
  `results_3.4_draft_20260330.md`'s exact wording but not yet confirmed).
- **BLOCKING — planted-signal/pure-noise fixtures don't recover their claimed R² at theory.md's
  literal single-seed parameters.** Independently reproduced: seed=42 gives R²=0.469 (ridge, not
  "≈0.8"); seeds 1-7 range -0.67 to 0.87. Fixed: Decision 6 redesigns both fixtures as N=20-seed
  averages (`n_traits=3`, empirically verified mean separation ~1.0 between signal and noise),
  asserting a comfortable margin between means rather than closeness to a single-realization
  target. Flagged as a `theory.md` (Tier 0) erratum candidate, not just a Tier 3 fixture fix.
  Verified with real sklearn (`uv run python`) during this review round, not just reasoned about
  — 50-seed sweep across `n_traits` ∈ {3,5,8,10} and Ridge `alpha` ∈ {1.0, 0.1}.
- **IMPORTANT — `representative_indices` name-vs-index ambiguity.** `select_cluster_representatives`
  returns trait names, never integer positions; `logo_cv_predict`'s bare-`ndarray` `X` had no way
  to resolve names to columns. Fixed: Decision 7 — `X` becomes a labeled `DataFrame`,
  `representative_indices` renamed to `representative_names: Optional[Sequence[str]]`.
- **IMPORTANT — `predictor_source` "runtime guard" language oversold what Tier 3 actually
  validates.** No spec scenario, task, or function parameter ever defined what the guard checks.
  Fixed: Decision 8 downgrades this to plain provenance metadata on
  `CrossPlatformPredictionResult`; the real guard is confirmed as Tier 3.5's
  `PredictionConfig.__post_init__` scope, not this tier's.
- **IMPORTANT — missing input-validation scenarios/tests.** Mismatched array lengths, invalid
  `reduction_method`, `representative_names=None` with `reduction_method="representatives"`, too
  few genotypes for LOGO-CV, zero-variance `y`, NaN in `X` (concretely reachable via Tier 1's own
  NaN-column contract on `08_blup_adjusted_means.csv`, which this tier's manual validation task
  directly consumes) were entirely unaddressed. Fixed: tasks.md gains an explicit input-validation
  test subsection in Section 3; the `cross-platform-prediction` spec gains corresponding
  scenarios.
- **IMPORTANT — `proposal.md`'s Impact section mislabeled `serializable-result-types` as
  "(MODIFIED)"** when the delta is entirely `## ADDED Requirements` and the proposal itself states
  no existing behavior is touched. Fixed: label corrected to `(ADDED)`.
- **IMPORTANT — Section 6 (`CrossPlatformPredictionResult`) had no test-first task for its
  package-root export**, unlike Section 7's `fit_pca_on_fold`/`logo_cv_predict` export (which
  correctly has 7.1 before 7.2). Precedent: `test_blupresult_importable_from_root`. Fixed: tasks.md
  6.6 split into a failing-test task followed by the export.
  `docs/CROSS_PLATFORM_ANALYSIS.md` (the actual narrative home for cross-platform program docs,
  per the immediately-preceding `pc_correlations` tier's precedent) was missing from the Docs
  section entirely. Fixed: tasks.md gains a 9.4 documenting `logo_cv_predict`/`fit_pca_on_fold`
  there, mirroring the `pc_correlations` section's shape.
- **IMPORTANT (statistical, documented not fixed) — Ridge `alpha`, RMSE cross-trait scale,
  Spearman p at small n, X/y-exclusion contract.** See Decision 9 — all four resolved by explicit
  documentation rather than new code.
- **SUGGESTION — Decision 4 restated near-verbatim in 3+ places.** Not reconciled by trimming in
  this round (low priority relative to the BLOCKING items above); left as a candidate follow-up
  cleanup, not required before approval.
- **SUGGESTION — `__init__.py`'s `__all__` grouping-by-comment-header convention** (see the
  `pc_correlations` precedent at `__init__.py:362`) should be followed when adding the new names —
  noted in tasks.md 6.6/7.2 rather than as a design.md decision (purely a code-style pointer, not a
  design choice).
