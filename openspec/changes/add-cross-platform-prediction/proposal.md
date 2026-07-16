## Why

The wheat EDPIE cross-platform paper currently reports **correlation** between platforms
(Turface19/Turface150/Cylinder/Field) on shared traits. Per Wolfgang's 2026-04-03 ask, this should
be reframed as **predictability**: given genotype BLUPs estimated within one platform (Tier 1,
merged — `add-blup-extraction`, #189/#190), test whether they **predict** genotype BLUPs in
another platform via ridge regression / Partial Least Squares (PLS) with leave-one-genotype-out
(LOGO) cross-validation.

This is Tier 3 (`add-cross-platform-prediction`, tracking issue
[talmolab/sleap-roots-analyze#194](https://github.com/talmolab/sleap-roots-analyze/issues/194))
of the program. Tier 0 (statistical-hygiene audit + `theory.md` grounding) and Tier 1
(`add-blup-extraction`) are done. Tier 2 (`add-heritability-fixed-effects`, merged #193) is also
done — its `fixed_effects` parameter improves field-BLUP quality but is not a hard dependency;
this tier proceeds with genotype-only BLUPs regardless. See the program roadmap and statistical
grounding at
`c:\vaults\sleap-roots\wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md` (external
to this repo; referenced here for provenance only).

## What Changes

- **New module `src/sleap_roots_analyze/cross_platform_prediction.py`** (plain functions, no
  pipeline step — that's Tier 3.5):
  - **`fit_pca_on_fold(X_train, X_test, n_components=1) -> np.ndarray`** — theory.md §5's exact
    contract: a fresh `sklearn.decomposition.PCA` fit on `X_train` only, `X_test` projected onto
    the resulting components. Raises `ValueError` before calling sklearn if
    `X_train.shape[1] < n_components`. Deliberately distinct from the pipeline-level `PCA` step in
    `pca.py` — reusing that step would fit loadings on all genotypes before the fold loop,
    leaking the held-out genotype's position into the PC axes (theory.md §3.1).
  - **`logo_cv_predict(X, y, genotypes, reduction_method="pls_latent", representative_names=None) -> LOGOCVResult`**
    (see design.md for the exact return shape) — implements the CV-hygiene contract from
    theory.md §2-3: a fresh `sklearn.Pipeline` is instantiated and fit **inside** each
    `LeaveOneOut` fold. `X` is a labeled `pandas.DataFrame` (`(n_genotypes, n_traits)`, columns
    named by trait, index by genotype), not a bare `np.ndarray` — see design.md Decision 7 (fixes
    a real name-vs-index integration gap between this function and
    `select_cluster_representatives`'s string-name output, found during `/review-openspec`).
    Three `reduction_method` values: `pls_latent` (default) — `StandardScaler`
    + `PLSRegression(n_components=1, fixed — see design.md Decision 1)` fit directly on the full
    trait matrix, no separate reduction step (PLS supervises its own dimensionality reduction);
    `representatives` — `StandardScaler` + `Ridge()` (default `alpha=1.0`, an accepted but
    undiscussed choice — design.md Decision 9) fit on trait columns selected once, pre-loop, by
    variance-based cluster-representative selection (`representative_names`, unsupervised, safe to
    fix up front per theory.md §2.2); `pc1` — `StandardScaler` + `Ridge()` fit on the single PC1
    score computed **per fold** via `fit_pca_on_fold`. Explicit precondition (design.md Decision
    9): `X`'s columns must never include the target trait's own values.
  - Aggregate CV **R², RMSE, and Spearman ρ** computed once over the concatenated leave-one-out
    predictions (matching theory.md §4.3's `logo_cv_r2` reference implementation exactly — see
    design.md Decision 4 for why this, not a per-single-genotype metric, is the correct reading of
    "per fold").
  - **`predictor_source`**: `{blup, genotype_means}`, stored as **provenance metadata** on
    `CrossPlatformPredictionResult`, not validated by `logo_cv_predict` itself — functions accept
    whatever `X` DataFrame they're given regardless of provenance. The actual pre-flight *guard*
    (rejecting an unresolvable BLUP path) is Tier 3.5's `PredictionConfig.__post_init__` scope, not
    this tier's (design.md Decision 8 — the original "runtime guard" framing overstated what this
    tier validates).
- **New `CrossPlatformPredictionResult` frozen dataclass** in `result_types.py`, following the
  `BLUPResult`/`HeritabilityResult` template exactly: `to_dict()`/`to_json(allow_nan=False)`
  finite-float contract, a `from_*` adapter, added to `__all__` and the package `__init__.py`. One
  instance per (platform pair, reduction method); nested `TargetPrediction` list covers each
  cluster-representative target trait plus PC1 (see design.md Decision 5).
- **Trait-set identity oracle: BLOCKED, not part of this proposal's implementable scope yet.**
  `/review-openspec` found the original design (reproduce 28 cylinder + 14 field traits directly
  from `select_cluster_representatives()` on BLUP-adjusted means) tests the wrong substrate and
  very likely the wrong quantity entirely — see design.md Decision 2's full revision. A real,
  already-committed fixture in this repo shows `select_cluster_representatives` alone gives 28
  field / 121 cylinder representatives, not 14/28; the real Section 3.4 figure appears to be a
  downstream artifact of clustering *plus* cross-platform correlation filtering, not clustering
  alone. **A handoff investigation has been requested** (see the separately-delivered vault
  handoff prompt) to confirm the real mechanism before this oracle can be correctly specced.
  `cluster_correlated_traits`/`select_cluster_representatives` are still reused unchanged for the
  `reduction_method="representatives"` *prediction* path (Section 3) — that does not depend on
  this resolution.
- **Explicit leakage regression test**: theory.md §4's mechanism implemented against a
  **redesigned** planted-signal fixture (design.md Decision 6) — `n_traits=3`, N=20-seed averaged
  R², not theory.md's literal single-seed recipe, which `/review-openspec` empirically found does
  not reliably recover its claimed R² at this program's actual scale (n=19 genotypes). Asserts
  `mean(r2_outside_across_seeds) / mean(r2_inside_across_seeds) >= 1.10`.
- **PC1 per-fold oracle**: `fit_pca_on_fold` is exercised inside the LOGO loop; its R² is reported
  as a separate `TargetPrediction` entry from the representative-trait path, never mixed into the
  same aggregate.
- **Synthetic non-EDPIE generalizability fixture**: a fixture unrelated to the wheat EDPIE
  trait/genotype structure, confirming the machinery isn't accidentally coupled to EDPIE-specific
  column names or genotype counts.
- **Non-CI manual validation task** (tasks.md, gates PR alongside `/pre-merge-check`): regenerate
  real `08_blup_adjusted_means.csv` outputs for the 4 EDPIE platforms by rerunning Tier 1's
  pipeline against the real platform configs, then run `logo_cv_predict` via the Python API on the
  4 directed pairs (Turface19→Cylinder, Turface19→Field, Cylinder→Field, Turface150→Turface19),
  sanity-checking the resulting R²/RMSE/ρ against the known correlation numbers already in the
  roadmap (e.g. Turface19→Cylinder correlation = 0.67, p = 0.002).

No breaking changes — this tier adds a new module and a new result type only; no existing
function, config, or pipeline behavior is touched.

## Design decisions (resolved via brainstorming this session, then revised during `/review-openspec` round 1 — full rationale and alternatives in `design.md`)

- PLS component count fixed at `n_components=1` — no inner-CV search (statistical + Tier-4-runtime
  reasons, design.md Decision 1).
- **BLOCKED, revised round 1:** the trait-set identity oracle's substrate/mechanism — the original
  "genotype-mean/BLUP-level" framing conflated two different matrices, and the target quantity
  (28/14) is very likely not reproducible from clustering alone. Pending a handoff investigation
  (design.md Decision 2).
- Tier 4's permutation-null runtime is estimated and documented (≈152,000 fits, well under the
  30-minute feasibility gate) rather than designed around with parallelization scaffolding now;
  `logo_cv_predict` is written stateless so Tier 4 can wrap it without refactoring (design.md
  Decision 3).
- Reported CV R²/RMSE/ρ are aggregate metrics over concatenated LOO predictions, not a distinct
  score per single held-out genotype — **flagged explicitly for review**, since this is an
  interpretation of ambiguous roadmap/issue phrasing, not a confirmed fact (design.md Decision 4).
- `CrossPlatformPredictionResult` nests one `TargetPrediction` per (representative trait or PC1);
  `comparison_methods` produce separate result instances, not a third nesting level (design.md
  Decision 5).
- **Added round 1:** planted-signal/pure-noise fixtures redesigned as N=20-seed averages after
  `/review-openspec` empirically found theory.md's literal single-seed recipe doesn't hold at this
  program's scale (design.md Decision 6).
- **Added round 1:** `logo_cv_predict` takes a labeled `DataFrame` and `representative_names`
  (trait name list), not a bare `ndarray` + integer indices — fixes a real integration gap with
  `select_cluster_representatives`'s string-name output (design.md Decision 7).
- **Added round 1:** `predictor_source` is provenance metadata, not a validated runtime guard in
  this tier — the real guard is Tier 3.5's `PredictionConfig` scope (design.md Decision 8).
- **Added round 1:** Ridge `alpha` default, RMSE cross-trait scale, Spearman p at small n, and the
  X/y-exclusion precondition are documented as known limitations/contracts, not solved with new
  code (design.md Decision 9).

## Impact

### Affected specs

- `cross-platform-prediction` (ADDED) — new capability: `fit_pca_on_fold`, `logo_cv_predict`, the
  CV-hygiene contract, the leakage regression test, and the synthetic-fixture oracles. The
  trait-set identity oracle requirement is present but its mechanism is **blocked** pending a
  handoff investigation (design.md Decision 2) — do not implement against it until resolved.
- `serializable-result-types` (ADDED) — new `CrossPlatformPredictionResult` /
  `TargetPrediction` requirement, following the existing frozen-dataclass /
  `to_json(allow_nan=False)` / `from_*` adapter / `__all__` export pattern. (Corrected from an
  earlier draft's "MODIFIED" label, per `/review-openspec` round 1 — this delta is purely
  additive, matching the actual `## ADDED Requirements` delta file.)

### Affected code

- `src/sleap_roots_analyze/cross_platform_prediction.py` (new) — `fit_pca_on_fold`,
  `logo_cv_predict`, supporting CV-metric helpers.
- `src/sleap_roots_analyze/result_types.py` — new `CrossPlatformPredictionResult` and
  `TargetPrediction` dataclasses, `from_*` adapter, `__all__` entries.
- `src/sleap_roots_analyze/__init__.py` — export the new public names.
- `tests/fixtures.py` — new planted-signal, pure-noise, and synthetic non-EDPIE fixtures.
- `tests/test_cross_platform_prediction.py` (new) — all oracle tests from issue #194's
  acceptance criteria.
- `docs/API.md`, `docs/CHANGELOG.md`, `docs/result-types.md` — new module/result-type entries.
- `docs/CROSS_PLATFORM_ANALYSIS.md` — new section documenting `logo_cv_predict`/`fit_pca_on_fold`,
  mirroring the immediately-preceding `pc_correlations` tier's section shape (added per
  `/review-openspec` round 1 — this is the actual narrative home for cross-platform program docs
  and was missing from the original Impact list).

### Explicitly out of scope

- Tier 3.5 (`PredictionConfig`, `PredictCrossPlatformStep`, CLI wiring, the manual all-4-pairs
  pipeline integration test) — separate future change.
- Tier 4 (permutation null, figures) — separate future change.
- Any change to `calculate_heritability_estimates`, `extract_blup_table`, `BLUPResult`,
  `cluster_correlated_traits`, or `select_cluster_representatives` — all reused as-is.
- PLS component-count search (inner-CV) — deferred; see design.md Decision 1's alternatives.
- `joblib.Parallel` support for the permutation loop — Tier 4's concern if the measured runtime
  requires it; not built speculatively now (design.md Decision 3).
