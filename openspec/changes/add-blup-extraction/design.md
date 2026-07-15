## Context

`calculate_heritability_estimates` (`statistics.py:195-448`) fits
`smf.mixedlm("value ~ 1", model_data, groups=genotype)` per trait with REML and
extracts only `result.cov_re` (σ²_G) and `result.scale` (σ²_E) to compute H².
`result.random_effects` — a dict mapping genotype → estimated random effect
(the BLUP) — is a lazy property on the same fitted `result` object and is
never touched. This is the entry point for Tier 1 of the wheat EDPIE
cross-platform genotype prediction program (see `c:\vaults\sleap-roots\
wheat-edpie-paper\cross-platform-prediction\{roadmap,theory}.md`, external to
this repo): BLUP-adjusted genotype means are the predictor substrate for later
tiers (ridge/PLS + leave-one-genotype-out CV across platforms), which are out
of scope here.

The serializable-result-types epic (#130) already established the pattern this
change follows: `PCAResult`, `HeritabilityResult`, `ClusterResult`/
`KMeansResult`/`GMMResult`, and `UMAPResult` are frozen dataclasses built from
a legacy dict via a `from_*_dict()` adapter, holding only JSON-native fields
with a `to_json(allow_nan=False)` finite-floats contract. `HeritabilityResult`
in particular already separates succeeded traits (`per_trait`) from failed
ones (`failed_traits`, names only) — the direct precedent for how `BLUPResult`
handles a trait whose mixed model failed.

The roadmap flagged one design question as blocking for this tier: return
BLUPs inside the existing heritability dict, or as a new `BLUPResult`
dataclass? This was resolved via `superpowers:brainstorming` this session (see
resolutions below), after reading `result_types.py` to confirm the existing
pattern.

## Goals / Non-Goals

- **Goals:** extract `result.random_effects` (accessed exactly once per
  successful trait fit) and `result.fe_params["Intercept"]`; a genotype ×
  trait adjusted-means table (`extract_blup_table()`) with a genuine `NaN`
  column for failed traits; a `BLUPResult` dataclass satisfying the
  finite-floats contract; a `08_blup_adjusted_means.csv` pipeline output behind a
  default-`True` config flag; no breaking changes to
  `calculate_heritability_estimates`'s two existing return shapes.
- **Non-Goals:** changing the H² formula or variance-component extraction;
  Tier 2's fixed-effects formula change (`R_j` handling); any LOGO-CV,
  ridge/PLS, or prediction machinery (Tier 3+); a new pipeline step (BLUP
  output is folded into the existing `StatisticalAnalysisStep`, not a separate
  step); reconciling `08_blup_adjusted_means.csv`'s trait set with
  `FilterHeritabilityStep`'s later low-H² trait removal — `StatisticalAnalysisStep`
  always calls `calculate_heritability_estimates(..., remove_low_h2=False)`
  (heritability-based filtering is Step 9, not Step 8), so the BLUP CSV
  intentionally includes columns for traits a later step may drop from the
  final high-heritability trait set. Reconciling the two is a Tier 2+ concern.

## Decisions

### Decision 1: `blup`/`intercept` are additive keys, not a new opt-in parameter

**What:** When a trait's mixed model succeeds, its per-trait result dict gains
`blup: dict[str, float]` and `intercept: float`, unconditionally — no new
parameter on `calculate_heritability_estimates`.

**Why:** The extraction is free (the model is already fit; accessing
`result.random_effects` and `result.fe_params` costs nothing beyond what's
already computed). An opt-in parameter would need to be threaded through the
one caller (`StatisticalAnalysisStep`) for no benefit, and would let a caller
silently miss BLUPs by forgetting to pass it. Both existing return shapes
(dict / 4-tuple) are dicts-of-dicts already, so adding keys to an inner dict
cannot change the outer shape — genuinely non-breaking.

**Implementation detail that matters for correctness:** the per-trait success
dict literal at `statistics.py:414-428` is shared by *two* independent success
paths — the mixed-model branch (`if use_mixed_model:`, has a fitted `result`)
and the ANOVA-based branch (`else:`, `force_method="anova_based"`, computes
variance components directly from groupby arithmetic and never calls
`smf.mixedlm` — no `result` object exists). A third success path, the
no-variance short-circuit (`subset[trait].nunique() == 1`), builds its own
dict and `continue`s before reaching that shared literal at all, so it's
unaffected either way. `blup`/`intercept` MUST be computed as local variables
inside the mixed-model branch only (defaulting to unset/`None` otherwise) and
added to the shared dict conditionally — an unconditional
`result.random_effects` reference at the shared literal would raise
`UnboundLocalError` whenever a trait takes the ANOVA-based path, since
`force_method="anova_based"` is a real, user-selectable code path
(`calculate_heritability_estimates(..., force_method="anova_based")`), not a
hypothetical.

**Alternatives considered:**
- **New `extract_blups: bool = False` parameter.** Rejected: more
  conservative, but adds signature complexity for a change that costs nothing
  to always perform, and risks the pipeline step forgetting to opt in.
- **Separate function that refits the same mixedlm models.** Rejected:
  wastes a full model refit per trait; theory.md is explicit that "the model
  only needs to be fit once per trait."

### Decision 2: `BLUPResult` dataclass, not BLUPs folded into the heritability dict

**What:** A new frozen `BLUPResult` in `result_types.py`, built via
`BLUPResult.from_blup_table(df, intercepts=...)` from the `extract_blup_table()`
DataFrame — not a modification to `HeritabilityResult`'s shape.

**Why:** This was the roadmap's explicitly flagged blocking question,
recommended answer `BLUPResult`, confirmed by reading `result_types.py`
first: `HeritabilityResult` is per-trait-shaped (one `TraitHeritability` row
per trait); BLUPs are genotype × trait-shaped (one row per genotype, columns
of traits) — a fundamentally different axis of aggregation that doesn't fit
`HeritabilityResult`'s per-trait row schema. A separate type keeps each
dataclass's shape aligned with what it actually represents, matching how
`PCAResult` (component-indexed) and `ClusterResult` (sample-indexed) are
already separate types for different aggregation axes.

**Alternatives considered:**
- **Add a `blup_table` field to `HeritabilityResult`.** Rejected: forces a
  genotype × trait matrix onto a dataclass whose every other field is
  per-trait-indexed; `from_heritability_dict()` would need a second input
  shape (the DataFrame) that today's `HeritabilityResult` adapter doesn't
  take.

### Decision 3: Failed traits — NaN in the CSV, excluded (not NaN) in the dataclass

**What:** `extract_blup_table()` returns a `pd.DataFrame` with a genuine `NaN`
column for any failed trait (the CSV oracle: "not silently dropped, not
zero"). `BLUPResult.from_blup_table()` then splits that DataFrame: finite
columns become `trait_names`/`adjusted_means`; all-`NaN` columns become
entries in `failed_traits` (names only, no values) and are excluded from the
numeric matrix.

**Why:** `BLUPResult.to_json()` enforces `allow_nan=False`, matching every
sibling result type's finite-floats contract. Storing `NaN` inside
`adjusted_means` would either violate that contract (raise on every normal run
with any failed trait) or require weakening it (deviating from the shared
contract other result types enforce, letting a `NaN` leak across the JSON
boundary a strict consumer like bloom-mcp rejects). Excluding failed columns
and naming them separately mirrors `HeritabilityResult.per_trait`/
`failed_traits` exactly — the same split, one level up (genotype × trait
matrix instead of per-trait scalar).

**Alternatives considered:**
- **`to_json(allow_nan=True)` override on `BLUPResult` only.** Rejected:
  breaks the shared contract other result types enforce for no reason other
  than convenience; the CSV already satisfies the "must show NaN" oracle, so
  the dataclass doesn't need to duplicate it.
- **Zero-filled matrix with a boolean success mask.** Rejected: the roadmap
  oracle explicitly says "not zero" — a placeholder zero risks a consumer
  reading it as a real adjusted mean if the mask is forgotten.

**Two edge cases the "all-NaN column" framing doesn't cover on its own:**
- **Cell-level gaps.** `calculate_heritability_estimates` computes each
  trait's genotype set independently (`subset = df[[trait, genotype_col]].dropna()`
  per trait), so two *both-succeeded* traits can legitimately cover different
  genotype sets. A genotype missing from one succeeded trait's `blup` dict but
  present in another's produces a single `NaN` cell in an otherwise-finite
  column — not a whole failed column. `BLUPResult.from_blup_table()` treats
  this the same as a fully-failed trait (excluded from `trait_names`, name in
  `failed_traits`): a column needs to be *entirely* finite to enter the
  always-finite matrix, so a partially-finite column is not eligible either.
  This is simpler than a per-cell exception and preserves the "matrix is
  rectangular and complete" property `PCAResult`/`ClusterResult` also rely on.
- **Zero succeeded traits.** `pd.Series([], dtype=float).notna().all()`
  evaluates to `True` in pandas (vacuous truth) — so a naive
  `df[col].notna().all()` partition would misclassify an all-failed-traits
  input (zero genotype universe, zero-row columns) as "all succeeded." The
  adapter's partition logic must explicitly special-case a zero-row column as
  failed, not rely on `.notna().all()` alone.

### Decision 4: Module placement — `statistics.py` / `result_types.py` split

**What:** `extract_blup_table()` lives in `statistics.py`; `BLUPResult` lives
in `result_types.py`. No new `blup.py` module.

**Why:** Matches the existing one-way dependency rule stated in
`result_types.py`'s module docstring ("this module imports nothing from the
analytical modules") and the precedent of every other analytical
function/result-type pair (`perform_pca_analysis`/`PCAResult`,
`calculate_heritability_estimates`/`HeritabilityResult`,
`perform_umap_analysis`/`UMAPResult`).

**Alternatives considered:**
- **New `blup.py` module** for both. Rejected for Tier 1: no evidence yet that
  BLUP logic needs a dedicated file; if Tier 2/3 substantially grow the BLUP
  surface, this can be revisited then without churn now.

### Decision 5: Config flag on `StatisticsConfig`, default `True`

**What:** `StatisticsConfig.generate_blup_table: bool = True`, gated on
`calculate_heritability` also being `True`.

**Why:** `StatisticsConfig` already owns `calculate_heritability` (the flag
this one is a free byproduct of); grouping them keeps the gating relationship
in one dataclass. Default `True` per the roadmap ("free once the model is
fit") and matches how a brand-new pipeline run should get the new CSV without
any config change — purely additive from a user's point of view.

**Alternatives considered:**
- **New field on `HeritabilityConfig`** (alongside `generate_diagnostics`).
  Rejected: `HeritabilityConfig.enabled` gates *filtering*, a downstream step
  (`FilterHeritabilityStep`) — BLUP extraction happens in
  `StatisticalAnalysisStep`, before filtering, and depends only on
  `calculate_heritability`, not on whether filtering is enabled. Coupling it to
  `HeritabilityConfig` would make BLUP output depend on a step it doesn't
  actually run before.
- **Warn (not raise) when `generate_blup_table=True` and
  `calculate_heritability=False`**, mirroring the `umap.enabled`-on-an-
  unwired-path precedent (`pipeline/config/utils.py` warns that setting
  "will be ignored"). Tried first, then rejected on closer inspection during
  round-2 review: the analogy doesn't hold up. `umap.enabled` defaults
  `False`, so its warning only fires on a rare, deliberate opt-in into a
  feature that would otherwise look silently broken. `generate_blup_table`
  defaults `True`, so the equivalent warning would fire whenever a user does
  the ordinary, legitimate thing of disabling heritability — confirmed
  concretely: four existing tests in `tests/test_step_statistical_analysis.py`
  already construct exactly this configuration
  (`calculate_heritability=False`, `generate_blup_table` left at its default)
  and would start emitting the warning on every run once implemented.
  Warning from both `validate_viz_config()` (config-load time) and the step's
  runtime path (needed for the QC pipeline, which has no config-validation
  entry point) also meant a Viz-pipeline user would see the same warning fire
  twice per run. Unlike `umap.enabled` (an explicit opt-in into a stub),
  "BLUPs need the same model fit heritability does" is a self-evident
  consequence already stated in the config field's own docstring — closer to
  the *already-accepted* silent no-op precedent
  (`calculate_heritability=True` + `heritability.enabled=False`) than to the
  UMAP stub case.

**Decision (revised after round-2 review): silent no-op, documented in the
config field's docstring, no warning.** `generate_blup_table=True` while
`calculate_heritability=False` simply produces no BLUP output — no exception,
no warning — consistent with the project's other "one flag makes another
irrelevant" precedent.

## Risks / Trade-offs

- **`StatisticalAnalysisStep` runs in both the QC and Viz pipelines, but only
  `VizPipelineConfig` composes `StatisticsConfig`.** `QCPipelineConfig` has no
  `statistics` field; the step already resolves this for
  `calculate_heritability` via `getattr(config, "statistics", None)`
  (defaulting to `True` when absent). `generate_blup_table` must use the same
  guard, or reading `config.statistics.generate_blup_table` directly raises
  `AttributeError` on every QC-pipeline run — this is not a hypothetical edge
  case, it is the primary pipeline's default code path.
- **Two boolean flags gating one output** (`calculate_heritability` and
  `generate_blup_table`) could confuse a user who sets `generate_blup_table=
  True` while `calculate_heritability=False` and gets no CSV. A warning was
  tried first (see Decision 5) but reversed after round-2 review found it
  would fire on every run of an ordinary, legitimate configuration
  (heritability disabled), not just a rare deliberate misconfiguration —
  concretely demonstrated by four existing tests that already hit this
  combination, and it would additionally double-fire for the Viz pipeline
  (once in `validate_viz_config()`, once at runtime). Mitigated instead by
  clear docstring documentation on `StatisticsConfig.generate_blup_table`
  stating the dependency explicitly, consistent with the project's other
  accepted silent-no-op precedent (`calculate_heritability=True` +
  `heritability.enabled=False`).
- **`BLUPResult.intercepts` duplicates information also derivable from
  `adjusted_means` and a genotype's raw BLUP** — kept as an explicit field
  (not reconstructed) because the raw per-genotype BLUP values themselves are
  *not* stored on `BLUPResult` (only the already-summed adjusted means are),
  so `intercepts` is the only way to recover `blup[g] = adjusted_means[g] -
  intercepts[trait]` if a future consumer needs the decomposition.
- **`extract_blup_table()` takes the `heritability_results` dict, not raw
  data** — consistent with the accepted precedent from
  `add-heritability-diagnostics` (Decision 1 in that design: "Diagnostic
  Functions Accept Pre-Calculated Results... avoids redundant computation").

## Migration Plan

Purely additive — no existing caller changes required:
- `calculate_heritability_estimates` callers reading `result[trait]["heritability"]`
  etc. are unaffected by the new `blup`/`intercept` keys.
- `StatisticalAnalysisStep` gains one new output file, on by default; a config
  that doesn't set `generate_blup_table` gets it for free.
- `BLUPResult` and `extract_blup_table` are new, opt-in surface.

No rollback concerns beyond reverting the additive commits — no existing
behavior is modified.

## Open Questions

None blocking Tier 1. Tier 2's `R_j` (fixed vs. random effect) question is
answered tentatively in `proposal.md` for continuity but is out of scope here.
