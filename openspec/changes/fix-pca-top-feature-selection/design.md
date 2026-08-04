## Context

`pca.n_top_features` is read in exactly one place,
`PCAAnalysisStep.execute()` (`pca_analysis.py:109-115`):

```python
top_feature_indices = select_top_features_from_pca(
    loadings=pca_results["loadings"],
    eigenvalues=pca_results["eigenvalues"],
    n_features_total=len(feature_names),
    n_features_to_select=config.pca.n_top_features,
    method=config.pca.feature_selection_strategy,
)
```

No `pc_indices` is passed, so `select_top_features_from_pca()` defaults to
`[0, 1]` (#203). `n_top_features` itself is a hand-picked integer with no
principled basis — every active config that uses
`feature_selection_strategy: "extreme"` sets a different value (`1` or `5`),
and #206 asks: is there a non-arbitrary alternative?

For `"extreme"`, yes: "1 most-positive-loading trait + 1 most-negative-
loading trait per retained PC" requires no count at all — it falls out of
calling `select_top_features_from_pca(n_features_to_select=1,
pc_indices=list(range(n_components)))`. For `"top_variance"`, the codebase
already has a precedent for count vs. threshold: `PCAConfig.n_components`
(`< 1` = variance-ratio threshold, `>= 1` = fixed count).

## Goals / Non-Goals

- Goals:
  - Resolve #203: `pca_analysis.py` passes `pc_indices` explicitly, scoped
    to all retained PCs.
  - Resolve #206 Part 1: `"extreme"` selection has no arbitrary count.
  - Resolve #206 Part 2: `"top_variance"` supports a variance-fraction
    threshold, mirroring `n_components`'s convention.
  - Leave `select_top_features_from_pca()`'s public signature unchanged
    (bloommcp calls it/its siblings directly with plain-int counts for
    unrelated purposes — `top_n_features` on `create_pca_biplot`,
    `n_traits` on `create_umap_colored_by_top_traits`).
- Non-Goals:
  - Threshold support for `"top_absolute"` / `"top_contribution"` — no
    active config uses them; deferred, and explicitly guarded against being
    invoked with a `< 1` value so the gap is loud, not silent.
  - Widening `pca.feature_selection_strategy`'s validation enum to accept
    `"vector_length"` — see Decision 3.
  - Changing `create_pca_biplot`'s `top_n_features` or
    `static_viz.pca_biplot_top_features` — a separate, unrelated config
    field.
  - A step-level `PCAAnalysisStep` defense-in-depth guard mirroring
    `FilterHeritabilityStep`'s — see Decision 3.

## Decisions

### Decision 1: What happens to `n_top_features` under `"extreme"`?

**Chosen: document it as ignored (issue's option "b"), not a validation
error (option "a").**

Rationale:

1. **Direct precedent in the same function.** `select_top_features_from_pca()`'s
   own docstring already documents that `pc_indices` is "ignored entirely by
   `top_variance`, which always ranks across every retained PC ... regardless
   of what is passed here" (`pca.py:44-48`) — no runtime warning, just an
   explicit doc contract. `n_top_features` under `"extreme"` is the same
   shape of problem (a parameter some methods don't use) and should follow
   the same, already-established resolution for consistency within one
   file.
2. **"Explicitly set" is not cleanly detectable.** Option (a) requires
   distinguishing "the user wrote `n_top_features: 10`" from "the user
   didn't set it and got the default of `10`" — both parse to the identical
   dataclass value once OmegaConf resolves the config. Doing this properly
   would mean adding raw-YAML-presence tracking (the "explicit config
   validation" tier project.md describes for a few load-bearing fields like
   `cleanup.max_nan_fraction`) purely to gate a warning on an otherwise
   harmless, ignored field — disproportionate machinery for this problem.
   A simpler value-based check (`n_top_features != default`) is worse: it
   would misfire on every config that happens to type the same number as
   the default.
3. **It isn't a footgun.** Unlike the `feature_selection_strategy` enum
   check (an invalid string breaks execution) or the new `< 1` validation
   added in Decision 3 below (a fractional value breaks count-based
   methods), a stale `n_top_features` value under `"extreme"` doesn't
   produce a wrong or crashing result — it produces *no* result, cleanly,
   because it's never read. The "no silent no-ops" precedent from #204 was
   about a field masquerading as load-bearing while doing nothing with zero
   documentation trail; here the ignoring is documented at the point of
   definition.

**Cheap additional safeguard, adopted after adversarial review**: although
no validation error fires, `PCAAnalysisStep.execute()` SHALL emit a single
`logger.info()` whenever `feature_selection_strategy == "extreme"`, stating
plainly that `n_top_features` is not read for this method. This costs
nothing (no explicit-vs-default detection needed — the statement is true
unconditionally whenever `"extreme"` is selected, default or not, so there
is no "misfire" concept to guard against) and closes the gap an adversarial
review pass correctly identified: a config from before this change (in
this repo, a fork, or a collaborator's local copy) that still carries a
stale `n_top_features: 5` next to `"extreme"` will otherwise get a silently
different `top_features.csv` with zero runtime signal, only a docstring
nobody reads mid-run. `logger.info` (not `warnings.warn`) is chosen
specifically so it cannot be mistaken for an error and cannot break any
existing test that asserts on `warnings.catch_warnings()` emitting nothing
for the zero-variance-handling paths already covered by this step.

Consequence: the 28 active configs pairing `feature_selection_strategy:
"extreme"` with an explicit `n_top_features` (27 under
`configs/active/viz/`, plus the flat pre-reorg duplicate
`configs/active/viz_turface_150genotypes.yaml`; verified via
`grep -rln "n_top_features" configs/active/` intersected with
`grep -rln 'feature_selection_strategy:.*extreme' configs/active/` — zero
matches under `configs/active/qc/`) get that line **removed** (not merely
left alone) as part of this change, so no config file implies a count that
no longer exists for this method.

### Decision 2: `select_n_features_by_variance()` as a new, separate helper

`select_top_features_from_pca()` is a shared function — `create_pca_biplot`
and `create_umap_colored_by_top_traits` also call it, with plain-int counts
(`top_n_features`, `n_traits`) that are unrelated to this redesign and out
of scope for #206. Changing `select_top_features_from_pca()`'s own
`n_features_to_select` parameter to accept `< 1` thresholds would force
every caller to reason about the overload, including two functions this
issue explicitly does not touch.

Instead, `select_n_features_by_variance(feature_contributions_df,
threshold) -> int` is a new, small function that only `PCAAnalysisStep`
calls, structurally mirroring `select_n_components()`
(`pca.py:144-192`):

```python
def select_n_features_by_variance(feature_contributions_df, threshold):
    if len(feature_contributions_df) == 0:
        raise ValueError("feature_contributions_df must have at least one row")
    cumulative = feature_contributions_df["fractional_contribution"].cumsum()
    if threshold <= 0:
        n = 1
    elif cumulative.iloc[-1] >= threshold:
        n = int(np.argmax(cumulative.to_numpy() >= threshold)) + 1
    else:
        n = len(feature_contributions_df)
    return max(1, min(n, len(feature_contributions_df)))
```

(`threshold <= 0` is handled explicitly rather than left to fall out of
`np.argmax(cumulative >= 0)`, which is technically also `0` → `n=1` for any
non-empty, non-negative-contribution input — the explicit branch documents
the intent rather than relying on that coincidence.)

It consumes `perform_pca_analysis()`'s existing `feature_contributions`
DataFrame (`pca.py:899-904`, already sorted descending by
`total_contribution`, with a `fractional_contribution` column summing to 1)
— no new computation, just a cumulative-sum stopping rule. `PCAAnalysisStep`
calls it only when `feature_selection_strategy == "top_variance"` and
`n_top_features < 1`, to resolve a concrete int, then calls
`select_top_features_from_pca()` exactly as it does today with that int.

**Revised after PR code review (originally "considered and deliberately not
done" — reversed below):** the first draft of this section argued against
extracting a shared `_first_index_crossing_threshold(cumulative, threshold,
total) -> int` helper for `select_n_features_by_variance()` and
`select_n_components()`'s independently-implemented
`np.argmax(cumulative >= threshold) + 1` crossing rule, reasoning that two
call sites didn't justify the abstraction. An adversarial PR review pass
found a second, more consequential instance of the *same* pattern —
`select_top_features_from_pca()`'s `"top_variance"` branch and
`perform_pca_analysis()`'s `feature_contributions` construction
independently recompute the identical `Σ eigenvalue · loading²` formula
(see Decision 5 below) — and that the resulting float-summation-order
divergence, while negligible today, is exactly the kind of "two
independent re-derivations of one formula" this project's reproducibility
values ask to be treated as a real, not theoretical, concern. Given a
second occurrence of the same underlying pattern surfaced independently,
"only two call sites" was the wrong bar — extract both shared helpers now
rather than wait for a third:

```python
def _first_index_crossing_threshold(
    cumulative: np.ndarray, threshold: float, total: int
) -> int:
    if cumulative[-1] >= threshold:
        n = int(np.argmax(cumulative >= threshold)) + 1
    else:
        n = total
    return max(1, min(n, total))
```

`select_n_components()` calls it with `cumulative_variance`,
`explained_variance_threshold`, `max_components`; `select_n_features_by_variance()`
calls it with the `fractional_contribution` cumulative sum, `threshold`,
`n_total` (after its own `threshold <= 0` special-case, which has no
`select_n_components()` analog and stays local to that function). Both
callers' existing test suites are unchanged assertions — this is a
behavior-preserving refactor (byte-identical outputs), not a new
capability, verified by running the full existing regression suite for
both functions after the extraction.

**Known, accepted footgun**: `n_top_features == 1.0` takes the `>= 1`
count branch (selects exactly 1 feature), not "100% of variance." This
diverges from the more intuitive reading a user might bring from
`n_components: 1.0`-style thinning, where `1.0` is a boundary threshold
value, not a count. This is an intentional consequence of reusing the
existing `< 1` / `>= 1` convention exactly as `n_components` already
defines it (mirroring, not reinterpreting, that convention per the Goals
above) — documented in the `PCAConfig.n_top_features` docstring and in the
spec's scenario for the `1.0`/boundary case (see the `visualization-pipeline`
delta), not silently left to surprise a user.

### Decision 5: unify the two independent variance-contribution formulas (added after PR review)

`perform_pca_analysis()` computes each feature's total variance
contribution vectorized — `total_contributions = np.sum(loadings_used**2 *
eigenvalues_used, axis=1)` (`pca.py:925`, pre-refactor line numbers) — to
build the `feature_contributions` DataFrame `select_n_features_by_variance()`
consumes. `select_top_features_from_pca()`'s `"top_variance"` method
independently re-derives the *same* quantity via a Python accumulation
loop — `for i in range(n_pcs): contributions += eigenvalues[i] *
loadings[:n_features, i] ** 2` (`pca.py:117-125`, in the same, pre-existing
function this proposal does not otherwise modify). Mathematically
identical; not guaranteed bit-identical, since `np.sum`'s internal
pairwise-summation order differs from a naive per-column accumulation
loop. In the vanishingly rare case of an exact tie at a selection boundary,
this could make `select_n_features_by_variance()`'s resolved *count*
(computed from one summation order) disagree by one feature with what
`select_top_features_from_pca(method="top_variance")`'s actual *selection*
(the other summation order) would rank as "top N" — deterministic on any
given run, but a silent discrepancy between the two, and exactly the kind
of "two paths computing the same scientific quantity, unverified to
agree" risk this project's statistical-rigor conventions ask to be
eliminated at the source rather than documented as a caveat.

**Fix**: extract a single shared helper both call:

```python
def _total_variance_contribution(loadings, eigenvalues, n_features=None):
    if n_features is not None:
        loadings = loadings[:n_features]
    n_pcs = min(loadings.shape[1], len(eigenvalues))
    return np.sum(loadings[:, :n_pcs] ** 2 * eigenvalues[:n_pcs], axis=1)
```

`perform_pca_analysis()` calls it with its full `loadings`/`eigenvalues`
(no `n_features` restriction — it already operates on the complete
feature set); `select_top_features_from_pca()`'s `"top_variance"` branch
calls it with its own `n_features` (already computed at the top of that
function as `min(n_features_total, loadings.shape[0])`). Given the same
`loadings`/`eigenvalues` inputs — which is always true along the
`PCAAnalysisStep` → `perform_pca_analysis()` → `select_top_features_from_pca()`
call chain this proposal's own scoping fix (#203) established — the two
now compute the *literal same array*, not just a numerically close one.
This is a behavior-preserving refactor for every existing caller of
`select_top_features_from_pca()`, including the two out-of-scope callers
(`create_pca_biplot`, `create_umap_colored_by_top_traits`) that pass their
own `top_n_features`/`n_traits` counts — their `"top_variance"` behavior is
unchanged (same formula, now computed one way instead of two).

### Decision 3: Reject invalid `n_top_features` values for methods that don't support a threshold, at config-validation time — and explicitly not with a step-level guard

`"top_absolute"` and `"top_contribution"` still read `n_top_features` as a
plain count inside `select_top_features_from_pca()` (e.g.
`np.argsort(...)[::-1][:n_features_to_select]`). Two distinct footguns need
rejecting:

1. A `< 1` float passed through unchanged would silently truncate —
   `int(0.5)` is `0`, so `[:0]` selects nothing, with no error.
2. A non-integer `>= 1` float (e.g. `5.7`) would also silently truncate —
   `int(5.7)` is `5` — with no indication the fractional part was dropped.
   This applies to `"top_variance"`'s own `>= 1` count branch too, not just
   `"top_absolute"`/`"top_contribution"`: nothing about the count branch is
   `"top_variance"`-specific once `n_top_features >= 1`.

Both `validate_qc_config()` and `validate_viz_config()`
(`pipeline/config/utils.py`) already have a PCA-config validation block
(`n_components <= 0` → error, `feature_selection_strategy` enum check) —
this adds two sibling checks in the same block:

```python
strategy = config.pca.feature_selection_strategy
n_top = config.pca.n_top_features

if n_top < 1 and strategy not in ("extreme", "top_variance"):
    raise ValueError(
        "pca.n_top_features < 1 (variance-fraction threshold) is only "
        "supported for feature_selection_strategy='top_variance'; got "
        f"'{strategy}' with n_top_features={n_top}. Use an integer "
        ">= 1 for this strategy."
    )
if n_top >= 1 and strategy != "extreme" and abs(n_top - round(n_top)) > 1e-9:
    raise ValueError(
        f"pca.n_top_features must be a whole number when >= 1 (got "
        f"{n_top} for feature_selection_strategy='{strategy}'); the "
        "fractional part would be silently truncated."
    )
```

(A tolerance-based comparison, not exact `!=`, per adversarial review — free
hardening against a future config-generation path that computes
`n_top_features` rather than hand-typing it as a literal. As of this
proposal, every `n_top_features` value in this codebase's configs and
tests is a hand-typed literal — grepped and confirmed — so this is
defensive, not fixing a live bug.)

`"extreme"` is excluded from **both** checks because the value is ignored
entirely for that method (Decision 1) — any value is harmless there and
shouldn't be flagged as an error.

**`"vector_length"` is deliberately not named in either check.**
`pca.feature_selection_strategy`'s pre-existing validation enum in the same
two functions (`valid_pca_strategies = ["extreme", "top_absolute",
"top_contribution", "top_variance"]`) has never included `"vector_length"`
as a valid value for *this* field — that string is a valid value only for
the separate `create_pca_biplot(feature_selection=...)` parameter, a
different config surface entirely. Grepping `configs/` confirms no config
anywhere sets `pca.feature_selection_strategy: "vector_length"` (it would
already fail today's enum check if it did). Naming it in the two checks
above would describe an unreachable branch and make the new validation
untestable for that case; widening the enum to make it reachable would be
an unrelated scope expansion. Both are avoided — the checks above name only
`"top_absolute"` and `"top_contribution"` as the count-only strategies.

**No step-level (`PCAAnalysisStep`) defense-in-depth guard is added**,
unlike `FilterHeritabilityStep`'s guard for its own cross-field validation
(`config-management` spec, "FilterHeritabilityStep Defense-in-Depth Guard").
That guard exists for a specific historical reason: `heritability_results`
being empty is a runtime *data* condition (upstream step didn't compute
heritability) that config validation alone cannot see coming, so a
runtime guard is the only place it can be caught. The new
`n_top_features`/`feature_selection_strategy` checks here are pure
*config* conditions, fully knowable at validation time with no dependency
on runtime data — every config-driven pipeline entry point
(`configure-run-all`, the CLI, `run_pipeline`) already calls
`validate_qc_config()`/`validate_viz_config()` before execution per this
project's "Validation at pipeline start" convention (`project.md`,
Configuration Philosophy). A step-level guard would duplicate that same
check for the sole benefit of programmatically-constructed `PCAConfig`
objects that skip validation on purpose — a narrower, lower-value case than
`FilterHeritabilityStep`'s, and not required by #206/#203. Left as an
explicit non-goal (see `proposal.md`'s Impact section) rather than silently
omitted.

### Decision 4: New `ADDED` requirement, not a `MODIFIED` rewrite of "Pipeline Step Parameter Passing"

The existing `visualization-pipeline` requirement "Pipeline Step Parameter
Passing" (`openspec/specs/visualization-pipeline/spec.md:65-262`) is already
large — it covers `GenerateStaticFiguresStep`'s genotype-highlighting
passthrough, `PCAAnalysisStep`'s zero-variance handling, `create_pca_biplot`'s
`feature_selection` dispatch, and `create_umap_colored_by_top_traits`'s
round-robin extreme-selection construction (~20 scenarios, four distinct
functions). None of that existing text needs to change for this proposal —
the new normative behavior (pc_indices scoping, extreme/top_variance/
count-strategy resolution) is purely additive, not a modification of any
existing sentence. Folding it into that requirement via a `MODIFIED` delta
would have meant pasting all ~200 lines of unrelated pre-existing content
just to append a few new paragraphs, and would have grown an
already-overloaded requirement to a fifth concern.

Instead, this proposal adds a new, standalone requirement — "PCA Analysis
Step Feature Selection Resolution" — scoped specifically to
`PCAAnalysisStep`'s `pc_indices`/`n_top_features` resolution logic, under
`## ADDED Requirements` in the `visualization-pipeline` delta. This keeps
the delta small, testable independently, and doesn't touch the existing
requirement's text at all.

## Risks / Trade-offs

- **Output change for every `"extreme"` config.** `top_features.csv`
  content changes for all 28 active configs using `"extreme"` paired with
  an explicit `n_top_features` (scoped to all retained PCs instead of
  PC1/PC2, and exactly 1-per-direction regardless of the old count) —
  and, more broadly, for every active config using `"extreme"` at all
  (49 files total, verified via `grep -rl 'feature_selection_strategy:.*extreme'
  configs/active/`), since the `pc_indices` scoping fix (#203) applies
  regardless of whether `n_top_features` was explicitly set. This is the
  intended fix for #203/#206, not a regression — but it's a real output
  diff, not just a doc/config change. Mitigation: regression tests assert
  the new, principled behavior explicitly (see `tasks.md`); no
  golden-output regeneration is needed because `top_features.csv`/the
  `top_features` metadata list are consumed only by a summary count
  (`generate_summary_viz.py:144`), not compared against fixture goldens.
- **`n_top_features` type change (`int` → `float`) is not safely committed
  on its own.** `select_top_features_from_pca()` slices numpy arrays with
  `n_features_to_select` directly (`np.argsort(...)[::-1][:n_features_to_select]`,
  `pca.py:104,115,125,135`); a bare Python `float` in a slice raises
  `TypeError: slice indices must be integers...`. Changing
  `PCAConfig.n_top_features`'s default to `10.0` in isolation — before the
  call site casts to `int(...)` for the count branches — would crash
  **every** strategy, including `"top_variance"` (the schema default), not
  just `"extreme"`. The schema change, the call-site rewrite, and the
  updated `test_pca_different_feature_selection_strategies` assertion must
  land in one atomic commit (see `tasks.md`); they are not independently
  committable in the order tasks are numbered.
- **Squash-merge means a full revert undoes everything, not just the new
  validation.** If Decision 3's validation later proves too strict for a
  real config, `git revert` of the (squash-merged) PR commit would also
  undo the #203 `pc_indices` fix that other work may have since built on.
  The safe rollback path for an over-strict validation specifically is a
  small, targeted follow-up patch to `pipeline/config/utils.py`, not a
  revert of this change.

## Migration Plan

1. Land `select_n_features_by_variance()` in `pca.py` with unit tests
   (mirrors `select_n_components()` test patterns in `test_pca.py`),
   including the `threshold <= 0` and exact-boundary cases. This function
   is uncalled by production code at this point — zero blast radius,
   independently committable.
2. In a **single commit**: update `PCAConfig.n_top_features`'s type
   (`int` → `float`) and docstring in `components.py`; update
   `PCAAnalysisStep.execute()`'s call site to always pass
   `pc_indices=list(range(n_components))` and branch on
   `feature_selection_strategy`/`n_top_features` to resolve
   `n_features_to_select` per Decisions 1–2 (casting to `int(...)` for
   every count branch); and update `test_step_pca_analysis.py`'s
   regression tests, including rewriting
   `test_pca_different_feature_selection_strategies`'s
   `len(top_features) >= config.pca.n_top_features` assertion, which no
   longer holds for `"extreme"`. See the Risks section above for why these
   three pieces cannot be split across commits without a red-CI window.
3. Add the Decision 3 validation to both `validate_qc_config()` and
   `validate_viz_config()`, with tests in `tests/test_pipeline_config.py`
   and `tests/test_viz_pipeline_config.py` respectively (where each
   function's existing PCA-validation tests already live).
4. Update docs/configs (proposal's "Docs" section): the 28 files
   identified in Decision 1, plus the additional stale-comment locations
   catalogued in `tasks.md` section 4.2.
5. Rollback: see the squash-merge caveat in Risks above — prefer a
   targeted follow-up patch over a full revert if only Decision 3's
   validation needs walking back post-merge.

## Open Questions

None outstanding — both design questions #206 raised are resolved above.
