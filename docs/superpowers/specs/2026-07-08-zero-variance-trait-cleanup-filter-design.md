# Design: configurable zero-variance / constant-trait filter in cleanup

**Date:** 2026-07-08
**Status:** Proposed (brainstormed design, pre-implementation)
**Tracking issue:** [talmolab/sleap-roots-analyze#177](https://github.com/talmolab/sleap-roots-analyze/issues/177)
**Repo:** [talmolab/sleap-roots-analyze](https://github.com/talmolab/sleap-roots-analyze)

## Summary

Add a configurable **zero-variance / constant-trait filter** to the data-cleanup
step so constant traits are dropped **and explicitly named at cleaning time**,
rather than surfacing later as a silent drop inside PCA (or as a hard error in a
downstream wrapper). The fix lives upstream in cleanup so **every consumer
benefits**, not just one downstream caller.

## Motivation

A trait whose values are all identical (zero variance) carries no information for
multivariate analysis. Today the codebase handles such traits **silently and
late**:

- **PCA drops them silently.** `perform_pca_analysis` → `standardize_data`
  (`src/sleap_roots_analyze/pca.py`) drops zero-variance columns
  (keeps `var(ddof=0) > 0`) and only raises when *nothing* survives
  (`df_clean.empty` → `"No numeric columns with non-zero variance found"`). It
  never names what it dropped.
- **Cleanup has no variance filter.** `clean_traits_for_analysis`
  (`src/sleap_roots_analyze/data_cleanup.py`) composes `apply_data_cleanup_filters`
  (which runs `remove_zero_inflated_traits` — zero-*inflation*, not
  zero-*variance* — plus the many-NaN, NaN-sample, and low-sample filters) and
  then validation. **No step drops constant traits.**
- **Validation only checks the aggregate.** `clean_traits_for_analysis`'s
  validation check 4 asserts that *at least one* non-constant numeric trait
  survives (`var(ddof=0) > 0`); it never drops the *individual* constant traits.

So a constant trait passes cleaning into the "analysis-ready" set, then PCA
silently drops it. In the QC pipeline this is reconstructed *after the fact*
(the PCA step diffs input traits against surviving `feature_names` and reports
`excluded_zero_variance_traits`) — honest, but **late** (post-fit) and only on
the full-pipeline path. Consumers that call `clean_traits_for_analysis` +
`perform_pca_analysis` directly (e.g. the bloom-mcp `pca_analysis` tool) get no
such report, so a "certified-clean" trait turning out constant becomes a hard
error at PCA time.

The honest, systemic fix is to drop constant traits **where the other trait
filters already live** — in cleanup — and to **name them in the cleanup log** so
every downstream consumer can surface the drop truthfully.

### Downstream trigger

- [Salk-Harnessing-Plants-Initiative/bloom#412](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/412):
  bloom-mcp's `pca_analysis` MCP tool hard-errors (`assumption_violated`) when a
  supposedly-clean trait turns out constant and PCA's trait count shrinks after
  fit. Resolving this **upstream** (cleanup drops + names the trait) makes that
  guard provably unreachable, without a behavior change in bloom-mcp and without
  a silent drop.

## Goals / non-goals

**Goals**

- Drop constant traits during cleanup, gated by config, defaulting on.
- Name each dropped trait in `cleanup_log` (`reason="zero_variance"`) so
  downstream tools surface it.
- Guarantee the `clean_traits_for_analysis` output is constant-free, even when a
  caller loosens sample-level NaN handling.
- Keep behavior consistent with `standardize_data` (`var(ddof=0)`).

**Non-goals**

- Changing `standardize_data`'s return signature or removing its drop (kept as a
  belt-and-suspenders no-op — see §7).
- Byte-equivalence between `clean_traits_for_analysis` output and the pipeline's
  `02_data_samples_cleaned.csv` (already not a goal upstream).
- Any bloom-mcp code change as part of *this* repo's work (tracked separately —
  see *Follow-up: landing this in bloom-mcp* below).

## Design

### 1. New granular filter (internal)

Mirror the existing trait-filter contract:

```python
def remove_zero_variance_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    min_variance: float = 0.0,
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """Remove traits whose population variance is <= min_variance.

    Uses var(ddof=0) to match standardize_data. With min_variance=0.0 this drops
    exactly-constant traits; set min_variance < 0 to disable (variance is always
    >= 0). Returns (filtered_df, remaining_trait_cols, removal_details), matching
    the other remove_* trait filters.
    """
```

- For each trait present in `df`, compute `var(ddof=0)`; if `var <= min_variance`,
  record `{"reason": "zero_variance", "variance": float(var), "threshold": min_variance}`.
- Drop the flagged columns; return the filtered frame, remaining traits, and the
  removal-details dict — identical shape to `remove_zero_inflated_traits` et al.
- **Edge cases:** an empty frame yields `var == NaN`, and `NaN <= x` is `False`,
  so nothing is flagged (degenerate cases are handled by the validation checks,
  not here). A single-row frame yields `var == 0`, correctly flagged constant.
- **Not exported.** The sibling granular filters (`remove_zero_inflated_traits`,
  `remove_traits_with_many_nans`, `remove_low_sample_traits`) are not in the
  package's public `__all__`; only the orchestrators are. The new filter follows
  suit.

### 2. Wire into `apply_data_cleanup_filters` as the final step

- Add parameter `min_variance: float = 0.0` to the signature.
- Run `remove_zero_variance_traits` **last** (after the existing low-sample step),
  so variance is evaluated on the **post-sample-removal** frame. This is
  essential: a trait can become constant *only after* NaN-carrying rows are
  dropped, so variance must be measured on the reduced frame.
- Append each removal to `cleanup_log["removed_traits"]` and add a step entry:

```python
{"step": "remove_zero_variance_traits", "traits_removed": <n>, "remaining_traits": <n>}
```

- The orchestrator **does not raise** when this empties the trait set (consistent
  with the other filters); emptiness is handled by the entry point / validation.

### 3. Re-check inside `clean_traits_for_analysis` (second drop site)

`clean_traits_for_analysis` runs its own `clean_df.dropna(subset=surviving)`
*after* `apply_data_cleanup_filters` returns. With the canonical
`max_nans_per_sample=0.0` that dropna is a no-op, but a caller overriding it
(`> 0`) can leave residual NaNs whose removal turns a surviving trait constant —
*after* the orchestrator already ran its variance step. To keep the promise that
the analysis-ready frame is constant-free:

- Add `min_variance` to the entry point's `threshold_names` tuple so it is popped
  from `cleanup_kwargs`, forwarded to `apply_data_cleanup_filters`, and recorded
  in `cleanup_log["effective_thresholds"]`.
- **After** the `dropna`, call `remove_zero_variance_traits(clean_df, surviving,
  min_variance=...)` once more; update `surviving`, drop the columns, and append
  any new removals to `cleanup_log["removed_traits"]` plus a `cleanup_steps`
  entry, so the log stays complete.

This closes the silent-drop-into-PCA hole on **both** the standard path and the
loosened-NaN path.

### 4. Validation check 4 — keep as a defensive guard

Check 4 ("no non-constant numeric trait remains after cleanup …") is retained
unchanged. With the new filter it now fires only when **every** trait was
constant (all dropped → `surviving` empty → zero non-constant), and its message
stays accurate. It remains a belt-and-suspenders assertion against silent
corruption.

### 5. Reporting shape (log contract)

```python
# cleanup_log["removed_traits"] entry
{"trait": "<name>", "reason": "zero_variance", "variance": <float>, "threshold": <min_variance>}

# cleanup_log["cleanup_steps"] entry
{"step": "remove_zero_variance_traits", "traits_removed": <n>, "remaining_traits": <n>}
```

Uses the key **`reason`** (not `removal_reason`) to match the three existing
trait filters; `removal_reason` is the key used for removed *samples*. Downstream
consumers (e.g. bloom-mcp `qc_clean`) already parse `reason` for the other trait
removals, so `reason="zero_variance"` slots in with no new parsing. The per-trait
`variance` value is included for debuggability (cheap; the value is already
computed).

### 6. Config surface

- Add `min_variance: float = 0.0` to `CleanupConfig`
  (`src/sleap_roots_analyze/pipeline/config/components.py`), documented as:
  *"Traits with `var(ddof=0) <= min_variance` are removed. `0.0` drops
  exactly-constant traits; set negative to disable."*
- `CleanupTraitsStep` passes `min_variance=config.cleanup.min_variance` into
  `apply_data_cleanup_filters`.
- Update the canonical-default drift guard
  (`tests/test_data_cleanup.py::TestCanonicalDefaultDriftGuard`,
  [talmolab/sleap-roots-analyze#167](https://github.com/talmolab/sleap-roots-analyze/issues/167)):
  add `min_variance` to both the canonical dict derived from `CleanupConfig()`
  and the pinned-literals dict, so the two layers cannot silently diverge.

### 7. PCA side — unchanged, now a redundant guard

- `standardize_data` is **left as-is** (silent `var(ddof=0) <= 0` drop, raise on
  empty, 3-tuple return). After this change, a cleaned frame never contains a
  constant trait, so this drop becomes a provably-unreachable no-op on the
  cleanup path — but it still protects callers who invoke PCA directly without
  cleaning. No signature change; tightest blast radius.
- The pipeline PCA step's `excluded_zero_variance_traits`
  (`src/sleap_roots_analyze/pipeline/steps/pca_analysis.py`) will now always be
  empty when fed a cleaned frame. No code change required; it stays as a
  defensive report.

## Configuration behavior (chosen options)

- **Threshold-only, always-on.** One knob, `min_variance` (default `0.0`), no
  enable/disable boolean — matching the existing filter idiom, where thresholds
  both tune and (at their extreme) disable a filter. Disable by setting it
  negative.
- **Two drop sites** — orchestrator (final step) + entry-point re-check — for a
  complete constant-free guarantee.
- **PCA drop left untouched** as belt-and-suspenders.

## Backward compatibility

Default-on changes the **cleaned frame and the cleanup log** for any dataset that
contains a genuinely-constant trait: that column is now **absent** from cleaned
output and **appears** in `cleanup_log["removed_traits"]` with
`reason="zero_variance"`. **Analysis results (PCA / UMAP / clustering) are
unchanged**, because those paths already dropped constant traits before fitting.
Consumers that read the cleaned CSV directly will see one fewer column per
constant trait. The pipeline's `excluded_zero_variance_traits` becomes empty. To
retain the old behavior (constants kept in the cleaned frame), set
`min_variance` negative.

## Testing

- **Unit — `remove_zero_variance_traits`:** constant trait dropped; near-constant
  trait kept; `min_variance` threshold honored (boundary at `<=`); `ddof=0`
  semantics; empty-frame and single-row edges; removal-details shape.
- **`apply_data_cleanup_filters`:** a trait made constant *by sample removal* is
  dropped at the final step and logged with `reason="zero_variance"`; step entry
  present.
- **`clean_traits_for_analysis`:**
  - constant trait dropped and named;
  - with `max_nans_per_sample>0` override, a trait made constant by the entry
    point's `dropna` is dropped by the re-check and named;
  - all-constant input → validation check 4 raises;
  - `effective_thresholds` records `min_variance`.
- **Drift guard** updated (`TestCanonicalDefaultDriftGuard`).
- **Regression:** a dataset containing a constant trait yields a smaller
  analysis-ready set; PCA outputs on the surviving traits are unchanged.

## Sequencing

1. Land this change upstream in `sleap-roots-analyze`.
2. Version bump (currently `0.1.0a4` → next pre-release) and release.
3. Follow-up in bloom-mcp (see *Follow-up: landing this in bloom-mcp* below).

## Follow-up: landing this in bloom-mcp

bloom-mcp (repo [Salk-Harnessing-Plants-Initiative/bloom](https://github.com/Salk-Harnessing-Plants-Initiative/bloom))
work, tracked and done **separately** from this repo:

1. **Bump the dependency pin.** Raise the `sleap-roots-analyze>=` constraint to
   the release that contains this filter.
2. **Surface the new reason in `qc_clean`.** Its removed-traits view already reads
   `reason`; confirm `reason="zero_variance"` renders alongside the existing
   reasons (`too_many_zeros`, `too_many_nans`, `insufficient_samples`).
   Refs: [bloom#338](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/338),
   [bloom#356](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/356).
3. **Downgrade or keep the `pca_analysis` guard.** The `assumption_violated`
   error on post-fit trait-count shrink is now unreachable for cleaned inputs.
   Either keep it as defense-in-depth (recommended: downgrade to a warning/log)
   or remove it. Refs:
   [bloom#308](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/308),
   [bloom#412](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/412).
4. **Update bloom-mcp tests/fixtures** that asserted the old hard-error behavior.
5. **Close [bloom#412](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/412)**
   once the pin is bumped and the guard is downgraded, noting it was resolved
   upstream.

## References

- This repo: [talmolab/sleap-roots-analyze](https://github.com/talmolab/sleap-roots-analyze)
  - [#177](https://github.com/talmolab/sleap-roots-analyze/issues/177) — tracking issue for this design.
  - [PR #166](https://github.com/talmolab/sleap-roots-analyze/pull/166) — `clean_traits_for_analysis` entry point (commit `f053b89`).
  - [#167](https://github.com/talmolab/sleap-roots-analyze/issues/167) — canonical cleanup-default drift guard.
  - [#80](https://github.com/talmolab/sleap-roots-analyze/issues/80) — related bug (downstream `trait_names` after PCA zero-variance filtering); preempted by this design.
- bloom-mcp: [Salk-Harnessing-Plants-Initiative/bloom](https://github.com/Salk-Harnessing-Plants-Initiative/bloom)
  - [#412](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/412) — `pca_analysis` hard-error on constant trait (the trigger).
  - [#308](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/308) — `pca_analysis` tool.
  - [#338](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/338), [#356](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/356) — `qc_clean` tool.
  - [#164](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/164), [#166](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/166) — `clean_traits_for_analysis` entry-point consumption.
