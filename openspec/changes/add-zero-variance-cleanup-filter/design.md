## Context

Constant (zero-variance) traits are dropped **silently and late** today — inside
`standardize_data` at PCA time — rather than at cleaning time where the other trait filters
live. Direct consumers of `clean_traits_for_analysis` + `perform_pca_analysis` (e.g.
bloom-mcp's `pca_analysis` tool) therefore hit a hard error when a "clean" trait turns out
constant. The full brainstormed design is checked in at
`docs/superpowers/specs/2026-07-08-zero-variance-trait-cleanup-filter-design.md`; this file
records the load-bearing decisions.

## Goals / Non-Goals

- **Goals:** drop constant traits during cleanup (config-gated, default-on); name each in
  `cleanup_log` (`reason="zero_variance"`); guarantee `clean_traits_for_analysis` output is
  constant-free even when a caller loosens per-sample NaN handling; keep the variance test
  consistent with `standardize_data` (`var(ddof=0)`).
- **Non-Goals:** changing `standardize_data`'s signature or removing its drop (kept as a
  belt-and-suspenders no-op); byte-equivalence between the entry point's output and the
  pipeline's `02_data_samples_cleaned.csv` (already not a goal); any bloom-mcp code change.

## Decisions

- **Threshold-only, always-on knob.** One `min_variance` (default `0.0`), no enable/disable
  boolean — matching the existing filter idiom where a threshold both tunes and (at its
  extreme) disables. Disable by setting it negative (variance is always `>= 0`).
  - *Alternative considered:* a separate `drop_constant_traits: bool`. Rejected — redundant
    with the threshold and inconsistent with the sibling filters.
- **Two drop sites.** (1) `apply_data_cleanup_filters` final step, evaluated
  post-sample-removal because a trait can go constant only after NaN rows are dropped; (2) a
  re-check in `clean_traits_for_analysis` after its own `dropna`, which under a loosened
  `max_nans_per_sample` can turn a trait constant *after* the orchestrator already ran its
  variance step. Both are needed for a complete constant-free guarantee.
- **Log key `reason` (not `removal_reason`).** Matches the three existing trait filters;
  `removal_reason` is the removed-*samples* key. Downstream consumers already parse `reason`
  for trait removals, so `reason="zero_variance"` slots in with no new parsing. The per-trait
  `variance` value is included for debuggability (already computed).
- **PCA drop left untouched.** Once cleanup guarantees no constant trait, `standardize_data`'s
  drop is a provably-unreachable no-op on the cleanup path but still guards direct PCA callers.

## Risks / Trade-offs

- **Default-on changes the cleaned frame + log** for datasets with a genuinely-constant
  trait: the column is now absent from cleaned output and appears in `removed_traits`.
  *Mitigation:* analysis results (PCA/UMAP/clustering) are unchanged (those paths already
  dropped constants pre-fit); the QC PCA step's `excluded_zero_variance_traits` simply becomes
  empty; old behavior is recoverable via `min_variance < 0`.
- **Tests asserting a constant trait's survival** through cleanup will need updating — a
  deliberate consequence of the fix, not a regression. The full suite is run to find them.

## Migration Plan

Additive and behind a default-on threshold. No data migration. Consumers wanting the prior
behavior set `CleanupConfig.min_variance` (or the `min_variance` kwarg) negative. bloom-mcp
bumps its `sleap-roots-analyze` pin and downgrades its now-unreachable guard — tracked
separately in the `bloom` repo.

## Open Questions

None outstanding — the design is fully specified in the checked-in design doc.
