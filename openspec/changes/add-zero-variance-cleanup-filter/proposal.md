# Proposal: Configurable Zero-Variance / Constant-Trait Filter in Cleanup

## Why

A trait whose values are all identical (zero variance) carries no information for
multivariate analysis, yet today the codebase handles such traits **silently and late**:

- **PCA drops them silently.** `perform_pca_analysis` → `standardize_data`
  (`src/sleap_roots_analyze/pca.py:641`) keeps only `var(ddof=0) > 0` columns and raises
  only when *nothing* survives (`"No numeric columns with non-zero variance found"`). It
  never names what it dropped.
- **Cleanup has no variance filter.** `apply_data_cleanup_filters`
  (`src/sleap_roots_analyze/data_cleanup.py:653`) runs zero-*inflation*, many-NaN,
  NaN-sample and low-sample filters — none drops a *constant* trait.
- **Validation only checks the aggregate.** `clean_traits_for_analysis`'s check (4)
  asserts that *at least one* non-constant numeric trait survives; it never drops the
  *individual* constant traits.

So a constant trait passes cleaning into the "analysis-ready" set, then PCA silently drops
it. On the QC-pipeline path this is reconstructed *after* the fact (the PCA step reports
`excluded_zero_variance_traits`) — honest, but late. Consumers that call
`clean_traits_for_analysis` + `perform_pca_analysis` directly (e.g. the bloom-mcp
`pca_analysis` tool) get no such report, so a "certified-clean" trait turning out constant
becomes a hard error at PCA time
([bloom#412](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/412)).

The honest, systemic fix is to drop constant traits **where the other trait filters already
live** — in cleanup — and to **name them in the cleanup log** so every downstream consumer
surfaces the drop truthfully. Tracked by
[talmolab/sleap-roots-analyze#177](https://github.com/talmolab/sleap-roots-analyze/issues/177);
design handoff in PR #178.

## What Changes

### New internal granular filter

Add `remove_zero_variance_traits(df, trait_cols, min_variance=0.0)` in `data_cleanup.py`,
mirroring the existing `remove_zero_inflated_traits` / `remove_traits_with_many_nans` /
`remove_low_sample_traits` contract: it returns `(filtered_df, remaining_trait_cols,
removal_details)`, flagging traits whose `var(ddof=0) <= min_variance` (matching
`standardize_data`'s `ddof=0`). `min_variance=0.0` drops exactly-constant traits; a negative
value disables it (variance is always `>= 0`). **Not exported** — like its sibling granular
filters, it stays out of `__all__`.

### Wire into the cleanup orchestrator as the final step

Add `min_variance: float = 0.0` to `apply_data_cleanup_filters` and run
`remove_zero_variance_traits` **last** (after the low-sample step) so variance is evaluated
on the **post-sample-removal** frame — a trait can go constant *only after* NaN-carrying
rows are dropped. Each removal is appended to `cleanup_log["removed_traits"]` with
`reason="zero_variance"` and a `{"step": "remove_zero_variance_traits", ...}` entry is added
to `cleanup_log["cleanup_steps"]`.

### Re-check inside the analysis-ready entry point

`clean_traits_for_analysis` runs its own `dropna` *after* the orchestrator returns; with a
loosened `max_nans_per_sample > 0` that drop can turn a surviving trait constant. Add
`min_variance` to the entry point's `threshold_names` (so it is forwarded and recorded in
`effective_thresholds`) and call `remove_zero_variance_traits` once more **after** the
`dropna`, updating the surviving set and appending any new removals to the log. This closes
the silent-drop-into-PCA hole on **both** the standard and loosened-NaN paths. Validation
check (4) is retained unchanged as a defensive guard (now fires only when *every* trait was
constant).

### Config surface

Add `CleanupConfig.min_variance: float = 0.0`
(`src/sleap_roots_analyze/pipeline/config/components.py`); `CleanupTraitsStep` forwards
`min_variance=config.cleanup.min_variance` into `apply_data_cleanup_filters`. Update the
canonical-default drift guard (`TestCanonicalDefaultDriftGuard`, #167) to include
`min_variance` in both the config-derived and the pinned-literal dicts.

### PCA side — unchanged

`standardize_data` is left as-is (its silent `var(ddof=0) <= 0` drop becomes a
provably-unreachable no-op on the cleanup path, but still protects direct PCA callers). No
signature change; tightest blast radius.

## Impact

- **Affected specs:** `analysis-ready-cleanup` (cleanup orchestrator + entry-point
  behavior), `config-management` (new `CleanupConfig.min_variance` knob).
- **Affected code:**
  - `src/sleap_roots_analyze/data_cleanup.py` — new `remove_zero_variance_traits`; new
    `min_variance` param + final step in `apply_data_cleanup_filters`; entry-point re-check
    + `min_variance` in `threshold_names`.
  - `src/sleap_roots_analyze/pipeline/config/components.py` — `CleanupConfig.min_variance`.
  - `src/sleap_roots_analyze/pipeline/steps/cleanup_traits.py` — forward `min_variance`.
  - `tests/test_data_cleanup.py` — new `remove_zero_variance_traits` unit tests +
    orchestrator/entry-point behavior tests; update `TestCanonicalDefaultDriftGuard`.
- **Behavior change (default-on):** for any dataset with a genuinely-constant trait, that
  column is now **absent** from cleaned output and **appears** in
  `cleanup_log["removed_traits"]` with `reason="zero_variance"`. **Analysis results (PCA /
  UMAP / clustering) are unchanged** — those paths already dropped constant traits before
  fitting. The QC pipeline's PCA-step `excluded_zero_variance_traits` becomes empty when fed
  a cleaned frame. To retain the old behavior (constants kept in the cleaned frame), set
  `min_variance` negative.
- **Statistics outputs change:** a constant-but-nonzero trait previously flowed into
  `StatisticalAnalysisStep` and produced a degenerate heritability row (`heritability=0.0`,
  `model_type="no_variance"`) and ANOVA row (`f_statistic`/`p_value` = `NaN`). It is now
  dropped at cleanup, so those rows vanish and `n_traits_analyzed` decreases. The removed
  entries are statistically degenerate (H²=0, p=NaN), so this is a correctness improvement,
  documented in `docs/CHANGELOG.md`.
- **No public API surface change** (the new filter is internal; no `__all__` edit).
- **Downstream:** unblocks bloom-mcp #412 (guard becomes provably unreachable); preempts
  #80. bloom-mcp work is tracked separately.
