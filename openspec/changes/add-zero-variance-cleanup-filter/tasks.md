## 1. Tests first (red)

New feature tests live in a dedicated `tests/test_zero_variance_cleanup.py`; the drift-guard
update (2.6) stays in `tests/test_data_cleanup.py`.

- [x] 1.1 New test class `TestRemoveZeroVarianceTraits` in `tests/test_zero_variance_cleanup.py`:
      constant trait dropped and a varying trait kept; `removal_details` shape is
      `{trait: {"reason": "zero_variance", "variance": <float>, "threshold": <min_variance>}}`;
      return is the 3-tuple `(filtered_df, remaining_trait_cols, removal_details)` matching the
      sibling filters (removed columns absent from `filtered_df`, `remaining` excludes them).
- [x] 1.2 Test: a near-constant trait (tiny but non-zero variance) is **kept** at
      `min_variance=0.0`; boundary is `<=` (a trait whose `var(ddof=0)` exactly equals
      `min_variance` is dropped, one epsilon above is kept).
- [x] 1.3 Test: `ddof=0` (population) semantics — for values `[0, 2]`, `var(ddof=0)=1.0` and
      `var(ddof=1)=2.0`; with `min_variance=1.0` the trait is dropped (confirms `ddof=0`, not
      `ddof=1`).
- [x] 1.4 Test: edge cases — empty frame yields `var == NaN` so nothing is flagged
      (`NaN <= x` is `False`); a single-row frame yields `var == 0` and the trait is flagged;
      a trait name absent from `df.columns` is skipped without error.
- [x] 1.5 Test: disable — `min_variance` negative keeps even an exactly-constant trait.
- [x] 1.6 Test (`apply_data_cleanup_filters`): a trait constant across all surviving rows is
      dropped at the **final** step and logged in `removed_traits` with `reason="zero_variance"`
      plus `variance`/`threshold`; a `cleanup_steps` entry `"remove_zero_variance_traits"` is
      present with correct counts.
- [x] 1.7 Test (`apply_data_cleanup_filters`): a trait that becomes constant **only after
      sample removal** (varies across all rows, but the varying rows are dropped as NaN-heavy)
      is removed by the final variance step — proving post-sample-removal evaluation.
- [x] 1.8 Test (`apply_data_cleanup_filters`): `min_variance` negative disables the step (a
      constant trait survives the orchestrator).
- [x] 1.9 Test (`clean_traits_for_analysis`): a constant trait is dropped and named in
      `cleanup_log["removed_traits"]` with `reason="zero_variance"`; it is absent from the
      returned surviving traits and cleaned frame.
- [x] 1.10 Test (`clean_traits_for_analysis`): with `max_nans_per_sample>0` override, a trait
      made constant by the entry point's own `dropna` is dropped by the **re-check** and named.
- [x] 1.11 Test (`clean_traits_for_analysis`): an all-constant input still raises the existing
      check-(4) `ValueError` (`"no non-constant numeric trait"`).
- [x] 1.12 Test (`clean_traits_for_analysis`): `cleanup_log["effective_thresholds"]` records
      `min_variance` (default `0.0`).
- [x] 1.13 Test (`clean_traits_for_analysis`): a constant trait alongside a varying one — the
      constant is dropped, the varying one survives, and the call returns successfully.
- [x] 1.14 Test (config): `CleanupConfig().min_variance == 0.0`.
- [x] 1.15 Test (pipeline step): `CleanupTraitsStep` forwards `min_variance` so a constant
      trait is dropped and logged by the step (integration on a small fixture with a constant
      trait).

## 2. Implement (green)

- [x] 2.1 Add `remove_zero_variance_traits(df, trait_cols, min_variance=0.0) ->
      Tuple[pd.DataFrame, List[str], Dict]` to `data_cleanup.py`, mirroring
      `remove_zero_inflated_traits`: per trait in `df.columns` compute `var(ddof=0)`; if
      `var <= min_variance` record `{"reason": "zero_variance", "variance": float(var),
      "threshold": min_variance}`; drop flagged columns; return
      `(filtered_df, remaining_trait_cols, removal_details)`. Google-style docstring. **Do not**
      add to `__all__`.
- [x] 2.2 Add `min_variance: float = 0.0` to `apply_data_cleanup_filters`; run
      `remove_zero_variance_traits` as the final step (after low-sample), append removals to
      `cleanup_log["removed_traits"]` and a `"remove_zero_variance_traits"` `cleanup_steps`
      entry; recompute `final_traits` from the surviving set. Update the docstring's step list.
- [x] 2.3 In `clean_traits_for_analysis`, add `"min_variance"` to `threshold_names` (so it is
      popped, forwarded, and recorded in `effective_thresholds`); after the `dropna`, call
      `remove_zero_variance_traits(clean_df, surviving, min_variance=...)`, update `surviving`,
      drop the columns, and append any new removals to `cleanup_log["removed_traits"]` +
      `cleanup_steps`. Keep check (4) unchanged.
- [x] 2.4 Add `CleanupConfig.min_variance: float = 0.0`
      (`pipeline/config/components.py`) with a docstring: *"Traits with `var(ddof=0) <=
      min_variance` are removed. `0.0` drops exactly-constant traits; set negative to disable."*
- [x] 2.5 `CleanupTraitsStep.execute` passes `min_variance=config.cleanup.min_variance` into
      `apply_data_cleanup_filters`.
- [x] 2.6 Update `TestCanonicalDefaultDriftGuard`: add `"min_variance"` to
      `_canonical_from_config` (from `cfg.min_variance`) and to the pinned-literals dict
      (`0.0`).

## 3. Verify

- [x] 3.1 `uv run pytest tests/test_data_cleanup.py tests/test_clean_traits_entry_point.py
      tests/test_step_cleanup.py -q` green.
- [x] 3.2 Full suite `uv run pytest -m "not integration" tests/` green (the new default-on
      behavior touches cleanup callers; fix any test that asserted a constant trait's survival).
- [x] 3.3 `uv run black src tests` and `uv run ruff check src tests` clean.
- [x] 3.4 `docs/CHANGELOG.md` `[Unreleased] → ### Changed` entry noting cleanup now drops and
      names constant traits (`reason="zero_variance"`), gated by `CleanupConfig.min_variance`.
- [ ] 3.5 `openspec validate add-zero-variance-cleanup-filter --strict`. (NOT RUN: the
      `openspec` CLI is not installed in the dev WSL env — see the sibling change
      `add-outlier-removal-entry-point` task 5.5. The change markdown follows the strict
      format; run this on a machine with the CLI.)
