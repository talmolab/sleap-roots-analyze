# Proposal: Public Outlier-Removal Entry Point — `remove_outlier_samples`

## Why

The minimal-QC entry point `clean_traits_for_analysis` (#164, landed in `0.1.0a3`) hands
downstream consumers a **runnable** trait table — no NaNs, ≥2 samples, ≥1 varying trait — so
`perform_pca_analysis` / UMAP / clustering no longer silently drop rows. But "runnable" is
not "high quality": **outlier samples are not NaNs**, so PCA/UMAP *run fine* with them while
they **distort principal-component directions** and skew clustering. Trimming them is a
**quality** step, deliberately kept **separate from, and after**, the minimal cleanup
(explicit non-goal of #164).

The detection and removal logic already exists and is already public, but only as
**primitives** a consumer must re-stitch:

- `detect_outliers_mahalanobis` (`src/sleap_roots_analyze/outlier_detection.py:22`) and
  `detect_outliers_isolation_forest` (`:753`) return raw result **dicts**
  (`outlier_indices`, `threshold_*`, …) — the caller must know each method's keys.
- `remove_outliers_from_data` (`:1144`) takes those `outlier_indices` and drops the rows.
- The QC pipeline composes them across **steps 05 → 07** (`DetectOutliersStep` →
  `VisualizeOutliersStep` → `RemoveOutliersStep`), but that composition (method dispatch,
  index alignment, a removal log) lives **inside the pipeline steps** with no importable
  one-call equivalent.

So a consumer who wants "give me the trimmed frame" today must: pick a detector, call it with
the right per-method kwargs, read the right result key, hand the indices to
`remove_outliers_from_data`, and assemble their own report — **untested, duplicated** outside
`analyze` (the same re-stitching problem #116/#164 set out to undo).

This adds **one public, tested entry point** that *imports and composes those already-public
functions* — no new detection or removal algorithm — mirroring the #164 pattern.

Tracked by [talmolab/sleap-roots-analyze#165](https://github.com/talmolab/sleap-roots-analyze/issues/165).

## What Changes

### Add the entry point that composes the existing detect + remove functions

Add public **`remove_outlier_samples`** in a new module
`src/sleap_roots_analyze/outlier_removal.py` (kept out of the NaN-cleanup module
`data_cleanup.py` because this is a distinct quality concern) that:

1. **Rejects malformed input up front** with actionable `ValueError`s — empty input,
   duplicate column names, explicit `trait_cols` missing from `clean_df` or non-numeric, an
   **unknown `method`** (naming the supported methods), and a **non-unique index** (label-based
   removal would otherwise silently mis-drop rows that share a flagged outlier's label, since
   `clean_traits_for_analysis` does not reset the index).
2. **Resolves trait columns** via `get_trait_columns` when `trait_cols` is not passed
   (mirroring `clean_traits_for_analysis`).
3. **Enforces the clean-input precondition** — verifies the trait columns contain **no NaN**
   via the already-exposed `validate_clean_traits`, **wrapping** its error to add a pointer to
   `clean_traits_for_analysis` (the validator's own message does not mention the entry point).
   This is a correctness guard, not just defense: the detectors call PCA which silently
   `dropna()`s rows, so a NaN-carrying input would make the detector's `outlier_indices`
   misalign with `clean_df`'s rows. With NaN-free + unique-index inputs, detection labels align
   one-to-one with `clean_df`'s rows.
4. **Detects** outliers with a single selected `method` (`"mahalanobis"` default, or
   `"isolation_forest"`) by calling the **existing** `detect_outliers_mahalanobis` /
   `detect_outliers_isolation_forest`, forwarding `random_state` (for determinism) and any
   per-method `**detect_kwargs` (e.g. `contamination=`, `chi2_percentile=`). If the detector
   returns an `error` (degenerate PCA, empty/all-NaN), the entry point **raises** rather than
   silently reporting zero outliers.
5. **Removes** the flagged samples by calling the **existing** `remove_outliers_from_data`
   (`keep_metadata=True`, so metadata columns survive), yielding the trimmed frame.
6. **Guards output analysis-readiness** — the trimmed frame must still be safe to hand to
   PCA/UMAP/clustering, so it re-applies the #164 readiness gates (≥`MIN_SAMPLES_FOR_ANALYSIS`
   surviving samples, ≥1 non-constant `var(ddof=0) > 0` trait) and raises an actionable
   `ValueError` — **carrying `outlier_report` on the exception** so a caller can still inspect
   what would have been removed — rather than returning a frame too small or degenerate to
   analyze. It emits a `UserWarning` when the removed fraction is large (default > 0.5, before
   the readiness gates — the signature of a mis-set `contamination`/threshold), and re-applies
   #164's **`p > n` `UserWarning`** when trimming pushes surviving traits above surviving samples.
7. **Returns** `(trimmed_df, outlier_report)`, where `outlier_report` is an auditable,
   JSON-serializable dict (plain Python types only). Its full key list is the single source of
   truth in the spec's **Auditable Outlier Report** requirement: `method` (the dispatch key),
   effective `method_params`, `random_state`, `n_input_samples`, `n_outliers`,
   `n_output_samples`, `removal_fraction`, `outlier_indices`, `outlier_barcodes` (or `None`),
   and the detector's `threshold_type` / `threshold_value` — populated for Mahalanobis (with
   `n_components` and `goodness_of_fit`) and `None` for isolation forest, whose control is
   `contamination`.

### Public API + docs

8. **Export `remove_outlier_samples`** from `__init__.py` / `__all__`, with a Google-style
   docstring (Args/Returns/Raises) and `typing.get_type_hints()`-resolvable hints (#116
   acceptance bar; enforced by the package `test_public_api_docs` audit gate). The underlying
   `detect_outliers_mahalanobis`, `detect_outliers_isolation_forest`, and
   `remove_outliers_from_data` are **already** in `__all__`
   (`src/sleap_roots_analyze/__init__.py:72,74,80`), so the issue's "expose the detect/remove
   functions if not already" is satisfied at the API-surface level. **Note:** `__all__`
   membership ≠ documentation — `docs/API.md` currently documents only
   `detect_outliers_mahalanobis` of the three, so the API.md task also backfills the two missing
   primitive entries the new entry cross-references (see item 9).
9. Add `remove_outlier_samples` (and the two missing composed primitives) to `docs/API.md`, and
   a `docs/CHANGELOG.md` `[Unreleased]` entry — mirroring #164's acceptance.
10. **Register for the reproducibility gate.** Because `remove_outlier_samples` exposes
    `random_state`, the package-wide stochastic-determinism sweep
    (`tests/test_reproducibility.py`) auto-discovers it and fails unless it is registered. Add it
    to `tests/reproducibility_cases.py` `CASES` (or `EXCLUDED` with a reason) and update the
    pinned `EXPECTED_QUALNAMES` / case-count anchors in lockstep.

### What "single source of truth" means here

The entry point and the pipeline (steps 05/07) call the **same** detection
(`detect_outliers_*`) and removal (`remove_outliers_from_data`) functions, so the *outlier
algorithm and index-alignment semantics cannot drift*. As with #164, it does **not** guarantee
byte-identical *output* to the pipeline, because the pipeline operates on
**name-sanitized** columns and supports **multi-method consensus** strategies the entry point
deliberately omits (see Out of scope). The entry point operates on the **raw-named** clean
frame produced by `clean_traits_for_analysis`.

### Composition

`clean_traits_for_analysis` → **(optional)** `remove_outlier_samples` → `perform_pca_analysis`
/ UMAP / clustering. The function *is* the opt-in: outlier trimming runs only when a consumer
calls it, and is parameterized by `method` + per-method threshold/contamination kwargs. It
does **not** alter `clean_traits_for_analysis` or run as part of it.

### Out of scope (explicitly)

- The full `QCPipeline` and `DetectOutliersStep` / `RemoveOutliersStep` refactor — those keep
  their config-driven, multi-artifact behavior unchanged. This change adds *no* pipeline-step
  edits (contrast #164, which refactored step 03).
- **Multi-method consensus / subset removal** (`combine_outlier_methods`, the pipeline's
  `consensus` / `subset` strategies). The entry point trims with a **single** method to stay
  minimal and testable; multi-method consensus is a possible follow-up.
- **Outlier visualization** (pipeline step 06 `visualize_outliers.py` / the
  `outlier_visualization` plots). The report is data-only; rendering dashboards stays in the
  pipeline.
- Any change to `clean_traits_for_analysis`, the minimal-QC cleanup, or NaN handling.
- The downstream **bloom-mcp** quality tool (a `remove_outliers` tool composing after
  `qc_clean`, before viz) — that lives in the `bloom` repo and consumes this function; it is
  not built here.

## Impact

- Affected specs: **outlier-trimming** (new capability).
- Affected code:
  - `src/sleap_roots_analyze/outlier_removal.py` — **new** module with
    `remove_outlier_samples` (composition + report assembly + readiness guards).
  - `src/sleap_roots_analyze/__init__.py` — import + `__all__` entry for
    `remove_outlier_samples` (detect/remove functions already exported).
  - `docs/API.md` — new public function + the two missing composed-primitive entries;
    `docs/CHANGELOG.md` — `[Unreleased]` entry.
  - `tests/test_remove_outlier_samples.py` (new) — TDD coverage.
  - `tests/reproducibility_cases.py` + `tests/test_reproducibility.py` — register the new
    `random_state` function and update the pinned `EXPECTED_QUALNAMES` / count anchors (else the
    package-wide reproducibility-coverage gate fails CI).
- **No behavior change** to any existing function or pipeline step (additive module +
  one `__all__` entry).
- **Depends on #164** (`clean_traits_for_analysis`, `validate_clean_traits`,
  `MIN_SAMPLES_FOR_ANALYSIS`) — already landed in `0.1.0a3`. Independent of #167 (shared
  cleanup-default alignment).
- **Release coupling:** rides the next `analyze` pre-release (`0.1.0a4`) that the bloom-mcp
  quality tool will consume; `[Unreleased]` until then.
