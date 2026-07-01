## Context

`clean_traits_for_analysis` (#164, `data_cleanup.py:905`) produces a **runnable** frame but
deliberately leaves outliers in — PCA/UMAP run fine with them (they are not NaNs), so trimming
them is a separate **quality** step. The detection + removal primitives already exist and are
already public:

- `detect_outliers_mahalanobis(data, *, standardize=True, variance_threshold=0.95,
  use_chi_squared=True, chi2_percentile=97.5, distance_threshold=None,
  robust_covariance=False, random_state=42) -> dict` (`outlier_detection.py:22`) — keys:
  `outlier_indices`, `mahalanobis_distances`, `n_outliers`, `n_components`, `threshold_type`,
  `threshold_value`, `feature_names`, `goodness_of_fit`, `error`.
- `detect_outliers_isolation_forest(data, contamination=0.1, random_state=42) -> dict`
  (`:753`) — keys: `method`, `contamination`, `outlier_indices`, `n_outliers`,
  `anomaly_scores`, `outlier_labels`, `data_indices`, `error`.
- `remove_outliers_from_data(df, outlier_indices, keep_metadata=True, return_outliers=True,
  reset_index=False) -> (cleaned_df, outliers_df)` (`:1144`).

All three are already in `__all__` (`__init__.py:72,74,80`). The QC pipeline composes them
across steps 05 (`DetectOutliersStep`) → 07 (`RemoveOutliersStep`), but with config-driven
multi-method strategies and artifact emission — there is no importable "detect with one method,
remove, return a report" one-call function. This change adds exactly that, **importing** the
existing functions (the #116/#164 single-source-of-truth pattern).

## Goals / Non-Goals

- Goals:
  - One public `remove_outlier_samples(clean_df, trait_cols=None, *, method="mahalanobis", …)`
    that **imports and composes** `detect_outliers_*` + `remove_outliers_from_data` and returns
    `(trimmed_df, outlier_report)`.
  - Keep the trimmed frame **analysis-ready** (the #164 readiness gates) so it can flow
    straight into PCA/UMAP/clustering.
  - A clean-input precondition that prevents the silent `dropna()`-misalignment failure mode.
  - Deterministic given `random_state`; auditable, JSON-serializable report.
- Non-Goals: any new detection/removal algorithm; multi-method consensus
  (`combine_outlier_methods`, pipeline `consensus`/`subset`); outlier visualization (step 06);
  any change to `clean_traits_for_analysis`, the cleanup thresholds, or the pipeline steps; the
  downstream bloom-mcp tool (separate repo).

## Decisions

- **D1 — New module `outlier_removal.py`, not `data_cleanup.py`.** Outlier trimming is a
  distinct quality concern (the issue is emphatic it is *separate from, and after*, the minimal
  cleanup). A new module keeps the NaN-cleanup module focused and gives the new capability a
  clear home. Imports `detect_outliers_*` / `remove_outliers_from_data` from
  `outlier_detection.py` and `validate_clean_traits` / `get_trait_columns` /
  `MIN_SAMPLES_FOR_ANALYSIS` from `data_cleanup.py`.

- **D2 — Signature.**
  `remove_outlier_samples(clean_df, trait_cols=None, *, method="mahalanobis",
  barcode_col="Barcode", genotype_col="geno", replicate_col="rep", random_state=42,
  **detect_kwargs) -> tuple[pd.DataFrame, dict]`.
  - `trait_cols=None` → inferred via `get_trait_columns` (same ergonomics as
    `clean_traits_for_analysis`); `barcode_col`/`genotype_col`/`replicate_col` are all forwarded
    to `get_trait_columns` and used only to *exclude* metadata during inference (`barcode_col`
    additionally identifies the column read for `outlier_barcodes`). Column-name defaults match
    the sibling
    (`"Barcode"`/`"geno"`/`"rep"`) because the entry point consumes the **raw-named** clean
    frame the sibling produces (no name sanitization).
  - `**detect_kwargs` forwards per-method parameters to the chosen detector
    (`contamination=` for isolation forest; `chi2_percentile=`, `variance_threshold=`,
    `use_chi_squared=`, `distance_threshold=`, `robust_covariance=` for Mahalanobis) — mirrors
    `clean_traits_for_analysis`'s `**cleanup_kwargs` pass-through.

- **D3 — Default method = `"mahalanobis"` with the chi-squared threshold.** It is the module's
  primary, statistically principled detector (distance on PCA-transformed data, chi² threshold)
  and needs no contamination guess. `"isolation_forest"` is offered for callers who want a
  direct `contamination` knob. An unknown `method` raises a `ValueError` naming the supported
  set, so a typo fails fast instead of silently doing nothing.

- **D4 — Return `(trimmed_df, outlier_report)` (2-tuple).** Unlike the sibling's 3-tuple,
  `trait_cols` is **invariant** here — outlier removal drops *rows*, never trait *columns* — so
  re-returning it would be redundant; the caller reuses the `trait_cols` from
  `clean_traits_for_analysis`. The report parallels the sibling's enriched `cleanup_log`. (See
  Open Questions: confirm 2-tuple vs. a symmetric 3-tuple.)

- **D5 — `outlier_report` schema (auditable + JSON-serializable).** The spec's **Auditable
  Outlier Report** requirement is the single source of truth for the key list; this is the
  rationale. `method` (str — the **dispatch key**, e.g. `"isolation_forest"`, set by the entry
  point, NOT the detector's internal `"IsolationForest"` label), `method_params` (effective
  per-method params used), `random_state` (int), `n_input_samples`/`n_outliers`/`n_output_samples`
  (ints), `removal_fraction` (float), `outlier_indices` (list of removed sample **labels**),
  `outlier_barcodes` (list when `barcode_col` is a column of `clean_df`, else `None`).
  Method-dependent: for **Mahalanobis**, `threshold_type` / `threshold_value`, the PCA basis the
  distances were computed on (`n_components`, effective `variance_threshold`), and the chi²
  `goodness_of_fit` (which flags when the percentile threshold's distributional assumption is
  violated). For **isolation forest** these are `None` — its return dict has no
  `threshold_type`/`threshold_value`/`goodness_of_fit` (verified: it returns `method`,
  `contamination`, `outlier_indices`, `n_outliers`, `anomaly_scores`, `outlier_labels`,
  `data_indices`), and its control knob (`contamination`) lives in `method_params`. All values
  are cast to plain Python scalars/lists (numpy ints/floats → `int()`/`float()`) so `json.dumps`
  succeeds. Large per-sample arrays (`mahalanobis_distances`, `anomaly_scores`) are **not**
  copied in — compact + serializable; a caller who needs them calls the detector directly. The
  report mirrors the pipeline's `07_outlier_removal_log.json` (`removal_fraction`,
  `removed_sample_barcodes`); note the deliberate key rename `removed_sample_barcodes` →
  `outlier_barcodes` (different object — the entry point's report, not the pipeline artifact).

- **D6 — Clean-input precondition (correctness, not just defense).** Before detecting, verify
  the trait columns are NaN-free via the exposed `validate_clean_traits`. The detectors run PCA
  which silently `df.dropna()`s rows and reports `outlier_indices` against the **post-dropna**
  index; feeding a NaN-carrying frame would therefore drop rows the caller never sees and
  misalign the indices handed to `remove_outliers_from_data`. On violation, raise a `ValueError`
  that names the offending traits and points to `clean_traits_for_analysis` — **wrapping**
  `validate_clean_traits`'s message, which is `"Validation failed: N NaN values found in trait
  columns!\nAffected traits: [...]"` and does **not** itself mention the entry point (so the
  pointer must be added by the wrapper, and the test must match the wrapper's text).

- **D6b — Unique-index precondition (the other half of alignment).** `clean_traits_for_analysis`
  does **not** `reset_index` (verified — it `dropna`s and returns the frame as-is), so the input
  inherits the caller's index, which can carry **duplicate labels** (e.g. `set_index("Barcode")`
  on replicated barcodes, or concatenated frames). Removal is **label-based**
  (`remove_outliers_from_data` uses `df.drop(index=...)` / `df.loc[...]`), so a duplicate label
  would silently drop or duplicate inlier rows sharing a flagged outlier's label —
  mis-removing rows. `validate_clean_traits` only checks NaN, not uniqueness. So the entry point
  also requires `clean_df.index.is_unique`, raising an actionable `ValueError` otherwise. NaN-free
  **and** unique-index together make the "labels align 1:1 with rows" guarantee sound.

- **D7 — Output readiness guard (reuse #164 gates) + p > n warning.** The whole point is a frame
  still safe for PCA/UMAP/clustering, so after removal re-check: ≥`MIN_SAMPLES_FOR_ANALYSIS` (2)
  surviving samples, and ≥1 non-constant trait (`var(ddof=0) > 0`, the same basis
  `standardize_data` uses). If trimming would breach either, raise a distinct, actionable
  `ValueError` naming the surviving count / the degeneracy — rather than returning a frame too
  small or constant to analyze. Checks run in a fixed order (samples, then non-constant) for a
  deterministic message. The raised `ValueError` **carries `outlier_report` as an attribute** so a
  caller who hit an over-aggressive trim can still inspect what would have been removed (resolving
  the "I wanted the frame/report anyway" tension noted in Risks). Additionally, because trimming
  *shrinks n*, an `n > p` frame can become `p > n`; the entry point re-applies #164's **`p > n`
  `UserWarning`** (surviving samples < surviving traits) so that statistically-fragile regime is
  not entered silently — a guardrail the first draft dropped.

- **D8 — Over-removal safety rail.** Emit a `UserWarning` when `removal_fraction` exceeds a
  guard (default `0.5`): removing the majority of samples almost always means a mis-set
  `contamination`/threshold, not real outliers. A warning (not a hard error) keeps the function
  usable on genuinely dirty data while flagging the likely-mistake case. It is emitted **before**
  the D7 readiness gates, so it is observable even when the same aggressive trim then fails a
  readiness gate (otherwise the warning pointing you at the over-removal would be swallowed by the
  raise). (Threshold configurability — e.g. a `max_removal_fraction` kwarg — is an Open Question.)

- **D9 — Determinism.** `random_state` (default 42, matching the detectors) is threaded into the
  detector and recorded in the report: same input + same `random_state` + same params ⇒ identical
  `outlier_indices` and identical trimmed frame. The seed is only *load-bearing* on stochastic
  paths — `method="isolation_forest"`, `robust_covariance=True` (MinCovDet), or large-n
  randomized SVD; for the **default exact-SVD Mahalanobis** path the result is identical
  regardless of seed (so a determinism test on that path is necessary but not sufficient — the
  determinism test must also exercise a seed-sensitive path, per tasks 1.12b). This satisfies the
  repo's stochastic-determinism / reproducibility gates — see D11 for the mandatory registry
  registration that gate enforces.

- **D10 — `remove_outliers_from_data(keep_metadata=True)`.** Preserve metadata columns
  (`Barcode`/`geno`/`rep`) in the trimmed frame so the result is a drop-in replacement for the
  clean frame in the downstream call chain, and so `outlier_barcodes` can be read from the
  removed rows.

- **D11 — Reproducibility-gate registration (mandatory, else CI red).** The repo's
  `tests/test_reproducibility.py` runs a package-wide sweep (`walk_packages`) that **fails if any
  public function with a `random_state` parameter is absent** from the `CASES` registry in
  `tests/reproducibility_cases.py` (or `EXCLUDED`), and additionally pins `EXPECTED_QUALNAMES` and
  the case count. `remove_outlier_samples` will be auto-discovered, so this change must register
  it in `CASES` (preferred — it *is* a determinism-relevant function) and update the
  `EXPECTED_QUALNAMES` / count anchors in lockstep. An in-suite determinism test (tasks 1.12) is
  necessary but **not sufficient** — the package-wide registry is the actual gate, and it also
  satisfies `openspec/specs/stochastic-determinism`'s "regression test enforces determinism"
  requirement.

## Risks / Trade-offs

- **Single-method only vs. the pipeline's consensus strategies.** A single method can over- or
  under-flag relative to a multi-method consensus. → Accepted for a minimal, tested entry point;
  multi-method consensus (`combine_outlier_methods`) is a named follow-up, and the `method`
  parameter leaves room to add a `"consensus"` value later without a signature break.
- **Detector returns an `error` key instead of raising** (e.g. empty/all-NaN data). → The
  entry point's empty-input + NaN-free preconditions make that path unreachable under normal
  flow; defensively, if a detector result carries `error` / lacks `outlier_indices`, raise a
  `ValueError` surfacing it rather than treating "no indices" as "no outliers".
- **Default `method="mahalanobis"` could surprise callers expecting a contamination knob.** →
  Documented in the docstring and report (`method`, `method_params`); `isolation_forest` is one
  kwarg away.
- **Readiness guard can reject aggressive trims (mild tension with "optional quality step").**
  The issue frames trimming as optional and "give me the trimmed frame"; a hard `ValueError` on a
  sub-threshold result discards that frame. A caller might want to *inspect what got removed* even
  at 1 survivor. → Keep the raise (the frame is not analyzable — PCA needs ≥2 samples / a varying
  trait — and it matches #164's contract), but **attach `outlier_report` to the raised exception**
  (D7) so the caller can still see the over-removal it would want to diagnose. A caller who wants
  the raw trimmed frame regardless can call the detector + `remove_outliers_from_data` directly.
  The error is actionable (loosen the threshold / use fewer-flagging params).
- **Report omits per-sample distance arrays.** → Intentional (compact, serializable); the
  detector remains public for callers who need full arrays. The audit-relevant scalars it *does*
  keep — `n_components`, effective `variance_threshold`, `goodness_of_fit`, `threshold_value` —
  are enough to reproduce the decision. Documented.
- **Default `chi2_percentile=97.5` trims ~2.5% by construction.** On a well-fit chi² tail the
  Mahalanobis threshold flags roughly the top 2.5% *regardless* of whether those samples are
  biologically aberrant. → Defensible as a documented, conventional default; the docstring/spec
  state the ~2.5%-by-design behavior so callers know the default removes ~2.5% even on clean data
  and can tighten `chi2_percentile` or switch to `isolation_forest` with an explicit
  `contamination`. The `goodness_of_fit` in the report surfaces when the chi² assumption (and thus
  the percentile's meaning) is violated.

## Migration Plan

Additive: one new module, one `__all__` entry, docs. No deprecations, no consumer migration, no
pipeline-step or `clean_traits_for_analysis` changes. `tasks.md` follows the repo's TDD
convention (red → green). Lands on `0.1.0a4` for the bloom-mcp quality tool to consume.

## Open Questions

- **Public name** `remove_outlier_samples` — the issue's suggestion ("maintainers' call");
  confirm before merge (alternatives: `trim_outlier_samples`, `remove_outliers_for_analysis`).
- **Return arity** — 2-tuple `(trimmed_df, outlier_report)` per the issue, vs. a 3-tuple
  `(trimmed_df, trait_cols, outlier_report)` symmetric with `clean_traits_for_analysis`.
  Recommend 2-tuple (columns are invariant); confirm.
- **Default method** — `"mahalanobis"` (recommended) vs. `"isolation_forest"` (direct
  `contamination` control). Confirm the default and whether to ship a `"consensus"` value now.
- **Over-removal rail** — fixed `UserWarning` at `removal_fraction > 0.5` vs. a configurable
  `max_removal_fraction` kwarg (warn or hard-error). Recommend the fixed warning for v1.
- **Non-unique index handling** — D6b *raises* on a non-unique index (strict, predictable). The
  alternative is to internally `reset_index` before detect/remove and map labels back. Recommend
  raising for v1 (explicit; avoids hidden index mutation), but confirm whether a convenience
  auto-reset is preferred for ergonomics.
