## Context

`apply_data_cleanup_filters` (`data_cleanup.py:642`) implements the smart cleanup (drop bad
traits → drop NaN rows → drop low-sample traits) and is called by `CleanupTraitsStep`
(step 02). `ValidateCleanStep` (step 03) asserts no NaN remains in trait columns, but that
logic is **inline** (`validate_clean.py:60-92`) with no importable function.
`perform_pca_analysis` (`pca.py:723`) silently `dropna()`s rows (`:775`) and, via
`standardize_data`, silently drops zero-variance *columns* (`pca.py:643-645`), raising only
after the fact. The issue asks for a public entry point that composes the *existing* cleanup
+ validate, with the pipeline and entry point sharing one implementation.

Per maintainer direction (mirroring #116 `expose-statistics-functions`): **single source of
truth = expose the functions the QC steps use, then import them in the entry point** — not a
parallel re-implementation.

## Goals / Non-Goals

- Goals:
  - Expose `apply_data_cleanup_filters` (step 02) and an extracted
    `validate_clean_traits` / `build_clean_validation_report` (step 03) in `__all__`,
    documented and type-hint-resolvable (#116 acceptance bar).
  - One public `clean_traits_for_analysis(df, trait_cols=None, …)` that **imports** those
    functions and returns `(clean_df, trait_cols, cleanup_log)`.
  - Prevent the silent-`dropna()` and zero-variance failure modes via explicit validation.
  - Refactor `ValidateCleanStep` onto the extracted functions with **no behavior change**.
- Non-Goals: outlier handling, stats/heritability/summaries, the full `QCPipeline`,
  trait-name sanitization, column-level zero-variance surfacing, any change to cleanup
  *thresholds* or *algorithm*.

## Decisions

- **D1 — Expose step-02 cleanup as-is.** `apply_data_cleanup_filters` is already a
  standalone function; add it to `__init__.py` imports and `__all__`. No signature change.

- **D2 — Extract step-03 validation into importable functions** in `data_cleanup.py`:
  - `build_clean_validation_report(df, trait_cols) -> dict` — pure; returns the report with
    the exact keys `ValidateCleanStep` builds today (`validation_passed`, `total_samples`,
    `nan_values_in_traits`, `trait_nan_counts`, …).
  - `validate_clean_traits(df, trait_cols) -> dict` — calls the builder; if
    `not report["validation_passed"]`, raises the **canonical** `ValueError` produced by a
    single shared formatter `_format_nan_validation_error(report)` returning exactly
    `"Validation failed: {n} NaN values found in trait columns!\nAffected traits: [...]"`
    (`validate_clean.py:89-92`). Returns the report on success.
  - `ValidateCleanStep.execute` becomes: `report = build_clean_validation_report(...)` →
    save `03_validation_report.json` (unchanged) → `if not report["validation_passed"]:
    raise ValueError(_format_nan_validation_error(report))`. This preserves the existing
    **save-then-raise** ordering (report artifact written even on failure) and the
    `StepResult.metadata` keys downstream steps read (`valid_trait_names`, `trait_names` —
    consumed by `exploratory_analysis.py:73`, `pca_analysis.py:54`, `detect_outliers.py:71`).

- **D3 — Entry-point signature.**
  `clean_traits_for_analysis(df, trait_cols=None, *, barcode_col="Barcode", genotype_col="geno", replicate_col="rep", **cleanup_kwargs) -> tuple[pd.DataFrame, list[str], dict]`.
  Cleanup threshold kwargs (`max_zeros_per_trait`, `max_nans_per_trait`,
  `max_nans_per_sample`, `min_samples_per_trait`) pass through to
  `apply_data_cleanup_filters` with **its** documented defaults (0.5 / 0.3 / 0.2 / 10) — no
  new defaults invented here. Column-name defaults match the cleanup/`get_trait_columns`
  defaults (`"Barcode"`/`"geno"`/`"rep"`); `replicate_col=None` is honored (issue #142).

- **D4 — Surviving trait derivation.** `trait_cols` returned =
  `[c for c in trait_cols if c in clean_df.columns]`. Removed traits are dropped from the
  frame by the cleanup helpers (`data_cleanup.py:559/597/635`), so column-membership is the
  robust, single-line derivation (equivalent to `CleanupTraitsStep`'s
  `removed_traits`-based reconstruction, without re-parsing the log). Pinned by test.

- **D5 — Validation order is fixed and each error is distinct/actionable:**
  1. **empty input** (no rows, or no resolvable trait columns) → entry point's own message
     (e.g. `"clean_traits_for_analysis: input has no trait columns / no rows"`), raised
     *before* any delegation so it never surfaces PCA's generic `"Empty DataFrame provided"`.
  2. **no NaN** in surviving traits → via `validate_clean_traits` (canonical message).
  3. **≥2 surviving samples** → `ValueError` naming the surviving count.
  4. **≥1 non-constant numeric trait** → `var(ddof=0) > 0`. Ordered *after* the no-NaN gate
     so `var` is computed on NaN-free data and agrees with `standardize_data`'s
     `variances > 0` test (`pca.py:643-645`). A single constant (zero-variance) trait →
     `ValueError("...no non-constant trait remains...")`.

- **D6 — `cleanup_log` is enriched, not verbatim** (closes the prior open question; addresses
  reproducibility review B1/I2). The returned log is the `apply_data_cleanup_filters` log
  plus `effective_thresholds` (the four threshold values actually used) and a
  `validation_summary` (`n_samples`, `n_surviving_traits`, `n_nonconstant_traits`). This
  makes "which thresholds produced this table" auditable and removes the silent-defaults
  hazard.

- **D7 — Do NOT route `CleanupTraitsStep` through `clean_traits_for_analysis`.** The step
  does pipeline-only work (name sanitization, artifact CSV/JSON emission, detail frames) and
  must not gain the entry point's stricter ≥2-sample / non-constant gating. Sharing is at the
  *function* level (D1/D2), which is what prevents drift.

## Risks / Trade-offs

- **Entry-point defaults ≠ pipeline config defaults** (cleanup function: 0.3/0.2; pipeline
  config: 0.2/0.0). → Mitigation: spec pins the default values, D6 records effective
  thresholds, docstring states the difference and that matched thresholds are required for
  parity. SSOT claim is scoped to *functions/semantics*, not *identical output*.
- **Near-constant (not exactly constant) surviving traits** can still break
  `StandardScaler` downstream; the ≥1-non-constant gate is a *runnability* floor, not a
  per-trait safety guarantee. → Documented as out of scope (quality follow-up); definition
  pinned to `var(ddof=0)` to at least agree with the downstream drop test.
- **Extracting `ValidateCleanStep`'s check could reword the error** → single shared
  `_format_nan_validation_error`; regression test pins the byte-exact message **and** the
  saved `03_validation_report.json` **and** `StepResult.metadata`.
- **≥2 samples is the mathematical floor**, not scientific adequacy → docstring/error notes
  meaningful multivariate analysis needs many more samples than traits; spec wording avoids
  over-claiming "usable".
- **`get_trait_columns` `set()` dedup** — verified deterministic: output order comes from
  `df.columns`, not the set (`data_cleanup.py:154`). Add a one-line comment to protect a
  future refactor; no functional change.

## Migration Plan

Additive. New public functions + `__all__` entries + a transparent refactor of one step's
inline check into shared functions. No deprecations, no consumer migration. `tasks.md`
follows TDD (red → green) per repo convention.

## Open Questions

- Final public name `clean_traits_for_analysis` — the issue's suggestion ("maintainers'
  call"); confirm before merge.
- Names for the extracted validation functions (`validate_clean_traits` /
  `build_clean_validation_report`) — confirm against any naming convention preference.
