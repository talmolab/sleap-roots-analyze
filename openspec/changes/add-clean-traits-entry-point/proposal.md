# Proposal: Public Minimal-QC Entry Point — `clean_traits_for_analysis`

## Why

Downstream consumers (bloom-mcp's Tier 3 `qc_clean` → `pca_analysis`, notebooks) need a
**clean, analysis-ready** trait table — no NaNs, enough samples, at least one varying
trait — but today there is no single, tested way to get one:

- **`perform_pca_analysis` does not guard NaNs — it silently `data.dropna()`s**
  (`src/sleap_roots_analyze/pca.py:775`), dropping *every row with any NaN*. A few
  NaN-heavy traits can wipe out most samples with no warning.
- **`apply_data_cleanup_filters`** (`src/sleap_roots_analyze/data_cleanup.py:642`) is the
  smart cleanup that `CleanupTraitsStep` (step 02) already uses — it drops bad **traits**
  first (zero-inflated / too-many-NaN / low-sample), *then* the remaining NaN rows,
  explicitly to **minimize sample loss** — but it is **not in `__all__`**
  (`src/sleap_roots_analyze/__init__.py`), so consumers cannot reach it as public API.
- **Step 03's validation** (no-NaN-in-traits check) lives **inline inside
  `ValidateCleanStep.execute`** (`src/sleap_roots_analyze/pipeline/steps/validate_clean.py:88-92`)
  — there is no importable function for it at all.
- The full **`QCPipeline`** is ~16 steps emitting many artifacts — far more than "give me
  a clean table" needs.
- Re-stitching steps `01 load → 02 cleanup → 03 validate` in each consumer would put that
  orchestration **untested and duplicated outside `analyze`** (exactly the
  `bloommcp/source/trait_statistics.py` duplication problem #116 set out to undo).

So: **expose the step-02 and step-03 functions in the public API** (the #116 pattern), and
add **one public, tested entry point** that *imports and composes those exposed functions*.

Tracked by [talmolab/sleap-roots-analyze#164](https://github.com/talmolab/sleap-roots-analyze/issues/164).

## What Changes

### A. Expose the step-02 / step-03 functions (single source of truth)

Following the #116 (`expose-statistics-functions`) pattern — extract & expose, then import:

1. **Export `apply_data_cleanup_filters`** (step 02's cleanup) from `__init__.py` / `__all__`.
   It is already a standalone function; this is an API-surface change only.
2. **Extract step 03's inline NaN validation into importable functions** in
   `data_cleanup.py`:
   - `build_clean_validation_report(df, trait_cols) -> dict` — pure builder of the
     validation report (same keys `ValidateCleanStep` produces today).
   - `validate_clean_traits(df, trait_cols) -> dict` — calls the builder, raises the
     **canonical** `ValueError` (byte-for-byte the message at `validate_clean.py:89-92`,
     via a single shared formatter) when NaNs remain, else returns the report.
   Export both in `__all__`.
3. **Refactor `ValidateCleanStep` to use these functions** — build the report via
   `build_clean_validation_report`, save `03_validation_report.json` exactly as today, then
   raise via the shared formatter. **No observable behavior change** (same report keys, same
   saved artifact, same error message, same `StepResult.metadata`).
4. **Audit docstrings / resolvable type hints** on the newly-public functions (Google-style
   Args/Returns/Raises; `typing.get_type_hints()` must not raise) and update `docs/API.md`
   + a `docs/CHANGELOG.md` `[Unreleased]` entry — mirroring #116's acceptance.

### B. Add the entry point that composes the exposed functions

5. Add public **`clean_traits_for_analysis`** in `data_cleanup.py` that:
   1. resolves trait columns via `get_trait_columns` when `trait_cols` is not passed;
   2. runs `apply_data_cleanup_filters(df, trait_cols, …)` — the exposed step-02 cleanup;
   3. derives the **surviving** trait columns as `[c for c in trait_cols if c in clean_df.columns]`
      (removed traits are dropped from the frame by the cleanup helpers);
   4. **validates** in a pinned order, each with a distinct, actionable `ValueError`:
      **(a)** empty input → its own message *before* delegating;
      **(b)** no NaN in surviving traits, via the exposed `validate_clean_traits`;
      **(c)** ≥2 surviving samples (message names the surviving count);
      **(d)** ≥1 non-constant numeric trait, *non-constant* defined as
      `var(ddof=0) > 0` — the **same** basis `perform_pca_analysis`/`standardize_data`
      uses (`pca.py:643-645`) so the gate and downstream agree;
   5. returns `(clean_df, trait_cols, cleanup_log)`, where `cleanup_log` is the
      `apply_data_cleanup_filters` log **enriched** with the *effective thresholds used* and
      a *validation summary*, so the result is auditable and reproducible.
6. **Export `clean_traits_for_analysis`** in `__all__`; add it to `docs/API.md` + CHANGELOG.

### What "single source of truth" means here (corrected per review)

The entry point and the pipeline share the **same functions** — `apply_data_cleanup_filters`
(cleanup) and `validate_clean_traits` (validation) — so the *algorithm and validation
semantics* cannot drift. They do **not** guarantee identical *output*, because:

- the pipeline **sanitizes/abbreviates trait names** (`sanitize_trait_names`, step 02) and
  passes its **config-driven thresholds** + sanitized column names (`"Genotype"`,
  `"Replicate"`) before calling the cleanup function;
- the entry point operates on the **raw** caller-supplied table, with the cleanup
  function's **own documented defaults** (`max_zeros_per_trait=0.5`, `max_nans_per_trait=0.3`,
  `max_nans_per_sample=0.2`, `min_samples_per_trait=10`) and column-name kwargs
  (`barcode_col="Barcode"`, `genotype_col="geno"`, `replicate_col="rep"`).

These defaults are the **function's**, not the pipeline's config defaults
(`max_nans_per_trait=0.2`, `max_nan_fraction=0.0`); identical results require passing
matched thresholds. The spec pins the default values, the entry point records the effective
thresholds into `cleanup_log`, and the docstring states this explicitly.

### Out of scope (explicitly)

- The full `QCPipeline`.
- **Outlier detection/removal** (`QCPipeline` steps 05–07) — PCA runs fine with outliers
  (they are not NaNs); a quality step, not a runnability one. Follow-up **#165**.
- Stats / heritability / summaries / dashboards; root-core / depth processing.
- **Column-level zero-variance loss inside PCA.** The ≥1-non-constant gate guarantees
  *runnability*, not that every surviving trait is safe to standardize; `standardize_data`
  may still drop additional near-zero-variance trait *columns* downstream. Surfacing that
  column-level drop is a quality concern tracked separately, not part of this runnability
  entry point.
- `sanitize_trait_names` — pipeline-only preprocessing, deliberately not part of the
  raw-input entry point (this is *why* entry-point and pipeline outputs can differ).

## Impact

- Affected specs: **analysis-ready-cleanup** (new capability).
- Affected code:
  - `src/sleap_roots_analyze/data_cleanup.py` — new `clean_traits_for_analysis`,
    `build_clean_validation_report`, `validate_clean_traits`, shared error formatter.
  - `src/sleap_roots_analyze/__init__.py` — import + `__all__` for
    `apply_data_cleanup_filters`, `validate_clean_traits`,
    `build_clean_validation_report`, `clean_traits_for_analysis`.
  - `src/sleap_roots_analyze/pipeline/steps/validate_clean.py` — call the extracted
    functions (transparent refactor).
  - `docs/API.md` + `docs/CHANGELOG.md` — new public functions.
  - `tests/test_clean_traits_entry_point.py` (new) + a steps-01–03 no-behavior-change
    regression assertion reusing the existing `TestQCPipelineIntegration` baseline.
- **Existing internal caller of `apply_data_cleanup_filters`** beyond `CleanupTraitsStep`:
  `visualization.py:830` (`create_trait_eda_plots`). Signature is unchanged; exposing the
  name in `__all__` does not affect it.
- **Release coupling:** must ride the `analyze` release bloom-mcp Tier 3 consumes — land
  before the `0.1.0a3` cut (tracked in #163, whose scope does **not** currently include
  #164), or plan a follow-up `0.1.0a4`.
