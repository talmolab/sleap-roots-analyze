## Context

`sleap-roots-contracts 0.1.0a1` (verified published on PyPI; `[pandas]` extra real; alpha resolves without
`--prerelease` flags; no pandas/numpy pin conflict) exposes:
- `validate_analysis_input(df, *, strict=False) -> ValidationResult` — pure, no coercion.
  `.raise_for_status()` raises on the universal structural errors (missing `genotype`, no numeric trait, bad
  role dtype); `.warnings` carries non-fatal issues (missing `sample_id`, unexpected/NaN metadata).
- `canonicalize_role_dtypes(df)` — casts the fixed canonical role columns (`genotype`, `sample_id`,
  `replicate`, `image_path`) to string. Does **not** rename.
- `sleap_roots_contracts.examples.load_analysis_input_example(name)` — 5 packaged example frames
  (`cylinder`, `cylinder_no_replicate`, `field`, `turface`, `genotype_means`) already role-string-cast.

Three name sets must be reconciled (the crux, per issue #144 discussion):

| role      | analyze raw-config (`ColumnConfig`) | contract canonical |
|-----------|-------------------------------------|--------------------|
| genotype  | `geno`                              | `genotype`         |
| sample id | `Barcode` (analyze's "barcode")     | `sample_id`        |
| replicate | `rep` / numeric / `None` (default)  | `replicate`        |

analyze has no `sample_id` concept — the sample id **is** the barcode role. The rename is consumer-side
(only analyze knows its `ColumnConfig`); only the dtype cast is shared.

**Footgun:** under analyze's *default* config against canonical CSVs, a numeric `replicate` leaks in as a
trait. The helper avoids this by renaming roles explicitly before selecting traits — never relying on
defaults.

## Goals / Non-Goals

- Goals: clear structured failure at the untrusted-input boundary; zero behavior change when enabled on
  trusted data; runs cleanly without the optional dependency; validate only the QC entry input.
- Non-Goals: the cross-platform boundary (deferred — `CrossPlatformConfig` lacks barcode/replicate roles and
  feeds a genotype-indexed frame; needs its own recipe + a new `validate_cross_platform_config`);
  re-validating internal intermediates; auto-coercing/renaming the pipeline frame; making contracts a hard
  dependency; the fixture-conformance tests (those are #147).

## Decisions

- **Decision: guarded import, module-level flag.** In `validation/input_contract.py`:
  ```python
  try:
      from sleap_roots_contracts import validate_analysis_input, canonicalize_role_dtypes
      CONTRACTS_AVAILABLE = True
  except ImportError:
      validate_analysis_input = canonicalize_role_dtypes = None
      CONTRACTS_AVAILABLE = False
  ```
  Mirrors the existing `UMAP_AVAILABLE` optional-import pattern (`src/sleap_roots_analyze/umap.py:11`).

- **Decision: one public helper.**
  `validate_entry_input(df, *, columns, mode, additional_exclude=None, logger=None) -> None`
  - `mode == "off"` → return immediately (no import, no work) — a true no-op even when contracts is installed.
  - contracts absent → `log.info` skip-notice and return, regardless of mode.
  - Otherwise build a **copy**: rename `{columns.genotype: "genotype", columns.barcode: "sample_id",
    columns.replicate: "replicate"}` for **only the keys whose source column is present and non-None**
    (`replicate` is `None` by default and may be absent — skip it without `KeyError`) → select
    `present_roles + get_trait_columns(renamed, ...)` → `canonicalize_role_dtypes` →
    `result = validate_analysis_input(check, strict=(mode == "strict"))`.
  - `strict`: `result.raise_for_status()` then log warnings.
  - `warn`: hard-fail only on the universal structural errors — call `raise_for_status()` (the validator
    emits *only* those as errors) and log everything in `result.warnings`. Trusted golden data never trips
    the structural errors, so `warn` is non-fatal in practice.

  Rationale: `validate_analysis_input` already classifies the universal structural problems as *errors* and
  everything else as *warnings*, so `warn` = "log warnings, surface structural errors" maps to
  `raise_for_status()` + log warnings; `strict` adds `strict=True`, which promotes boundary issues (e.g.
  missing `sample_id`) to errors. No separate `strict_source` parameter — the strict boundary is reached
  purely via `mode="strict"` in config (the Bloom/external adapter sets it, and enforces its own guard
  independently — out of scope here).

- **Decision: `validate_input` field placement.** Add `validate_input: str = "warn"` to `DataConfig`
  (co-located with `csv_path`, the load-boundary concern it gates). Enum (`off`/`warn`/`strict`) validated in
  `validate_qc_config` (`pipeline/config/utils.py:197`). The `DataConfig.validate_input` Google docstring is
  the **single canonical** user-facing description of the modes; templates/CHANGELOG/README point to it
  rather than re-explaining.

- **Decision: wiring point.** Call the helper in `LoadDataStep.execute` right after the entry DataFrame is in
  hand — covering **both** the CSV branch and the pre-loaded root-core `data` branch
  (`load_data.py:58-71`), before trait-column selection. Skip when `mode == "off"`. Pass
  `config.data.additional_exclude_cols` through so excluded metadata is dropped from the validation copy too.

## Risks / Trade-offs

- **Risk: canonicalization leaks into the pipeline frame** → the helper operates only on a local copy and
  returns `None`; the equivalence test (`off` == `warn`, golden `rtol=1e-6`) and an `assert_frame_equal`
  identity check on the downstream frame are the guards.
- **Risk: replicate-leak footgun** → explicit rename before `get_trait_columns`; unit test on the
  default-config + canonical-CSV case, plus a `replicate=None` (`cylinder_no_replicate`) case.
- **Risk: real `except ImportError` branch never runs in CI** (contracts is in the dev group) → covered by a
  `sys.modules["sleap_roots_contracts"] = None` + `importlib.reload` test (pattern at `tests/test_dag.py:457`),
  in addition to the `CONTRACTS_AVAILABLE=False` monkeypatch behavior test.
- **Risk: `build.yml` uses `uv sync --frozen`** → `uv.lock` must be regenerated and committed in the same
  commit as the `pyproject.toml` dep, or the release/build job fails on a frozen-lock mismatch.
- **Risk: contracts alpha churn** → pin `>=0.1.0a1`; the committed `uv.lock` is the reproducibility anchor.

## Migration Plan

Additive only. Default `warn` + warn-by-default semantics + optional dependency means existing configs and
trusted golden runs are unchanged. No migration needed; opt into `strict` at external boundaries. Existing
golden templates and `configs/active/*` need no regeneration — the absent field takes the default via
`OmegaConf.merge`.

## Open Questions

- None blocking. Cross-platform validation is explicitly deferred to a follow-up issue.
