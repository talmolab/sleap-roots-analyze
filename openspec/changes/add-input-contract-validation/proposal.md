## Why

Malformed analysis input (missing `genotype`, no numeric trait, wrong role dtype) currently fails deep
inside QC with confusing errors — or silently produces garbage. This change wires the optional
`sleap-roots-contracts` `validate_analysis_input` into the QC data-loading boundary so analyze rejects bad
input with a clear, structured error before any analysis runs (GitHub issue #144).

## What Changes

- The check is a **non-intrusive, optional side-check**: enabling it never moves a result, and analyze must
  still run when `sleap-roots-contracts` is not installed.
- Add `sleap-roots-contracts[pandas]>=0.1.0a1` as an **optional** dependency (extra + dev group), imported
  through a guarded import. Absence degrades validation to a **logged no-op** — never a hard `ImportError`.
- Add a `validate_input: off | warn | strict` config flag (default `warn`) to the QC `DataConfig`.
- Add a shared boundary helper (`canonicalize-then-validate` on a **copy**): rename config role names →
  canonical, drop non-trait metadata via `get_trait_columns`, `canonicalize_role_dtypes`, then
  `validate_analysis_input`. `strict` raises at the boundary; `warn` logs warnings and only hard-fails on the
  universal structural errors (missing `genotype`, no trait, bad dtype).
- Wire the helper into the QC entry-input boundary only — `LoadDataStep`. **No** re-validation of internal
  step-to-step intermediates. The canonicalized frame is built solely for the validate call and discarded —
  the frame fed to QC is never touched.
- **Out of scope (deferred to a follow-up):** the cross-platform load boundary (`LoadCrossPlatformDataStep`).
  `CrossPlatformConfig` has no barcode/replicate roles and feeds an already genotype-indexed frame, so it
  needs a distinct genotype-only recipe and its own config validator — tracked separately.

## Impact

- Affected specs: `input-contract-validation` (new capability)
- Affected code:
  - `pyproject.toml` — optional extra + dev dependency; `uv.lock` regenerated and committed
  - `src/sleap_roots_analyze/validation/__init__.py` + `input_contract.py` — new guarded helper module
  - `src/sleap_roots_analyze/pipeline/config/components.py` — `validate_input` on `DataConfig`
  - `src/sleap_roots_analyze/pipeline/config/utils.py` — `validate_qc_config` enum check
  - `src/sleap_roots_analyze/pipeline/steps/load_data.py` — call helper on QC entry input
  - `configs/templates/qc_template_grouped.yaml`, `qc_template_ungrouped.yaml` — commented `validate_input`
  - `docs/CHANGELOG.md`, `README.md` — `[Unreleased]` entry + `[contracts]` extra install note
- Reproducibility: equivalence proven against the #120/#146 `turface_19` golden (`off` == `warn`, and
  contracts-absent == contracts-present) on the deterministic QC/PCA stages, UMAP excluded.
- Related: downstream of contracts#3 (`v0.1.0a1`); runtime counterpart to fixture-conformance #147; sibling
  of the bloom-mcp data-access strict guard.
