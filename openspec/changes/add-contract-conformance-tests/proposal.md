## Why

`sleap-roots-contracts 0.1.0a1` is released, so we can now prove the committed EDPIE
post-QC fixtures and the package's canonical examples conform to the analysis-input
contract (`validate_analysis_input`). PR #146 landed the reproduction fixtures without
that check — it was split out along the contracts-dependency seam (issue #147) — and this
change closes the gap so downstream QC/viz/cross-platform code can rely on the schema.

## What Changes

This is a **tests + dev-dependency** change only — **no `src/` changes**.

- Add `sleap-roots-contracts[pandas]>=0.1.0a1` to the **dev** dependency group so the
  contract tests run for real (no `importorskip`). Because it is an alpha, the lockfile
  must resolve the pre-release and be committed alongside the `pyproject.toml` edit.
- Add a contracts-dependent test module that:
  - **Real-data conformance** (parametrized over all four platforms `turface_19`,
    `turface_150`, `cylinder`, `root_core`) — builds a **copy** of the post-QC fixture:
    rename native roles (`Genotype`→`genotype`, `Barcode`→`sample_id`,
    `Replicate`→`replicate`), drop non-trait metadata via `get_trait_columns` **called
    with explicit role kwargs** (`barcode_col="sample_id"`, `genotype_col="genotype"`,
    `replicate_col="replicate"` — defaults are `geno`/`rep`/`Barcode`, which would leak
    the numeric `replicate` into the trait set as a duplicate column), then
    `canonicalize_role_dtypes`; asserts `validate_analysis_input(check).raise_for_status()`
    under the default (non-strict) mode. It also asserts the rename actually happened
    (`genotype`/`sample_id` present, native names gone; `replicate`/`genotype` absent from
    the trait set) so a no-op rename cannot pass vacuously.
  - **Canonical-example conformance** — iterates every name from
    `analysis_input_example_names()` (the package registry: `cylinder`,
    `cylinder_no_replicate`, `field`, `turface`, `genotype_means`), loads each via
    `load_analysis_input_example`, and asserts it validates as-is. Iterating the registry
    (rather than a hardcoded list) keeps the contract's own source of truth authoritative
    and auto-covers future examples.
  - **Negative control** — a deliberately malformed frame (e.g. missing the `genotype`
    role) makes `validate_analysis_input(...).raise_for_status()` raise / `.ok is False`,
    proving the green asserts above are non-vacuous.
  - **Pipeline-input guard** — re-loads the fixture fresh and asserts
    `pd.testing.assert_frame_equal` against a deep pre-copy, proving canonicalization ran
    on a copy and never touched the frame that feeds QC/viz/cross-platform.
    (`canonicalize_role_dtypes` already returns a copy, so this is defense-in-depth.)
- Confirm no stale `*_validation.json` expected files remain (they were removed in commit
  `73583f9`; their `summary` shape never matched `ValidationResult`). Keep this as a guard
  assertion, not a delete step.

## Impact

- Affected specs: `contract-conformance` (new capability). Depends on the #120/#146
  reproduction fixtures (capability `reproduction-fixtures`) being present.
- Affected code: `pyproject.toml` (dev group) + `uv.lock`, and a new `tests/` contract
  test module. **No `src/` changes**; `get_trait_columns` is consumed, not modified.
- Guardrail: canonicalization runs on a **copy**, never the frame that feeds the pipeline.
  The #146/#120 reproduction golden tests (native names, `rtol=1e-6`) stay green as proof
  `run-all` output is unchanged. This change does **not** implement #144 (the runtime
  `validate_input` flag / `run-all` wiring) — only fixture conformance tests.
