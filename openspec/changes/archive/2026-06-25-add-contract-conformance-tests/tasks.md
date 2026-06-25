## 1. Canonical-example conformance test (write first — fails red on a clean checkout)

- [x] 1.1 Write `tests/test_contract_conformance.py` importing `sleap_roots_contracts`.
      On a clean checkout (contracts not installed) it fails red with `ModuleNotFoundError`
      — the genuine TDD red for a tests-only change.
- [x] 1.2 Add a test that iterates `analysis_input_example_names()`
      (`cylinder`, `cylinder_no_replicate`, `field`, `turface`, `genotype_means`), loads
      each via `load_analysis_input_example`, and asserts
      `validate_analysis_input(example).raise_for_status()` (default non-strict mode).

## 2. Real-data conformance test (parametrized over 4 platforms)

- [x] 2.1 Reuse the existing session loader `final_data_by_platform`
      (`tests/fixtures.py`) — a **session-scoped dict** `{platform: DataFrame}` over
      `expected/qc/{p}/10_final_data.csv`; do **not** add a second loader (DRY).
      Parametrize over `turface_19`, `turface_150`, `cylinder`, `root_core`. Because the
      fixture is shared, all transforms operate on `df.copy()` (never the dict's frame).
- [x] 2.2 Build a **copy**: rename `{Genotype→genotype, Barcode→sample_id,
      Replicate→replicate}`, select `roles + get_trait_columns(renamed,
      barcode_col="sample_id", genotype_col="genotype", replicate_col="replicate")` — the
      explicit kwargs are mandatory (defaults `geno`/`rep`/`Barcode` would duplicate the
      numeric `replicate` into the trait set) — then `canonicalize_role_dtypes`.
- [x] 2.3 Assert the rename happened and the trait set is clean:
      `{"genotype","sample_id"}.issubset(check.columns)`, native names gone, and
      `"replicate" not in trait_cols and "genotype" not in trait_cols` (anti-vacuous).
- [x] 2.4 Assert `validate_analysis_input(check).raise_for_status()` succeeds for every
      platform.

## 3. Pipeline-input guard + negative control

- [x] 3.1 Deep-copy the shared `final_data_by_platform[platform]` frame, run the full
      build, then assert `pd.testing.assert_frame_equal(shared_frame, pre_copy)` — proves
      canonicalization ran on a copy and the session fixture is unmutated (catches an
      in-place cast / `rename(inplace=True)` regression that would corrupt other tests).
- [x] 3.2 Negative control: a deliberately malformed frame (e.g. drop the `genotype` role)
      makes `validate_analysis_input(bad).raise_for_status()` raise `ValueError`
      (`.ok is False`), proving the post-QC green asserts are non-vacuous.

## 4. Dev dependency (turns the red tests green)

- [x] 4.1 Add `sleap-roots-contracts[pandas]>=0.1.0a1` to the `dev` group in
      `pyproject.toml`; `uv add --dev --prerelease=allow` resolves the alpha. NOTE: that
      flag also bumped pydantic to a needless `2.14.0a1` alpha, so re-pin stable with
      `uv lock --upgrade-package pydantic --upgrade-package pydantic-core` (contracts only
      needs `pydantic>=2.7`). Commit `pyproject.toml` + `uv.lock` together so
      `uv sync --frozen` stays green; only `sleap-roots-contracts` is a prerelease.
- [x] 4.2 Confirm the lockfile resolves the alpha and the `[pandas]` extra on the CI
      matrix (Ubuntu/Windows/macOS, Python 3.11); the tests now run unskipped and pass.

## 5. Cleanup guard

- [x] 5.1 Assert no `*_validation.json` expected files exist under `tests/fixtures/`
      (removed in `73583f9`; verify `git ls-files 'tests/**_validation.json'` is empty —
      a guard, not a delete step). The contract is asserted live, never against stored JSON.

## 6. Verify

- [x] 6.1 Run the full suite: contract tests run (no skip) and pass; the #146/#120
      reproduction golden tests stay green.
- [x] 6.2 `/lint` clean (black + ruff + pydocstyle, google docstrings).
- [x] 6.3 `openspec validate add-contract-conformance-tests --strict` passes.

## Commit plan (CI green at each step)

1. `chore: add sleap-roots-contracts[pandas] dev dependency for contract conformance (#147)`
   — `pyproject.toml` + `uv.lock` (lands first so the test module imports cleanly in CI).
2. `test: assert post-QC EDPIE fixtures + canonical examples conform to analysis-input contract (#147)`
   — `tests/test_contract_conformance.py` (real-data + examples + guard + negative control).

Single PR off `add-contract-conformance-tests-147`.
