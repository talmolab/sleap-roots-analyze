## Why

PR #153 (issue #144) wired optional input-contract validation into the **QC** load boundary only and
explicitly deferred the cross-platform loader. This change completes the work (issue #154): apply the same
optional contract validation to `LoadCrossPlatformDataStep` without changing any golden output.

## What Changes

- Add a `validate_input: off | warn | strict` flag (default `warn`) to `CrossPlatformConfig`, enum-validated
  in its existing `__post_init__`.
- Add a thin `validate_cross_platform_experiment` helper that reuses #153's `validate_entry_input`: the
  aligned experiment frames already carry canonical `genotype`/`replicate` columns (and no `sample_id`), so
  the recipe is the same canonicalize-then-validate on a **copy**, with fixed canonical role names.
- Wire the helper into `LoadCrossPlatformDataStep.execute`, validating **each** loaded experiment frame
  (exp1, exp2) once. The frames fed to alignment/correlation are never modified.
- Reuse the optional-dependency, copy-isolation, and severity semantics from #153 unchanged.

## Impact

- Affected specs: `input-contract-validation` (extends the capability added by #144 to the cross-platform
  boundary)
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/components.py` — `validate_input` on `CrossPlatformConfig` + enum check
  - `src/sleap_roots_analyze/validation/input_contract.py` — `validate_cross_platform_experiment` wrapper
  - `src/sleap_roots_analyze/pipeline/steps/load_cross_platform_data.py` — call helper on exp1/exp2
  - `docs/CHANGELOG.md` — `[Unreleased]` note
- Reproducibility: equivalence proven on the #120/#146 cross-platform golden (`off` == `warn`, and
  contracts-absent == contracts-present).
- Builds on (stacked atop) PR #153; depends on its `validate_entry_input` helper.
