## 1. Config flag

- [x] 1.1 (test) `CrossPlatformConfig.validate_input` defaults to `warn`; `__post_init__` rejects an invalid value naming `off | warn | strict`
- [x] 1.2 Add `validate_input: str = "warn"` to `CrossPlatformConfig` with docstring + enum check in `__post_init__`

## 2. Cross-platform helper

- [x] 2.1 (test) `validate_cross_platform_experiment`: validates an aligned (genotype/replicate-canonical) frame on a copy; good passes under warn **and** strict (strict injects a synthetic positional `sample_id` for the structurally-absent per-sample role); missing-`genotype` raises even under warn; `off` no-op; contracts-absent no-op
- [x] 2.2 Add `validate_cross_platform_experiment(df, *, mode, additional_exclude=None, logger=None)` to `input_contract.py`, reusing `validate_entry_input` with fixed canonical role names; under non-`off` modes inject a synthetic `sample_id` into the discarded copy so `strict` is usable on aligned frames (review)

## 3. Wire into cross-platform load boundary

- [x] 3.1 (test) `LoadCrossPlatformDataStep` invokes the helper once per experiment frame (exp1, exp2) with `config.validate_input`; returned frames unchanged; the helper receives `exp1_exclude_cols`/`exp2_exclude_cols` so the validated trait set matches the analyzed one (review)
- [x] 3.2 Call `validate_cross_platform_experiment` on `exp1_df`/`exp2_df` in `LoadCrossPlatformDataStep.execute` after load/align, before trait selection, passing `additional_exclude=config.exp{1,2}_exclude_cols` (review)
- [x] 3.3 Drop blank/NaN-genotype rows during `load_and_align_experiments` so the validator no longer aborts a previously-successful run under default `warn`, keeping `off`/`warn`/`strict` output-identical (review)

## 4. Equivalence + optional-dependency proofs

- [x] 4.1 (test) Equivalence: cross-platform load with `validate_input=off` vs `=warn` (and `=strict`) on a golden-shaped pair → identical output
- [x] 4.2 (test) Runs-without-contracts: monkeypatch `CONTRACTS_AVAILABLE=False` → no crash, identical output, skip logged
- [x] 4.3 (test) Strict on a clean aligned frame passes (synthetic `sample_id`) and does not mutate the input; strict still catches a structural error (review)
- [x] 4.4 (test) Blank/NaN-genotype pair: `off`/`warn` produce identical output and `warn` does not abort (review)
- [x] 4.5 (test) Validation does not pre-empt the existing "No common genotypes found" error (review)
- [x] 4.6 (test) YAML load path: `load_cross_platform_config` round-trips `validate_input` and `__post_init__` rejects an invalid value via `to_object` (review)

## 5. Docs + pre-merge

- [x] 5.1 Add `[Unreleased]` CHANGELOG note (cross-platform validation, #154)
- [x] 5.2 Documentation parity: surface the `validate_input` flag + strict caveat in the field doc and `docs/CROSS_PLATFORM_ANALYSIS.md` (review)
- [x] 5.3 `/lint` + `/fix-formatting` clean
- [ ] 5.4 Full `uv run pytest` green; coverage on new code paths
- [ ] 5.5 `openspec validate add-cross-platform-input-validation --strict` passes
