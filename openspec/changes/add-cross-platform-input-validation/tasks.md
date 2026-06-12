## 1. Config flag

- [x] 1.1 (test) `CrossPlatformConfig.validate_input` defaults to `warn`; `__post_init__` rejects an invalid value naming `off | warn | strict`
- [x] 1.2 Add `validate_input: str = "warn"` to `CrossPlatformConfig` with docstring + enum check in `__post_init__`

## 2. Cross-platform helper

- [x] 2.1 (test) `validate_cross_platform_experiment`: validates an aligned (genotype/replicate-canonical) frame on a copy; good passes under warn/strict; missing-`genotype` raises even under warn; `off` no-op; contracts-absent no-op
- [x] 2.2 Add `validate_cross_platform_experiment(df, *, mode, additional_exclude=None, logger=None)` to `input_contract.py`, reusing `validate_entry_input` with fixed canonical role names

## 3. Wire into cross-platform load boundary

- [x] 3.1 (test) `LoadCrossPlatformDataStep` invokes the helper once per experiment frame (exp1, exp2) with `config.validate_input`; returned frames unchanged
- [x] 3.2 Call `validate_cross_platform_experiment` on `exp1_df` and `exp2_df` in `LoadCrossPlatformDataStep.execute`, after load/align, before trait selection

## 4. Equivalence + optional-dependency proofs

- [x] 4.1 (test) Equivalence: cross-platform load with `validate_input=off` vs `=warn` on a golden-shaped pair → identical output
- [x] 4.2 (test) Runs-without-contracts: monkeypatch `CONTRACTS_AVAILABLE=False` → no crash, identical output, skip logged

## 5. Docs + pre-merge

- [x] 5.1 Add `[Unreleased]` CHANGELOG note (cross-platform validation, #154)
- [x] 5.2 N/A — no cross-platform golden template exists
- [x] 5.3 `/lint` + `/fix-formatting` clean
- [ ] 5.4 Full `uv run pytest` green; coverage on new code paths
- [ ] 5.5 `openspec validate add-cross-platform-input-validation --strict` passes
