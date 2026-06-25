## 1. Optional dependency

- [x] 1.1 Add `[project.optional-dependencies]` `contracts = ["sleap-roots-contracts[pandas]>=0.1.0a1"]` to `pyproject.toml`, and add the same spec to the `dev` dependency group
- [x] 1.2 Run `uv lock` and **commit the updated `uv.lock`** in the same commit (build.yml uses `uv sync --frozen` and will fail on a stale lock)

## 2. Config flag

- [x] 2.1 (test) `data.validate_input` defaults to `warn` on `DataConfig`; `validate_qc_config` rejects an invalid value naming `off | warn | strict`
- [x] 2.2 Add `validate_input: str = "warn"` to `DataConfig` with the canonical Google-style docstring describing `off`/`warn`/`strict` + the optional-dependency caveat (single source of truth for the modes)
- [x] 2.3 Add the `off|warn|strict` enum check to `validate_qc_config` (`pipeline/config/utils.py`)

## 3. Boundary helper module

- [x] 3.1 Create `src/sleap_roots_analyze/validation/__init__.py` (new package) and `input_contract.py` skeleton with the guarded import (`CONTRACTS_AVAILABLE`)
- [x] 3.2 (test) Build malformed fixtures (3 distinct: missing-`genotype`, no numeric trait, bad role dtype) + a non-fatal one (missing `sample_id`); good-input via `sleap_roots_contracts.examples.load_analysis_input_example`
- [x] 3.3 (test) `validate_entry_input`: `mode="off"` → no-op even when contracts installed (validator not called); validation runs on a copy (downstream frame unchanged via `assert_frame_equal`); numeric `replicate` not leaked as a trait; `columns.replicate=None` (`cylinder_no_replicate`) → rename skipped, no error; `additional_exclude` columns dropped from the validation copy
- [x] 3.4 (test) Severity + logging (`caplog`): good passes; missing-`sample_id` warns under `warn` (WARNING logged) and raises under `strict`; missing-`genotype`/no-trait/bad-dtype raise even under `warn`
- [x] 3.5 (test) Real import-absent branch: `sys.modules["sleap_roots_contracts"]=None` + `importlib.reload(input_contract)` → `CONTRACTS_AVAILABLE is False`, `validate_entry_input` logs a skip and returns; restore in `finally`
- [x] 3.6 Implement `validate_entry_input(df, *, columns, mode, additional_exclude=None, logger=None)` per design.md (canonicalize-then-validate on a copy; returns `None`)

## 4. Wire into QC load boundary

- [x] 4.1 (test) `LoadDataStep` invokes the helper exactly once on the entry frame (both CSV and pre-loaded root-core branches) with `config.data.validate_input`; returned `StepResult.data` unchanged; not called on downstream steps
- [x] 4.2 Call `validate_entry_input` in `LoadDataStep.execute` after the entry frame is obtained, before trait selection; skip when `mode == "off"`

## 5. Equivalence + optional-dependency proofs

- [x] 5.1 (test) Equivalence: run the QC pipeline on the #120/#146 `turface_19` reference with `validate_input=off` vs `=warn` → identical deterministic QC + PCA output at `rtol=1e-6` (UMAP coordinates excluded), reusing the `tests/test_pipeline_reproduction.py` golden/pattern
- [x] 5.2 (test) Runs-without-contracts behavior: monkeypatch `CONTRACTS_AVAILABLE=False` (null the imported names too) → no crash, identical output, skip logged

## 6. Docs + pre-merge

- [x] 6.1 Add `[Unreleased] → Added` entry to `docs/CHANGELOG.md` (the new flag + optional extra + #144)
- [x] 6.2 Document the `pip install "sleap-roots-analyze[contracts]"` extra in `README.md` Installation
- [x] 6.3 Add a commented `validate_input: warn` block (pointing to the docstring) to `configs/templates/qc_template_grouped.yaml` and `qc_template_ungrouped.yaml`; do NOT regenerate `configs/active/*` (additive, default-valued, golden-equivalent)
- [x] 6.4 `/lint` + `/fix-formatting` (black 88, ruff, pydocstyle) clean
- [ ] 6.5 Full `uv run pytest --cov --cov-branch` green; coverage ≥ 90% on `validation/input_contract.py` (confirm `--cov-branch` covers: import-absent, `mode=off` early return, contracts-absent early return, `replicate=None` rename skip)
- [ ] 6.6 `openspec validate add-input-contract-validation --strict` passes

## 7. Follow-up (not this change)

- [ ] 7.1 File a follow-up issue: wire validation into the cross-platform boundary (`LoadCrossPlatformDataStep`) with a genotype-only recipe and a new `validate_cross_platform_config`
