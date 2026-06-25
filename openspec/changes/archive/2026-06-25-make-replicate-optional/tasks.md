# Tasks

## 1. Config validation accepts None

- [x] 1.1 Write a failing test: `validate_qc_config()` on a config with
  `columns.replicate=None` reports no replicate error (and still errors when
  `columns.genotype` is missing).
- [x] 1.2 Remove the `config.columns.replicate is None` required-field check in
  `pipeline/config/utils.py`.
- [x] 1.3 Run the test green.

## 2. Heritability runs without a replicate column

- [x] 2.1 Write a failing test: `calculate_heritability_estimates(..., replicate_col=None)`
  on a no-replicate fixture returns H² without raising.
- [x] 2.2 Write a failing equivalence test: on a fixture *with* a replicate column,
  H² is identical whether `replicate_col` is the column name or `None`.
- [x] 2.3 In `statistics.py` `calculate_heritability_estimates`, gate
  `required_cols`, the `dropna` subset, and the subset column rename on
  `replicate_col is not None`.
- [x] 2.4 Make `analyze_trait_variance` accept `replicate_col=None`
  (`Optional[str]` signature + gate the subset) so public diagnostics stay usable.
- [x] 2.5 Run tests green.

## 3. Trait detection handles None

- [x] 3.1 Write a regression test: `get_trait_columns(df, replicate_col=None)`
  returns all numeric trait columns and miscounts none (already guarded by
  `if replicate_col:`).
- [x] 3.2 Run green.

## 4. Cylinder-shaped end-to-end path

- [x] 4.1 Add a cylinder-shaped fixture (genotype → multiple plants, no replicate
  column) and a test that the QC heritability step runs and produces real H².
- [x] 4.2 NOT IN ORIGINAL ISSUE: `StatisticalAnalysisStep` hardcoded
  `replicate_col = "Replicate"`, so with `columns.replicate=None` heritability
  returned an error dict instead of H². Gate it to `None` when
  `columns.replicate is None`. (The issue mislabeled `statistical_analysis.py` as
  pass-through; it is the in-pipeline heritability consumer.)

## 5. Field root-core regression

- [x] 5.1 Regression test that a root-core fixture with a hardcoded `"Rep"` column
  aggregates identically regardless of `columns.replicate` (parametrized over
  None / "rep" / "block").

## 6. Docs

- [x] 6.1 Update `ColumnConfig.replicate` docstring to state it is optional and
  not a model term.
- [x] 6.2 Update config-authoring example docs (`QC_PIPELINE_GUIDE.md`) to show
  `replicate` as optional.

## 7. Validation

- [x] 7.1 `openspec validate make-replicate-optional --strict` passes.
- [ ] 7.2 `/lint` clean; full `pytest` green.
