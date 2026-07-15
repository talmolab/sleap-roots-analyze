## ADDED Requirements

### Requirement: BLUP Table Output Config Gating

`StatisticsConfig` SHALL expose `generate_blup_table: bool = True`, controlling
whether `StatisticalAnalysisStep` writes `08_blup_adjusted_means.csv`. This flag
SHALL only take effect when `calculate_heritability` is also `True` — BLUP
extraction reuses the same mixed-model fit as heritability, so it cannot run
independently of it. Setting `generate_blup_table=True` while
`calculate_heritability=False` SHALL NOT raise (unlike the contradictory
`heritability.enabled`/`calculate_heritability` combination, which raises) and
SHALL NOT warn — it SHALL simply produce no BLUP output, since there is no
model fit to extract from. This mirrors the project's other accepted
"one flag makes another irrelevant" precedent (`calculate_heritability=True`
+ `heritability.enabled=False`), rather than the `umap.enabled`-on-an-unwired-
path warning precedent: unlike that case (a rare, deliberate opt-in into a
feature that would otherwise look silently broken), `generate_blup_table`
defaults `True`, so a warning would fire on the ordinary, common act of
disabling heritability. The dependency SHALL instead be documented in the
field's own docstring.

`StatisticalAnalysisStep` runs in both the QC pipeline (`qc_pipeline.py`) and
the Viz pipeline (`viz_pipeline.py`). Only `VizPipelineConfig` composes
`StatisticsConfig`; `QCPipelineConfig` has no `statistics` field. The step
already resolves this via `getattr(config, "statistics", None)`, treating a
QC-pipeline config (which has no `statistics` field to set) as
`calculate_heritability=True` by default — `generate_blup_table` SHALL be
resolved the same way, so a QC-pipeline run always gets the default (`True`)
for both flags and cannot configure `generate_blup_table` any other way.

#### Scenario: Both flags enabled — CSV is written

- **WHEN** a pipeline runs `StatisticalAnalysisStep` with
  `statistics.calculate_heritability=True` and
  `statistics.generate_blup_table=True` (the defaults)
- **THEN** `08_blup_adjusted_means.csv` SHALL be written to the run's `data/`
  directory
- **AND** its row count SHALL equal the number of distinct genotypes in the
  input data
- **AND** its column count SHALL equal the number of trait columns passed to
  the step, including any trait whose heritability estimation failed (those
  columns are `NaN`, per `extract_blup_table()`'s contract) — not just the
  traits that succeeded

#### Scenario: generate_blup_table disabled — no CSV

- **WHEN** a pipeline runs `StatisticalAnalysisStep` with
  `statistics.calculate_heritability=True` and
  `statistics.generate_blup_table=False`
- **THEN** `08_blup_adjusted_means.csv` SHALL NOT be written
- **AND** `08_heritability_results.csv` SHALL still be written unchanged

#### Scenario: calculate_heritability disabled — no CSV, no warning, no exception

- **WHEN** a pipeline runs `StatisticalAnalysisStep` with
  `statistics.calculate_heritability=False` and
  `statistics.generate_blup_table=True` (the default)
- **THEN** `08_blup_adjusted_means.csv` SHALL NOT be written
- **AND** no exception SHALL be raised
- **AND** no warning SHALL be issued — this is an ordinary, legitimate
  configuration (heritability disabled entirely), not a misconfiguration to
  flag

#### Scenario: calculate_heritability disabled and generate_blup_table also disabled — no CSV

- **WHEN** a pipeline runs `StatisticalAnalysisStep` with
  `statistics.calculate_heritability=False` and
  `statistics.generate_blup_table=False`
- **THEN** `08_blup_adjusted_means.csv` SHALL NOT be written
- **AND** no exception or warning SHALL be raised

#### Scenario: Default config produces BLUP output

- **WHEN** a `StatisticsConfig` is constructed with default values
  (`calculate_heritability=True`, `generate_blup_table=True`)
- **THEN** `StatisticalAnalysisStep` SHALL write `08_blup_adjusted_means.csv`
- **AND** existing configs that do not set `generate_blup_table` explicitly
  SHALL get this default (backward-compatible, additive behavior — an
  existing pipeline run gains one new output file with no other change)

#### Scenario: QC-pipeline config has no way to configure generate_blup_table

- **WHEN** `StatisticalAnalysisStep.execute()` runs with a `QCPipelineConfig`
  (which has no `statistics` field)
- **THEN** `generate_blup_table` SHALL resolve to `True` (the same default
  fallback already used for `calculate_heritability` when `config.statistics`
  is absent)
- **AND** the implementation MUST resolve this via a
  `getattr(config, "statistics", None)` guard rather than a direct
  `config.statistics.generate_blup_table` attribute access — the latter would
  raise `AttributeError` on `.statistics` itself, before `.generate_blup_table`
  is ever reached, since `QCPipelineConfig` has no `statistics` attribute at
  all
- **AND** `StatisticalAnalysisStep.execute()` SHALL NOT raise `AttributeError`
  when run with such a config
