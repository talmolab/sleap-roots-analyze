## ADDED Requirements

### Requirement: Public Trait-Level Correlation Enrichment Workflow

The package SHALL export a public function `trait_correlation_enrichment` from
the top-level `sleap_roots_analyze` namespace that tests whether the number of
nominally significant trait-level cross-platform correlations deviates from
chance using a binomial test, in a single call. The function SHALL accept
`correlation_files` (mapping of platform-pair label → path to a
`cross_platform_correlations.csv`), `output_dir`, and the keyword parameters
`alpha` (default `0.05`), `p_value_column` (default `"spearman_p"`),
`confidence_level` (default `0.95`), and `make_figure` (default `True`). The
function SHALL have complete type hints and a Google-style docstring and SHALL
return a structured `dict`.

#### Scenario: Function and result type are importable from the package root

- **WHEN** a consumer runs `from sleap_roots_analyze import trait_correlation_enrichment, EnrichmentResult`
- **THEN** the import SHALL succeed
- **AND** both `trait_correlation_enrichment` and `EnrichmentResult` SHALL be
  listed in `sleap_roots_analyze.__all__`

### Requirement: Binomial Enrichment Over Existing Correlation Files

For each supplied platform-pair correlation file the workflow SHALL count the
tests with `p_value_column` below `alpha` and run a binomial test against the
null proportion `alpha`, reporting fold enrichment, two-sided and one-sided
(enrichment and depletion) p-values, a Clopper-Pearson confidence interval on
the observed proportion, and an interpretation of `enriched`, `depleted`, or
`null`. The workflow SHALL also compute a `Combined` result pooling all files.

#### Scenario: Per-pair and combined enrichment are computed

- **WHEN** the workflow runs over three platform-pair correlation files
- **THEN** the result SHALL contain one `EnrichmentResult` per pair plus a
  `Combined` result whose `n_tests` equals the sum of the per-pair test counts

#### Scenario: Depletion is detected

- **WHEN** a correlation file has far fewer significant tests than `n_tests * alpha`
- **THEN** that pair's interpretation SHALL be `depleted` with a one-sided
  depletion p-value below `0.05`

### Requirement: Enrichment Workflow Is Independent of the PC Workflow

The trait-level enrichment workflow SHALL run from existing trait-level
`cross_platform_correlations.csv` files alone and SHALL NOT require any output of
`cross_platform_pc_correlations`. Its inputs are produced by the repository's
existing trait-level cross-platform pipeline.

#### Scenario: Runs with no PC-workflow outputs present

- **WHEN** `trait_correlation_enrichment` is called with only
  `cross_platform_correlations.csv` fixtures and no PC-workflow artifacts exist
- **THEN** the workflow SHALL complete and produce its enrichment results

### Requirement: Reproducible Enrichment Artifacts

The workflow SHALL write `enrichment_results.csv`, a `metadata.json` recording
parameters, provenance (input file paths), and a results summary, and — when
`make_figure=True` — an `enrichment_summary` figure under `output_dir`. The
function SHALL also return the in-memory results list so it is testable without
reading files back.

#### Scenario: Results CSV and metadata are written

- **WHEN** the workflow completes with `make_figure=False`
- **THEN** `enrichment_results.csv` and `metadata.json` SHALL exist under
  `output_dir`
- **AND** the returned `dict` SHALL expose the list of enrichment results whose
  row count matches `enrichment_results.csv`
