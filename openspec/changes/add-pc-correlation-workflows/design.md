# Design — PC-Correlation and Trait-Enrichment Workflows

## Context

The wheat EDPIE paper ships a `pc_correlations/` package (aggregate, correlate,
enrichment, visualize) driven by two scripts. We port it into
`sleap-roots-analyze` as a public subpackage. Two design tensions need an
explicit decision before implementation.

## Decision 1 — Public API shape supersedes the #119 sketch

Issue #119 sketched an **in-memory** signature
(`cross_platform_pc_correlations(platforms: dict[str, DataFrame], trait_cols,
n_components) -> CrossPlatformPCResult`) that re-runs PCA internally. The
maintainer-confirmed reference implementation (`run_analysis.py` + Box golden
outputs) instead consumes **pre-computed pipeline PCA outputs from disk** and
writes artifacts.

**Decision:** follow the reference implementation's **orchestration** shape:

```python
def cross_platform_pc_correlations(
    pipeline_run: Path,
    platform_config: dict,
    output_dir: Path,
    fdr_methods: list[str] | None = None,
    primary_fdr_method: str = "fdr_by",
    alpha: float = 0.05,
    confidence_level: float = 0.95,
    fdr_scope: str = "both",
    make_figures: bool = True,
) -> dict
```

Rationale: it matches the real paper code and golden outputs (so the regression
is meaningful), and it preserves the **sample-level-PCA → genotype-mean →
correlate** ordering #119 requires — the pipeline already did sample-level PCA
(`pc_scores.csv` is sample-level); `aggregate_pc_scores_by_genotype` averages
to genotype means *after* PCA, never average-then-PCA.

**Testability is preserved by the DAG nodes, not the orchestrator.** The pure
nodes (`load_platform_data`, `aggregate_pc_scores_by_genotype`,
`align_genotypes_across_platforms`, `calculate_all_platform_correlations`) are
independently unit-testable without IO; the orchestrator is a thin driver. The
returned `dict` bundles the in-memory artifacts (correlation DataFrame, summary,
aligned genotype means, output paths) so tests assert on values, not just files.

## Decision 2 — Two independent capabilities, not a pipeline

`trait_correlation_enrichment` is **not** downstream of
`cross_platform_pc_correlations`. They operate at different levels:

| | PC workflow | Enrichment workflow |
|---|---|---|
| Level | principal components | individual traits |
| Input | pipeline-run PCA dir + QC final_data | existing `cross_platform_correlations.csv` |
| Produces | PC correlations + FDR | binomial enrichment/depletion |

The enrichment input is produced by the repo's **existing** trait-level
`cross_platform` pipeline. Wiring them in series would be scientifically wrong
(PC tests ≠ trait tests). They share only the `visualize.save_figure` helper and
the `metadata.json` provenance convention. Tests explicitly verify enrichment
runs from correlation-CSV fixtures alone, with no PC-workflow outputs present.

## Decision 3 — Reuse existing statistics helpers

`cross_experiment_analysis.py` already defines `calculate_correlation_ci`,
`achieved_power`, and `minimum_detectable_correlation` (these are documented
requirements in the `cross-platform-analysis` spec). The ported `correlate.py`
imports them rather than re-defining, keeping a single source of truth. Only
genuinely new logic is added: multi-scope FDR (`combined` vs `per_pair`),
pairwise PC × PC enumeration, and the binomial `EnrichmentResult`.

## Decision 4 — `fdr_scope` column contract

- `combined`: one `multipletests` call across all pairs' p-values →
  `significant_combined_<method>`, `p_adj_combined_<method>`.
- `per_pair`: `multipletests` within each platform pair →
  `significant_per_pair_<method>`, `p_adj_per_pair_<method>`.
- `both` (default): emit both column families for sensitivity comparison.

`combined fdr_by` is the paper's primary scope/method (0 significant); the
`metadata.json` documents that the trait-level pipeline uses per-pair `fdr_by`.

## Decision 5 — Split the two workflows by their natural home (review feedback)

The two workflows have different shapes, so they get different homes:

- **Trait enrichment → a config-gated step in the existing cross-platform DAG.**
  Its input is the pipeline's own `cross_platform_correlations.csv`, so it slots
  in as a pure downstream node: `… → CalculateCorrelations →
  [CalculateTraitEnrichment] → Visualize`. Gated by `enrichment_enabled`
  (default off), it runs automatically under `cross-platform` and `run-all`. The
  step is **per-pair** (the pipeline is pairwise) and passes its input through so
  visualization is unaffected. The public `trait_correlation_enrichment` remains
  the thin ad-hoc entry point over historical CSVs and is where the **combined /
  all-pairs** pooled ("Combined") result lives — an all-platforms synthesis,
  above the pairwise DAG.

- **PC correlations → the all-platforms synthesis case.** They need viz's
  `pc_scores.csv` (a different pipeline) and span all pairs at once (the 47
  tests), so they do not fit the pairwise DAG; the orchestrator-reads-a-run-dir
  shape is justified, and a `pc-correlations` CLI subcommand is added.

## Decision 6 — Typed results and separated FDR validation (review feedback)

- The PC workflow returns a typed **`CrossPlatformPCResult`** (mirroring
  `EnrichmentResult`), pioneering the serializable-result-types pattern
  (#127–#130) for clean bloom-mcp wrapping.
- PC FDR methods are validated by the workflow's **own** validator (the full
  `statsmodels` family, incl. `bonferroni`) — never routed through the
  trait-level `CrossPlatformConfig` validator (which only allows
  `fdr_bh`/`fdr_by`/`none`).
- Enrichment uses the **nominal** p + an exact binomial test (no FDR). The
  config validates that `enrichment_p_value_column` matches `correlation_method`,
  so it cannot silently count the wrong column.
- **Representative population**: enrichment counts the rows of the produced
  correlation CSV — representative-only under `trait_reduction_method=
  "clustering"`, full-trait otherwise — an implicit inheritance made explicit
  and pinned by a test. PC correlations are full-trait PCA by construction.

## Module layout

```
src/sleap_roots_analyze/pc_correlations/
  __init__.py        # re-export workflow fns + EnrichmentResult
  aggregate.py       # load_pc_scores, load_genotype_mapping,
                     # aggregate_pc_scores_by_genotype, load_platform_data,
                     # align_genotypes_across_platforms, example WHEAT_EDPIE_PLATFORMS
  correlate.py       # calculate_correlation, calculate_cross_platform_pc_correlations,
                     # calculate_all_platform_correlations, summarize_correlations
                     # (CI/power/min-detectable imported from cross_experiment_analysis)
  enrichment.py      # EnrichmentResult, calculate_enrichment_test, count_significant,
                     # calculate_all_enrichment_tests, create_enrichment_figure, ...
  visualize.py       # heatmap, summary, scatter, sensitivity, save_figure
  workflow.py        # cross_platform_pc_correlations, trait_correlation_enrichment
```

## Risks / trade-offs

- **Disk-coupled orchestrator** is harder to fuzz than a pure function;
  mitigated by testing DAG nodes directly + a small synthetic pipeline-run dir.
- **matplotlib import cost / headless safety**: figure code uses the `Agg`
  backend inside workflows and `make_figures=False` keeps tests fast.
- **Golden regression depends on real post-QC data** (issue #120 fixtures);
  skip-guarded until present, mirroring the existing reproduction tests.
