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
