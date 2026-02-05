## Why

Cross-platform correlation analysis tests ~28,665 trait pairs, making FDR correction extremely conservative. With n≈18 genotypes, even correlations with adequate power (≥80%) cannot survive BY or BH correction due to the massive multiple testing burden. Many traits in exp2 are highly correlated (e.g., `Seminal_Length_Mean` vs `Seminal_Length_Median`), inflating the test count without adding independent information.

Reducing trait redundancy before correlation analysis will:
1. Decrease the number of tests by 90-95% (from ~28K to ~500-1000)
2. Make FDR correction tractable
3. Improve interpretability (one representative per biological construct)
4. Preserve statistical power for remaining tests

## What Changes

- **ADDED**: `TraitRedundancyConfig` dataclass with parameters for trait clustering
- **ADDED**: `ReduceTraitRedundancyStep` pipeline step that clusters correlated traits and selects representatives
- **ADDED**: Config options: `trait_reduction_method`, `trait_clustering_threshold`, `trait_clustering_linkage`
- **MODIFIED**: `CrossPlatformConfig` to include trait reduction parameters
- **MODIFIED**: Cross-platform pipeline to insert trait reduction step between data loading and correlation calculation
- **ADDED**: Cluster membership output file for traceability (`trait_clusters.csv`)
- **ADDED**: Metadata reporting: original trait counts, reduced counts, method used

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/pipeline/config/components.py` (CrossPlatformConfig)
  - `src/sleap_roots_analyze/pipeline/steps/reduce_trait_redundancy.py` (new)
  - `src/sleap_roots_analyze/pipeline/pipelines/cross_platform_pipeline.py`
  - `src/sleap_roots_analyze/cross_experiment_analysis.py` (clustering functions)
- Backward compatible: Default `trait_reduction_method: "none"` preserves existing behavior
- Related issue: #50

## Scientific Rationale

### Correlation-Based Clustering

Hierarchical clustering with complete linkage groups traits where **all** members are highly correlated (|r| ≥ threshold). This is conservative—a trait is only grouped if it correlates strongly with every other member.

**Algorithm**:
1. Compute pairwise Spearman correlations within experiment traits
2. Convert to distance: d = 1 - |r| (range [0, 1])
3. Apply hierarchical clustering with complete linkage
4. Cut dendrogram at threshold t = 1 - clustering_threshold
5. Select representative per cluster (highest variance = most informative)

**Why complete linkage**: Ensures clusters are compact. A trait joins a cluster only if it correlates ≥ threshold with ALL existing members, not just one (contrast with single linkage).

**Why highest variance representative**: Traits with higher variance carry more information and are more likely to show true correlations. Low-variance traits may appear constant due to measurement noise.

### Reproducibility Guarantees

- Deterministic: No random components
- Traceable: Cluster membership saved to CSV
- Reversible: Original traits preserved; only correlation step uses representatives

## References

- Storey & Tibshirani (2003). Statistical significance for genomewide studies. PNAS.
- Hastie, Tibshirani & Friedman (2009). The Elements of Statistical Learning, Ch. 14.
