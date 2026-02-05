## 1. Test Infrastructure (TDD: Red Phase)

- [x] 1.1 Create test file `tests/test_trait_redundancy.py`
- [x] 1.2 Write test fixtures for synthetic correlated traits
  - Fixture with 10 traits in 3 clusters (r=0.95 within, r=0.1 between)
  - Fixture with all independent traits (r≈0)
  - Fixture with one dominant cluster + noise
- [x] 1.3 Write unit tests for `cluster_correlated_traits()` function
  - Test correct cluster count with known synthetic data
  - Test threshold sensitivity (higher threshold = more clusters)
  - Test determinism (same input = same output)
  - Test edge cases: single trait, all identical traits, all NaN trait
- [x] 1.4 Write unit tests for `select_cluster_representatives()` function
  - Test selects highest variance trait per cluster
  - Test handles ties deterministically (alphabetical fallback)
  - Test preserves trait names correctly
- [x] 1.5 Write integration tests for `ReduceTraitRedundancyStep`
  - Test step metadata includes reduction statistics
  - Test output includes cluster membership CSV
  - Test backward compatibility with `method: "none"`

## 2. Configuration (TDD: Green Phase)

- [x] 2.1 Add trait reduction parameters to `CrossPlatformConfig`
  - `trait_reduction_method: str = "none"` (options: "none", "clustering")
  - `trait_clustering_threshold: float = 0.8`
  - `trait_clustering_linkage: str = "complete"`
- [x] 2.2 Add validation in `__post_init__`
  - Threshold must be in (0, 1]
  - Linkage must be in ["complete", "average", "single"]
  - Method must be in ["none", "clustering"]
- [x] 2.3 Update config template with new parameters and documentation

## 3. Core Algorithm Implementation

- [x] 3.1 Implement `cluster_correlated_traits(df, threshold, linkage)` in `cross_experiment_analysis.py`
  - Input: DataFrame with traits as columns, genotypes as rows
  - Output: Dict mapping cluster_id -> list of trait names
  - Uses scipy.cluster.hierarchy for hierarchical clustering
- [x] 3.2 Implement `select_cluster_representatives(df, clusters)` in `cross_experiment_analysis.py`
  - Input: DataFrame, cluster dict
  - Output: List of representative trait names (one per cluster)
  - Selection criterion: highest variance within cluster
- [x] 3.3 Add docstrings with mathematical notation and examples
- [x] 3.4 Add logging for cluster statistics

## 4. Pipeline Step Implementation

- [x] 4.1 Create `ReduceTraitRedundancyStep` in `pipeline/steps/reduce_trait_redundancy.py`
  - Implements `PipelineStep` interface
  - Accepts prev_result from LoadCrossPlatformDataStep
  - Clusters traits in exp2 (typically larger trait set)
  - Optional: Also cluster exp1 if enabled
- [x] 4.2 Add step metadata:
  - `original_exp2_traits`: count before reduction
  - `reduced_exp2_traits`: count after reduction
  - `n_clusters`: number of clusters formed
  - `trait_reduction_method`: method used
  - `trait_clustering_threshold`: threshold used
- [x] 4.3 Generate cluster membership output:
  - `trait_clusters.csv`: trait, cluster_id, is_representative, variance
- [x] 4.4 Update `data` dict with reduced trait lists for downstream steps

## 5. Pipeline Integration

- [x] 5.1 Register step in `pipeline/steps/__init__.py`
- [x] 5.2 Add step to cross-platform pipeline between load and calculate
- [x] 5.3 Conditionally execute based on `trait_reduction_method`
- [x] 5.4 Update downstream steps to use reduced trait lists

## 6. Documentation

- [x] 6.1 Add "Trait Redundancy Reduction" section to `docs/CROSS_PLATFORM_ANALYSIS.md`
  - Explain the problem (test multiplicity)
  - Explain the solution (hierarchical clustering)
  - Document configuration options
  - Provide example with expected reduction
- [x] 6.2 Update config template with examples

## 7. Validation

- [x] 7.1 Run full test suite: `uv run pytest tests/`
- [x] 7.2 Run on real data: verify reduction from ~28K to ~500-1000 tests
- [x] 7.3 Validate cluster membership output is traceable
- [x] 7.4 Verify lint passes: `uv run black --check && uv run ruff check`
