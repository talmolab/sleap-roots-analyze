# stochastic-determinism Specification

## Purpose
TBD - created by archiving change audit-stochastic-determinism. Update Purpose after archive.
## Requirements
### Requirement: Seeded Stochastic Public Functions

Every public function whose result depends on a random number generator SHALL accept
a `random_state` parameter and propagate it to the underlying sklearn/umap estimator,
so a fixed seed yields a reproducible result.

#### Scenario: Stochastic functions expose and propagate a seed

- **WHEN** any of `perform_pca_analysis`, `perform_umap_analysis`,
  `perform_kmeans_clustering`, `perform_gmm_clustering`,
  `detect_outliers_isolation_forest`, `detect_outliers_kmeans`,
  `detect_outliers_gmm`, or `detect_outliers_mahalanobis` is called
- **THEN** it SHALL accept a `random_state` argument
- **AND** that value SHALL reach the underlying random-using estimator

#### Scenario: Deterministic functions need no seed

- **WHEN** `perform_hierarchical_clustering` is called twice with the same input
- **THEN** it SHALL return an identical linkage matrix without requiring a
  `random_state` argument

### Requirement: Reproducible Outputs Under a Fixed Seed

Stochastic public functions SHALL produce identical output, within a documented
numerical tolerance, when called repeatedly with the same `random_state` and input.

#### Scenario: Two runs with the same seed match

- **WHEN** a stochastic public function is run twice with the same `random_state`
  and the same input data
- **THEN** integer cluster labels and outlier indices SHALL be exactly equal
- **AND** floating-point arrays (embeddings, transformed data, distances) SHALL be
  equal within `rtol=1e-6`

#### Scenario: A regression test enforces determinism

- **WHEN** the test suite runs
- **THEN** `tests/test_reproducibility.py` SHALL execute each stochastic public
  function twice and assert output equality, failing if any function becomes
  non-deterministic under a fixed seed

### Requirement: Documented Reproducibility Policy

The project SHALL document its seeding and numerical-tolerance policy so consumers
(including golden-value tests) know what reproducibility to expect.

#### Scenario: Reproducibility policy is documented

- **WHEN** `docs/reproducibility.md` is viewed
- **THEN** it SHALL list the stochastic public functions and their default seed
- **AND** it SHALL state the determinism guarantee (same seed + input + environment
  → identical output)
- **AND** it SHALL state the float-comparison tolerance (`rtol=1e-6`, integer labels
  exact) and the cross-platform/BLAS caveat
- **AND** it SHALL note that `perform_hierarchical_clustering` is deterministic and
  requires no seed

