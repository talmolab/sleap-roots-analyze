# Reproducibility & Seeding Policy

This document defines how `sleap-roots-analyze` guarantees reproducible results from
its stochastic analyses, and the numerical tolerance to expect when comparing outputs
(e.g. for golden-value tests). It backs issue #118 and is enforced by
[`tests/test_reproducibility.py`](../tests/test_reproducibility.py).

## Seeding policy

Every public function whose result depends on a random number generator accepts a
`random_state` parameter (default `42`) and forwards it to the underlying
sklearn / umap estimator. Pass an explicit integer for reproducible output; pass
`None` to opt into non-deterministic behavior.

| Function | Module | Default `random_state` | Underlying RNG |
| --- | --- | --- | --- |
| `perform_pca_analysis` | `pca` | `42` | sklearn `PCA` (matters for the randomized SVD solver) |
| `perform_umap_analysis` | `umap` | `42` | `umap.UMAP` |
| `perform_kmeans_clustering` | `clustering` | `42` | sklearn `KMeans` |
| `perform_gmm_clustering` | `clustering` | `42` | sklearn `GaussianMixture` |
| `detect_outliers_isolation_forest` | `outlier_detection` | `42` | sklearn `IsolationForest` |
| `detect_outliers_kmeans` | `outlier_detection` | `42` | via `perform_kmeans_clustering` |
| `detect_outliers_gmm` | `outlier_detection` | `42` | via `perform_gmm_clustering` |
| `detect_outliers_mahalanobis` | `outlier_detection` | `42` | via `perform_pca_analysis` |

### Deterministic functions (no seed needed)

`perform_hierarchical_clustering` uses scipy `linkage`, which is **deterministic**:
the same input always produces the same linkage matrix. It intentionally does **not**
take a `random_state` — adding one would misrepresent the API. Its determinism is
still covered by the regression test.

`create_phenotype_variation_plot` adds plotting jitter seeded with
`np.random.seed(42)`, so its visual output is reproducible. The `create_umap_*` plot
helpers consume a **precomputed** embedding and run no RNG of their own.

## Determinism guarantee

> Given the same `random_state`, the same input data, and the same environment
> (library versions + BLAS backend), each stochastic function returns identical
> output.

This is verified on every test run: `tests/test_reproducibility.py` calls each
function twice with `random_state=42` and asserts the reproducibility-bearing outputs
match. Within a single machine the outputs are bit-for-bit identical (UMAP embeddings
included).

## Tolerance policy

When comparing outputs across runs or machines, use these tolerances:

- **Integer cluster labels and outlier indices:** compare for **exact** equality.
- **Floating-point arrays** (UMAP embeddings, PCA transformed data / loadings /
  eigenvalues, distances, cluster centers): compare with **`rtol=1e-6`** (and a small
  `atol`, e.g. `1e-9`). This is the typical reliable tolerance for sklearn outputs.

```python
import numpy as np

# integer labels — exact
assert np.array_equal(a["cluster_labels"], b["cluster_labels"])

# float arrays — within tolerance
assert np.allclose(a["embedding"], b["embedding"], rtol=1e-6, atol=1e-9)
```

### Cross-platform / BLAS caveat

Bit-for-bit equality is guaranteed **within a single environment**. Across operating
systems or BLAS implementations (OpenBLAS vs MKL vs Accelerate), floating-point
reductions can reorder and produce differences at roughly the `1e-6`–`1e-12` level.
Therefore:

- Do **not** assert exact float equality across machines — use `rtol=1e-6`.
- Cluster labels can in principle flip if two samples are near a decision boundary;
  for golden tests prefer comparing them up to a label permutation, or assert on
  stable derived quantities (counts, sorted distances) when an exact match is needed.
- We do **not** pin a BLAS backend. If a future golden test proves sensitive at the
  `1e-6` level, pin the BLAS implementation in that test's environment (e.g. via the
  `threadpoolctl` / `OPENBLAS_*` env) and document it alongside the test.
