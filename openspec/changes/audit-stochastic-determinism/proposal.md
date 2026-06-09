# Proposal: Audit `random_state`/Seed Handling for Determinism

## Why

bloom-mcp's Phase 2 golden-value tests will assert that past analyses — including
the wheat EDPIE paper's PCA, clustering, and UMAP results — reproduce within
tolerance on every run and CI machine. If any stochastic function returns different
output across runs with the same seed, those golden tests fail non-deterministically
and waste debugging time (e.g. a flaky `perform_umap_analysis` embedding).

Tracked by issue #118.

**Audit result (this change is mostly verification + lock-in):** every genuinely
stochastic public function *already* accepts a `random_state` and propagates it to
its underlying sklearn/umap call, and an empirical two-run check confirms identical
output (UMAP embeddings are bit-identical with a fixed seed). The gaps are not in the
implementation but in the *guarantees*: there is no regression test pinning this
behavior, and no documented tolerance policy. This change adds both, so the
determinism contract is enforced and the cross-platform expectations are written
down before the intern's golden tests land.

### Audit findings

| Function | Module | `random_state` | Propagated | Deterministic (2-run) |
| --- | --- | --- | --- | --- |
| `perform_pca_analysis` | `pca` | ✅ (=42) | ✅ via `fit_pca`/`select_n_components` | ✅ |
| `perform_umap_analysis` | `umap` | ✅ (=42) | ✅ to `umap.UMAP` | ✅ (exact) |
| `perform_kmeans_clustering` | `clustering` | ✅ (=42) | ✅ to `KMeans` | ✅ |
| `perform_gmm_clustering` | `clustering` | ✅ (=42) | ✅ to `GaussianMixture` | ✅ |
| `perform_hierarchical_clustering` | `clustering` | ➖ none | n/a | ✅ (deterministic) |
| `detect_outliers_isolation_forest` | `outlier_detection` | ✅ (=42) | ✅ to `IsolationForest` | ✅ |
| `detect_outliers_kmeans` | `outlier_detection` | ✅ (=42) | ✅ via `perform_kmeans_clustering` | ✅ |
| `detect_outliers_gmm` | `outlier_detection` | ✅ (=42) | ✅ via `perform_gmm_clustering` | ✅ |
| `detect_outliers_mahalanobis` | `outlier_detection` | ✅ (=42) | ✅ via `perform_pca_analysis` | ✅ |

`perform_hierarchical_clustering` uses scipy `linkage`, which is **deterministic** —
the same input always yields the same linkage matrix. Adding a no-op `random_state`
would misrepresent the API, so it is documented as seed-free and covered by the
determinism test instead.

No other public function computes a stochastic result without a seed: the
`create_umap_*` plot helpers consume a precomputed embedding (they do not run UMAP),
and `create_phenotype_variation_plot` already calls `np.random.seed(42)` before
adding jitter.

## What Changes

1. **Add `tests/test_reproducibility.py`** — for each stochastic public function,
   run it twice with the same `random_state` on a shared small synthetic dataset and
   assert output equality: exact for integer labels/indices and discrete metadata,
   `rtol=1e-6` for float arrays (embeddings, transformed data, distances). Include
   `perform_hierarchical_clustering` (no seed) to lock in its determinism. Add a
   smoke check that each function accepts `random_state=None` without error.

2. **Add `docs/reproducibility.md`** — the seeding and tolerance policy: which
   functions are stochastic and their default seed (`42`); the determinism guarantee
   (same seed + same input + same environment → identical output); the
   float-comparison tolerance (`rtol=1e-6`, integer labels exact) and the
   cross-platform/BLAS caveat; and the note that hierarchical clustering is
   deterministic and seed-free.

3. **No production code change is required** — the audit confirmed every stochastic
   function already exposes and propagates `random_state`. (If the determinism test
   surfaces any propagation gap, the minimal fix is folded in under this change.)

## Impact

- Affected specs: **stochastic-determinism** (new capability).
- Affected code:
  - `tests/test_reproducibility.py` (new)
  - `docs/reproducibility.md` (new)
- **No behavior change** to any analysis function — this adds a regression test and
  documentation that pin existing behavior.
