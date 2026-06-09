# Tasks: Audit `random_state`/Seed Handling for Determinism

## 1. Determinism regression test (write the check)

- [x] 1.1 Add `tests/test_reproducibility.py` with a shared small synthetic dataset
  fixture (clustered, NaN-free, a few dozen samples × handful of features).
- [x] 1.2 For each seeded stochastic function (`perform_pca_analysis`,
  `perform_umap_analysis`, `perform_kmeans_clustering`, `perform_gmm_clustering`,
  `detect_outliers_isolation_forest`, `detect_outliers_kmeans`, `detect_outliers_gmm`,
  `detect_outliers_mahalanobis`): run twice with `random_state=42` and assert key
  outputs equal — exact for integer labels/indices, `rtol=1e-6` for float arrays.
- [x] 1.3 Add a `perform_hierarchical_clustering` case: run twice (no seed) and assert
  the linkage matrix is identical.
- [x] 1.4 Add a smoke test that each seeded function accepts `random_state=None`
  without raising.
- [x] 1.5 Confirm the suite passes (audit found all functions already deterministic);
  if any function is non-deterministic, fix propagation minimally and note it here.

## 2. Documentation

- [x] 2.1 Add `docs/reproducibility.md`: stochastic-function inventory + default seed,
  the determinism guarantee, the `rtol=1e-6` / exact-labels tolerance policy, the
  cross-platform/BLAS caveat, and the hierarchical-is-deterministic note.

## 3. Validation

- [x] 3.1 `uv run pytest tests/test_reproducibility.py` passes.
- [x] 3.2 `uv run black --check` + `uv run ruff check` clean on changed files.
- [x] 3.3 `openspec validate audit-stochastic-determinism --strict` passes.
