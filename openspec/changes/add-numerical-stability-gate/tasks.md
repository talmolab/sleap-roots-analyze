## 1. Measure the reference dataset (grounds every threshold)

- [x] 1.1 Measured same-stack spread on `turface_19` under the canonical stack
      (py3.11): UMAP Procrustes-aligned max|Δ| ≈ 2.8e-17 (bit-identical), KMeans
      ARI = 1.0 (sizes `[32, 86, 35]`, `n_clusters=3`), trait summary max rel Δ = 0.0.
- [x] 1.2 Derived thresholds from measurement and recorded them as commented constants in
      `tests/numerical_stability_recompute.py`: `ATOL_PROCRUSTES=1e-6` (a real 1e-3 nudge
      → aligned Δ ≈ 2.6e-5, so 1e-6 is sensitive yet noise-immune), `ARI_FLOOR=0.95`,
      `TRAIT_RTOL=1e-10`. Confirmed Procrustes on the **aligned matrices** flags the nudge.

## 2. Test skeleton + negative controls (red before golden exists)

- [x] 2.1 `tests/test_numerical_stability.py` carries
      `pytestmark = skipif(sys.platform != CANONICAL_PLATFORM)` (macOS — the OS the golden
      is generated on); shared `SEED`/`N_CLUSTERS`/`TRAIT_COLS` live in
      `tests/numerical_stability_recompute.py` (never imported from `scripts/`).
- [x] 2.2 Test loads goldens via a session fixture and recomputes; a missing input fails
      loudly with a regen pointer (`test_missing_input_fails_loudly`).
- [x] 2.3 Added `numerical_stability_golden` session loader to `tests/fixtures.py`; input
      is the committed `turface_19_final_data.csv` the regen script also consumes.

## 3. Regeneration script + golden artifacts (make it green)

- [x] 3.1 `scripts/regenerate_numerical_stability_golden.py` recomputes the three goldens
      from the committed input. Decision: a **distinct** golden is justified — the existing
      `expected/viz/turface_19/viz_umap_embedding.csv` comes from the full viz pipeline on
      heritability-filtered traits, whereas the gate uses an explicit 12-trait list on the
      post-QC input (self-contained, no dependence on the heritability step).
- [x] 3.2 Script emits `golden_provenance.json` via `importlib.metadata` (OS, machine,
      python, numpy/pandas/umap-learn/numba/scipy/scikit-learn versions, seed, n_clusters,
      tolerances).
- [x] 3.3 Script asserts `result["n_clusters"] == N_CLUSTERS` after KMeans; trait summary
      is computed on **raw** values (UMAP/KMeans standardize internally).
- [x] 3.4 Generated + committed the goldens + provenance (Darwin / py3.11.15). Test passes.

## 4. Assertions + remaining negative controls

- [x] 4.1 UMAP: `procrustes` then `np.allclose(aligned, atol)`; message names "UMAP
      embedding". Sub-tests prove a rigid transform passes and a structural nudge fails.
- [x] 4.2 Clusters: `adjusted_rand_score > ARI_FLOOR`; `n_clusters == N_CLUSTERS` asserted;
      message names "cluster labels".
- [x] 4.3 Traits: `assert_frame_equal(rtol=1e-10)`; message names "trait summary".
- [x] 4.4 Negative control: mutating each golden past tolerance fails its assertion.
- [x] 4.5 Anti-tautology: a different-seed recompute fails the gate (proves recompute is
      live, not golden-vs-golden).
- [x] 4.6 Edge cases: missing input fails loudly; a new NaN in a recomputed output fails.

## 5. CI integration

- [x] 5.1 Added a dedicated single-OS `numerical-stability` job (macos-latest) to
      `.github/workflows/ci.yml`, running the test by path, with an honest comment
      (drift detector, not determinism; single-OS rationale).
- [x] 5.2 The `skipif` keeps the cross-platform `tests` matrix green (ubuntu/windows skip;
      macOS runs). Shared constant stays out of `scripts/`.
- [x] 5.3 Green locally: `uv run --python 3.11 pytest tests/test_numerical_stability.py`
      (11 passed).

## 6. Documentation

- [x] 6.1 `tests/fixtures/README.md`: added the `numerical_stability/` node to the Layout
      tree (single-dataset, not per-platform) and extended the Regenerate policy with the
      script-driven (non-harness) path + dependency-bump trigger + provenance.
- [x] 6.2 `docs/reproducibility.md`: added the "Numerical-stability golden gate" section,
      reconciled `rtol=1e-10` vs the standing `rtol=1e-6` policy, cross-referenced the
      ARI/permutation rule + BLAS caveat, and added the gate to the CI-enforcement list +
      local-run command.
- [x] 6.3 `docs/CHANGELOG.md`: added an `[Unreleased] / ### Added` entry.

## 7. Validation

- [x] 7.1 `openspec validate add-numerical-stability-gate --strict` passes.
- [x] 7.2 Local run green: gate (11 passed) + `black --check` + `ruff check` on new files.
