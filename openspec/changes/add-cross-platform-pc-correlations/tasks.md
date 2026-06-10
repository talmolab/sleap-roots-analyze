## 1. Result model

- [x] 1.1 Add `src/sleap_roots_analyze/cross_platform_pc.py` with `from __future__ import annotations`
- [x] 1.2 Define `CrossPlatformPCResult` (frozen `@dataclass`): per-platform PCA results, genotype-
      mean PC-score matrices, tidy per-test correlation table, pooled FDR p-values, CIs, power,
      significant subset, and a `summary` dict (n_tests, n_genotypes, n_fdr_significant, …)

## 2. Per-platform PCA → genotype-mean PC scores

- [x] 2.1 Test (red): genotype-mean PC score == mean of that genotype's sample-level PC scores
      (`test_genotype_pc_means_are_aggregated_after_pca`)
- [x] 2.2 Implement `_platform_pc_means` using `perform_pca_analysis` on sample rows (NaN-row-drop
      kept aligned to genotypes), then `calculate_genotype_means` on the PC scores

## 3. Cross-platform correlations + pooled FDR

- [x] 3.1 Test (red): 3 platforms with components 3/4/5 → 47 pooled tests; global genotype alignment
- [x] 3.2 Implement pairwise PC×PC correlation over the shared genotype panel (Spearman default)
- [x] 3.3 Pool all tests and apply `multipletests(method=correction_method, alpha=alpha)` once;
      attach corrected p-values + significance flags (`test_fdr_is_pooled_across_all_tests`)

## 4. CIs + power

- [x] 4.1 Test (red): every test row carries CI bounds + achieved power
- [x] 4.2 Compute Fisher-z CI (`calculate_correlation_ci`) and `achieved_power` per test from the
      shared-panel genotype count

## 5. Public function + API export

- [x] 5.1 Test (red): import both names from `sleap_roots_analyze` and assert they are in `__all__`
- [x] 5.2 Implement `cross_platform_pc_correlations(...)`; full type hints + Google docstring
- [x] 5.3 Import + add both names to `src/sleap_roots_analyze/__init__.py` `__all__`

## 6. Tests

- [x] 6.1 `tests/test_cross_platform_pc.py`: synthetic 3-platform unit tests (47-count, CI/power,
      pooled FDR, ordering pin, pearson signal), plus edge cases (disjoint genotypes don't raise;
      <2 platforms raises)
- [x] 6.2 Skip-guarded wheat-EDPIE regression test asserting 47 tests / 19 genotypes / 0 FDR;
      `skipif` on fixture absence (issue #120). **Verified locally against the real post-QC data:
      47 / 19 / 0 reproduced (min q ≈ 0.73).**

## 7. Verify

- [x] 7.1 `uv run black --check` (src + new test) && `uv run ruff check src` clean
- [x] 7.2 `uv run pytest tests/test_cross_platform_pc.py` → 9 passed, 1 skipped (golden)
- [ ] 7.3 `openspec validate add-cross-platform-pc-correlations --strict`
- [ ] 7.4 Full suite green (run at pre-merge)
