## 1. Result model

- [ ] 1.1 Add `src/sleap_roots_analyze/cross_platform_pc.py` with `from __future__ import annotations`
- [ ] 1.2 Define `CrossPlatformPCResult` (frozen `@dataclass`): per-platform PCA results, genotype-
      aligned PC-score matrices (name → DataFrame), per-pair correlation tables, pooled FDR p-values,
      CIs, power, significant-correlations list, and a small `summary` dict (n_tests, n_genotypes,
      n_fdr_significant). Write the dataclass test first.

## 2. Per-platform PCA → genotype-mean PC scores

- [ ] 2.1 Test (red): for one platform with multiple samples/genotype, the genotype-mean PC score
      equals the mean of that genotype's sample-level PC scores (pins sample-PCA-then-aggregate)
- [ ] 2.2 Implement `_platform_pc_scores(df, trait_cols, n_components, genotype_col, random_state)`
      using `perform_pca_analysis` on sample rows, then aggregate `transformed_data` by genotype

## 3. Cross-platform correlations + pooled FDR

- [ ] 3.1 Test (red): 3 platforms with components 3/4/5 → 47 pooled tests; alignment on common genotypes
- [ ] 3.2 Implement pairwise PC×PC correlation over common genotypes (one table per unordered pair)
- [ ] 3.3 Pool all pair tests and apply `multipletests(method=correction_method, alpha=alpha)` once;
      attach corrected p-values + significance flags

## 4. CIs + power

- [ ] 4.1 Test (red): every test row carries CI bounds + achieved power
- [ ] 4.2 Compute Fisher-z CI (`calculate_correlation_ci`) and `achieved_power` per test from its
      common-genotype count

## 5. Public function + API export

- [ ] 5.1 Test (red): `from sleap_roots_analyze import cross_platform_pc_correlations, CrossPlatformPCResult`
      and both names are in `__all__`
- [ ] 5.2 Implement `cross_platform_pc_correlations(platforms, trait_cols, n_components, *, genotype_col,
      alpha, correction_method, random_state)` orchestrating steps 2–4; full type hints + docstring
- [ ] 5.3 Import + add both names to `src/sleap_roots_analyze/__init__.py` `__all__`

## 6. Tests

- [ ] 6.1 `tests/test_cross_platform_pc.py`: synthetic 3-platform unit test (shape, 47-count for
      3/4/5, CI/power present, pooled FDR), plus edge cases (disjoint genotypes don't raise)
- [ ] 6.2 Skip-guarded wheat-EDPIE regression test: assert 47 tests / 19 genotypes / 0 FDR-significant
      when the post-QC fixture exists; `pytest.mark.skipif` on fixture absence (issue #120)

## 7. Verify

- [ ] 7.1 `uv run black --check src tests` && `uv run ruff check src tests`
- [ ] 7.2 `uv run pytest tests/test_cross_platform_pc.py -v`
- [ ] 7.3 `openspec validate add-cross-platform-pc-correlations --strict`
- [ ] 7.4 Full suite green
