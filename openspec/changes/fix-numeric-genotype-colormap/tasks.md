# Implementation Tasks

## 1. Tests (write BEFORE implementation — TDD Red phase)

### Unit tests for `create_pca_biplot` in tests/test_pca_biplot_colormap.py

- [ ] 1.1a Test: integer genotype column produces N distinct scatter collections (one per genotype)
      (currently fails — produces single scatter with continuous colormap)
- [ ] 1.1b Test: integer genotype column produces no colorbar (no `plt.colorbar` call)
- [ ] 1.1c Test: integer genotype column legend labels match str(int_id) for each unique value
- [ ] 1.1d Test: string genotype column still produces N distinct scatter collections (regression guard)
- [ ] 1.1e Test: string genotype column legend labels match original string values (regression guard)
- [ ] 1.1f Test: `genotypes_to_color` filter still works when genotype column is integer
- [ ] 1.1g Test: `highlight_genotypes` still works when genotype column is integer

## 2. Implementation

- [ ] 2.1 In `create_pca_biplot` (visualization.py ~line 2073): before the `dtype == "object"`
      check, if `pd.api.types.is_integer_dtype(df_pca[color_by])`, cast the column to string
      so downstream categorical (tab10) logic applies. Float columns unchanged.

## 3. Verify RED → GREEN

- [ ] 3.1 Run `uv run pytest tests/test_pca_biplot_colormap.py -v` — confirm tests 1.1a–1.1c
      fail before implementation (RED)
- [ ] 3.2 Implement task 2.1
- [ ] 3.3 Run `uv run pytest tests/test_pca_biplot_colormap.py -v` — confirm all tests pass (GREEN)
- [ ] 3.4 Run full test suite: `uv run pytest tests/` — no regressions

## 4. Lint

- [ ] 4.1 `uv run black --check src/sleap_roots_analyze tests && uv run ruff check src/sleap_roots_analyze`
- [ ] 4.2 Fix any formatting or lint errors
