## 1. TDD: Write Unit Tests (Red Phase)

- [x] 1.1 Add test `test_interactive_umap_colored_by_genotype` to `test_step_generate_interactive.py`
- [x] 1.2 Add test `test_interactive_umap_shows_barcode_on_hover` to `test_step_generate_interactive.py`
- [x] 1.3 Add test `test_interactive_umap_matches_pca_style` to `test_step_generate_interactive.py`
- [x] 1.4 Run tests, confirm all FAIL (red phase complete)

## 2. TDD: Write Integration Tests (Red Phase)

- [x] 2.1 Add test `test_umap_metadata_preserved_through_interactive_step` to verify:
  - `umap_results` flows from UMAPAnalysisStep to GenerateInteractiveStep
  - `image_paths` preserved in output metadata
  - `trait_names` preserved in output metadata
- [x] 2.2 Add test `test_interactive_umap_uses_clean_indices_for_alignment` to verify:
  - DataFrame is aligned with UMAP embedding using `clean_indices`
  - Barcode values match between hover data and original samples
- [x] 2.3 Run integration tests, confirm FAIL (red phase complete)

## 3. Implementation (Green Phase)

- [x] 3.1 Update `_create_interactive_umap()` in `generate_interactive.py`:
  - Import `create_interactive_scatter_plot` instead of `create_interactive_umap_with_hover_highlight`
  - Add UMAP coordinates (UMAP1, UMAP2) to DataFrame
  - Use `create_interactive_scatter_plot` with `color_by=genotype_col`
  - Include Barcode and Genotype in `hover_data`
  - Add UMAP parameters to plot title
- [x] 3.2 Keep image hover variant using `create_interactive_umap_with_images`
- [x] 3.3 Ensure metadata preservation with `**prev_result.metadata` pattern
- [x] 3.4 Run all tests, confirm PASS (green phase complete)

## 4. Verification

- [x] 4.1 Run full test suite: `uv run pytest tests/`
- [x] 4.2 Run linting: `uv run black src tests && uv run ruff check src`
- [x] 4.3 Run Viz pipeline and verify interactive UMAP output
- [x] 4.4 Compare UMAP and PCA interactive plots for style consistency
- [x] 4.5 Verify metadata flows through entire pipeline (image_paths, umap_results, trait_names)

## Verification Commands

```bash
# Run new tests only
uv run pytest tests/test_step_generate_interactive.py::TestInteractiveUMAPVisualizationQuality -v

# Run all interactive tests
uv run pytest tests/test_step_generate_interactive.py -v

# Run full suite
uv run pytest tests/

# Lint
uv run black src tests && uv run ruff check src

# Manual test
sleap-roots-analyze viz configs/active/viz/viz_turface_19genotypes.yaml
```
