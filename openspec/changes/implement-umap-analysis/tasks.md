## 1. TDD: UMAPAnalysisStep Unit Tests (Red Phase)

Write all tests BEFORE implementation. Tests MUST fail initially.

- [x] 1.1 Create `tests/test_step_umap_analysis.py` with test class structure
- [x] 1.2 Test: `test_step_initialization` - Verify step name and description
- [x] 1.3 Test: `test_basic_execution_when_enabled` - Step runs without error
- [x] 1.4 Test: `test_skip_when_disabled` - Returns early with status="disabled"
- [x] 1.5 Test: `test_skip_when_umap_not_installed` - Graceful handling if umap-learn missing
- [x] 1.6 Test: `test_metadata_preservation` - All previous metadata preserved (`**prev_result.metadata`)
- [x] 1.7 Test: `test_image_paths_preserved` - Critical: `image_paths` must flow through
- [x] 1.8 Test: `test_umap_results_in_metadata` - `umap_results` dict available in output
- [x] 1.9 Test: `test_embedding_shape` - Embedding has shape (n_samples, 2)
- [x] 1.10 Test: `test_reproducibility` - Same random_state produces identical results
- [x] 1.11 Test: `test_artifacts_saved` - Files created in `data/umap/` directory
- [x] 1.12 Run tests, confirm all FAIL (red phase complete)

## 2. Implement UMAPAnalysisStep (Green Phase)

- [x] 2.1 Replace stub in `umap_analysis.py` with full implementation
- [x] 2.2 Get trait columns from metadata (handle multiple key names)
- [x] 2.3 Clean data by dropping NaN rows
- [x] 2.4 Call `perform_umap_analysis()` with config parameters
- [x] 2.5 Handle UMAP not installed gracefully
- [x] 2.6 Create `data/umap/` directory and save artifacts:
  - `umap_embedding.csv` - Embedding coordinates with sample indices
  - `umap_parameters.json` - n_neighbors, min_dist, random_state
- [x] 2.7 Build metadata dict with `**prev_result.metadata` (critical for image_paths)
- [x] 2.8 Return StepResult with data and metadata
- [x] 2.9 Run tests, confirm all PASS (green phase complete)

## 3. TDD: Metadata Flow Integration Tests

- [x] 3.1 Create `tests/test_umap_metadata_flow.py`
- [x] 3.2 Test: `test_image_paths_flows_through_umap_step` - image_paths preserved across stats → pca → umap
- [x] 3.3 Test: `test_umap_results_available_in_interactive_step` - GenerateInteractiveStep receives umap_results
- [x] 3.4 Test: `test_full_viz_pipeline_with_umap_enabled` - End-to-end integration
- [x] 3.5 Run integration tests, confirm PASS

## 4. Enable Static UMAP Visualizations

- [x] 4.1 Test: `test_static_umap_plot_generated_when_enabled` - Add to test_step_generate_static_figures.py
- [x] 4.2 Add UMAP plot generation to `_create_umap_plots()` in generate_static_figures.py:
  - Basic UMAP scatter colored by genotype
  - `create_umap_colored_by_top_traits()` if PCA results available
- [x] 4.3 Wire `_create_umap_plots()` into execute() method, guarded by `config.static_viz.create_umap_plots`
- [x] 4.4 Run tests, confirm PASS

## 5. Enable Interactive UMAP Visualizations

- [x] 5.1 Test: `test_interactive_umap_generated_when_enabled` - Add to test_step_generate_interactive.py
- [x] 5.2 Verify existing `_create_interactive_umap()` in generate_interactive.py receives umap_results
- [x] 5.3 Test: `test_interactive_umap_with_images_when_image_paths_available`
- [x] 5.4 Run tests, confirm PASS

## 6. Documentation and Validation

- [x] 6.1 Update config examples: Set `umap.enabled: true` in a test config
- [x] 6.2 Run full test suite: `uv run pytest tests/`
- [x] 6.3 Run lint: `uv run black src tests && uv run ruff check src`
- [x] 6.4 Manual validation: Run Viz pipeline with UMAP enabled, verify outputs
- [x] 6.5 Compare UMAP output with notebook output for feature parity

## 7. Code Review Checklist

- [x] 7.1 All tests pass
- [x] 7.2 No regressions in existing functionality
- [x] 7.3 Metadata propagation verified (image_paths flows through)
- [x] 7.4 UMAP artifacts saved to correct directory
- [x] 7.5 Code follows existing patterns (PCAAnalysisStep as reference)
- [x] 7.6 Docstrings complete for new/modified functions
- [x] 7.7 No hardcoded values (all from config)

## Verification Commands

```bash
# Run UMAP step tests only
uv run pytest tests/test_step_umap_analysis.py -v

# Run metadata flow tests
uv run pytest tests/test_umap_metadata_flow.py -v

# Run all tests with coverage
uv run pytest --cov=src/sleap_roots_analyze --cov-report=term-missing tests/

# Lint
uv run black src tests && uv run ruff check src

# Manual test: Run Viz pipeline with UMAP
sleap-roots-analyze viz configs/active/viz/viz_turface_19genotypes.yaml
```
