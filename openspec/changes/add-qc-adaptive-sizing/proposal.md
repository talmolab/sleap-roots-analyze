# Add Adaptive Figure Sizing to QC Pipeline

**Status**: 📋 PROPOSED  
**Created**: 2025-12-01  
**Type**: Feature Enhancement

## Why

The QC pipeline uses fixed figure sizes, creating legibility problems for datasets with varying numbers of traits/genotypes:
- Correlation heatmaps with 50+ traits → tiny, unreadable labels
- Heritability bar plots with 30+ traits → overlapping text  
- Batched visualizations use fixed `(16, 16)` regardless of trait count

The Viz pipeline already has adaptive sizing (`viz_utils.py`, `AdaptiveSizingConfig`), but QC pipeline doesn't use it.

## What Changes

### 1. Add `adaptive_sizing` to `QCPipelineConfig`

**File**: `src/sleap_roots_analyze/pipeline/config/qc_config.py`

```python
adaptive_sizing: AdaptiveSizingConfig = field(default_factory=AdaptiveSizingConfig)
```

### 2. Update Steps with Adaptive Sizing

**Step 4** (exploratory_analysis.py): Use `calculate_correlation_matrix_size()` for correlation heatmap  
**Step 8** (statistical_analysis.py): Use `calculate_barplot_size()` for heritability plot  
**Step 9** (filter_heritability.py): Use adaptive sizing for variance decomposition & threshold plots

### 3. Update Example Config

Add optional `adaptive_sizing` section (disabled by default for backwards compatibility)

## Impact

**Files modified**: 4-5 pipeline files  
**Backwards compatible**: Yes (disabled by default)  
**Test coverage**: Update 3 test files  

**Benefits**:
- Figures scale appropriately (10 traits vs 100 traits)
- Consistent with Viz pipeline
- User controllable via config
- Professional output for any dataset size

**Risks**: Low (infrastructure exists, disabled by default)

## Implementation Checklist

- [ ] Add `adaptive_sizing` field to `QCPipelineConfig`
- [ ] Update Step 4 with adaptive sizing for correlation heatmap
- [ ] Update Step 8 with adaptive sizing for heritability plot
- [ ] Update Step 9 with adaptive sizing for diagnostic plots
- [ ] Update example config
- [ ] Add tests for adaptive sizing enabled/disabled
- [ ] Create capability spec
- [ ] Run full test suite

## Timeline

**Estimated**: 4-5 hours total

## Related

- Viz Pipeline already has full adaptive sizing
- Issue #19: QC pipeline test coverage (completed)
- Font config enhancement (similar pattern)
