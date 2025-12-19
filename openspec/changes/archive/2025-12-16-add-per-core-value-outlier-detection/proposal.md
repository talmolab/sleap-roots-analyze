# Per-Core Value Outlier Detection for Root Coring QC

**Status**: COMPLETED (2025-12-16)

## Why

Root coring experiments collect 3 cores per plot to measure root biomass and root counts at multiple depths. Currently, the QC pipeline aggregates these 3 cores using median (or mean) without detecting individual cores that may have measurement errors or sampling issues. This "aggregate first, QC later" approach fails when a single bad core is present:

**Real-world problem (EDPIE dataset, GH_7371 Rep 1):**
- Core 0: 0.7636 g (normal)
- Core 1: 0.7071 g (normal)
- Core 2: 0.3132 g (56% below median - likely measurement error or damaged core)

**Current pipeline behavior:**
- Median of [0.76, 0.71, 0.31] = 0.71
- Value 0.71 is NOT flagged as outlier at trait level (not extreme enough)
- Stays in dataset → inflates residual variance → **heritability drops from >0.50 to 0.27-0.45**
- Biomass traits removed from final data (fail H² ≥ 0.5 threshold)

**Nov 30 notebook behavior (high heritability):**
- Used different aggregation that excluded bad core
- Mean of [0.76, 0.71] = 0.735 (without core 2)
- Value 0.735 flagged as outlier and removed
- Result: **57 samples, H² ≥ 0.50**, biomass traits retained

**Root cause:** With only N=3 cores per plot, median aggregation is NOT robust enough against extreme outliers. A single bad core can shift the aggregated value enough to evade trait-level outlier detection while still inflating variance and destroying heritability.

**Solution:** Detect and remove outlier cores BEFORE aggregation using per-group quality control with conservative thresholds. This is a **measurement error detection** approach (quality control), not statistical hypothesis testing. The method respects the nested experimental structure (cores within plots) and uses simple, transparent criteria suitable for small sample sizes.

## What Changes

- **Update `CoreQCConfig`** dataclass (`pipeline/config/components.py`) with new parameters:
  - `detect_value_outliers: bool = True` - Enable value-based outlier detection
  - `max_deviation_from_median: float = 0.30` - Threshold for percent deviation (30%)
  - `min_cores_after_qc: int = 1` - Safety: Keep at least 1 core per group

- **Enhance `QCCoreLevelStep`** (`pipeline/steps/qc_core_level.py`) to add:
  - `_detect_value_outliers_per_group()` method - Flags cores with anomalous values
  - Per-group (Plot-Rep-Geno-Depth) analysis with N=3 cores
  - Detection method: Percent deviation from median (robust for small samples)
  - Formula: Flag if `|value - median| / median > threshold` (e.g., 0.30 = 30%)
  - Safety logic: Always keep at least `min_cores_after_qc` cores per group

- **Update default config** (`configs/qc_root_core_edpie.yaml`):
  - Change `core_qc.enabled: false` → `true` (enable core QC)
  - Add `detect_value_outliers: false` (opt-in, disabled by default)
  - Add `max_deviation_from_median: 0.30` (conservative threshold when enabled)
  - Update comments to explain rationale, statistical considerations, and tuning guidance

- **Add comprehensive test coverage** (`tests/test_step_qc_core_level.py`):
  - Test GH_7371 real-world case (56% deviation)
  - Test normal variation not flagged (<30%)
  - Test safety: keeps minimum cores
  - Test edge cases: median=0, only 1-2 cores available, all cores flagged
  - Test combined: missing data + value outliers

- **Add documentation**:
  - Update `qc_core_level.py` docstrings with new functionality
  - Update config template with threshold tuning guide
  - Add troubleshooting section to CLAUDE.md

**Breaking changes:**
- None - value outlier detection is **opt-in** (disabled by default)
- **Migration path:** Existing configs continue to work unchanged
- **Opt-in:** Set `detect_value_outliers: true` to enable measurement error detection
- **Backward compatible:** Old behavior (missing data detection only) is preserved

## Impact

- **Affected specs**: New `qc-pipeline-root-core` capability (no existing QC spec)
- **Affected code**:
  - `src/sleap_roots_analyze/pipeline/config/components.py` - Modify `CoreQCConfig`
  - `src/sleap_roots_analyze/pipeline/steps/qc_core_level.py` - Add value outlier detection
  - `configs/qc_root_core_edpie.yaml` - Update default config
  - `configs/qc_cylinder_edpie.yaml` - Update config (if exists)
  - `tests/test_step_qc_core_level.py` - Add comprehensive tests
  - `docs/CLAUDE.md` - Add configuration guidance
- **Dependencies**: Uses existing numpy, pandas (no new dependencies)
- **Documentation**: Update QC pipeline docs and config comments
- **Expected outcome (EDPIE dataset)**:
  - 1-3% more cores flagged at Step 00c
  - GH_7371 Rep 1 core 2 removed before aggregation
  - Aggregated value changes from 0.71 → ~0.74 (closer to true value)
  - Trait-level outlier detection now catches 0.74 → 57 samples retained
  - Heritability increases from 0.27-0.45 → >0.50 (biomass traits retained)
