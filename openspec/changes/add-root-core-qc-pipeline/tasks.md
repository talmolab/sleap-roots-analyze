# Implementation Tasks

## Phase 0: Setup and Configuration

- [ ] 0.1 Create new pipeline steps directory structure
- [ ] 0.2 Define `RootCoreConfig` dataclass in `pipeline/config/components.py`
- [ ] 0.3 Extend `QCPipelineConfig` to include `root_core: Optional[RootCoreConfig]`
- [ ] 0.4 Create example YAML configs for biomass and counting
- [ ] 0.5 Add validation logic for root core configuration

## Phase 1: Core Data Loading and Transformation (Steps 0a-0b)

### Step 0a: Load Raw Root Core Data

- [ ] 1.1 Create `pipeline/steps/load_root_core_data.py`
- [ ] 1.2 Implement `LoadRootCoreDataStep` class
- [ ] 1.3 Write tests for loading biomass CSV
- [ ] 1.4 Write tests for loading counting CSV
- [ ] 1.5 Add validation for required columns
- [ ] 1.6 Implement sample identifier creation using `create_sample_identifier()`
- [ ] 1.7 Add unique identifier validation using `validate_unique_identifiers()`
- [ ] 1.8 Handle missing core_n column (generate if needed)

### Step 0b: Transform to Long Format

- [ ] 2.1 Create `pipeline/steps/transform_depth_data.py`
- [ ] 2.2 Implement `TransformDepthDataStep` class
- [ ] 2.3 Write tests for biomass depth transformation (manual mapping)
- [ ] 2.4 Write tests for counting depth transformation (auto-parsing)
- [ ] 2.5 Add custom depth mapping configuration option
- [ ] 2.6 Validate depth calculations match notebook results
- [ ] 2.7 Add metadata tracking (transformation method, depth mappings)

## Phase 2: Core-Level QC (Step 0c)

- [ ] 3.1 Create `pipeline/steps/qc_core_level.py`
- [ ] 3.2 Implement `QCCoreLevelStep` class
- [ ] 3.3 Write tests for within-replicate outlier detection
- [ ] 3.4 Implement Mahalanobis distance calculation per Plot-Rep group
- [ ] 3.5 Add missing data flagging (>X% missing depths)
- [ ] 3.6 Create core-level spaghetti plot visualization
- [ ] 3.7 Color cores by outlier status in plots
- [ ] 3.8 Add configurable removal threshold
- [ ] 3.9 Generate core QC report HTML
- [ ] 3.10 Track removed cores in metadata

## Phase 3: Aggregation (Step 0d)

- [ ] 4.1 Create `pipeline/steps/aggregate_cores.py`
- [ ] 4.2 Implement `AggregateCoresStep` class
- [ ] 4.3 Write tests for mean aggregation
- [ ] 4.4 Write tests for median aggregation
- [ ] 4.5 Write tests for NaN handling strategies
- [ ] 4.6 Add support for custom aggregation functions
- [ ] 4.7 Validate aggregated values match collaborator's data
- [ ] 4.8 Add group-level sample size tracking (N cores aggregated)

## Phase 4: Pre-QC Visualization (Step 0e)

- [ ] 5.1 Create `pipeline/steps/visualize_depth_profiles_pre_qc.py`
- [ ] 5.2 Implement `VisualizeDepthProfilesPreQCStep` class
- [ ] 5.3 Write tests for biomass bar plot generation
- [ ] 5.4 Write tests for counting line plot generation
- [ ] 5.5 Add faceted plots by genotype
- [ ] 5.6 Add replicate spaghetti plots
- [ ] 5.7 Save plots to configured output directory
- [ ] 5.8 Add plot configuration options (colors, size, layout)

## Phase 5: Reshape for Trait QC (Step 0f)

- [ ] 6.1 Create `pipeline/steps/reshape_for_trait_qc.py`
- [ ] 6.2 Implement `ReshapeForTraitQCStep` class
- [ ] 6.3 Write tests for long → wide pivot
- [ ] 6.4 Implement column prefixing (`RootDW_`, `RootCount_`)
- [ ] 6.5 Write tests verifying no column name conflicts
- [ ] 6.6 Add metadata column preservation logic
- [ ] 6.7 Validate output format matches QC pipeline expectations

## Phase 6: Pipeline Integration

- [ ] 7.1 Modify `QCPipeline.__init__()` to instantiate root core steps conditionally
- [ ] 7.2 Update `QCPipeline.create_tasks()` to add Steps 0a-0f before Step 1
- [ ] 7.3 Add conditional logic: only run root core steps if `config.root_core` is set
- [ ] 7.4 Update task dependencies correctly (Step 1 depends on Step 0f)
- [ ] 7.5 Write integration tests with full pipeline
- [ ] 7.6 Test pipeline with biomass data only
- [ ] 7.7 Test pipeline with counting data only
- [ ] 7.8 Test pipeline with both biomass and counting data

## Phase 7: Above-Ground Trait Integration (Steps 11-12)

### Step 11: Load Above-Ground Traits

- [ ] 8.1 Create `pipeline/steps/load_above_ground_traits.py`
- [ ] 8.2 Implement `LoadAboveGroundTraitsStep` class
- [ ] 8.3 Write tests for loading above-ground CSV
- [ ] 8.4 Add Plot-Rep combination validation
- [ ] 8.5 Check for column name conflicts with root data
- [ ] 8.6 Add warning/error for duplicate columns

### Step 12: Merge All Traits

- [ ] 9.1 Create `pipeline/steps/merge_all_traits.py`
- [ ] 9.2 Implement `MergeAllTraitsStep` class
- [ ] 9.3 Write tests for simple merge (no duplicates)
- [ ] 9.4 Write tests for duplicate column detection
- [ ] 9.5 Write tests for mismatched Plot-Rep combinations
- [ ] 9.6 Add configurable join keys
- [ ] 9.7 Add configurable duplicate handling (fail/skip/rename)
- [ ] 9.8 Preserve all metadata columns from all sources
- [ ] 9.9 Add sample size validation (same N before/after merge)
- [ ] 9.10 Save merged CSV with descriptive filename

## Phase 8: Metadata Generation (Step 13)

- [ ] 10.1 Create `pipeline/steps/generate_root_core_metadata.py`
- [ ] 10.2 Implement `GenerateRootCoreMetadataStep` class
- [ ] 10.3 Write tests for metadata JSON structure
- [ ] 10.4 Add data source tracking (file paths, timestamps, checksums)
- [ ] 10.5 Add processing parameter logging (agg method, QC thresholds)
- [ ] 10.6 Add QC decision logging (outliers removed with sample IDs)
- [ ] 10.7 Add trait filtering logging (columns removed with reasons)
- [ ] 10.8 Add final sample size summary (N samples, N genotypes, N traits)
- [ ] 10.9 Add software version tracking (pandas, numpy, sleap-roots-analyze)
- [ ] 10.10 Save metadata JSON alongside final CSV

## Phase 9: Post-QC Visualization (Step 14)

- [ ] 11.1 Create `pipeline/steps/visualize_depth_profiles_post_qc.py`
- [ ] 11.2 Implement `VisualizeDepthProfilesPostQCStep` class
- [ ] 11.3 Regenerate depth profile plots with cleaned data
- [ ] 11.4 Add pre/post comparison plots (side-by-side)
- [ ] 11.5 Add cross-correlation heatmap (root vs above-ground)
- [ ] 11.6 Add heritability by depth bar plots
- [ ] 11.7 Add genotype ranking plots (top/bottom by total root biomass)
- [ ] 11.8 Add PCA with all traits (root + above-ground)
- [ ] 11.9 Generate comprehensive HTML visualization report

## Phase 10: Configuration and Documentation

- [ ] 12.1 Create example config: `configs/root_core_biomass_qc.yaml`
- [ ] 12.2 Create example config: `configs/root_core_counting_qc.yaml`
- [ ] 12.3 Create example config: `configs/root_core_merged_qc.yaml`
- [ ] 12.4 Update `pipeline/config/README.md` with root core options
- [ ] 12.5 Write user guide: `docs/root_core_qc_pipeline.md`
- [ ] 12.6 Add code examples to guide
- [ ] 12.7 Document column naming conventions
- [ ] 12.8 Document QC decision points
- [ ] 12.9 Add troubleshooting section
- [ ] 12.10 Update main README with root core pipeline section

## Phase 11: Testing and Validation

### Unit Tests

- [ ] 13.1 Write tests for all new Step classes
- [ ] 13.2 Write tests for RootCoreConfig validation
- [ ] 13.3 Write tests for duplicate column detection
- [ ] 13.4 Write tests for metadata JSON schema
- [ ] 13.5 Achieve >95% coverage for new modules

### Integration Tests

- [ ] 14.1 Create test dataset: `tests/data/root_core_test_biomass.csv`
- [ ] 14.2 Create test dataset: `tests/data/root_core_test_counting.csv`
- [ ] 14.3 Create test dataset: `tests/data/above_ground_test.csv`
- [ ] 14.4 Write integration test: biomass → QC → merge
- [ ] 14.5 Write integration test: counting → QC → merge
- [ ] 14.6 Write integration test: both → QC → merge
- [ ] 14.7 Write integration test: verify no duplicate columns

### Validation with Real Data

- [ ] 15.1 Run pipeline on EDPIE biomass data
- [ ] 15.2 Run pipeline on EDPIE counting data
- [ ] 15.3 Compare aggregated values to collaborator's `Field_2024_clean.csv`
- [ ] 15.4 Validate merged CSV has expected columns
- [ ] 15.5 Validate no duplicate trait columns
- [ ] 15.6 Review QC decisions (outliers, filtered traits)
- [ ] 15.7 Generate final visualization report
- [ ] 15.8 Manual inspection of depth profile plots

## Phase 12: Quality Assurance

- [ ] 16.1 Run `uv run pytest` - all tests pass
- [ ] 16.2 Run `uv run pytest --cov --cov-branch` - verify >95% coverage
- [ ] 16.3 Run `uv run black src/sleap_roots_analyze tests`
- [ ] 16.4 Run `uv run ruff check src/sleap_roots_analyze tests`
- [ ] 16.5 Verify all docstrings follow Google format
- [ ] 16.6 Run `openspec validate add-root-core-qc-pipeline --strict`
- [ ] 16.7 Manual code review of all new modules
- [ ] 16.8 Check for security issues (path injection, etc.)

## Phase 13: Final Deliverables

- [ ] 17.1 Create example script: `examples/run_root_core_qc.py`
- [ ] 17.2 Create example notebook: `examples/root_core_qc_walkthrough.ipynb`
- [ ] 17.3 Generate API documentation for new steps
- [ ] 17.4 Update CHANGELOG.md
- [ ] 17.5 Create PR with comprehensive description
- [ ] 17.6 Add screenshots of visualizations to PR
- [ ] 17.7 Request review from team

---

**Total Tasks**: 143
**Estimated Time**: 4-6 days
**Priority**: High (blocking EDPIE paper data finalization)