# Design: Notebook-Config Reproducibility Audit

## Context

For scientific publication of root trait analysis results, we need absolute certainty that pipeline configurations exactly replicate the notebook analyses that generated the results. We have 4 main datasets, each with QC and visualization notebooks, plus cross-platform comparison notebooks.

**Stakeholders:**
- Research team publishing results
- Journal reviewers verifying reproducibility
- Future researchers replicating analyses
- Collaborators using the pipelines

**Constraints:**
- Must be done before manuscript submission
- Cannot change scientific conclusions (only fix configs to match notebooks)
- Some notebooks are very large (>1000 cells)
- Some parameters may be implicit (using function defaults)

**Current State:**
- Notebooks contain the ground truth analysis parameters
- Configs were created to match notebooks but may have drifted
- REPLICATION_GUIDE.md shows one example where config mismatch affected results
- Some configs have excellent documentation, others are minimal

## Goals / Non-Goals

### Goals
1. **Identify all parameter inconsistencies** between notebooks and configs
2. **Fix critical mismatches** that would prevent exact replication
3. **Document intentional variations** with scientific rationale
4. **Create parameter reference tables** for Methods section and supplementary materials
5. **Establish audit process** for future datasets

### Non-Goals
1. **NOT** changing analysis parameters (configs match notebooks, not vice versa)
2. **NOT** re-running all analyses (verify configs produce same outputs)
3. **NOT** building automated validation infrastructure (nice-to-have, but not required for publication)
4. **NOT** unifying parameters across datasets (intentional variation is acceptable with documentation)

## Decisions

### Decision 1: Notebook as Ground Truth
**Choice:** Notebooks are the authoritative source; configs must match notebooks

**Rationale:**
- Notebooks contain the actual analysis that produced published results
- Changing notebooks would invalidate existing results
- Configs are meant to replicate notebook analyses via pipelines

**Alternatives considered:**
- Make configs authoritative and update notebooks → Would require re-analysis, delays publication
- Allow divergence with documentation → Fails reproducibility requirements

### Decision 2: Systematic Parameter Extraction
**Choice:** Extract ALL parameters from notebooks, not just suspected mismatches

**Rationale:**
- Ensures no parameters are missed
- Provides comprehensive documentation for Methods section
- Enables future validation automation
- Builds institutional knowledge

**Alternatives considered:**
- Spot-check known problem areas → Risk missing subtle inconsistencies
- Manual review without extraction → Not systematic, hard to verify completeness

### Decision 3: OpenSpec Organization
**Choice:** Use OpenSpec change proposal to organize audit findings and fixes

**Rationale:**
- Prevents stray markdown files that get lost or outdated
- Clear tracking of what needs to be done (tasks.md)
- Spec defines requirements for future configs (spec.md)
- Can be archived after publication for historical record

**Alternatives considered:**
- Ad-hoc documentation in project root → Gets messy, hard to maintain
- Direct config updates without design doc → Loses rationale and context

### Decision 4: Parameter Reference Table Location
**Choice:** Store in OpenSpec specs, not in each config file

**Rationale:**
- Centralized location easier to reference in Methods section
- Can compare across datasets more easily
- Config files stay clean and focused
- Can generate supplementary tables from spec documentation

**Alternatives considered:**
- Embed in each config header → Duplicates information, hard to maintain
- Separate docs/ file → Would become another stray markdown file

### Decision 5: Intentional Variation Documentation
**Choice:** Document dataset-specific rationale in spec, not in tasks

**Rationale:**
- Spec is long-lived documentation
- Tasks are ephemeral (archived after completion)
- Spec can be referenced in paper Methods section
- Provides scientific context, not just technical checklist

## Architecture

### Audit Process Flow

```
1. Notebook Parameter Extraction
   ├─ Read latest QC notebook for dataset
   ├─ Read latest viz notebook for dataset
   ├─ Read cross-platform notebooks
   ├─ Extract all parameters to structured format
   └─ Document cell numbers for traceability

2. Config Comparison
   ├─ Load corresponding config file
   ├─ Compare each parameter category
   ├─ Flag mismatches (notebook ≠ config)
   └─ Categorize: Critical | Minor | Intentional

3. Reference Table Creation
   ├─ Build table: Dataset | Parameter | Notebook | Config | Status
   ├─ Add notebook cell references
   ├─ Add rationale for intentional variations
   └─ Format for spec documentation

4. Config Fixes
   ├─ Update configs to match notebooks
   ├─ Add source notebook reference to headers
   ├─ Document verification date
   └─ Test pipeline produces expected outputs

5. Validation
   ├─ Run QC pipeline → Compare sample/trait counts
   ├─ Run viz pipeline → Compare PCA structure
   ├─ Run cross-platform → Compare correlations
   └─ Document any acceptable differences
```

### Parameter Categories

Organized by pipeline stage and importance:

**Critical Parameters (Must Match Exactly):**
- Cleanup thresholds (directly affect which samples/traits are analyzed)
- Outlier detection methods and thresholds (affect sample exclusion)
- Heritability threshold (affects trait retention)
- Column mappings (must match source data structure)

**Important Parameters (Should Match):**
- PCA variance threshold (affects dimensionality)
- Feature selection strategy (affects which traits are highlighted)
- Genotype highlighting lists (affects published figures)

**Contextual Parameters (Document if Different):**
- Data file paths (expected to differ between notebook runs and pipeline runs)
- Output directories (expected to differ)
- Figure DPI (notebook vs publication quality)

### Findings Organization

```
openspec/changes/audit-notebook-config-reproducibility/
├── proposal.md          # Why, what, impact
├── tasks.md             # Systematic checklist
├── design.md            # This file - decisions and architecture
└── specs/
    └── config-management/
        └── spec.md      # Requirements and parameter reference tables
```

## Known Issues from Initial Investigation

### Critical Mismatches Found

1. **Cylinder Viz Config - PCA Variance**
   - Notebook: `PCA_EXPLAINED_VARIANCE_THRESHOLD = 0.75`
   - Config: `pca.n_components: 0.95`
   - **Impact:** Config would use MORE principal components than notebook
   - **Fix:** Change config to 0.75

2. **Cylinder Viz Config - Missing Genotype Highlighting**
   - Notebook: `GENOTYPES_TO_COLOR = ["GH_7293", "GH_7378", "GH_7327"]`
   - Config: Parameter missing
   - **Impact:** Published figures can't be reproduced exactly
   - **Fix:** Add genotypes_to_color and highlight_genotypes to config

### Intentional Variations (Document, Don't Fix)

1. **Heritability Thresholds Vary by Dataset**
   - Turface 150: H² ≥ 0.40
   - Turface 19: H² ≥ 0.60
   - Cylinder: H² ≥ 0.60
   - Root coring: H² ≥ 0.50
   - **Rationale:** Dataset-specific based on trait count and data quality

2. **PCA Variance Thresholds Vary**
   - Turface 150 viz: 0.80
   - Cylinder viz: 0.75
   - Root coring viz: 0.75
   - **Rationale:** Visualization clarity vs. information retention trade-off

### Acceptable Differences (Document)

1. **Data Paths**
   - Notebooks: Point to specific run directories with timestamps
   - Configs: Point to generic locations or latest pipeline outputs
   - **Status:** Expected and acceptable

2. **Cross-Platform Top N Settings**
   - Notebook: top_n_correlations = 20
   - Config: top_n_correlations = 30
   - **Status:** Config provides MORE output, which is fine

## Risks / Trade-offs

### Risk 1: Notebook Parameter Location
**Risk:** Parameters scattered across multiple cells, some may be missed

**Mitigation:**
- Systematic cell-by-cell review
- Use grep/search for common parameter patterns (MAX_, THRESHOLD, _COL)
- Cross-reference with config structure to ensure all categories covered

### Risk 2: Implicit Parameters
**Risk:** Notebooks may use function defaults not explicitly visible

**Mitigation:**
- Check function signatures in source code
- Document when notebook uses defaults
- Add those defaults to config explicitly

### Risk 3: Notebook Evolution
**Risk:** Notebooks may have been re-run with different parameters after initial analysis

**Mitigation:**
- Use LATEST notebook for each dataset
- Check notebook cell execution order and timestamps
- Verify results match published/reported values

### Risk 4: Time Investment
**Risk:** Comprehensive audit is time-consuming

**Mitigation:**
- Focus on critical parameters first (cleanup, outlier, heritability)
- Parallel extraction for multiple datasets
- Use agent to help with systematic parameter extraction

### Trade-off: Completeness vs. Speed
**Choice:** Comprehensive audit even if slower

**Rationale:**
- Publication stakes are high (retractions are career-damaging)
- Audit provides long-term value for future datasets
- One-time cost with lasting benefit

## Migration Plan

Not applicable - this is config verification/correction, not a system migration.

## Validation Approach

### Verification Tests

For each dataset config that was modified:

1. **Sample Count Test**
   ```bash
   # Run QC pipeline
   uv run python -m sleap_roots_analyze.pipeline.cli qc configs/qc_[dataset].yaml

   # Check final sample count matches notebook
   # Expected: Notebook "Final samples: X" matches pipeline Step 10 sample count
   ```

2. **Trait Count Test**
   ```bash
   # Check traits after heritability filtering
   # Expected: Notebook "Final traits: Y" matches pipeline Step 10 trait count
   ```

3. **PCA Structure Test**
   ```bash
   # Run viz pipeline
   uv run python -m sleap_roots_analyze.pipeline.cli viz configs/viz_[dataset].yaml

   # Check number of PCs used
   # Expected: Matches notebook PC count at specified variance threshold
   ```

4. **Correlation Count Test (Cross-Platform)**
   ```bash
   # Check total correlations calculated
   # Expected: N_traits_exp1 × N_traits_exp2 = Total correlations
   ```

### Acceptance Criteria

- ✅ All critical parameters match between notebook and config
- ✅ Intentional variations are documented with rationale
- ✅ Pipeline outputs (sample counts, trait counts) match notebook results
- ✅ Parameter reference table complete for all 4 datasets
- ✅ Config headers updated with notebook source references

## Open Questions

1. **Q:** Should we create a viz config for turface_19genotypes if one doesn't exist?
   **A:** Check if visualizations were done in notebook. If yes, create config.

2. **Q:** How to handle data paths that point to user-specific directories?
   **A:** Document both the original notebook path and the generic config path pattern.

3. **Q:** Should cross-platform configs reference specific QC run timestamps?
   **A:** Yes in comments for reproducibility, but allow flexible paths for reuse.

4. **Q:** What if notebook has parameters that don't map to any config field?
   **A:** Document in spec as "notebook-specific" and note if they affect results.

5. **Q:** How to handle parameters that changed mid-analysis (e.g., REPLICATION_GUIDE.md)?
   **A:** Use the FINAL notebook values that produced the correct high-heritability results.

## Future Enhancements (Post-Publication)

Once the immediate publication need is met, consider:

1. **Automated Parameter Extraction**
   - Parse notebooks programmatically
   - Extract parameter dictionaries
   - Compare against configs automatically

2. **CI/CD Validation**
   - Add pytest tests that verify configs match reference parameters
   - Alert on config changes without documentation updates
   - Validate new configs before merge

3. **Notebook-to-Config Generation**
   - Tool to generate config YAML from notebook parameters
   - Reduces manual transcription errors
   - Ensures consistency from the start

4. **Parameter Provenance Tracking**
   - Git commit linking config changes to notebook updates
   - Automated changelog for parameter modifications
   - Version control for parameter evolution

## Summary

This audit uses a systematic approach to verify that all pipeline configurations exactly match the notebook analyses that produced publishable results. By organizing the work in OpenSpec, we ensure findings are well-documented, fixes are tracked, and future datasets can follow the same process. The audit prioritizes critical parameters that affect scientific conclusions while documenting intentional variations with appropriate rationale.
