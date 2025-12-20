## 1. Write Failing Tests (TDD)

- [x] 1.1 Create test that verifies CrossPlatformConfig accepts exp1_exclude_cols parameter
- [x] 1.2 Create test that verifies CrossPlatformConfig accepts exp2_exclude_cols parameter
- [x] 1.3 Create test that verifies LoadCrossPlatformDataStep excludes columns from exp1 traits
- [x] 1.4 Create test that verifies LoadCrossPlatformDataStep excludes columns from exp2 traits
- [x] 1.5 Create integration test with metadata columns (Ent, Sub) that should be excluded
- [x] 1.6 Run tests to confirm they fail with current implementation

## 2. Implementation

- [x] 2.1 Add `exp1_exclude_cols: Optional[List[str]] = None` to CrossPlatformConfig
- [x] 2.2 Add `exp2_exclude_cols: Optional[List[str]] = None` to CrossPlatformConfig
- [x] 2.3 Update CrossPlatformConfig docstring to document new parameters
- [x] 2.4 Modify LoadCrossPlatformDataStep to pass exp1_exclude_cols to get_trait_columns() for exp1
- [x] 2.5 Modify LoadCrossPlatformDataStep to pass exp2_exclude_cols to get_trait_columns() for exp2
- [x] 2.6 Update step docstring to document exclusion behavior

## 3. Configuration Updates

Each cross-platform config needs exclusion lists based on the source data:

### Data Source Exclusion Lists (from QC configs)

**Root Core EDPIE** (exp1 in rootcore_vs_* configs):
```yaml
exp1_exclude_cols:
  - "Ent"              # Entry number
  - "Sub"              # Sub-entry
  - "Cid"              # Cross ID
  - "Sid"              # Selection ID
  - "GID"              # Germplasm ID
  - "Cross name"       # Cross pedigree
  - "Sel_Hist"         # Selection history
  - "Origin"           # Germplasm origin
  - "core_id"          # Core identifier
  - "plant_identifier" # Plant identifier
```

**Turface 150 Genotypes**:
```yaml
exp1_exclude_cols:  # or exp2_exclude_cols depending on config
  - "File.me"
  - "region"
  - "set"
  - "Salk_geno"
  - "Entry"
  - "GID"
  - "scanner"
```

**Turface 19 Genotypes**: No additional exclusions needed (empty list or omit)

**Cylinder EDPIE**: No additional exclusions needed (empty list or omit)

**Field 2024 Clean**: Same as Root Core EDPIE (merged data)

### Config File Updates

- [x] 3.1 Update `cross_platform_rootcore_vs_turface19.yaml`:
  - exp1_exclude_cols: Root Core EDPIE list (Ent, Sub, Cid, etc.)
  - exp2_exclude_cols: [] (Turface 19 has no extra metadata)

- [x] 3.2 Update `cross_platform_rootcore_vs_cylinder.yaml`:
  - exp1_exclude_cols: Root Core EDPIE list
  - exp2_exclude_cols: [] (Cylinder has no extra metadata)

- [x] 3.3 Update `cross_platform_turface19_vs_cylinder.yaml`:
  - exp1_exclude_cols: [] (Turface 19 has no extra metadata)
  - exp2_exclude_cols: [] (Cylinder has no extra metadata)

- [x] 3.4 Update `cross_platform_turface_150vs19_genotypes.yaml`:
  - exp1_exclude_cols: Turface 150 list (File.me, region, etc.)
  - exp2_exclude_cols: [] (Turface 19 has no extra metadata)

- [x] 3.5 Update `cross_platform_field_vs_cylinder.yaml`:
  - exp1_exclude_cols: Root Core EDPIE list (Field data has same metadata)
  - exp2_exclude_cols: [] (Cylinder has no extra metadata)

- [x] 3.6 Update `cross_platform_turface19_vs_field.yaml`:
  - exp1_exclude_cols: [] (Turface 19 has no extra metadata)
  - exp2_exclude_cols: Root Core EDPIE list (Field data has same metadata)

- [x] 3.7 Update cross-platform config template with documented exclude_cols examples

## 4. Verification

- [x] 4.1 Run failing tests to confirm they now pass
- [x] 4.2 Run full test suite to ensure no regressions
- [x] 4.3 Run cross-platform pipeline with updated configs
- [x] 4.4 Verify Ent, Sub, Cid, etc. are no longer in correlation results
- [x] 4.5 Verify only biological traits remain in analysis