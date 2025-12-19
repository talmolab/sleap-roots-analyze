# Implementation Tasks

**Status**: ✅ COMPLETED
**Commit**: dc970ea - "fix: convert float columns to int in sample_id and Barcode creation"

## Tasks

- [x] Fix 1: Refactor LoadRootCoreDataStep to use shared `create_sample_identifier()` function
- [x] Fix 2: Convert numeric metadata columns to int immediately after loading CSV
- [x] Fix 3: Update ReshapeForTraitQCStep to convert Rep/Plot to int before creating Barcode
- [x] Fix 4: Add tests to catch float dtype issues in sample_id and Barcode generation
- [x] Fix 5: Document outlier_flag column behavior in pipeline documentation
