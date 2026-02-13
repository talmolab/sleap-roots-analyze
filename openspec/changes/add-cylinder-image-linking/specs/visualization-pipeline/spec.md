## ADDED Requirements

### Requirement: Image Linking Method Configuration
The `DataConfig` dataclass SHALL provide `image_linking_method` and `scan_path_col` parameters to support different image organization patterns.

#### Scenario: Configure RhizoVision image linking (default)
- **WHEN** user omits `image_linking_method` or sets it to "rhizovision"
- **THEN** the pipeline SHALL use `link_rhizovision_images_to_samples()` function
- **AND** the function SHALL look for files like `{barcode}_c1_p1_{image_type}` in flat directory
- **AND** backward compatibility SHALL be maintained

#### Scenario: Configure cylinder image linking
- **WHEN** user sets `image_linking_method` to "cylinder" in config
- **THEN** the pipeline SHALL use `link_cylinder_images_from_scan_path()` function
- **AND** the function SHALL read `scan_path` column from data
- **AND** the function SHALL build paths like `{base_dir}/{scan_path}/{image_type}`

#### Scenario: Cylinder linking without scan_path column
- **WHEN** user configures cylinder linking but data lacks `scan_path` column
- **THEN** the pipeline SHALL log a warning
- **AND** no images SHALL be linked
- **AND** the pipeline SHALL continue execution without image grids

#### Scenario: Custom scan_path column name
- **WHEN** user sets `scan_path_col` to a custom column name
- **THEN** the pipeline SHALL use that column for cylinder image linking
- **AND** the default SHALL remain "scan_path" for compatibility

### Requirement: Image Paths Metadata Flow
The visualization pipeline SHALL preserve `image_paths` metadata through all analysis steps.

#### Scenario: Metadata preserved through StatisticalAnalysisStep
- **WHEN** `LoadDataAndImagesStep` produces `image_paths` in metadata
- **AND** `StatisticalAnalysisStep` executes
- **THEN** `image_paths` SHALL be present in the step's output metadata
- **AND** downstream steps SHALL have access to the image paths

#### Scenario: Metadata preserved through UMAPAnalysisStep
- **WHEN** `image_paths` metadata exists from previous steps
- **AND** `UMAPAnalysisStep` executes
- **THEN** `image_paths` SHALL be present in the step's output metadata
- **AND** the UMAP step SHALL NOT modify or remove image_paths

#### Scenario: GenerateStaticFiguresStep receives image_paths
- **WHEN** `GenerateStaticFiguresStep` executes
- **THEN** it SHALL receive `image_paths` from accumulated metadata
- **AND** it SHALL pass image_paths to `_create_genotype_image_grids()` function
- **AND** genotype image grids SHALL be generated with correct images

### Requirement: Image Paths Format Handling
The `_create_genotype_image_grids` function SHALL handle both nested dict format and legacy Series format for `image_paths`.

#### Scenario: Handle nested dict format from link functions
- **WHEN** `image_paths` is in format `Dict[barcode, Dict[image_type, Path]]`
- **THEN** the function SHALL detect this as nested dict format
- **AND** the function SHALL use the paths directly (with Path conversion)
- **AND** genotype image grids SHALL be created successfully

#### Scenario: Handle legacy Series format
- **WHEN** `image_paths` is a pd.Series indexed by DataFrame row numbers
- **THEN** the function SHALL detect this as legacy format
- **AND** the function SHALL convert to `Dict[barcode, Dict[image_type, Path]]` format
- **AND** backward compatibility SHALL be maintained