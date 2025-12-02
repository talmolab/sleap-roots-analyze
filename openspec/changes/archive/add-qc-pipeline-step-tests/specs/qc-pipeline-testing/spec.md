# QC Pipeline Testing

## ADDED Requirements

### Requirement: Validate Clean Step Testing
The test suite SHALL provide comprehensive unit tests for the ValidateCleanStep (Step 3) to verify data validation behavior.

#### Scenario: Basic validation passes
- **WHEN** ValidateCleanStep executes with valid cleaned data
- **THEN** step completes successfully and returns data unchanged
- **AND** generates validation report file
- **AND** updates manifest with validation status

#### Scenario: Validation detects unexpected NaNs
- **WHEN** ValidateCleanStep receives data with unexpected NaN values
- **THEN** step raises ValueError with descriptive message
- **AND** identifies which columns contain unexpected NaNs

#### Scenario: Empty data handling
- **WHEN** ValidateCleanStep receives empty DataFrame
- **THEN** step raises ValueError indicating insufficient data
- **AND** provides clear error message for debugging

### Requirement: Exploratory Analysis Step Testing
The test suite SHALL provide comprehensive unit tests for the ExploratoryAnalysisStep (Step 4) to verify exploratory statistics generation.

#### Scenario: Generates correlation matrix
- **WHEN** ExploratoryAnalysisStep executes with multi-trait data
- **THEN** step generates correlation matrix CSV file
- **AND** correlation values are between -1 and 1
- **AND** matrix is symmetric

#### Scenario: Computes trait statistics
- **WHEN** ExploratoryAnalysisStep processes trait data
- **THEN** step generates trait statistics CSV with mean, std, min, max
- **AND** all statistics are numerically valid
- **AND** statistics match expected values for known data

#### Scenario: Single trait edge case
- **WHEN** ExploratoryAnalysisStep receives data with single trait
- **THEN** step completes successfully with 1x1 correlation matrix
- **AND** generates valid statistics for single trait

### Requirement: Detect Outliers Step Testing
The test suite SHALL provide comprehensive unit tests for the DetectOutliersStep (Step 5) to verify outlier detection methods.

#### Scenario: Mahalanobis distance detection
- **WHEN** DetectOutliersStep executes with Mahalanobis method enabled
- **THEN** step calculates Mahalanobis distances for all samples
- **AND** identifies outliers based on chi-squared threshold
- **AND** returns outlier indices in result metadata

#### Scenario: Isolation Forest detection
- **WHEN** DetectOutliersStep executes with Isolation Forest enabled
- **THEN** step applies Isolation Forest algorithm
- **AND** identifies outliers based on contamination parameter
- **AND** returns outlier scores and indices

#### Scenario: Combined outlier detection
- **WHEN** DetectOutliersStep executes with multiple methods enabled
- **THEN** step runs all configured detection methods
- **AND** combines outlier results according to strategy (union/intersection)
- **AND** returns comprehensive outlier report

#### Scenario: No outliers detected
- **WHEN** DetectOutliersStep processes well-behaved data
- **THEN** step completes successfully with empty outlier list
- **AND** generates report indicating no outliers found
- **AND** propagates all data to next step

#### Scenario: Insufficient samples for detection
- **WHEN** DetectOutliersStep receives fewer than minimum required samples
- **THEN** step raises ValueError or warning
- **AND** provides guidance on minimum sample size requirements

### Requirement: Visualize Outliers Step Testing
The test suite SHALL provide comprehensive unit tests for the VisualizeOutliersStep (Step 6) to verify outlier visualization generation.

#### Scenario: PCA outlier plot generation
- **WHEN** VisualizeOutliersStep executes with PCA results and outlier indices
- **THEN** step generates PCA scatter plot highlighting outliers
- **AND** saves plot in configured formats (PNG, PDF, SVG)
- **AND** outliers are visually distinguished from normal samples

#### Scenario: Outlier distance plots
- **WHEN** VisualizeOutliersStep has distance metrics available
- **THEN** step generates distance distribution plots
- **AND** marks outlier threshold on plots
- **AND** saves plots with descriptive filenames

#### Scenario: No outliers visualization
- **WHEN** VisualizeOutliersStep receives empty outlier list
- **THEN** step generates plots showing all samples as normal
- **AND** includes annotation indicating no outliers detected

#### Scenario: Missing PCA results handling
- **WHEN** VisualizeOutliersStep executes without PCA results in metadata
- **THEN** step skips PCA-based visualizations
- **AND** logs warning about missing PCA data
- **AND** continues with available visualizations

### Requirement: Statistical Analysis Step Testing
The test suite SHALL provide comprehensive unit tests for the StatisticalAnalysisStep (Step 8) to verify statistical calculations.

#### Scenario: Heritability calculation
- **WHEN** StatisticalAnalysisStep executes with genotype and replicate data
- **THEN** step calculates broad-sense heritability for all traits
- **AND** H² values are between 0.0 and 1.0
- **AND** generates heritability results CSV file

#### Scenario: ANOVA analysis
- **WHEN** StatisticalAnalysisStep performs ANOVA tests
- **THEN** step calculates F-statistics and p-values for each trait
- **AND** p-values are between 0.0 and 1.0
- **AND** results are saved to CSV file

#### Scenario: Insufficient genotypes for statistics
- **WHEN** StatisticalAnalysisStep receives data with single genotype
- **THEN** step raises ValueError indicating insufficient variance
- **AND** provides clear error message about minimum genotype requirement

#### Scenario: Missing replicate information
- **WHEN** StatisticalAnalysisStep receives data without replicate column
- **THEN** step raises ValueError identifying missing column
- **AND** provides guidance on required data structure

### Requirement: Filter Heritability Step Testing
The test suite SHALL provide comprehensive unit tests for the FilterHeritabilityStep (Step 9) to verify trait filtering by heritability.

#### Scenario: Basic heritability filtering
- **WHEN** FilterHeritabilityStep executes with H² threshold of 0.3
- **THEN** step removes traits with H² < 0.3
- **AND** returns filtered DataFrame with only high-heritability traits
- **AND** tracks removed traits in metadata

#### Scenario: All traits pass threshold
- **WHEN** FilterHeritabilityStep processes data where all traits have high H²
- **THEN** step returns data unchanged
- **AND** reports zero traits removed
- **AND** continues pipeline normally

#### Scenario: All traits fail threshold
- **WHEN** FilterHeritabilityStep processes data where no traits meet threshold
- **THEN** step raises ValueError indicating no traits remaining
- **AND** provides list of removed traits and their H² values
- **AND** suggests lowering threshold or reviewing data quality

#### Scenario: Threshold edge cases
- **WHEN** FilterHeritabilityStep executes with threshold 0.0 or 1.0
- **THEN** step handles extreme thresholds correctly
- **AND** threshold 0.0 keeps all traits
- **AND** threshold 1.0 removes all traits except perfect H²=1.0

#### Scenario: Missing heritability data
- **WHEN** FilterHeritabilityStep executes without heritability results in metadata
- **THEN** step raises ValueError indicating missing prerequisite step
- **AND** provides guidance to run StatisticalAnalysisStep first

### Requirement: Generate Summary Step Testing
The test suite SHALL provide comprehensive unit tests for the GenerateSummaryStep (Step 10) to verify pipeline summary generation.

#### Scenario: Complete pipeline summary
- **WHEN** GenerateSummaryStep executes after full pipeline run
- **THEN** step generates comprehensive JSON summary file
- **AND** summary includes sample counts, trait counts, and QC statistics
- **AND** summary includes file paths for all generated outputs

#### Scenario: Summary statistics aggregation
- **WHEN** GenerateSummaryStep compiles pipeline results
- **THEN** step aggregates statistics from all previous steps
- **AND** includes counts for removed samples, removed traits, detected outliers
- **AND** includes execution timestamps and configuration used

#### Scenario: File list compilation
- **WHEN** GenerateSummaryStep generates summary
- **THEN** step lists all CSV, plot, and report files generated
- **AND** verifies all listed files exist
- **AND** includes file sizes and modification times

#### Scenario: Minimal pipeline summary
- **WHEN** GenerateSummaryStep executes after partial pipeline run
- **THEN** step generates summary with available information
- **AND** marks unavailable sections as N/A or null
- **AND** does not fail due to missing optional steps

#### Scenario: JSON schema validation
- **WHEN** GenerateSummaryStep writes summary JSON
- **THEN** output is valid JSON format
- **AND** follows consistent schema structure
- **AND** includes version information for reproducibility

### Requirement: Test Fixture Reusability
The test suite SHALL provide reusable fixtures for QC pipeline step testing following established patterns.

#### Scenario: Centralized fixture usage
- **WHEN** tests import fixtures from conftest.py or fixtures.py
- **THEN** fixtures provide consistent test data across all step tests
- **AND** fixtures follow established naming conventions
- **AND** fixtures are documented with clear docstrings

#### Scenario: Step-specific fixtures
- **WHEN** individual test files need specialized fixtures
- **THEN** fixtures are defined within the test file
- **AND** fixtures are scoped appropriately (function, class, module)
- **AND** complex fixtures include usage examples in docstrings

### Requirement: Test Coverage Standards
The test suite SHALL achieve minimum 90% code coverage for all tested QC pipeline steps.

#### Scenario: Coverage measurement
- **WHEN** pytest runs with coverage flags (--cov --cov-branch)
- **THEN** coverage report shows ≥90% line coverage for each step module
- **AND** coverage report shows ≥85% branch coverage
- **AND** uncovered lines are intentionally excluded or documented

#### Scenario: Edge case coverage
- **WHEN** tests execute for each step
- **THEN** tests cover success paths, error paths, and edge cases
- **AND** tests validate error messages and exception types
- **AND** tests verify boundary conditions and extreme inputs
