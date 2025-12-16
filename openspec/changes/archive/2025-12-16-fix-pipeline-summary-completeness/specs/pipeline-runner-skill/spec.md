## MODIFIED Requirements

### Requirement: Run Summary Generation

The system SHALL generate a comprehensive markdown summary document after each run.

#### Scenario: Summary document created

- **WHEN** pipeline run completes
- **THEN** a `SUMMARY.md` file SHALL be created in the run directory
- **AND** the summary SHALL include generation timestamp
- **AND** the summary SHALL include git commit hash
- **AND** the summary SHALL include manifest file reference
- **AND** the file SHALL be written with UTF-8 encoding to properly display Unicode characters (H², mm², etc.)

#### Scenario: QC results in summary

- **WHEN** QC pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Dataset, Samples, Traits, Genotypes, H² Threshold, Mean H², Status, Run Path
- **AND** for each QC run, the summary SHALL read `10_pipeline_summary.json` to extract scientific metrics
- **AND** the summary SHALL include a "Removed Traits" subsection listing traits filtered by heritability threshold for each dataset
- **AND** each removed trait SHALL display its heritability value in parentheses (e.g., "Depth (mm) (H²=0.32)")

#### Scenario: QC results with failed pipeline

- **WHEN** a QC pipeline fails during execution
- **THEN** the summary table SHALL show "Failed" status for that config
- **AND** numeric columns (Samples, Traits, etc.) SHALL display "N/A"
- **AND** other successful QC runs SHALL still display full metrics

#### Scenario: Viz results in summary

- **WHEN** Viz pipelines complete
- **THEN** the summary SHALL include a table with columns: Dataset, Figures Generated, Interactive Plots, Status, Time, Run Path
- **AND** the summary SHALL count static figures from the `static_figures/` directory
- **AND** the summary SHALL count interactive plots from the `pca/` and `umap/` directories

#### Scenario: Cross-Platform results in summary

- **WHEN** Cross-Platform pipelines complete successfully
- **THEN** the summary SHALL include a table with columns: Comparison, Common Genotypes, Exp1 Samples, Exp1 Traits, Exp2 Samples, Exp2 Traits, Top Correlation, Status, Run Path
- **AND** the summary SHALL read `cross_platform_alignment_summary.csv` or `pipeline_summary.json` to extract alignment metrics
- **AND** the summary SHALL read `cross_platform_correlations.csv` to extract the top correlation value
- **AND** the CSV parser SHALL use the actual column names: `genotype`, `exp1_sample_count`, `exp2_sample_count`

#### Scenario: Cross-Platform results with missing data

- **WHEN** Cross-Platform pipeline completes but alignment CSV is missing
- **THEN** the summary SHALL display "N/A" for genotype/sample/trait counts
- **AND** the summary SHALL still display status and run path

#### Scenario: Methods section template

- **WHEN** summary is generated
- **THEN** the summary SHALL include a "## Methods" section with publication-ready template text
- **AND** the template SHALL describe the QC pipeline methodology (cleanup, outlier detection, heritability filtering)
- **AND** the template SHALL describe the Viz pipeline methodology (statistical analysis, visualization generation)
- **AND** all placeholders SHALL be replaced with actual config values (e.g., `{h2_threshold}` becomes "0.4")
- **AND** if configs differ across datasets, the Methods section SHALL note "varied by dataset" with a footnote

#### Scenario: Summary statistics figure generated

- **WHEN** QC pipelines complete with at least 2 datasets
- **THEN** the summary directory SHALL contain a `summary_statistics.png` bar chart
- **AND** the chart SHALL show sample count, trait count, and genotype count per dataset
- **AND** the chart SHALL use a grouped bar layout for easy comparison

#### Scenario: Heritability distribution figure generated

- **WHEN** QC pipelines complete with heritability filtering enabled
- **THEN** the summary directory SHALL contain a `heritability_distribution.png` visualization
- **AND** the visualization SHALL show H² distributions for retained vs. removed traits
- **AND** the visualization SHALL include the threshold line

## NEW Requirements

### Requirement: Cross-Platform Output Directory Naming

The system SHALL create output directories with filesystem-safe names.

#### Scenario: Experiment names sanitized

- **WHEN** cross-platform pipeline creates output directory
- **THEN** spaces in experiment names SHALL be replaced with underscores
- **AND** special characters (parentheses, quotes) SHALL be removed or replaced
- **AND** the resulting path SHALL be valid on Windows, macOS, and Linux

### Requirement: Input Data Checksums

The system SHALL record checksums of input data files for reproducibility verification.

#### Scenario: Checksum recorded in summary

- **WHEN** pipeline summary is generated
- **THEN** the summary JSON SHALL include an `input_checksums` field
- **AND** each input CSV file SHALL have its MD5 hash recorded
- **AND** the checksum format SHALL be: `{"filename": "path/to/file.csv", "md5": "abc123..."}`

### Requirement: Cross-Reference Links

The system SHALL provide navigation links between related pipeline outputs.

#### Scenario: QC to Viz cross-reference

- **WHEN** a Viz pipeline uses QC output as input
- **THEN** the Viz `SUMMARY.md` SHALL link to the source QC summary
- **AND** the link SHALL be a relative path from the Viz output directory

#### Scenario: Run summary cross-references

- **WHEN** the main `SUMMARY.md` is generated
- **THEN** each pipeline output path SHALL be a clickable link to that pipeline's summary
- **AND** the links SHALL work when viewing the markdown in common editors (VS Code, GitHub)

### Requirement: Package Dependency Recording

The system SHALL record versions of key scientific packages used in the analysis.

#### Scenario: Dependencies in code snapshot

- **WHEN** pipeline summary JSON is generated
- **THEN** the `code_snapshot` section SHALL include a `dependencies` field
- **AND** the field SHALL list versions of: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn
- **AND** versions SHALL be retrieved from the installed packages at runtime

#### Scenario: Missing package handled gracefully

- **WHEN** a dependency is not installed
- **THEN** the version SHALL be recorded as "not installed"
- **AND** the summary generation SHALL not fail

### Requirement: Output Files Inventory

The system SHALL maintain an inventory of all files generated by the pipeline.

#### Scenario: Files generated list populated

- **WHEN** pipeline completes successfully
- **THEN** the `files_generated` field in the summary JSON SHALL contain all output files
- **AND** each entry SHALL include: relative path, file size in bytes, and file type
- **AND** the list SHALL be sorted alphabetically by path

#### Scenario: Files categorized by type

- **WHEN** files are recorded
- **THEN** files SHALL be categorized as: "data" (CSV/JSON), "figure" (PNG/SVG/PDF), "report" (MD/HTML)
- **AND** the summary SHALL include counts per category
