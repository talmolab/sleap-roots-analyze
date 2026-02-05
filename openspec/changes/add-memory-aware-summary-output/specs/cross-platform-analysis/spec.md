## ADDED Requirements

### Requirement: Memory-Aware Summary Image Handling

The system SHALL support configurable image handling modes for cross-platform analysis summaries with the following options:

- `file_path` (default): Use relative file paths for all images
- `embed`: Embed images as base64 data URIs if total size < threshold, otherwise fallback to file_path with warning
- `auto`: Automatically select based on total image size

The default embed threshold SHALL be 10MB (10,485,760 bytes).

#### Scenario: File path mode generates small summary

- **WHEN** generating a cross-platform summary with `image_mode="file_path"`
- **THEN** the SUMMARY.md file SHALL be less than 1MB
- **AND** all image references SHALL use relative file paths
- **AND** images SHALL be viewable when SUMMARY.md is opened in VS Code markdown preview

#### Scenario: Embed mode respects size threshold

- **WHEN** generating a cross-platform summary with `image_mode="embed"`
- **AND** total image size is less than 10MB
- **THEN** images SHALL be embedded as base64 data URIs

#### Scenario: Embed mode falls back when over threshold

- **WHEN** generating a cross-platform summary with `image_mode="embed"`
- **AND** total image size exceeds the threshold
- **THEN** a warning SHALL be logged
- **AND** the system SHALL fall back to file_path mode
- **AND** the summary SHALL be generated successfully

#### Scenario: Auto mode selects appropriate method

- **WHEN** generating a cross-platform summary with `image_mode="auto"`
- **THEN** the system SHALL calculate total image size
- **AND** embed images if total size < threshold
- **AND** use file paths if total size >= threshold

### Requirement: HTML Summary Output

The system SHALL support generating HTML output format for cross-platform summaries that can be viewed directly in web browsers.

#### Scenario: HTML output generated with markdown

- **WHEN** generating a cross-platform summary with `output_format="both"`
- **THEN** both SUMMARY.md and SUMMARY.html SHALL be created
- **AND** SUMMARY.html SHALL include embedded CSS styling
- **AND** SUMMARY.html SHALL render correctly in Chrome/Firefox

#### Scenario: HTML output contains proper structure

- **WHEN** generating HTML summary output
- **THEN** the HTML SHALL include proper DOCTYPE and charset
- **AND** tables SHALL be styled with borders and padding
- **AND** images SHALL be properly referenced
- **AND** the file SHALL be self-contained (embedded styles)

### Requirement: Image Size Calculation

The system SHALL calculate total image size before generating summaries to support memory-aware decisions.

#### Scenario: Total size calculated for all images

- **WHEN** preparing to generate a summary
- **THEN** the system SHALL calculate the total size of all visualization images
- **AND** estimate base64 overhead (approximately 1.37x raw size)
- **AND** make this information available for mode selection

#### Scenario: Missing images handled gracefully

- **WHEN** calculating total image size
- **AND** some image files do not exist
- **THEN** missing files SHALL be skipped without error
- **AND** a warning SHALL be logged for each missing file

## MODIFIED Requirements

### Requirement: Summary Generation

The system SHALL generate comprehensive markdown summaries from cross-platform correlation analysis outputs using the `CrossPlatformSummaryGenerator` class.

The `to_markdown()` method SHALL accept the following parameters:
- `image_mode`: One of "file_path", "embed", or "auto" (default: "file_path")
- `embed_threshold_bytes`: Maximum bytes for embedded images (default: 10,485,760)

The `generate()` method SHALL return a `CrossPlatformSummary` object that can be converted to markdown or HTML format.

#### Scenario: Default generates file-path markdown

- **WHEN** `CrossPlatformSummaryGenerator.generate()` is called
- **AND** `summary.to_markdown()` is called with no arguments
- **THEN** the output SHALL use relative file paths for images
- **AND** the file size SHALL be less than 1MB for typical analyses

#### Scenario: Embedded images requested under threshold

- **WHEN** `summary.to_markdown(image_mode="embed")` is called
- **AND** total visualization size is under 10MB
- **THEN** images SHALL be embedded as base64 data URIs
- **AND** the markdown SHALL be self-contained

#### Scenario: Summary written by pipeline runner

- **WHEN** the pipeline runner generates SUMMARY.md
- **THEN** it SHALL use `image_mode="file_path"` by default
- **AND** optionally accept `--embed-images` flag for portable output
- **AND** optionally accept `--html` flag for HTML output
