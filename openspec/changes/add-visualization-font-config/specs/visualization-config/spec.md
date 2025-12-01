# Visualization Configuration Specification

**Component**: VisualizationConfig  
**Version**: 2.0  
**Status**: Draft

## Overview

The `VisualizationConfig` dataclass provides comprehensive configuration for all visualization parameters in the sleap-roots-analyze pipeline, including font sizes, figure formats, and matplotlib savefig parameters.

## ADDED Requirements

### Requirement: Font Size Configuration
The system SHALL provide configurable font sizes for all text elements in generated plots:
- Title font size (main plot titles)
- Label font size (axis labels)
- Tick font size (axis tick labels)
- Legend font size (legend text)

#### Scenario 2: Journal-Specific Font Requirements

**Given**: Journal requires 16pt titles, 14pt labels

```yaml
visualization:
  dpi: 300
  figsize: [10, 8]
  title_fontsize: 16
  label_fontsize: 14
  tick_fontsize: 12
  legend_fontsize: 12
```

**When**: Pipeline generates plots

**Then**:
- All titles use 16pt font
- All axis labels use 14pt font
- All tick labels use 12pt font
- All legends use 12pt font

### Requirement: Figure Format Configuration
The system SHALL support multiple output formats:
- Single format mode: `figure_format` for pipeline steps
- Multi-format mode: `figure_formats` list for batch generation
- Supported formats: PNG, PDF, SVG, EPS
- Format validation at config load time

#### Scenario 3: Vector Format for Publication

**Given**: Journal requires PDF format

```yaml
visualization:
  dpi: 300
  figure_format: "pdf"
  title_fontsize: 14
  label_fontsize: 12
```

**When**: ExploratoryAnalysis step runs

**Then**:
- All figures save as `.pdf` files
- High-quality vector graphics
- Scalable without quality loss

#### Scenario 4: Multiple Format Generation

**Given**: Need both PNG (preview) and PDF (publication)

```yaml
static_viz:
  enabled: true
  formats: ["png", "pdf"]
  dpi: 300
```

**When**: GenerateStaticFigures step runs

**Then**:
- Each figure saved in both PNG and PDF formats
- `pca_scree.png` and `pca_scree.pdf` both created
- Manifest lists all generated files

#### Scenario 7: Invalid Format

**Given**: User specifies unsupported format

```yaml
visualization:
  figure_format: "jpg"
```

**When**: Config is loaded

**Then**:
- Validation error: "figure_format must be one of {'png', 'pdf', 'svg', 'eps'}, got 'jpg'"
- Pipeline does not start

### Requirement: Savefig Parameter Configuration
The system SHALL provide control over matplotlib savefig parameters:
- Bounding box mode (`bbox_inches`: "tight" or None)
- Figure face color (`facecolor`: color string or None)
- Figure edge color (`edgecolor`: color string or None)
- Transparency (`transparent`: boolean)

#### Scenario 5: Transparent Background

**Given**: Need figures with transparent backgrounds for presentations

```yaml
visualization:
  dpi: 300
  transparent: true
  facecolor: null
```

**When**: Pipeline generates plots

**Then**:
- All saved figures have transparent backgrounds
- No white background in PNG files
- Suitable for overlay on colored slides

### Requirement: Backward Compatibility
All new parameters SHALL have defaults matching previous hardcoded values. Existing configs without new parameters SHALL work unchanged. Existing tests SHALL pass without modification.

#### Scenario 1: Default Configuration (Backward Compatibility)

**Given**: User has existing config without new parameters

```yaml
visualization:
  dpi: 300
  figsize: [10, 8]
```

**When**: Pipeline runs

**Then**: 
- All font sizes default to previous hardcoded values (14, 12, 10, 10)
- Figures save as PNG format
- bbox_inches defaults to "tight"
- All existing tests pass

### Requirement: Consistent Application
All pipeline steps SHALL use configured font sizes. All visualization functions SHALL accept optional font size parameters. All savefig calls SHALL use configured format and bbox parameters.

#### Scenario: Consistent Font Application

**Given**: Config with custom font sizes

```yaml
visualization:
  title_fontsize: 16
  label_fontsize: 14
```

**When**: Multiple pipeline steps generate plots (ExploratoryAnalysis, VisualizeOutliers, FilterHeritability)

**Then**:
- All plot titles across all steps use 16pt font
- All axis labels across all steps use 14pt font
- Font sizes are consistent across entire pipeline output

### Requirement: Config Validation
Parameters SHALL have clear, descriptive names. Default values SHALL be suitable for publication-quality figures. Config validation SHALL provide helpful error messages.

#### Scenario 6: Invalid Configuration

**Given**: User specifies invalid font size

```yaml
visualization:
  title_fontsize: -5
```

**When**: Config is loaded

**Then**:
- Validation error raised: "title_fontsize must be positive, got -5"
- Pipeline does not start
- Clear error message guides user to fix config

## References

- Design documentation: [design.md](design.md)
- Matplotlib savefig documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html
- Current VisualizationConfig: `src/sleap_roots_analyze/pipeline/config/components.py:52-60`
- Example configs: `configs/qc_turface_150genotypes.yaml`
