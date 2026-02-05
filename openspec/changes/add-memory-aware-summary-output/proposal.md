# Add Memory-Aware Summary Output

## Why

Cross-platform analysis summaries with embedded base64 images create files exceeding 70MB, which cannot be rendered in VS Code (data URIs blocked for security) or browsers (memory limits exceeded). Users currently have no practical way to view the generated SUMMARY.md with visualizations.

**Constraints discovered:**
- VS Code markdown preview: Blocks data URIs entirely for security; file path references work
- Chrome/Firefox DOM: Rendering degrades significantly above 10-20MB
- Base64 overhead: Images are ~33% larger when base64 encoded
- Current default: `embed_images=True` hardcoded in pipeline_runner.py

## What Changes

- **Default output mode**: Change from `embed_images=True` to file path references (viewable in VS Code)
- **Size threshold**: Add 10MB threshold for embedded content; warn and fallback if exceeded
- **HTML export**: Add option to generate browser-viewable HTML alongside markdown
- **Auto mode**: Intelligent selection based on total image size
- **Portable mode**: Explicit flag for single-file sharing (with size warning)

### Configuration Options

```python
class SummaryOutputConfig:
    image_mode: Literal["file_path", "embed", "auto"] = "file_path"
    output_format: Literal["markdown", "html", "both"] = "markdown"
    embed_size_threshold_mb: float = 10.0
```

### Behavior by Mode

| Mode | Behavior | Use Case |
|------|----------|----------|
| `file_path` | Always use relative paths | VS Code viewing (default) |
| `embed` | Embed if < threshold, else warn + fallback | Portable sharing |
| `auto` | Embed if < threshold, else file_path | Smart default |

## Impact

- Affected specs: `cross-platform-analysis`
- Affected code:
  - `src/sleap_roots_analyze/summary/cross_platform_summary.py`
  - `src/sleap_roots_analyze/pipeline_runner.py`
  - `.claude/commands/cross-platform-summary.md`
  - `scripts/convert_summary_to_html.py`

## Non-Breaking

This change is **non-breaking** for existing workflows:
- Users who embed summaries programmatically can still request `embed_images=True`
- The default changes from embedded to file-path, which is strictly better for viewing
