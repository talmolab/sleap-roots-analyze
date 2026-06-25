## MODIFIED Requirements

### Requirement: Pipeline Summary Config Population

The pipeline summary JSON SHALL include the complete configuration used for the run.

#### Scenario: Summary includes config dict
- **GIVEN** a pipeline run that completes
- **WHEN** `pipeline_summary.json` is loaded
- **THEN** the `config` field SHALL contain a non-empty dictionary
- **AND** the dictionary SHALL include all config parameters

#### Scenario: Summary config is JSON serializable
- **GIVEN** a config with complex types (Path objects, dataclasses)
- **WHEN** the pipeline saves the summary
- **THEN** all config values SHALL be serialized to JSON-compatible types
- **AND** Path objects SHALL be converted to POSIX strings via `Path.as_posix()` (forward-slash separators on every OS)

## ADDED Requirements

### Requirement: Provenance Path Serialization Is Centralized

The pipeline provenance manifest (`pipeline_summary.json`) SHALL record every filesystem path through the central JSON serializer (`convert_to_json_serializable`) so that path normalization happens in exactly one place. Producing steps SHALL store `Path` objects in `files_generated` and `metadata`; they SHALL NOT pre-stringify paths with `str(path)`, which on Windows yields backslash separators and bypasses the serializer's `Path.as_posix()` normalization. The `files_generated` field SHALL be typed `List[Path]` so the type cannot invite per-producer divergence.

#### Scenario: Generated file paths normalize to POSIX in the manifest

- **GIVEN** a pipeline step that records a generated file in `files_generated`
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized path SHALL equal `Path(p).as_posix()`
- **AND** it SHALL use forward-slash (`/`) separators regardless of the host platform's native separator

#### Scenario: Metadata path values normalize to POSIX in the manifest

- **GIVEN** a pipeline step that records a path under `metadata` (e.g. `output_csv`, `dashboard_path`, or a relative `Path.relative_to(run_dir)`)
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized path value SHALL equal `Path.as_posix()` of the stored path, preserving its relative-vs-absolute form
- **AND** it SHALL use forward-slash (`/`) separators on every OS

#### Scenario: Optional path values serialize to JSON null

- **GIVEN** a step that records an optional path under `metadata` whose value is `None` (e.g. `reps_plot` when no replicate plot was produced)
- **WHEN** `pipeline_summary.json` is written
- **THEN** the serialized value SHALL be JSON `null`
- **AND** dropping the producer-side `str(path)` SHALL NOT change a `None` value into the string `"None"`

#### Scenario: Top-level and standalone-manifest paths normalize to POSIX

- **GIVEN** the pipeline writes `output_directory` into `pipeline_summary.json` and per-step `*_manifest.json` / `summary.json` files containing `Path` values
- **WHEN** those files are written on any OS
- **THEN** every serialized path SHALL use forward-slash (`/`) separators
- **AND** all serializer sinks (`convert_to_json_serializable`, the `save_json` default hook, and the viz `summary.json` writer) SHALL normalize paths via the same `PurePath.as_posix()` predicate

#### Scenario: Producers do not pre-stringify paths

- **GIVEN** any pipeline step that contributes a path to `files_generated` or `metadata`
- **WHEN** the step records the path
- **THEN** it SHALL store a `Path` object (never `str(path)`)
- **AND** a CI-enforced source guard SHALL fail if a `str(path)` pre-stringification is reintroduced into a step's `files_generated`/`metadata`

#### Scenario: Serialization round-trip gate is green cross-OS

- **GIVEN** the result-object JSON round-trip gate (`serialization-gate` CI job, issue #156) running on ubuntu, windows, and macos
- **WHEN** a `PipelineSummary` carrying `Path` values is serialized and reloaded
- **THEN** it SHALL round-trip to identical POSIX strings on all three platforms
