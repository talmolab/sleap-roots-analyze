## Why

The `LoggingConfig` dataclass has `log_file` and `log_to_file` fields that are defined and validated but never actually used by the CLI. Users expect that setting `logging.log_file: "pipeline.log"` in their YAML config will create log files, but logs are only written when using the `--log-file` CLI flag.

## What Changes

- CLI commands (`qc`, `viz`, `cross-platform`) use config's `logging.log_file` as default when `--log-file` CLI flag is not provided
- Log files are saved relative to the **output directory** (not CWD), making paths portable
- CLI `--log-file` flag **overrides** config value when explicitly provided
- Config's `log_to_file: false` is respected as the default, but CLI `--log-file` can override it

**Precedence (CLI overrides config):**
1. `--log-file path` on CLI → use that path (overrides config)
2. `--log-file` not specified + config `log_to_file: true` → use `output_dir / config.logging.log_file`
3. `--log-file` not specified + config `log_to_file: false` → no file logging

## Impact

- Affected specs: New `cli-pipeline` spec (none exists currently)
- Affected code: [cli.py:29-54](src/sleap_roots_analyze/cli.py#L29-L54) - `setup_logging()` and command handlers
- Non-breaking: Existing CLI usage unchanged; new behavior activates config's logging settings