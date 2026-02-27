## ADDED Requirements

### Requirement: Git-Anchored Active Configs

Active pipeline configuration files in `configs/active/` SHALL be committed to git before any pipeline run so that the exact configuration used for an analysis can be permanently recovered from version history.

This requirement exists because pipeline outputs (large CSV files, figures) are gitignored, but the configuration that produced them must be reproducible. A git SHA on the config commit is the minimal reproducibility artifact that links a result to its exact analysis parameters.

#### Scenario: Config commit created before pipeline run

- **WHEN** a user finalizes pipeline configs using `/configure-run-all`
- **THEN** the command SHALL commit the config files to git with a descriptive message
- **AND** the commit message SHALL include: the run_name from the manifest, the dataset path, and the ISO date
- **AND** the resulting git SHA SHALL be reported to the user as the reproducibility anchor

#### Scenario: Commit message format

- **WHEN** the git commit is created
- **THEN** the commit message SHALL follow the format:
  ```
  chore: configure analysis "{run_name}" ({date})

  Dataset: {csv_path}
  Config files: {list of committed config paths}
  ```
- **AND** the message SHALL be machine-parseable for future tooling

#### Scenario: Git anchor preserved in config header

- **WHEN** configs are committed and the SHA is known
- **THEN** the run manifest header SHOULD include a comment referencing the commit SHA
- **AND** this allows the manifest itself to document its own reproducibility anchor

#### Scenario: Git commit failure handled gracefully

- **WHEN** a git commit cannot be created (e.g., no changes staged, detached HEAD, repository not initialized)
- **THEN** the system SHALL issue a clear warning to the user explaining that configs are NOT yet anchored to git
- **AND** the system SHALL NOT crash or refuse to write config files
- **AND** the warning SHALL instruct the user to manually run `git add configs/active/ && git commit -m "..."`

---

### Requirement: Backup Before Overwrite

The system SHALL protect existing active configuration files from accidental overwrite by offering timestamped backups before any modification.

This requirement exists because active configs represent scientific decisions. Overwriting one silently destroys the record of those decisions unless they were previously committed to git.

#### Scenario: Backup offered when active config exists

- **WHEN** a user is about to write a new config to a path that already exists in `configs/active/`
- **THEN** the system SHALL detect the existing file
- **AND** the system SHALL offer to back it up to `configs/archive/<original-filename>_backup_<YYYYMMDD_HHMMSS>.yaml`
- **AND** the system SHALL NOT proceed with the overwrite until the user explicitly confirms

#### Scenario: Backup naming is unambiguous

- **WHEN** a backup is created
- **THEN** the backup filename SHALL include the original filename stem, the literal string `_backup_`, and a timestamp in `YYYYMMDD_HHMMSS` format
- **AND** two backups created in the same second SHALL NOT overwrite each other (timestamp resolution is sufficient for interactive use)

#### Scenario: Archive directory is gitignored

- **WHEN** backups are written to `configs/archive/`
- **THEN** the `configs/archive/` directory SHALL be listed in `.gitignore`
- **AND** backup files SHALL NOT be committed to git (they are local safety nets, not reproducibility artifacts)
- **AND** the committed configs in `configs/active/` are the canonical reproducibility artifacts

#### Scenario: No backup needed for new files

- **WHEN** a config path in `configs/active/` does not yet exist
- **THEN** no backup SHALL be created or offered
- **AND** the file SHALL be written directly without prompting
