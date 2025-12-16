# Project Documentation Specification

## MODIFIED Requirements

### Requirement: CLAUDE.md Content Structure
The CLAUDE.md file SHALL contain only the OpenSpec managed block and a reference to project.md.

#### Scenario: AI assistant reads CLAUDE.md
- **GIVEN** an AI assistant opens CLAUDE.md
- **WHEN** reading the file content
- **THEN** the file contains:
  1. OpenSpec managed block (lines 1-18)
  2. Single reference line to openspec/project.md
  3. No additional project guidelines or documentation

#### Scenario: OpenSpec update command
- **GIVEN** the OpenSpec update command is run
- **WHEN** updating the CLAUDE.md managed block
- **THEN** no conflicts occur with custom content below the managed block

#### Scenario: Finding project context
- **GIVEN** a developer needs project guidelines
- **WHEN** they read CLAUDE.md
- **THEN** they are directed to openspec/project.md for complete information

---

### Requirement: Project Guidelines Location
All project guidelines SHALL be documented in openspec/project.md, not CLAUDE.md.

#### Scenario: Code style guidelines
- **GIVEN** a developer needs code style information
- **WHEN** they consult openspec/project.md
- **THEN** they find formatting rules, naming conventions, and docstring requirements

#### Scenario: Testing guidelines
- **GIVEN** a developer needs testing information
- **WHEN** they consult openspec/project.md
- **THEN** they find test coverage goals, fixture usage, and testing best practices

#### Scenario: Module documentation
- **GIVEN** a developer needs module interface information
- **WHEN** they consult openspec/project.md
- **THEN** they find module organization and key function descriptions

---

### Requirement: Single Source of Truth
Project documentation SHALL exist in only one location to prevent drift and duplication.

#### Scenario: Updating project guidelines
- **GIVEN** project guidelines need to be updated
- **WHEN** making documentation changes
- **THEN** changes are made only in openspec/project.md, not in CLAUDE.md

#### Scenario: Discovering inconsistencies
- **GIVEN** documentation exists in multiple files
- **WHEN** the files contain conflicting information
- **THEN** this violates the single source of truth requirement

#### Scenario: Maintenance burden
- **GIVEN** documentation is duplicated across files
- **WHEN** updates are needed
- **THEN** maintainers must update multiple locations, increasing error risk