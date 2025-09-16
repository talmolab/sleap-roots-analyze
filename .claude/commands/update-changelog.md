# Update CHANGELOG.md

This command helps maintain the project's CHANGELOG.md file following the Keep a Changelog format.

## Usage

Update the CHANGELOG.md file when:
- Adding new features
- Fixing bugs
- Making breaking changes
- Updating dependencies
- Improving documentation
- Refactoring code

## CHANGELOG Format

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that have been added

### Changed
- Changes to existing functionality

### Fixed
- Bug fixes

### Deprecated
- Features that will be removed in future versions

### Removed
- Features that have been removed

### Security
- Security fixes and improvements

## [0.2.0] - YYYY-MM-DD

### Added
- ...
```

## Categories

Use these categories for organizing changes:

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Vulnerability fixes

## Steps to Update

1. **Check current changes**
```bash
# View recent commits since last tag
git log --oneline $(git describe --tags --abbrev=0)..HEAD

# Or view all uncommitted changes
git diff HEAD
```

2. **Identify change category**
- Is it a new feature? → Added
- Is it a bug fix? → Fixed
- Does it change existing behavior? → Changed
- Does it remove something? → Removed

3. **Write clear descriptions**
- Start with a verb (Added, Fixed, Updated, etc.)
- Be concise but descriptive
- Include PR numbers if applicable
- Reference issues if applicable

## Examples

### Good Examples
```markdown
### Added
- Statistical analysis module with heritability estimation and ANOVA (#2)
- Modular data cleanup functions for zero-inflated and NaN-heavy traits
- Comprehensive test suite achieving 95% coverage

### Fixed
- Duplicate imports in test files causing confusion
- Private function incorrectly exposed in public API
- Line ending consistency issues on Windows

### Changed
- Renamed `link_images_to_samples` to `link_rhizovision_images_to_samples` for clarity
- Made `_convert_to_json_serializable` part of public API
- Added configurable alpha parameter to ANOVA function (default: 0.05)
```

### Poor Examples
```markdown
### Added
- New stuff  # Too vague
- Fixed things  # Wrong category and vague
- Updated code  # Not descriptive
```

## Version Numbering

Follow Semantic Versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: Add functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

## Releasing a Version

When ready to release:

1. **Move Unreleased items to new version**
```markdown
## [Unreleased]
(empty or future items)

## [0.2.0] - 2024-01-04
### Added
- (move items from Unreleased here)
```

2. **Update version in pyproject.toml**
```bash
uv version 0.2.0
```

3. **Commit and tag**
```bash
git add CHANGELOG.md pyproject.toml
git commit -m "Release version 0.2.0"
git tag -a v0.2.0 -m "Release version 0.2.0"
```

## Template for Current PR

Based on recent changes, here's a template for updating the CHANGELOG:

```markdown
## [Unreleased]

### Added
- Statistical analysis module (`statistics.py`) with heritability estimation and ANOVA
- Modular data cleanup functions: `remove_zero_inflated_traits`, `remove_traits_with_many_nans`, `remove_low_sample_traits`
- Comprehensive test suite for statistical functions with known-answer fixtures
- Claude command for reviewing GitHub PR comments (`.claude/commands/review-pr.md`)

### Changed
- Renamed `link_images_to_samples` to `link_rhizovision_images_to_samples` to clarify Rhizovision-specific functionality
- Made `_convert_to_json_serializable` public API by removing underscore prefix
- Added configurable `alpha` parameter to `perform_anova_by_genotype` (default: 0.05)
- Improved test fixtures with mathematically validated expected values

### Fixed
- Duplicate imports in `test_statistics.py`
- Misplaced docstring between test classes
- Brittle test dependency in heritability tests
- Metadata key conflict risk by changing `_metadata` to `__calculation_metadata__`
```

## Best Practices

1. **Update as you go**: Don't wait until release to update the CHANGELOG
2. **Be user-focused**: Write from the user's perspective, not implementation details
3. **Include breaking changes**: Clearly mark any breaking API changes
4. **Credit contributors**: Mention PR authors when applicable
5. **Link to issues/PRs**: Include links for more context

## Quick Command

```bash
# Quick add to Unreleased section
echo "- Your change description" >> docs/CHANGELOG.md
```

Then manually organize into the correct category.