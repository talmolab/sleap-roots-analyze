# Refactor CLAUDE.md to OpenSpec Structure

## Why

CLAUDE.md currently contains 340+ lines of project guidelines that duplicate content already in `openspec/project.md`. This violates OpenSpec best practices, which specify that CLAUDE.md should contain ONLY the OpenSpec managed block (lines 1-18).

**Current Problems:**
1. **Duplication**: Project overview, code style, testing guidelines, module docs all duplicated between CLAUDE.md and project.md
2. **Maintenance burden**: Updates must be made in two places, leading to drift
3. **Violates OpenSpec convention**: Managed blocks should not be extended with custom content
4. **Confusing for AI assistants**: Two sources of truth for the same information

**OpenSpec Best Practice** (from `openspec/AGENTS.md`):
- `CLAUDE.md` = OpenSpec managed block ONLY (updated by `openspec update`)
- `openspec/project.md` = ALL project context, conventions, and guidelines
- `openspec/specs/` = Requirements and system behavior

## What Changes

- **BREAKING**: Remove 340+ lines of content from CLAUDE.md (lines 20-361)
- Simplify CLAUDE.md to contain only:
  1. OpenSpec managed block (lines 1-18)
  2. Single reference line directing to `openspec/project.md`
- Audit `openspec/project.md` to ensure no information loss
- Update any missing content in `openspec/project.md` if needed

## Impact

**Affected files:**
- `CLAUDE.md` - Major reduction (361 lines → ~20 lines)
- `openspec/project.md` - Potential minor additions if gaps found

**Affected specs:**
- None directly affected
- This is purely a documentation structure change

**Benefits:**
1. ✅ **Single source of truth**: All project info in `openspec/project.md`
2. ✅ **Maintainability**: Update once, not twice
3. ✅ **OpenSpec compliance**: Follows best practices from `AGENTS.md`
4. ✅ **Clearer separation**: Instructions (AGENTS.md) vs Context (project.md) vs Requirements (specs/)
5. ✅ **Automatic updates**: `openspec update` can manage CLAUDE.md without conflicts

**Migration:**
- No code changes required
- AI assistants will use `openspec/project.md` instead of CLAUDE.md for context
- Existing workflows unaffected (OpenSpec handles file discovery)

**Validation:**
- Ensure `openspec/project.md` contains all critical information from CLAUDE.md
- Run `openspec validate --strict` to verify structure
- Test AI assistant can find all information via OpenSpec discovery
