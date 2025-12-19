# Design Document: CLAUDE.md Refactoring

## Context

CLAUDE.md was created before OpenSpec best practices were established. Over time, it accumulated 340+ lines of project guidelines that now duplicate content in `openspec/project.md`. OpenSpec's design expects CLAUDE.md to contain only the managed block, with all project context in `project.md`.

**Stakeholders:**
- AI assistants (Claude Code, etc.) - Need clear, single source of project info
- Project maintainers - Want to update docs in one place
- OpenSpec framework - Expects clean separation of concerns

**Constraints:**
- Must not lose any critical information
- Must maintain backward compatibility for AI assistant workflows
- Must follow OpenSpec conventions from AGENTS.md

## Goals / Non-Goals

**Goals:**
1. Align CLAUDE.md with OpenSpec best practices (managed block only)
2. Ensure single source of truth for project context (project.md)
3. Maintain all critical information (zero information loss)
4. Improve maintainability (update once, not twice)

**Non-Goals:**
1. Change OpenSpec framework behavior
2. Modify AI assistant discovery mechanisms
3. Restructure openspec/specs/ directory
4. Add new project guidelines (just reorganize existing)

## Decisions

### Decision 1: Minimal CLAUDE.md
**What**: Keep ONLY OpenSpec managed block + one reference line

**Why**:
- OpenSpec's `update` command manages CLAUDE.md header
- Custom content below managed block creates conflicts
- AGENTS.md explicitly states this is best practice
- Simplifies maintenance (one file to update)

**Alternatives Considered:**
1. **Keep some guidelines in CLAUDE.md** (e.g., "Quick Start")
   - ❌ Still creates duplication
   - ❌ Will drift from project.md over time
   - ❌ Violates OpenSpec convention

2. **Use CLAUDE.md as index to project.md sections**
   - ❌ Adds layer of indirection
   - ❌ AI assistants already discover project.md
   - ❌ More maintenance overhead

### Decision 2: Content Audit Before Deletion
**What**: Manually compare all 340 lines before removing

**Why**:
- Ensure no unique information is lost
- Some content may have been added to CLAUDE.md only
- Safer than bulk delete + hope

**Process:**
1. Read CLAUDE.md sections
2. Find corresponding project.md sections
3. Note any gaps
4. Add missing content to project.md
5. Then delete from CLAUDE.md

### Decision 3: Add Reference Line to project.md
**What**: After managed block, add: `See openspec/project.md for project guidelines.`

**Why**:
- Explicit pointer for anyone opening CLAUDE.md
- Makes it clear where to find information
- Prevents re-adding content to CLAUDE.md

**Alternative**: Leave completely blank after managed block
- ❌ Less helpful for human readers
- ❌ Might encourage adding content back

## Risks / Trade-offs

### Risk: Information Loss
**Risk**: Accidentally delete unique content from CLAUDE.md

**Mitigation**:
1. Manual audit before deletion (task 1.1-1.3)
2. Git history preserves original content
3. Create list of all moved content in IMPLEMENTATION_SUMMARY.md
4. Review by maintainer before merge

### Risk: AI Assistant Confusion
**Risk**: AI assistants expect guidelines in CLAUDE.md

**Mitigation**:
- OpenSpec framework handles discovery automatically
- AI assistants already read project.md
- Reference line points to correct location
- This aligns with OpenSpec design, not against it

### Trade-off: One-Time Effort vs Long-Term Gain
**Trade-off**: Effort to migrate content now vs maintaining two files forever

**Accepted Because:**
- Duplication creates more work long-term
- Single update is simpler than synchronized updates
- OpenSpec compliance benefits all OpenSpec projects
- Sets good example for future contributors

## Migration Plan

### Phase 1: Audit (Task 1)
1. Open CLAUDE.md and project.md side-by-side
2. For each section in CLAUDE.md (lines 20-361):
   - Find equivalent in project.md
   - Note if missing or different
3. Create markdown list of gaps
4. Estimate: 30 minutes

### Phase 2: Fill Gaps (Task 2)
1. For each identified gap:
   - Add content to appropriate project.md section
   - Preserve formatting and examples
   - Update any stale references
2. Estimate: 15 minutes (expect few/no gaps)

### Phase 3: Simplify CLAUDE.md (Task 3)
1. Delete lines 20-361
2. Add reference line
3. Verify managed block intact
4. Estimate: 5 minutes

### Phase 4: Validate (Task 4)
1. Run `openspec validate --strict`
2. Fix any validation errors
3. Test AI assistant can find info
4. Estimate: 10 minutes

### Rollback Plan
If issues discovered:
1. Revert CLAUDE.md changes (git revert)
2. Keep project.md improvements
3. Re-evaluate approach
4. Git history preserves all original content

## Open Questions

1. **Q**: Are there any custom commands/scripts that parse CLAUDE.md?
   **A**: Unknown - needs check. If yes, update to use project.md instead.

2. **Q**: Should we add a linter to prevent content being added to CLAUDE.md?
   **A**: Out of scope for this change. OpenSpec's `update` command should handle this.

3. **Q**: Do other projects using OpenSpec follow this pattern?
   **A**: Assumed yes based on AGENTS.md guidance, but not verified.

## Success Metrics

1. ✅ CLAUDE.md contains only managed block + reference line (~20 lines total)
2. ✅ All project guidelines accessible via openspec/project.md
3. ✅ Zero information loss (all unique content preserved)
4. ✅ `openspec validate --strict` passes
5. ✅ AI assistant can find all information via OpenSpec discovery