# Implementation Tasks

## 1. Audit Current Content
- [ ] 1.1 Compare CLAUDE.md lines 20-361 with openspec/project.md
- [ ] 1.2 Create list of content present in CLAUDE.md but missing from project.md
- [ ] 1.3 Document any unique information that needs preservation

## 2. Update openspec/project.md (if needed)
- [ ] 2.1 Add any missing project context from CLAUDE.md
- [ ] 2.2 Add any missing code conventions from CLAUDE.md
- [ ] 2.3 Add any missing troubleshooting info from CLAUDE.md
- [ ] 2.4 Verify all references and links are correct

## 3. Simplify CLAUDE.md
- [ ] 3.1 Remove lines 20-361 (all content after OpenSpec managed block)
- [ ] 3.2 Add single line referencing openspec/project.md
- [ ] 3.3 Verify OpenSpec managed block (lines 1-18) remains intact

## 4. Validation
- [ ] 4.1 Run `openspec validate --strict`
- [ ] 4.2 Verify no validation errors
- [ ] 4.3 Confirm AI can discover project.md via OpenSpec

## 5. Documentation
- [ ] 5.1 Update IMPLEMENTATION_SUMMARY.md with changes made
- [ ] 5.2 Document any content moved from CLAUDE.md to project.md