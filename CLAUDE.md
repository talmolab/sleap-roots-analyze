<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open `@/openspec/AGENTS.md` when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use `@/openspec/AGENTS.md` to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

For project context, guidelines, and conventions, see **[openspec/project.md](openspec/project.md)**.

## GitHub CLI

When using `gh` commands, always prefix with `unset GITHUB_TOKEN` to avoid token lifetime restrictions from the `talmolab` organization:

```bash
unset GITHUB_TOKEN && gh pr create ...
```

This ensures `gh` falls back to its own auth rather than the environment variable.