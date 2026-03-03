## Context

The `/configure-run-all` command is a Claude Code slash command (a `.md` file under `.claude/commands/`) that runs interactively in the Claude Code CLI. It does not modify Python source code—it is pure instruction text that tells Claude how to conduct the interactive session with the user.

The existing `/run-pipelines` skill runs the pipeline. This new command is its companion: "configure first, then run."

Key stakeholder concern: **reproducibility**. A plant biologist who runs this command in 2026 must be able to exactly reproduce the analysis in 2028. The git SHA anchor on committed configs is the mechanism for this.

## Goals / Non-Goals

**Goals:**
- Guide users through creating scientifically sound configs without requiring deep knowledge of every parameter
- Prevent silent statistical errors (wrong column names, underpowered heritability, too-small groups)
- Ensure configs are committed to git before any pipeline runs
- Protect existing active configs from accidental overwrites
- Produce configs that are self-documenting (headers explain choices)

**Non-Goals:**
- Replacing the YAML config format or config schema
- Automatically running the pipeline (user must explicitly invoke `/run-pipelines` after)
- Validating that the CSV data itself is scientifically correct
- Generating cross-platform configs (out of scope for v1; add later)

## Decisions

### Decision: Slash command as `.md` instruction file, not Python code

The interactive workflow is implemented as a Claude Code slash command definition (`.claude/commands/configure-run-all.md`), not as a Python CLI subcommand. This keeps all interactivity in the AI layer where it belongs and avoids a complex Python wizard implementation.

**Alternative considered:** Python `sleap-roots-analyze configure` CLI wizard. Rejected because: (1) requires rich TUI library (questionary/click), (2) harder to add nuanced statistical reasoning, (3) Claude can inspect CSVs and make context-aware recommendations that a script cannot.

### Decision: git commit after config write, not before

Configs are committed AFTER the user reviews and approves them. The commit message includes the run_name, dataset path, and git SHA of the input data (if available). This preserves the exact analysis intent.

**Alternative considered:** commit before showing user. Rejected because: user may reject after seeing the full config, resulting in a noisy commit history.

### Decision: Backup to `configs/archive/`, gitignored

Backups go to `configs/archive/` which is gitignored. Backups are local safety nets, not reproducibility artifacts. The committed configs in `configs/active/` are the reproducibility artifacts.

**Alternative considered:** Backup to a git-committed branch. Rejected because: overly complex for what is a simple "undo" mechanism.

### Decision: Warn, not block, on statistical feasibility issues

When a group has < 30 samples (Mahalanobis chi-squared reliability threshold) or < 3 replicates per genotype (heritability estimation requirement), the command warns the user and asks them to confirm before proceeding. It does NOT refuse to write the config—the user may have domain knowledge that overrides the warning (e.g., using a non-chi-squared Mahalanobis variant).

## Risks / Trade-offs

- **Risk**: Users skip reading the critical parameter review section. **Mitigation**: The command asks the user to explicitly confirm each high-stakes parameter.
- **Risk**: Dataset inspection fails on unusual CSV formats. **Mitigation**: Inspection is best-effort; the command falls back to asking the user to supply values manually.
- **Risk**: git commit fails (no git repo, dirty state). **Mitigation**: Treat git anchoring as strongly recommended but not blocking; warn clearly if it fails.

## Open Questions

- Should cross-platform config creation be in scope for v1 or deferred? (Current recommendation: defer—it requires knowing which two QC outputs to pair, which is a separate decision tree.)
- Should the command support creating multiple QC configs in one session (e.g., one per experiment platform)? (Current recommendation: one analysis at a time for simplicity.)
