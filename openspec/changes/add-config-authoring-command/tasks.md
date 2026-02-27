## 1. Slash Command Definition

- [x] 1.1 Create `.claude/commands/configure-run-all.md` with full interactive workflow instructions
- [x] 1.2 Define the Q&A sequence: dataset path → column detection → group_by → outlier settings → heritability → PCA/UMAP → image dir → output dir
- [x] 1.3 Define dataset inspection step: read CSV, report sample count, infer column candidates, detect group_by column candidates, flag sparse groups
- [x] 1.4 Define critical parameter review section with statistical rationale for each key setting
- [x] 1.5 Define backup-before-overwrite behavior (timestamped backup to `configs/archive/`)
- [x] 1.6 Define user validation gate: present final configs, highlight critical values, wait for approval
- [x] 1.7 Define git-anchoring step: `git add` configs, commit with analysis metadata in commit message, report SHA to user

## 2. Backup Infrastructure

- [x] 2.1 Document `configs/archive/` as the backup destination in the command
- [x] 2.2 Define backup naming convention: `<original-name>_backup_<YYYYMMDD_HHMMSS>.yaml`
- [x] 2.3 Add `configs/archive/` to `.gitignore` (backups are local-only, not committed)

## 3. Statistical Accuracy Guardrails

- [x] 3.1 Define minimum sample warnings: warn if any group has < 30 samples (Mahalanobis chi-squared requires n≥30)
- [x] 3.2 Define heritability feasibility check: warn if < 3 replicates per genotype or < 3 genotypes in any group
- [x] 3.3 Define recommendation logic for heritability threshold based on sample size and experimental design
- [x] 3.4 Define UMAP n_neighbors recommendation: suggest min(15, n_samples // 4)

## 4. Tests

- [x] 4.1 Test backup naming convention and archive behavior (unit test)
- [x] 4.2 Test statistical guardrail warning thresholds (unit test)
- [x] 4.3 Integration test: simulate full configure-run-all workflow with test fixture data

## 5. Documentation

- [x] 5.1 Add `/configure-run-all` to README or docs as the recommended starting point for new analyses
- [x] 5.2 Update `configs/templates/README.md` to reference the slash command
