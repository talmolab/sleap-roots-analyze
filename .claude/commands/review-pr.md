# Review GitHub PR Comments

This command helps review and address GitHub Copilot or human review comments on pull requests.

## Usage

```bash
# List current PRs
gh pr list --author @me

# View PR with comments
gh pr view <PR_NUMBER> --comments

# Get inline code review comments
gh api repos/<OWNER>/<REPO>/pulls/<PR_NUMBER>/comments --jq '.[] | {path: .path, line: .line, body: .body}'

# Get review summaries
gh api repos/<OWNER>/<REPO>/pulls/<PR_NUMBER>/reviews --jq '.[].body'
```

## Example for this repository

```bash
# View PR #2 with all comments
gh pr view 2 --comments

# Get inline comments with file paths and line numbers
gh api repos/talmolab/sleap-roots-analyze/pulls/2/comments --jq '.[] | {path: .path, line: .line, body: .body}'

# Get review summaries
gh api repos/talmolab/sleap-roots-analyze/pulls/2/reviews --jq '.[].body'
```

## Action Plan Template

When addressing PR comments, create an action plan:

1. **List all comments** with their locations
2. **Prioritize** by severity (errors > warnings > suggestions > nitpicks)
3. **Group related changes** (e.g., all import fixes together)
4. **Test after each group** of changes
5. **Document decisions** if not implementing a suggestion

## Categories of Changes

### Critical (Must Fix)
- Duplicate code/imports
- Broken functionality
- Security issues

### Important (Should Fix)
- API inconsistencies
- Misleading documentation
- Code maintainability issues

### Nice to Have (Consider)
- Style improvements
- Performance optimizations (unless critical)
- Additional features

## Responding to Reviews

After addressing comments:

```bash
# Post a comment summarizing changes
gh pr comment <PR_NUMBER> --body "Addressed review comments:
- Fixed duplicate imports
- Updated docstrings
- Made function public
..."

# Request re-review
gh pr review <PR_NUMBER> --request-changes --body "Please review the changes"
```