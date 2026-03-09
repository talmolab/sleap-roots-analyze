# Tasks: unify-boxplot-style

## Task 1: Write failing test for consistent boxplot style (TDD Red Phase)
- [x] 1.1 Add `test_boxplot_horizontal_uses_unfilled_style` — generate horizontal boxplot with 12 genotypes; verify no filled patches exist (seaborn adds filled PathPatch, matplotlib default does not)
- [x] 1.2 Add `test_boxplot_vertical_and_horizontal_same_style` — generate both orientations; verify both produce matplotlib-style unfilled boxes (no filled patches)
- [x] 1.3 Run tests, confirm FAIL on current code (horizontal produces seaborn filled boxes)

## Task 2: Implement the fix (TDD Green Phase)
- [x] 2.1 Replace `sns.boxplot()` horizontal code path with `ax.boxplot(..., orientation="horizontal")` in `create_trait_boxplots_by_genotype()`
- [x] 2.2 Group data by genotype to produce per-genotype arrays for `ax.boxplot()`
- [x] 2.3 Set y-tick labels to genotype names via `tick_labels` parameter
- [x] 2.4 Match `df.boxplot()` style: blue box/whisker outlines, green medians, black caps, gridlines
- [x] 2.5 Remove unused local `import seaborn as sns` from the function
- [x] 2.6 Run new tests, confirm PASS (green)

## Task 3: Verify no regressions
- [x] 3.1 All existing boxplot tests pass unchanged
- [x] 3.2 Full visualization test suite passes (168 tests)
- [x] 3.3 Linting and formatting pass (`uv run ruff check`, `uv run black --check`)

## Task 4: Visual QA
- [x] 4.1 Generate boxplot PNGs for 5, 8, 12, and 25 genotype scenarios
- [x] 4.2 Verify both orientations produce consistent style (blue outlines, green medians, gridlines)
- [x] 4.3 Verify genotype labels are readable and correctly positioned
- [x] 4.4 Visual QA approved by user
- [x] 4.5 Clean up QA artifacts
