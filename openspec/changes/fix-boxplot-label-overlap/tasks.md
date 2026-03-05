# Tasks: fix-boxplot-label-overlap

## Task 1: Write failing tests -- layout timing (TDD Red Phase)
- [ ] 1.1 Add `test_boxplot_suptitle_not_overlapping` -- create batched boxplots with suptitle; verify suptitle y-position is above the top subplot (no overlap)
- [ ] 1.2 Add `test_boxplot_no_tight_layout_before_suptitle` -- verify `create_trait_boxplots_by_genotype()` does NOT call `tight_layout()` internally (use mock)
- [ ] 1.3 Run tests, confirm FAIL (red) on current code

## Task 2: Write failing tests -- horizontal threshold (TDD Red Phase)
- [ ] 2.1 Add `test_boxplot_horizontal_with_10_genotypes` -- 10 genotypes with default settings; verify horizontal orientation is used
- [ ] 2.2 Add `test_boxplot_vertical_with_7_genotypes` -- 7 genotypes with default settings; verify vertical orientation is used
- [ ] 2.3 Run tests, confirm FAIL on current code (10 genotypes still vertical with threshold=15)

## Task 3: Write failing tests -- adaptive subplot sizing (TDD Red Phase)
- [ ] 3.1 Add `test_boxplot_subplot_width_scales_with_genotypes` -- 20 genotypes; verify figure width is larger than 4 subplots * 4.0 inches
- [ ] 3.2 Add `test_boxplot_label_fontsize_decreases_for_many_genotypes` -- 20 genotypes; verify x-tick label fontsize is smaller than default 10
- [ ] 3.3 Run tests, confirm FAIL on current code

## Task 4: Implement the fixes (TDD Green Phase)
- [ ] 4.1 Remove `plt.tight_layout()` call from `create_trait_boxplots_by_genotype()`
- [ ] 4.2 In `create_trait_boxplots_by_genotype_batched()`, add `fig.tight_layout(rect=[0, 0, 1, 0.96])` after `fig.suptitle()`
- [ ] 4.3 Change `horizontal_threshold` default from 15 to 8 in both functions
- [ ] 4.4 Add adaptive subplot width calculation for vertical orientation: `subplot_width = max(4.0, n_genotypes * 0.5)`
- [ ] 4.5 Add label font size scaling: reduce fontsize for high genotype counts
- [ ] 4.6 Run all new tests, confirm all PASS (green)

## Task 5: Write integration tests (TDD Red->Green)
- [ ] 5.1 Add `test_eda_step_boxplots_readable_with_many_genotypes` -- run ExploratoryAnalysisStep with 20 genotypes; verify figure files exist and dimensions are reasonable
- [ ] 5.2 Add `test_static_figures_boxplots_no_overlap` -- run GenerateStaticFiguresStep with 15 genotypes; verify all figure files generated without error

## Task 6: Visual QA -- generate and inspect figures
- [ ] 6.1 Write a throwaway script that generates boxplot figures for 3 scenarios: (a) 5 genotypes with short IDs, (b) 12 genotypes with medium-length IDs, (c) 25 genotypes with long IDs (e.g., "GENOTYPE_ACCESSION_12345678"). Save each as PNG.
- [ ] 6.2 Open each PNG with the Read tool and visually confirm:
  - No excessive whitespace around or between subplots
  - All axes (x and y) are fully visible and not clipped
  - All genotype labels are readable and not overlapping
  - Boxplots are not vertically or horizontally stretched/squashed
  - Suptitle (batch title) does not overlap the top row of subplots
  - Font sizes are appropriate (not too small to read, not too large)
- [ ] 6.3 For scenario (b) and (c), confirm horizontal orientation is used (genotypes on y-axis)
- [ ] 6.4 For scenario (a), confirm vertical orientation is used (genotypes on x-axis, labels rotated 90 degrees)
- [ ] 6.5 If any visual issue is found, fix the implementation and re-generate until all scenarios pass visual inspection
- [ ] 6.6 Delete the throwaway script after visual QA passes

## Task 7: Verify no regressions
- [ ] 7.1 All existing `TestCreateTraitBoxplots` tests pass unchanged
- [ ] 7.2 All existing `TestCreateTraitBoxplotsBatched` tests pass unchanged
- [ ] 7.3 Full test suite passes (`uv run pytest`)
- [ ] 7.4 Linting and formatting pass (`uv run ruff check`, `uv run black --check`)
