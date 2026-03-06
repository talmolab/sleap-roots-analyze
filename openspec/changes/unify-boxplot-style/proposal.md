## Why
Horizontal boxplots use seaborn (`sns.boxplot`) which renders filled blue boxes with orange medians and no gridlines, while vertical boxplots use matplotlib (`df.boxplot`) which renders unfilled outlined boxes with blue outlines, green medians, and gridlines. This visual inconsistency is confusing when comparing plots across orientations.

## What Changes
- Replace `sns.boxplot()` horizontal code path in `create_trait_boxplots_by_genotype()` with `ax.boxplot(..., orientation="horizontal")` to produce unfilled outlined boxes
- Match `df.boxplot()` style: blue box/whisker outlines (`#1f77b4`), green medians (`#2ca02c`), black caps, and gridlines enabled
- Group data per genotype manually since `ax.boxplot()` requires pre-grouped arrays
- Remove unused local `import seaborn as sns` from the function
- Preserve all existing behavior: axis labels, title, genotype ordering

## Impact
- Affected specs: `visualization-pipeline` (modifies boxplot readability requirement)
- Affected code: `src/sleap_roots_analyze/visualization.py` (~15 lines changed in horizontal branch)
- No API changes — function signatures and return types unchanged
