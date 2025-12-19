## Context

The cross-platform analysis pipeline has a data consistency bug where visualization functions recalculate correlation statistics independently instead of using pre-computed values from the correlation step. This causes:

1. **Different n_genotypes**: Visualization includes genotypes excluded by `min_samples_per_genotype`
2. **Different correlation values**: Computed on different data subsets
3. **Different p-values**: Derived from different sample sizes

This violates the single source of truth principle and can mislead researchers interpreting results.

## Goals / Non-Goals

**Goals:**
- Ensure visualization displays exactly the same values as CSV output
- Maintain backward compatibility for direct API usage of visualization functions
- Follow DRY principle - compute once, display consistently
- Add regression tests to prevent future inconsistencies

**Non-Goals:**
- Change the correlation calculation methodology
- Modify the CSV output format
- Break existing API signatures (parameters are optional)

## Decisions

### Decision 1: Optional parameters with fallback

Add optional correlation parameters to `create_joint_plot` with fallback to recalculation for backward compatibility:

```python
def create_joint_plot(
    exp1_means: pd.DataFrame,
    exp2_means: pd.DataFrame,
    exp1_trait: str,
    exp2_trait: str,
    exp1_name: str = "Experiment 1",
    exp2_name: str = "Experiment 2",
    figsize: Tuple[int, int] = (10, 10),
    color: str = "#4CB391",
    line_color: str = "#2E6E73",
    # NEW: Pre-computed correlation values (single source of truth)
    correlation: Optional[float] = None,
    p_value: Optional[float] = None,
    n_genotypes: Optional[int] = None,
) -> plt.Figure:
```

**Rationale:** This preserves backward compatibility for direct API users while allowing the pipeline to pass authoritative values.

### Decision 2: Pipeline passes pre-computed values

`VisualizeCrossPlatformStep` will extract correlation values from `correlation_df` and pass them to visualization functions:

```python
for i in range(n_joint_plots):
    row = correlation_df.iloc[i]
    trait1 = row["exp1_trait"]
    trait2 = row["exp2_trait"]

    fig = create_joint_plot(
        exp1_means,
        exp2_means,
        trait1,
        trait2,
        exp1_name=exp1_name,
        exp2_name=exp2_name,
        # Pass pre-computed values from CSV
        correlation=row["correlation"],
        p_value=row["p_value"],
        n_genotypes=row["n_genotypes"],
    )
```

**Rationale:** The correlation_df is the authoritative source computed in Step 2. Step 3 should display these values, not recompute them.

### Alternatives Considered

1. **Filter genotype means in visualization step** - Would fix the n_genotypes issue but still recalculates correlations. Violates DRY.

2. **Remove fallback calculation entirely** - Would break backward compatibility for users calling `create_joint_plot` directly without pre-computed values.

3. **Store genotype means in correlation_df** - Bloats the CSV with redundant data.

## Risks / Trade-offs

| Risk | Mitigation |
|------|------------|
| Backward compatibility break | Optional parameters with fallback |
| Inconsistent API behavior | Document when fallback is used |
| Additional parameters clutter | Group as optional kwargs |

## Migration Plan

1. Add optional parameters (no breaking change)
2. Update pipeline to pass values
3. Add deprecation warning when fallback is used (future consideration)

## Open Questions

None - the design is straightforward.
$