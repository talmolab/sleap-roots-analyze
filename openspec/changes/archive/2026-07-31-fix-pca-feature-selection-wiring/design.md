## Context

Issue #202 documents two independent bugs where `feature_selection`
(sourced from `pca.feature_selection_strategy` in config) fails to actually
control which features get selected/plotted. Both live in
`visualization.py` and both were introduced as incomplete follow-through on
earlier work (`create_pca_biplot`'s mapping block was rewritten in PR #37
after `top_variance` already existed elsewhere; `create_feature_contribution_plot`'s
parameter was added in the same commit as an unused
`select_top_features_from_pca` import, per the issue's git archaeology).

## Gotcha: `select_top_features_from_pca`'s `top_variance` method ignores `pc_indices`

`pca.py:105-113`:

```python
elif method == "top_variance":
    # Use total variance contribution across all available PCs
    n_pcs = min(loadings.shape[1], len(eigenvalues))
    contributions = np.zeros(n_features)

    for i in range(n_pcs):
        contributions += eigenvalues[i] * loadings[:n_features, i] ** 2

    return np.argsort(contributions)[::-1][:n_features_to_select].tolist()
```

This loops over `range(n_pcs)` — every retained PC — regardless of whatever
`pc_indices` argument was passed. Verified empirically: calling this method
with `pc_indices=[0, 1]` explicit vs. omitted (defaults to `[0, 1]` inside
the function) produces identical output either way, because the
`top_variance` branch never reads `pc_indices` at all. The shared docstring
for `pc_indices` (`pca.py:36`) doesn't call this out explicitly; only the
`top_variance` bullet's parenthetical "(all PCs)" hints at it.

`create_umap_colored_by_top_traits` already has a correct workaround for
this (`visualization.py:2776-2783`):

```python
top_indices = select_top_features_from_pca(
    ...
    method=feature_selection,
    pc_indices=pc_indices if feature_selection != "top_variance" else None,
)
```

**Decision**: `create_pca_biplot`'s fix must apply the same workaround —
when `method == "top_variance"`, pass `pc_indices=None` rather than
`[pc_x_idx, pc_y_idx]`. Passing the biplot's 2 displayed PC indices would
silently no-op (the method ignores them anyway) and produce no visible
error, but it would misleadingly suggest the biplot's 2-PC scope is
respected when it isn't. `pc_indices=None` makes the "ranked across all
retained PCs, not just the 2 shown" behavior explicit at the call site,
matching what actually happens.

This also means `top_variance` and `top_contribution` are related but
distinct: `top_variance` is mathematically `top_contribution` summed over
*all* retained PCs rather than an arbitrary subset — they only coincide
when a run retains exactly 2 PCs total (verified: with >2 retained PCs they
select different features). `top_variance` is also not equivalent to
`vector_length` (eigenvalue-weighted variance contribution vs. unweighted
Euclidean norm in the PC plane) — confirmed empirically to select different
feature sets — so the previous silent `vector_length` fallback was a real
behavior change, not a harmless substitution between equivalent methods.

## Decision: remove, don't wire up, `create_feature_contribution_plot`'s `feature_selection`

Two options were considered for `create_feature_contribution_plot`:

1. **Wire it up**: call `select_top_features_from_pca(method=feature_selection, ...)`
   to choose `top_traits`, keep displaying true per-PC variance-contribution
   bars for whichever traits get picked, and make the title dynamic (e.g.
   `f"Top {n} Features by {feature_selection}: Variance Contribution"`) so
   it doesn't misdescribe non-contribution-selected content.
2. **Remove the parameter**: hardcode variance-based selection, delegating
   to `select_top_features_from_pca(method="top_variance", pc_indices=None, ...)`
   instead of duplicating the ranking formula inline.

**Chosen: option 2.** The chart's bars always plot true eigenvalue-weighted
variance contribution regardless of which traits are shown — that's the one
thing this chart visualizes. Selecting traits by a different criterion
(e.g. `extreme`, which surfaces traits that dominate a PC's *direction* but
can contribute little to that PC's *variance* if its eigenvalue is small)
would make the unconditional title
(`f"Top {n} Feature Contributions to First {k} PCs"`, asserting displayed
traits *are* the top contributors) false. There's also direct in-repo
precedent: `create_feature_contribution_heatmap`
(`visualization.py:3114-3203`) — same purpose, same era, same author — has
no `feature_selection` parameter at all; it hardcodes top-by-contribution
selection. That is this codebase's own prior answer to "should a
*contribution* chart accept a non-contribution selection criterion?", and
the answer is no. Option 1 is not ruled out permanently — if a genuine
cross-check use case appears (e.g. in a notebook or an explicit feature
request), it can be revisited then — but nothing today asks for it, and
inventing it now would be scope creep beyond #202.

## Non-goals

- Not touching `create_feature_contribution_heatmap` — it already has the
  correct, parameter-free behavior this change moves
  `create_feature_contribution_plot` toward; no fix needed there.
- Not changing `PCAAnalysisStep`'s hardcoded `pc_indices=[0, 1]` (#203) — a
  different call site with its own scoping question.
- Not redesigning `n_top_features` semantics (#206) or fixing the UMAP
  top-traits PC1-negative-only bug (#207).
