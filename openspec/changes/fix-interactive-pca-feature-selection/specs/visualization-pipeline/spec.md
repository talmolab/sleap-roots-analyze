## ADDED Requirements

### Requirement: Interactive PCA Loading Feature Selection

`create_interactive_pca_with_images` and `create_interactive_pca_plot` SHALL
select the features shown as loading arrows (when `show_loadings=True`) via
`select_top_features_from_pca()`, using a `feature_selection` parameter that
accepts the same method names as `create_pca_biplot`'s `feature_selection`
parameter, rather than a hardcoded ranking.

#### Scenario: Default feature_selection matches prior hardcoded behavior

- **WHEN** `create_interactive_pca_with_images` or `create_interactive_pca_plot`
  is called with `show_loadings=True` and no `feature_selection` argument
- **THEN** the function SHALL default `feature_selection` to `"top_variance"`
- **AND** the selected loading-arrow features SHALL match those previously
  produced by `feature_contributions.nlargest(n_loadings, "total_contribution")`

#### Scenario: Alternate feature_selection method changes selected features

- **WHEN** `create_interactive_pca_with_images` or `create_interactive_pca_plot`
  is called with `show_loadings=True` and `feature_selection` set to
  `"extreme"`, `"top_absolute"`, `"top_contribution"`, or `"vector_length"`
- **THEN** the selected loading-arrow features SHALL exactly match the
  indices returned by directly calling `select_top_features_from_pca()`
  with the same `loadings`, `eigenvalues`, `n_features_to_select=n_loadings`,
  `method=feature_selection`, and `pc_indices=[pc_x, pc_y]`

#### Scenario: PC indices passed to selection are not re-adjusted

- **WHEN** `create_interactive_pca_with_images` or `create_interactive_pca_plot`
  is called with `components=(pc_x, pc_y)` (already 0-indexed, as used
  elsewhere in these functions) and a `feature_selection` other than
  `"top_variance"`
- **THEN** `select_top_features_from_pca()` SHALL be called with
  `pc_indices=[pc_x, pc_y]` unmodified — no additional offset SHALL be
  applied, unlike `create_pca_biplot`'s 1-indexed-to-0-indexed conversion,
  which does not apply here

#### Scenario: Unrecognized feature_selection value rejected regardless of show_loadings

- **WHEN** `create_interactive_pca_with_images` or `create_interactive_pca_plot`
  is called with a `feature_selection` value not in `VALID_SELECTION_METHODS`
- **THEN** the function SHALL raise a `ValueError` naming the invalid value
  and listing the valid options
- **AND** this validation SHALL occur regardless of whether `show_loadings`
  is `True` or `False`
