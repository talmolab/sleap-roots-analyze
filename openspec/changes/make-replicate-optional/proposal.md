# Make columns.replicate optional

## Why

`columns.replicate` is currently a **required** config field, but its values are
**never used in any computation**. A four-agent adversarial sweep of every
consumer (~26 files) found no `groupby`/`pivot`/aggregation/dedup/merge keyed on
the config replicate column. Heritability — its only statistical consumer — fits
`value ~ 1 + (1|genotype)` (`statistics.py`) and weights by `mean_n_reps`
(rows per genotype), neither of which reads the replicate values.

Requiring the field is a misfit for datasets with no replicate factor — e.g.
Bloom cylinder data, where the plant/`barcode` is the replicate unit and there is
no separate block column. Making it optional is the upstream prerequisite for the
bloom-mcp **analysis-input contract** (`talmolab/sleap-roots-contracts#3`), which
needs `replicate` genuinely optional. Removing it **changes zero numbers**; the
code only *assumes the column is present* at a few access points (which `KeyError`,
not miscompute).

Closes #142.

## What Changes

- **Config validation** (`pipeline/config/utils.py`): stop rejecting
  `columns.replicate = None`. The type is already `Optional[str]`; only the
  required-field check forbids `None` today.
- **Heritability** (`statistics.py` `calculate_heritability_estimates`): drop
  `replicate_col` from `required_cols` and the `dropna` subset when it is `None`,
  and rename the subset columns accordingly. Gating on "≥2 rows per genotype"
  (already how `mean_n_reps` works) is unchanged.
- **`get_trait_columns`** (`data_cleanup.py`): already guards the exclusion with
  `if replicate_col:` — add a regression test confirming `replicate_col=None`
  does not miscount a trait.
- **Public diagnostics robustness** (`statistics.py` `analyze_trait_variance`):
  accept `replicate_col=None` so the public diagnostic helpers
  (`diagnose_heritability_issues`, `compare_trait_heritabilities`) stay usable on
  replicate-free data. (Not in the pipeline path, but shares the same contract.)
- **Docs**: state in `ColumnConfig.replicate` and config-authoring examples that
  replicate is optional and is not a model term.

### Explicitly NOT changed

The field / root-core `"Rep"` column is a **separate, hardcoded** column from the
root-core schema (`aggregate_cores.py`, `qc_core_level.py`,
`merge_all_traits.py` `join_keys`), **not** `columns.replicate`. It IS
load-bearing for field aggregation/merge and must remain untouched.

## Impact

- Affected specs: `config-management` (new requirement: Optional Replicate Column).
- Affected code: `pipeline/config/utils.py`, `statistics.py`, `data_cleanup.py`
  (test only), `pipeline/config/components.py` (docstring), config example docs.
- Mostly backward compatible: existing configs that set `replicate` explicitly
  keep working identically. **Default-behavior change:** `ColumnConfig.replicate`
  now defaults to `None` (was `"rep"`), so omitting the key disables replicate
  instead of silently assuming a `"rep"` column. No shipped config relies on the
  old implicit default (every replicate-bearing config sets it explicitly, and
  `configs/qc_cylinder_edpie.yaml` already omits it); a stray `rep` column under
  the old default would only have been reclassified as a trait, never miscomputed.
- The four golden QC templates (`configs/templates/qc_*`) now present `replicate`
  as optional (commented out / omittable) rather than a REQUIRED field.
