# config-management Specification

## ADDED Requirements

### Requirement: Optional Replicate Column

The `columns.replicate` config field SHALL be optional. Config validation SHALL
accept `columns.replicate = None` (or an omitted field), and every consumer in
the general trait path SHALL operate correctly when no replicate column is
present, producing identical results to the replicate-present case. The replicate
value SHALL NOT be a term in any statistical model. The hardcoded root-core
`"Rep"` column is a separate field and is out of scope of this requirement.

#### Scenario: Config validation accepts a missing replicate column

- **WHEN** `validate_qc_config()` is run on a config whose `columns.replicate` is
  `None`
- **THEN** validation SHALL NOT report an error for `columns.replicate`
- **AND** a config that still sets `columns.replicate` to a column name SHALL
  continue to validate as before

#### Scenario: Heritability is computed without a replicate column

- **WHEN** `calculate_heritability_estimates` is called with `replicate_col=None`
  on data that has a genotype column and ≥2 rows per genotype but no replicate
  column
- **THEN** it SHALL return heritability (H²) estimates without raising
- **AND** for data that also contains a replicate column, the H² values SHALL be
  identical whether `replicate_col` is set to that column or left `None`

#### Scenario: Trait detection ignores an absent replicate column

- **WHEN** `get_trait_columns` is called with `replicate_col=None`
- **THEN** it SHALL return the numeric trait columns without excluding or
  miscounting any trait on account of a replicate column

#### Scenario: Field root-core Rep column is unaffected

- **WHEN** the root-core (field) aggregation/merge path runs on data containing a
  hardcoded `"Rep"` column
- **THEN** it SHALL aggregate and merge on `"Rep"` identically regardless of the
  value of `columns.replicate`
