## ADDED Requirements

### Requirement: Golden Numerical-Stability Smoke Test

The system SHALL provide a numerical-stability smoke test that asserts the outputs of
the UMAP, clustering, and pandas trait-aggregation paths match committed golden
artifacts within documented tolerances. This gate SHALL be distinct from, and
complementary to, the same-machine determinism sweep: it detects drift introduced by
dependency upgrades (e.g. `numba`, `numpy`, `umap-learn`, `pandas`), which a double-run
determinism check cannot catch because both runs share the upgraded libraries.

#### Scenario: Outputs match committed golden artifacts

- **WHEN** the smoke test runs against the committed reference dataset on the current
  dependency stack
- **THEN** the recomputed UMAP embedding, cluster labels, and trait-summary table SHALL
  each compare equal to their committed golden artifact within the documented tolerance
- **AND** the test SHALL pass

#### Scenario: Silent drift fails the gate with a named artifact

- **WHEN** a dependency upgrade changes a recomputed output beyond the documented
  tolerance for that artifact
- **THEN** the smoke test SHALL fail
- **AND** the failure message SHALL name which artifact drifted (UMAP embedding, cluster
  labels, or trait summary)

### Requirement: Tolerance-Based UMAP Embedding Assertion

The smoke test SHALL assert UMAP embedding stability using Procrustes superimposition
against the golden embedding, and SHALL NOT use raw or bit-exact coordinate equality.
Procrustes is invariant to translation, scale, rotation, and reflection, so it absorbs
BLAS/numba coordinate-frame wobble while still flagging structural change in the
manifold. Because the Procrustes *disparity scalar* alone is insensitive to small but
real per-point drift, the assertion SHALL compare the Procrustes-aligned coordinate
matrices element-wise, not only the disparity scalar.

#### Scenario: Embedding compared on Procrustes-aligned coordinates

- **WHEN** the recomputed embedding is superimposed onto the golden embedding via
  `scipy.spatial.procrustes`
- **THEN** the test SHALL assert the two aligned coordinate matrices match via
  `np.allclose(..., atol=ATOL_PROCRUSTES)`
- **AND** `ATOL_PROCRUSTES` SHALL be a documented value derived from the reference
  dataset's measured same-stack spread (recorded next to the assertion), not a value
  copied without measurement

#### Scenario: Assertion tolerates rigid transforms but not structural drift

- **WHEN** the golden embedding is rigidly transformed (rotated, reflected, translated,
  or uniformly scaled) and re-compared
- **THEN** the Procrustes assertion SHALL still pass
- **AND** **WHEN** the golden embedding is perturbed by a structural change above the
  documented sensitivity floor, the assertion SHALL fail

### Requirement: Tolerance-Based Cluster-Label Assertion

The smoke test SHALL assert cluster-label stability using the Adjusted Rand Index (ARI)
against golden cluster assignments, with a pinned number of clusters. ARI is invariant
to label permutation, so the integer names of clusters do not matter. This realizes the
"compare up to a label permutation" guidance already in the project tolerance policy.

#### Scenario: Cluster labels compared via ARI threshold

- **WHEN** cluster labels are recomputed on the reference dataset with the pinned seed
  and pinned cluster count
- **THEN** the test SHALL compute `adjusted_rand_score(golden_labels, new_labels)`
- **AND** assert it is greater than a documented threshold (initially `0.95`, justified
  against the dataset's measured same-stack ARI)

#### Scenario: Cluster count is honored

- **WHEN** clustering runs with the pinned cluster count `N`
- **THEN** the test SHALL assert the recomputed result reports exactly `N` non-empty
  clusters, guarding against silent internal cluster-count clamping

### Requirement: Tolerance-Based Pandas Trait-Table Assertion

The smoke test SHALL assert pandas trait-table stability using
`pandas.testing.assert_frame_equal` with a tight relative tolerance, and SHALL NOT use
raw `DataFrame` equality, so that pandas Copy-on-Write representation changes do not
cause spurious failures while genuine value drift is still caught. The tight tolerance
is justified because this comparison runs same-stack on a fixed committed input through
pure groupby float arithmetic with no RNG.

#### Scenario: Trait table compared with assert_frame_equal

- **WHEN** the reference per-genotype trait-summary table is recomputed via a
  deterministic groupby/aggregate on the committed input
- **THEN** the test SHALL compare it to the golden table with
  `pd.testing.assert_frame_equal(new, golden, rtol=1e-10)`
- **AND** SHALL NOT rely on raw `==` equality

### Requirement: Pinned Stochastic Configuration

Every stochastic step in the smoke test SHALL use a fixed seed (`random_state=42`), and
the cluster count SHALL be a fixed, recorded value (no silhouette auto-selection), so
that the only source of variation across runs is the dependency stack.

#### Scenario: Stochastic steps are seeded and pinned

- **WHEN** the smoke test computes the UMAP embedding and cluster labels
- **THEN** each stochastic call SHALL pass `random_state=42`
- **AND** the KMeans cluster count SHALL be the pinned recorded value, not silhouette
  auto-selected

### Requirement: Single-OS Execution

The smoke test SHALL execute only on the operating system the golden artifacts were
generated on, because its tolerances are tighter than the cross-OS BLAS floor and the
golden is generated on a single machine. On other operating systems the test SHALL be
skipped (not failed), so the cross-platform test matrix stays green.

#### Scenario: Skipped on non-golden operating systems

- **WHEN** the test suite is collected on an operating system other than the
  golden-generating OS
- **THEN** the smoke test SHALL report as skipped with a reason citing the single-OS
  design
- **AND** SHALL NOT execute its tolerance assertions

#### Scenario: Runs on the golden-generating OS

- **WHEN** the test suite runs on the golden-generating OS
- **THEN** the smoke test SHALL execute its assertions

### Requirement: Committed Golden Artifacts With Provenance

The repository SHALL commit golden artifacts for the smoke test derived from a fixed
reference dataset — a golden UMAP embedding, golden cluster labels, and a golden
trait-summary table — alongside a machine-readable provenance record, stored with the
existing curated fixtures.

#### Scenario: Golden artifacts are present and loadable

- **WHEN** the smoke test runs on the golden-generating OS
- **THEN** it SHALL load committed golden artifacts for the reference dataset (the
  `turface_19` post-QC final-data slice)
- **AND** the three golden artifacts SHALL exist under the curated `expected/` fixtures
  tree

#### Scenario: Provenance record accompanies the golden

- **WHEN** the golden artifacts are regenerated
- **THEN** a provenance record SHALL be written beside them capturing the generating OS,
  the resolved versions of `numpy`, `pandas`, `umap-learn`, `numba`, `scipy`, and
  `scikit-learn`, the seed, the pinned cluster count, and the tolerances
- **SO THAT** a reviewer can determine from a diff whether the golden is stale

### Requirement: Robust Failure Modes

The smoke test SHALL fail loudly and informatively when its inputs are missing or
malformed, rather than erroring opaquely or skipping silently, so that the gate cannot
pass vacuously.

#### Scenario: Missing or unreadable golden or input fails clearly

- **WHEN** a required golden artifact or the input fixture is absent or cannot be parsed
- **THEN** the test SHALL fail with a message naming the missing file and pointing to the
  regeneration script
- **AND** SHALL NOT be reported as passed or silently skipped

#### Scenario: New NaN in a recomputed output is caught

- **WHEN** a recomputed output contains a NaN that the golden does not
- **THEN** the corresponding assertion SHALL fail

### Requirement: CI Gate Integration

The smoke test SHALL run in continuous integration within a dedicated single-OS job,
distinct from the determinism gate, invoked by path, so it can be configured as its own
required status check in branch protection and labeled honestly as a drift detector
rather than a determinism check.

#### Scenario: Smoke test runs in its own gate job

- **WHEN** the CI numerical-stability job runs on a pull request
- **THEN** it SHALL execute the numerical-stability smoke test by path on the
  golden-generating OS
- **AND** a drift failure SHALL fail the job

#### Scenario: Cross-platform test matrix does not fail on the gate

- **WHEN** the cross-platform `tests` matrix collects the smoke test on Windows or macOS
- **THEN** the test SHALL skip there (per Single-OS Execution) and SHALL NOT turn the
  matrix red

### Requirement: Documented Golden Regeneration Procedure

The repository SHALL document how to regenerate the golden artifacts as a single source
of truth, including a regeneration script and explicit guidance on when regeneration is
appropriate, and SHALL reconcile the gate's tolerances against the project's standing
tolerance policy.

#### Scenario: Regeneration guidance distinguishes major from patch bumps

- **WHEN** a maintainer consults the reproducibility documentation
- **THEN** it SHALL describe a script that recomputes all golden artifacts and the
  provenance record from the committed reference input
- **AND** it SHALL state that golden artifacts are regenerated on major
  numba/numpy/umap-learn/pandas bumps that move numbers past tolerance (with reviewer
  approval)
- **AND** it SHALL state that golden artifacts are NOT regenerated on patch bumps that
  stay within tolerance, since that would defeat the gate

#### Scenario: Tolerance choices are reconciled in prose

- **WHEN** the documentation presents the gate's tolerances
- **THEN** it SHALL explain why the trait-table `rtol=1e-10` is tighter than the standing
  `rtol=1e-6` float-array policy (same-stack pure-float on fixed input vs cross-OS BLAS
  reduction), so the two tolerances do not read as a contradiction
