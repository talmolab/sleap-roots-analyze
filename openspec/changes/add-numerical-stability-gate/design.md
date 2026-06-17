## Context

`docs/reproducibility.md` already defines our seeding policy, tolerance policy, and the
two CI gates (determinism + serialization). The determinism gate verifies "same seed +
same environment → identical output" by double-running each function on one machine.
The gap this change fills is orthogonal: a **library-upgrade drift detector** that pins
output to a committed ground truth. Because both halves of a double-run share the
upgraded libraries, only a golden-vs-committed comparison can catch silent drift.

The reference dataset, fixture layout, tolerance vocabulary, and CI-gate convention all
already exist — this change reuses them rather than inventing new ones.

## Goals / Non-Goals

- **Goals**
  - Catch numerical drift introduced by `numba` / `numpy` / `umap-learn` / `pandas`
    upgrades in the UMAP, clustering, and pandas-aggregation paths.
  - Be robust to BLAS / numba floating-point reordering (no flaky failures on patch
    bumps) while still failing on real structural change.
  - Reuse the existing `turface_19` fixtures, the `reproducibility-gates` job, and the
    documented tolerance policy.
- **Non-Goals**
  - Bit-exact cross-OS reproduction of UMAP (explicitly impossible; `docs/reproducibility.md:78-91`).
  - Replacing the determinism sweep or the pipeline-reproduction golden tests.
  - Pinning a BLAS backend (only revisit if the gate proves flaky at `1e-6`).
  - Cross-OS matrix execution (golden is generated on one machine; start single-OS like
    the existing determinism gate).

## Decisions

- **Decision: New standalone test file, single-OS-enforced, run in a dedicated CI job.**
  `tests/test_numerical_stability.py`, fast (turface_19 is ~153 rows). It carries a
  module-level `pytest.mark.skipif(sys.platform != "linux")` so the cross-platform
  `tests` matrix (`-m "not integration" tests/`, ubuntu/windows/mac) *collects and skips*
  it — keeping the matrix green — while a dedicated single-OS `numerical-stability` CI
  job runs it by path on ubuntu.
  - *Critical correction (review finding B1):* an earlier draft left the test unmarked
    and claimed "single-OS" only by virtue of the gate job. That was wrong: an unmarked
    file in `tests/` is also collected by the cross-OS matrix, where the golden (generated
    on ubuntu) would fail at `rtol=1e-10` / Procrustes `atol` against Windows-MKL /
    macOS-Accelerate float reordering. The `skipif` is what actually makes single-OS real.
  - *Alternative considered:* `tests/integration/...` per the upstream issue's wording, or
    `--ignore` in the matrix job. Rejected — `integration`-marked tests are excluded from
    CI by design (#69), and `--ignore` is a brittle path string; `skipif` co-locates the
    constraint with the assertions and self-documents.
  - *Alternative considered:* folding into the existing `reproducibility-gates` job.
    Rejected — that job's name and comment say "determinism (same-machine double-run)";
    this is a drift detector with a different single-OS rationale. A dedicated job mirrors
    the existing `serialization-gate` precedent and gets its own required status check.

- **Decision: Reference dataset = `turface_19` post-QC final data.**
  Already shipped as a golden fixture (PR #146), small and deterministic, and exercises
  the real trait columns. Input:
  `tests/fixtures/real/wheat_edpie/inputs/post_qc/turface_19_final_data.csv`.
  - *Alternative considered:* a synthetic matrix. Rejected — would not exercise the real
    trait distributions the production paths see.

- **Decision: UMAP stability via Procrustes on the aligned matrices, not the disparity
  scalar.** `scipy.spatial.procrustes(golden, new)` superimposes the two embeddings
  (invariant to translation, uniform scale, rotation, reflection — exactly the wobble a
  BLAS/numba change introduces) and returns `(aligned_golden, aligned_new, disparity)`.
  Assert `np.allclose(aligned_golden, aligned_new, atol=ATOL_PROCRUSTES)`.
  - *Critical correction (review finding B2):* asserting only `disparity ~ 0` is too
    insensitive — on `turface_19` the same-stack disparity is ≈ `4.8e-32`, yet a real
    `1e-3` single-coordinate nudge yields disparity only ≈ `7e-10`, *still under* a naive
    `atol=1e-6`. Comparing the aligned coordinate matrices element-wise restores
    sensitivity. `ATOL_PROCRUSTES` is derived from the measured same-stack aligned-delta
    × a documented safety factor (task 1.2), not copied from the issue.

- **Decision: Cluster-label stability via Adjusted Rand Index, with the cluster count
  pinned and asserted.** Run `clustering.perform_kmeans_clustering` with
  `random_state=42` and `n_clusters=3` (pinned, no silhouette auto-selection), then
  `sklearn.metrics.adjusted_rand_score(golden_labels, new_labels)`. ARI is
  permutation-invariant. Measured same-stack ARI on `turface_19` is `1.0` (cluster sizes
  `[32, 86, 35]`), so `ARI > 0.95` has headroom. The test additionally asserts the
  recomputed `result["n_clusters"] == 3`, because `clustering.py:99` silently clamps to
  `max(2, len(df)//10)` — for 153 rows that is 15, so 3 is honored today, but the
  assertion prevents a future dataset/`k` change from making the golden a lie. The
  groupby trait summary is computed on **raw** values; KMeans/UMAP standardize internally.

- **Decision: Pandas trait table = per-genotype trait summary, compared whole.**
  Compute a deterministic `groupby("Genotype")[traits].agg(["mean", "std"])` on the
  final-data table — the exact groupby/aggregate path pandas CoW changes — and compare
  the full frame to a golden CSV with `pd.testing.assert_frame_equal(new, golden,
  rtol=1e-10, check_like=False)`. Pure float arithmetic on fixed input, so a tight
  `rtol` is safe; `assert_frame_equal` (not raw `==`) tolerates CoW representation
  churn.

- **Decision: One regeneration script + provenance sidecar + single-source policy.**
  `scripts/regenerate_numerical_stability_golden.py` recomputes all three golden
  artifacts plus `golden_provenance.json` (generating OS + resolved
  numpy/pandas/umap-learn/numba/scipy/scikit-learn versions + seed + n_clusters +
  tolerances, stamped via `importlib.metadata`) and writes them under
  `tests/fixtures/real/wheat_edpie/expected/numerical_stability/`. The provenance record
  makes "is the golden stale?" a reviewable diff rather than a guess. The
  regenerate-when policy is documented as a *single source of truth* by extending the
  existing `tests/fixtures/README.md:100-110` policy (not forking a second one):
  regenerate on **major** numba/numpy/umap-learn/pandas bumps past tolerance with
  reviewer sign-off; do **not** regenerate on patch bumps within tolerance — that defeats
  the gate. `docs/reproducibility.md` links to this policy and additionally reconciles
  the gate's `rtol=1e-10` against the standing `rtol=1e-6` float-array policy
  (same-stack pure-float on fixed input vs cross-OS BLAS reduction).

## Risks / Trade-offs

- **Procrustes/ARI thresholds too tight → flaky on patch bumps.** Mitigation: seed the
  thresholds from same-stack re-run spread, document the observed margin, and only widen
  with justification. Start at `atol=1e-6` / `ARI>0.95` per the issue.
- **Golden regenerated carelessly → gate rubber-stamps drift.** Mitigation: regeneration
  requires reviewer approval and a note on *why* numbers moved, per existing fixture
  policy; the script is separate from the test so regeneration is a deliberate act.
- **Single-OS only → misses a drift that appears solely on macOS/Windows.** Accepted: the
  gate targets *library-version* drift, which is OS-independent. The `skipif` makes this
  real (vs the earlier draft that would have run cross-OS with single-OS tolerances and
  gone red — finding B1). OS-matrix promotion is a future option if warranted, but would
  require per-OS goldens.
- **Gate passes vacuously (golden missing, or compares golden to itself).** Mitigation:
  Robust Failure Modes requirement + the anti-tautology negative control (task 4.5) that
  perturbs the *recompute* path, not just the golden.

## Migration Plan

Additive only — no behavior change to existing code or specs. Commit order keeps CI green
at every step (the golden must precede the test, the test must precede the CI job):
(1) measure spread + add the regen script and OpenSpec dir; (2) commit golden artifacts +
provenance (commit body records the regen command and resolved dep versions);
(3) add the test (with `skipif`); (4) add the dedicated CI job; (5) docs. The OpenSpec
change ships in this PR and is archived in a separate `chore:` PR post-merge per repo
convention. Branch: open a local tracking issue mirroring bloom#141 →
`evelyn/issue-<N>-numerical-stability-gate` (do not reuse `141`, which is a different
local issue). Rollback is deletion of the new files and the one CI job.

## Open Questions

- `ATOL_PROCRUSTES` and the ARI floor are *derived* in task 1.2 from the measured
  same-stack spread (≈ `4.8e-32` disparity, ARI `1.0`) — the exact constants land with
  the implementation, not the proposal.
- Whether the existing `expected/viz/turface_19/viz_umap_embedding.csv` is the same
  compute path as this gate's embedding (reuse it) or differs enough to justify a
  distinct golden — resolved in task 3.1 by comparing feature columns / standardization;
  do not commit an unexplained near-duplicate.
- Whether to add kNN-overlap as a second structural UMAP check — deferred; the
  aligned-matrix Procrustes assertion (not the disparity scalar) is sensitive enough for
  v1.
