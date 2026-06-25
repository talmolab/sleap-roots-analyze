## Why

Our dependencies are fully pinned via `uv.lock` against bleeding-edge scientific
packages (`numpy>=2.3.2`, `pandas>=2.3.2` heading toward the Copy-on-Write default,
`umap-learn` riding the numba/BLAS JIT stack). The reproducibility gate we already
ship — `tests/test_reproducibility.py` — is a **same-machine double-run**: it calls
each stochastic function twice with `random_state=42` and asserts the two runs agree.
That proves *determinism*, but it is structurally blind to *drift*: when a numba /
numpy minor bump or the pandas CoW switch silently changes a result, **both** runs
move together and the gate still passes (see `docs/reproducibility.md:47-57`).

Nothing in the suite pins our numerical output to a known-good **ground truth**.
For UMAP the golden-fixture test asserts only embedding *shape and finiteness*
(`tests/test_pipeline_reproduction.py:266-275`), never coordinate values; no golden
cluster labels are committed at all; and pandas trait tables are checked only at the
scalar-summary level, never cell-by-cell. A dependency bump could silently rotate the
UMAP manifold, reshuffle clusters, or flip a pandas groupby/aggregate path with nobody
noticing. This mirrors upstream issue
[Salk-Harnessing-Plants-Initiative/bloom#141](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/141).

## What Changes

- Add a new **golden-vs-committed numerical-stability smoke test**
  (`tests/test_numerical_stability.py`) that asserts the numerical outputs of the
  UMAP, clustering, and pandas trait-aggregation paths match committed golden
  artifacts within documented tolerances — complementing, not replacing, the existing
  determinism sweep.
- Use **tolerance-based, not bit-exact** assertions, chosen to absorb BLAS / numba
  floating-point wobble while still catching real structural drift, with every threshold
  **derived from the reference dataset's measured same-stack spread** (recorded next to
  the assertion), not copied unmeasured:
  - **UMAP embedding** — Procrustes superimposition against the golden embedding, then
    `np.allclose` on the **aligned coordinate matrices** (not just the disparity scalar,
    which is too insensitive to flag small real drift). Procrustes is
    translation/rotation/reflection-invariant, so coordinate-frame wobble does not flag.
  - **Cluster labels** — Adjusted Rand Index (ARI) against golden assignments,
    asserted `> 0.95` (justified against the dataset's measured same-stack ARI), with the
    recomputed cluster count asserted equal to the pinned value.
  - **Pandas trait table** — `pd.testing.assert_frame_equal(..., rtol=1e-10)` against a
    golden CSV, NOT raw equality, to tolerate pandas CoW representation changes.
- Commit golden artifacts derived from the existing **`turface_19`** reference dataset
  (`tests/fixtures/real/wheat_edpie/inputs/post_qc/turface_19_final_data.csv`): a golden
  UMAP embedding (reusing the existing `expected/viz/turface_19/viz_umap_embedding.csv`
  if it shares the same compute path, else a justified new artifact), golden cluster
  labels, a golden per-genotype trait-summary table, and a **`golden_provenance.json`**
  recording the dependency versions / seed / cluster count / tolerances the golden was
  generated under (so staleness is a diff, not a guess).
- Use a **fixed seed (`random_state=42`)** and a **pinned cluster count (`n_clusters=3`)**
  throughout.
- **Enforce single-OS execution**: the golden is generated on one OS and the tolerances
  are below the cross-OS BLAS floor, so the test `skipif`s on non-golden OSes (the
  cross-platform `tests` matrix collects-and-skips it, staying green) and runs in a
  **dedicated single-OS `numerical-stability` CI job** (by path), separate from the
  determinism gate so it is not mislabeled as a determinism check.
- Document the **golden-regeneration procedure** with a single source of truth in
  `tests/fixtures/README.md` (extending its existing regenerate policy with the
  dependency-bump trigger + the script), cross-linked from `docs/reproducibility.md`,
  which additionally **reconciles** the gate's `rtol=1e-10` against the document's
  standing `rtol=1e-6` float-array policy (same-stack pure-float vs cross-OS BLAS) and
  adds the gate to the CI-enforcement list. Add a `docs/CHANGELOG.md` `[Unreleased]`
  entry.
- **No new dependencies**: `scipy` (Procrustes) and `scikit-learn` (ARI) are already in
  `dependencies`.

## Impact

- Affected specs: **numerical-stability-gate** (new capability). Does NOT modify the
  existing `umap-analysis` "UMAP Reproducibility" requirement — that is same-machine
  determinism, orthogonal to this golden-drift gate.
- Affected code:
  - `tests/test_numerical_stability.py` (new; module-level `skipif` to non-golden OS)
  - `tests/fixtures.py` — new session-scoped loaders for the golden artifacts (reuse
    existing input/embedding loaders rather than reading CSVs inline)
  - `tests/fixtures/real/wheat_edpie/expected/numerical_stability/` golden artifacts +
    `golden_provenance.json` (new)
  - `scripts/regenerate_numerical_stability_golden.py` (new regeneration helper; emits
    the provenance record). The shared seed / `n_clusters` constant lives in the test (or
    a tiny importable shared module), NOT imported from `scripts/`, so a script-only edit
    cannot break test collection un-CI'd.
  - `.github/workflows/ci.yml` — new dedicated single-OS `numerical-stability` job
  - `docs/reproducibility.md` (gate semantics + tolerance reconciliation + CI list),
    `tests/fixtures/README.md` (single-source regenerate policy + layout), `docs/CHANGELOG.md`
- Related: upstream bloom#141; this repo's #118 (tolerance policy), #133 (reproducibility
  gates), #120/#146 (golden fixtures rollout).
