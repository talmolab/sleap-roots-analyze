# Reproducibility & Seeding Policy

This document defines how `sleap-roots-analyze` guarantees reproducible results from
its stochastic analyses, the numerical tolerance to expect when comparing outputs
(e.g. for golden-value tests), and the contract for serializable result objects. It
backs issues #118 and #133 and is enforced in CI by the **Reproducibility gates** job
(see [CI enforcement](#ci-enforcement)).

## Seeding policy

Every function whose result depends on a random number generator accepts a
`random_state` parameter (default `42`) and forwards it to the underlying
sklearn / umap estimator. Pass an explicit integer for reproducible output; pass
`None` to opt into non-deterministic behavior.

The **authoritative inventory** of stochastic functions is the case registry in
[`tests/reproducibility_cases.py`](../tests/reproducibility_cases.py). A coverage
guard walks every module in the package and fails CI if any function accepting
`random_state` is absent from that registry, so the list below cannot silently drift —
it is an illustrative summary, not a second source of truth. The registry covers the
top-level entry points *and* lower-level helpers such as `pca.fit_pca`,
`pca.select_n_components`, `pca.perform_pca_with_variance_threshold`, and
`clustering.calculate_optimal_k_kmeans`.

| Function | Module | Default `random_state` | Underlying RNG |
| --- | --- | --- | --- |
| `perform_pca_analysis` | `pca` | `42` | sklearn `PCA` (matters for the randomized SVD solver) |
| `perform_umap_analysis` | `umap` | `42` | `umap.UMAP` |
| `perform_kmeans_clustering` | `clustering` | `42` | sklearn `KMeans` |
| `perform_gmm_clustering` | `clustering` | `42` | sklearn `GaussianMixture` |
| `detect_outliers_isolation_forest` | `outlier_detection` | `42` | sklearn `IsolationForest` |
| `detect_outliers_kmeans` | `outlier_detection` | `42` | via `perform_kmeans_clustering` |
| `detect_outliers_gmm` | `outlier_detection` | `42` | via `perform_gmm_clustering` |
| `detect_outliers_mahalanobis` | `outlier_detection` | `42` | via `perform_pca_analysis` |

### Deterministic functions (no seed needed)

`perform_hierarchical_clustering` uses scipy `linkage`, which is **deterministic**:
the same input always produces the same linkage matrix. It intentionally does **not**
take a `random_state` — adding one would misrepresent the API. Its determinism is
still covered by the regression test.

`create_phenotype_variation_plot` adds plotting jitter seeded with
`np.random.seed(42)`, so its visual output is reproducible. The `create_umap_*` plot
helpers consume a **precomputed** embedding and run no RNG of their own.

## Determinism guarantee

> Given the same `random_state`, the same input data, and the same environment
> (library versions + BLAS backend), each stochastic function returns identical
> output.

This is verified on every test run: `tests/test_reproducibility.py` calls each
registered function twice with `random_state=42` and asserts the reproducibility-bearing
outputs match, and the whole-package coverage guard ensures *every* stochastic function
is in that sweep. Within a single machine the outputs are bit-for-bit identical (UMAP
embeddings included).

## Tolerance policy

When comparing outputs across runs or machines, use these tolerances:

- **Integer cluster labels and outlier indices:** compare for **exact** equality.
- **Floating-point arrays** (UMAP embeddings, PCA transformed data / loadings /
  eigenvalues, distances, cluster centers): compare with **`rtol=1e-6`** (and a small
  `atol`, e.g. `1e-9`). This is the typical reliable tolerance for sklearn outputs.

```python
import numpy as np

# integer labels — exact
assert np.array_equal(a["cluster_labels"], b["cluster_labels"])

# float arrays — within tolerance
assert np.allclose(a["embedding"], b["embedding"], rtol=1e-6, atol=1e-9)
```

### Cross-platform / BLAS caveat

Bit-for-bit equality is guaranteed **within a single environment**. Across operating
systems or BLAS implementations (OpenBLAS vs MKL vs Accelerate), floating-point
reductions can reorder and produce differences at roughly the `1e-6`–`1e-12` level.
Therefore:

- Do **not** assert exact float equality across machines — use `rtol=1e-6`.
- Cluster labels can in principle flip if two samples are near a decision boundary;
  for golden tests prefer comparing them up to a label permutation, or assert on
  stable derived quantities (counts, sorted distances) when an exact match is needed.
- We do **not** pin a BLAS backend. If a future golden test proves sensitive at the
  `1e-6` level, pin the BLAS implementation in that test's environment (e.g. via the
  `threadpoolctl` / `OPENBLAS_*` env) and document it alongside the test.

## Numerical-stability golden gate

The determinism guarantee above is a **same-machine double-run**: it proves a function is
deterministic given a fixed environment, but it is structurally blind to *drift*. When a
`numba` / `numpy` / `umap-learn` / `pandas` upgrade silently changes a result, both halves
of the double-run move together and still agree. To catch that, the **numerical-stability
gate** (`tests/test_numerical_stability.py`) recomputes the UMAP, clustering, and pandas
trait-aggregation paths on the `turface_19` reference slice and compares them to
**committed golden artifacts** under `tests/fixtures/real/wheat_edpie/expected/numerical_stability/`.
A library upgrade that moves the numbers fails here.

Assertions are tolerance-based, with thresholds grounded in the dataset's measured
same-stack spread (recorded in `tests/numerical_stability_recompute.py`):

- **UMAP embedding** — `scipy.spatial.procrustes` superimposition (invariant to
  translation, scale, rotation, reflection), then `np.allclose` on the **aligned
  coordinate matrices** with `atol=1e-6`. The aligned matrices, not the disparity scalar,
  are compared: same-stack aligned spread is ≈ `3e-17`, yet a real `1e-3` coordinate nudge
  yields ≈ `2.6e-5`, so `1e-6` is far above noise yet below the drift it must catch.
- **Cluster labels** — Adjusted Rand Index `> 0.95`. ARI is permutation-invariant, the
  "compare up to a label permutation" rule from the [Tolerance policy](#tolerance-policy).
  The pinned cluster count (`n_clusters = 3`) is also asserted, guarding the clusterer's
  internal `len // 10` clamp.
- **Trait summary** — `pd.testing.assert_frame_equal(rtol=1e-10)` on a per-genotype
  `groupby().agg(["mean", "std"])` — the Copy-on-Write-sensitive path.

**Why `rtol=1e-10` here when the Tolerance policy says `rtol=1e-6`?** The `1e-6` figure is
the **cross-OS / BLAS-reorder** tolerance (see the [BLAS caveat](#cross-platform--blas-caveat)).
The trait-summary comparison is **same-stack, single-OS, pure-float groupby on a fixed
committed input** with no RNG and tiny per-genotype reductions, so it can — and should — be
much tighter to actually catch a CoW/groupby path change; a loose `1e-6` would let real
drift through. Conversely the UMAP `atol=1e-6` *is* the policy number, applied to a
manifold that rides the numba/BLAS JIT stack.

**Single-OS by design.** Because UMAP is not bit-reproducible across operating systems and
these tolerances are below the cross-OS float floor, the golden is generated on one OS
(macOS) and the gate **skips on other operating systems** (keeping the cross-platform
`tests` matrix green). The committed `golden_provenance.json` records the OS / Python /
dependency versions the golden was generated under, so staleness is a diff. To regenerate,
see the [Regenerate policy](../tests/fixtures/README.md#regenerate-policy) — major
numba/numpy/umap-learn/pandas bumps past tolerance only, with reviewer approval; never on
patch bumps within tolerance.

## Result-object serialization contract

The FAIR interoperability guarantee: every analytical result object must serialize to
JSON and round-trip without loss. This is enforced by
[`tests/test_result_serialization.py`](../tests/test_result_serialization.py), which is
**opt-in by construction** — it asserts only when a function returns a dataclass, and
skips functions that still return plain dicts.

**To give a new result type automatic round-trip coverage (e.g. #127 `PCAResult`,
#128 `HeritabilityResult`, #129 `ClusterResult`):**

1. Return a `@dataclass` from the analytical function.
2. Make every field JSON-projectable via
   [`convert_to_json_serializable`](../src/sleap_roots_analyze/data_utils.py) — numpy
   scalars, `ndarray`, `Path`, and nested dict/list of those are handled. Do **not**
   store raw estimator objects (e.g. a fitted `PCA`) as result fields: the serializer
   can only stringify them to a `"<PCA>"` placeholder, and the gate fails on such lossy
   stringification rather than passing vacuously.
3. Optionally add a `from_dict` classmethod; the gate then also asserts it reconstructs
   an equal object.
4. If the function is not already listed, add it to the round-trip case list in
   `tests/test_result_serialization.py`. No other test edits are needed — coverage
   activates the moment the function returns a dataclass.

"Lossless" is defined on the JSON-native projection (`convert_to_json_serializable` is
intentionally asymmetric: `ndarray`→list, unknown objects→`"<TypeName>"`). `NaN`
survives an in-process round-trip and is compared NaN-aware; note it is non-standard
JSON, so a stricter external consumer may reject it.

**Path normalization is centralized — store `Path`, never `str(path)`.** Producers
(pipeline steps, result objects) hand a `Path` object to the serializer and let it
normalize to a POSIX string (`Path.as_posix()`) exactly once. Pre-stringifying with
`str(path)` defeats this and bakes in OS-specific separators (backslashes on Windows),
producing `out\a.csv` instead of `out/a.csv` in `pipeline_summary.json` and the
standalone `*_manifest.json` files (#157). Both serialization sinks own this:
`convert_to_json_serializable` (for the summary) and the `BaseStep.save_json` `default`
hook (for standalone manifests). The `StepSummary.files_generated` field is typed
`List[Path]` so the rule can't silently regress; [`tests/test_no_path_prestringify.py`](../tests/test_no_path_prestringify.py)
is an AST guard that fails if a `str(path)` is reintroduced into a step.

## CI enforcement

The gates run on every pull request and can be set as required status checks in
branch protection ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)):

- **Reproducibility gates (determinism)** — single OS. Determinism is a same-machine
  double-run comparison; cross-OS bit-identity is not guaranteed by NumPy/BLAS, so a
  single OS is the correct, achievable claim.
- **Serialization round-trip gate** — full OS matrix (Ubuntu, Windows, macOS).
  Serialization is **not** OS-independent: `Path → str` differs by OS (Windows uses
  backslashes), exactly the class of bug this gate exists to catch, so it runs
  cross-OS.
- **Numerical-stability gate (golden drift)** — single OS (macOS). A drift detector, not
  a determinism check: it compares recomputed UMAP/cluster/trait outputs to committed
  golden artifacts. Single-OS because the golden is generated on one machine and its
  tolerances are tighter than the cross-OS float floor; the test self-skips elsewhere.
  See the [Numerical-stability golden gate](#numerical-stability-golden-gate) section.

Run the gates locally with:

```bash
uv run pytest tests/test_reproducibility.py tests/test_result_serialization.py \
  tests/test_numerical_stability.py
```
