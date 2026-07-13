## Context

The serializable-result-types epic (#130, `result_types.py`) gives each analytical
function a frozen, JSON-serializable dataclass view built by a `from_*_dict()` adapter,
under a strict convention: **the producer returns a plain dict; the adapter maps that
dict into the typed view (native casts only, no mutation).** `PCAResult` (#127) is the
exemplar — it excludes the fitted `PCA`/`StandardScaler` and keeps only serializable
science.

UMAP is the last function in the epic without a result type. `perform_umap_analysis`
returns `{embedding, reducer, scaler, n_neighbors, min_dist}` — `reducer`/`scaler` are
non-serializable, and the dict omits `feature_names`, `n_components`, and `random_state`.
Issue #180 offers two API shapes ("return it, **or** add `from_umap_dict(...)`") and two
seed typings ("required `int`, **or** `Optional[int]` for parity with `PCAResult`"); this
change picks one of each and adds a load-bearing-seed safety refinement.

## Goals / Non-Goals

- **Goals:** an opt-in `UMAPResult` that JSON round-trips with no non-serializable
  fields; `embedding` of shape `n_samples × n_components`; `feature_names` /
  `n_neighbors` / `min_dist` / `n_components` / `n_samples` / `random_state` populated in
  the serialized payload; the fitted `reducer`/`scaler` excluded; a `from_umap_dict`
  adapter whose signature parallels `from_pca_dict`.
- **Non-Goals:** changing `perform_umap_analysis`'s return **type** (would break every
  caller); a `standardize=False` path for UMAP; the `0.1.0a5` version bump (separate
  release PR).

## Decisions

- **Decision: opt-in `from_umap_dict()` adapter, not a changed return type.** Every
  sibling kept its producer dict and added a `from_*_dict` adapter, guarded by a "Non-
  Breaking Return Shape" requirement. `perform_umap_analysis`'s dict is consumed by the
  pipeline `UMAPAnalysisStep`, `interactive_visualization`, the reproducibility sweep, and
  the golden-embedding recompute — changing the return type breaks all of them.
  - *Alternative: make `perform_umap_analysis` return a `UMAPResult`.* Rejected —
    breaking, and inconsistent with the epic (a dataclass return is deferred to a future
    major bump with a `__getitem__` shim, per `docs/result-types.md`).

- **Decision: `n_samples` and `n_components` are stored fields, not `@property`
  derivations — a deliberate, documented exception to result-types rule 4.** Rule 4
  ("derivations are `@property`, so the serialized state has no redundant members") exists
  to keep the payload lean; both `n_samples == len(embedding)` and
  `n_components == len(embedding[0])` are embedding-derivable, so rule 4 would make them
  properties. They are kept as **fields** anyway because:
  - Issue #180 lists both among the result fields, and the bloom#425 consumer reads the
    result **after it has crossed the JSON boundary as a plain dict** — a `@property` is
    unavailable there, so a derivable-but-needed value must be materialized into the
    payload.
  - The pipeline already treats `n_samples` as payload, serializing it into its own UMAP
    metadata (`pipeline/steps/umap_analysis.py`) — in-repo precedent that these are
    payload-worthy dimensional provenance, not merely incidental derivations.
  This is the honest justification (consumer contract + dimensional provenance), *not*
  "they aren't derivable" — they are. Keeping both as fields also avoids the awkward
  asymmetry of storing one embedding-shape value while computing the other. `PCAResult`
  does not carry `n_samples`; UMAP diverges here because its named consumer needs it.
  - *Alternative: `n_samples`/`n_components` as `@property`.* Rejected — strict rule-4
    purity at the cost of the serialized consumer contract the epic exists to serve.

- **Decision: `from_umap_dict(d, *, random_state=None)` with a dict fallback; the
  producer additively echoes `random_state`.** The adapter keeps the sibling signature
  (`random_state` as a keyword arg), but resolves the seed as
  `random_state if random_state is not None else d.get("random_state")`. The producer is
  enriched to echo the actual seed it used into its dict, so a caller invoking
  `from_umap_dict(d)` with no argument records the **true** seed rather than one stamped
  on trust. This matters because UMAP's seed is load-bearing — it governs the entire
  stochastic embedding, so a wrong stamp yields a false-but-authoritative reproducibility
  record whose embedding cannot be regenerated. (PCA's documented full-SVD path is seed-
  insensitive, so its on-trust stamp is largely cosmetic; UMAP is not equivalent.) Since
  this change is already enriching the producer for `feature_names`, echoing the seed is
  nearly free.
  - *Alternative (a): pure stamp-on-trust (arg only, no producer echo), matching
    `from_pca_dict` exactly.* Rejected — buys signature symmetry (still preserved here)
    but leaves the load-bearing-seed footgun.
  - *Alternative (b): drop the `random_state` argument and read only from the dict.*
    Rejected — the keyword arg keeps the adapter family uniform and lets a caller override
    when reconstructing from a dict that lacks the echoed seed.
  - *The one shape avoided:* echo into the dict while the adapter ignores it (silent
    mismatch). The explicit-arg-wins-else-dict resolution rules this out.

- **Decision: `random_state: Optional[int] = None`.** Matches `PCAResult.random_state`
  (`Optional[int]` today) — one seed typing with the PCA sibling, and the value bloom-mcp
  already expects from that sibling. (The #179 change widens `ClusterResult.random_state`
  to `Optional[int]` too, but is **not yet merged** — on current `main` it is still a
  required `int`, so this decision rests on the `PCAResult` precedent, which is present
  fact.) A real `perform_umap_analysis` run always produces an `int` (its `random_state`
  defaults to `42`), so `None` only surfaces when a caller builds a `UMAPResult` from a
  seedless dict.
  - *Alternative: required `int`.* Rejected despite UMAP's seed being load-bearing —
    cross-sibling consistency wins, and the echo+fallback above already protects the seed's
    accuracy.

## Risks / Trade-offs

- **Additive dict keys could surprise a strict-key consumer** → mitigated: no test or
  caller asserts an exact key set on the UMAP dict (`tests/test_umap.py` checks only
  membership), so adding `feature_names`/`random_state` is safe.
- **`standardized` is always `True` from the real producer** — `perform_umap_analysis`
  has no `standardize` parameter and always fits a `StandardScaler`, so the `False` branch
  is reachable **only** via a hand-built dict with `scaler=None` (exercised by the spec's
  own scenario, which guards against a hardcoded `standardized=True` in the adapter). The
  field is kept for `PCAResult` parity and future-proofing; it is not producible by a real
  UMAP run today.
- **`n_samples`/`n_components` redundancy** (both derivable from `embedding`) → accepted
  deliberately for the consumer contract above; the redundant values are cheap and
  cross-checkable against the embedding.

## Migration Plan

Additive; no existing caller changes required — the new dataclass and adapter are opt-in,
and the `feature_names`/`random_state` keys are purely additional. This PR carries only a
`docs/CHANGELOG.md` `[Unreleased]` entry; `0.1.0a5` is cut in a **separate `chore`
release PR** via `uv version` (per #176), bundling this change with #179.

## Open Questions

- Confirm the exact `UMAPResult` field set (esp. `n_samples`/`n_components` as serialized
  fields) and the `from_umap_dict(d, *, random_state=None)` signature are what bloom-mcp's
  `umap_analysis` contract wrapper (bloom#425) expects to consume.
