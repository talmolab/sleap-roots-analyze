"""Regenerate the numerical-stability golden artifacts and provenance record.

Run this ONLY when a major ``numba`` / ``numpy`` / ``umap-learn`` / ``pandas`` bump moves
the reference numbers past tolerance (with reviewer approval). Do NOT regenerate on patch
bumps that stay within tolerance — that would defeat the gate. See the "Regenerate policy"
in ``tests/fixtures/README.md`` for the full procedure.

The artifacts MUST be generated on the gate's canonical OS
(``numerical_stability_recompute.CANONICAL_PLATFORM``); UMAP is not bit-reproducible
across operating systems.

Usage::

    uv run --python 3.11 python scripts/regenerate_numerical_stability_golden.py
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Import the shared recompute logic from the tests package (single source of truth).
# The constant/compute lives under tests/, never here, so a script-only edit cannot
# break test collection un-CI'd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.numerical_stability_recompute import (  # noqa: E402
    ARI_FLOOR,
    ATOL_PROCRUSTES,
    CANONICAL_PLATFORM,
    GOLDEN_DIR,
    GOLDEN_EMBEDDING,
    GOLDEN_LABELS,
    GOLDEN_PROVENANCE,
    GOLDEN_TRAIT_SUMMARY,
    N_CLUSTERS,
    PROVENANCE_PACKAGES,
    SEED,
    TRAIT_RTOL,
    load_input,
    recompute_embedding,
    recompute_labels,
    recompute_trait_summary,
)


def _package_versions() -> dict[str, str]:
    """Return the installed versions of the provenance-tracked packages."""
    versions: dict[str, str] = {}
    for name in PROVENANCE_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def main() -> None:
    """Recompute and write the three golden artifacts plus the provenance record."""
    if sys.platform != CANONICAL_PLATFORM:
        print(
            f"WARNING: generating golden on '{sys.platform}', but the gate's canonical "
            f"platform is '{CANONICAL_PLATFORM}'. UMAP is not bit-reproducible across "
            "operating systems — the committed golden must match the OS the gate runs "
            "on, or the gate will fail.",
            file=sys.stderr,
        )

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    df = load_input()

    embedding = recompute_embedding(df)
    embedding.to_csv(GOLDEN_EMBEDDING, index=False)

    labels, effective_k = recompute_labels(df)
    if effective_k != N_CLUSTERS:
        raise RuntimeError(
            f"KMeans returned {effective_k} clusters, expected {N_CLUSTERS}. The "
            "internal cluster-count clamp may have engaged; do not commit this golden."
        )
    labels.to_csv(GOLDEN_LABELS, index=False)

    trait_summary = recompute_trait_summary(df)
    trait_summary.to_csv(GOLDEN_TRAIT_SUMMARY)

    provenance = {
        "generated_on": date.today().isoformat(),
        "os": platform.system(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "seed": SEED,
        "n_clusters": N_CLUSTERS,
        "tolerances": {
            "procrustes_atol": ATOL_PROCRUSTES,
            "ari_floor": ARI_FLOOR,
            "trait_summary_rtol": TRAIT_RTOL,
        },
    }
    GOLDEN_PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"Wrote golden artifacts to {GOLDEN_DIR}:")
    for path in (
        GOLDEN_EMBEDDING,
        GOLDEN_LABELS,
        GOLDEN_TRAIT_SUMMARY,
        GOLDEN_PROVENANCE,
    ):
        print(f"  - {path.name}")
    print(
        f"Provenance: {provenance['os']} / py{provenance['python']} / {provenance['packages']}"
    )


if __name__ == "__main__":
    main()
