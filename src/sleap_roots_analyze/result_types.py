"""Serializable stdlib dataclass result types for analytical functions.

This module is the shared home for the serializable-result-types epic (#130):
flat ``@dataclass`` views that hold **only JSON-serializable science** — native
Python scalars and lists, never ``numpy`` arrays, ``pandas`` frames, or sklearn
objects. Because every field is JSON-native, ``json.dumps(asdict(result))``
round-trips with no custom serializer, which is what lets these results cross a
JSON boundary (bloom-mcp, a cached artifact, an API) and anchor the
reproducibility CI gate.

The first type is :class:`PCAResult` (issue #127, the detailed exemplar);
``HeritabilityResult`` (#128) and ``ClusterResult`` (#129) follow the same
convention here. This module imports nothing from the analytical modules
(``pca.py`` etc.) so dependencies stay one-way.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

__all__ = ["FeatureContribution", "PCAResult"]


@dataclass(frozen=True)
class FeatureContribution:
    """A single feature's contribution to the retained PCA components.

    Attributes:
        feature: Feature (column) name the contribution belongs to.
        total_contribution: Sum over retained components of
            ``loading**2 * eigenvalue`` for this feature (variance contributed).
        fractional_contribution: ``total_contribution`` normalized to sum to 1
            across all features.
    """

    feature: str
    total_contribution: float
    fractional_contribution: float


@dataclass(frozen=True)
class PCAResult:
    """JSON-serializable view of a PCA run (science only, no sklearn objects).

    Built from the legacy ``perform_pca_analysis`` dict via
    :meth:`from_pca_dict`. Component-indexed fields cover only the retained
    components; the fitted ``PCA``/``StandardScaler`` objects are intentionally
    excluded (still available via the legacy dict for in-process callers).

    Attributes:
        n_components: Number of retained principal components.
        feature_names: Feature (column) names in their original order; row order
            of ``loadings``.
        explained_variance_ratio: Per-component fraction of total variance
            explained, length ``n_components``.
        eigenvalues: Per-component eigenvalues, length ``n_components``.
        cumulative_variance_ratio: Per-component cumulative sum of
            ``explained_variance_ratio``, length ``n_components``.
        loadings: ``(n_features, n_components)`` nested list of component
            loadings.
        scores: ``(n_samples, n_components)`` nested list of transformed sample
            coordinates.
        standardized: Whether the data was standardized (a ``StandardScaler``
            was fitted) before PCA.
        random_state: Random state used for the run, stamped for reproducibility
            provenance; ``None`` if not supplied to the adapter.
        explained_variance_threshold: Cumulative-variance threshold used for
            component selection, stamped for provenance; ``None`` if not
            supplied to the adapter.
        feature_contributions: Per-feature contributions, ordered by
            ``total_contribution`` descending.
    """

    n_components: int
    feature_names: list[str]
    explained_variance_ratio: list[float]
    eigenvalues: list[float]
    cumulative_variance_ratio: list[float]
    loadings: list[list[float]]
    scores: list[list[float]]
    standardized: bool
    random_state: Optional[int] = None
    explained_variance_threshold: Optional[float] = None
    feature_contributions: list[FeatureContribution] = field(default_factory=list)

    @property
    def cumulative_variance(self) -> float:
        """Total fraction of variance explained by the retained components."""
        return float(sum(self.explained_variance_ratio))

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` view via :func:`dataclasses.asdict`."""
        return dataclasses.asdict(self)

    @classmethod
    def from_pca_dict(
        cls,
        d: dict,
        *,
        random_state: Optional[int] = None,
        explained_variance_threshold: Optional[float] = None,
    ) -> "PCAResult":
        """Build a :class:`PCAResult` from a ``perform_pca_analysis`` dict.

        Assumes the canonical key set returned by ``perform_pca_analysis``
        (``n_components_selected``, ``feature_names``, ``loadings``,
        ``eigenvalues``, ``explained_variance_ratio``,
        ``cumulative_variance_ratio``, ``transformed_data``, ``scaler``,
        ``feature_contributions``). Does not mutate ``d``.

        Args:
            d: The dict returned by ``perform_pca_analysis``.
            random_state: Random state to stamp into the result for provenance.
            explained_variance_threshold: Threshold to stamp into the result for
                provenance.

        Returns:
            A frozen :class:`PCAResult` holding only JSON-serializable science.
        """
        n = int(d["n_components_selected"])

        fc_df = d.get("feature_contributions")
        feature_contributions: list[FeatureContribution] = []
        if fc_df is not None:
            for idx, row in fc_df.iterrows():
                feature_contributions.append(
                    FeatureContribution(
                        feature=str(idx),
                        total_contribution=float(row["total_contribution"]),
                        fractional_contribution=float(row["fractional_contribution"]),
                    )
                )

        return cls(
            n_components=n,
            feature_names=[str(name) for name in d["feature_names"]],
            explained_variance_ratio=np.asarray(d["explained_variance_ratio"])[
                :n
            ].tolist(),
            eigenvalues=np.asarray(d["eigenvalues"])[:n].tolist(),
            cumulative_variance_ratio=np.asarray(d["cumulative_variance_ratio"])[
                :n
            ].tolist(),
            loadings=np.asarray(d["loadings"])[:, :n].tolist(),
            scores=np.asarray(d["transformed_data"])[:, :n].tolist(),
            standardized=d.get("scaler") is not None,
            random_state=random_state,
            explained_variance_threshold=explained_variance_threshold,
            feature_contributions=feature_contributions,
        )
