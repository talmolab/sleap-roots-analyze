"""Cross-platform genotype-effect prediction: LOGO-CV ridge/PLS machinery.

Tier 3 of the wheat EDPIE cross-platform genotype-prediction program (#194).
Reframes the cross-platform result from *correlation* to *predictability*:
given genotype BLUPs (or raw genotype means) estimated within one platform,
this module tests whether they predict genotype effects in another platform
via ridge regression / Partial Least Squares (PLS) with leave-one-genotype-out
(LOGO) cross-validation.

See the program's grounding document (``theory.md``, referenced from
``openspec/changes/add-cross-platform-prediction/design.md``) for the
CV-hygiene contract this module implements against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_VALID_REDUCTION_METHODS = ("pls_latent", "representatives", "pc1")


def fit_pca_on_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """Fit PCA on training data only, project held-out data onto it.

    A fresh ``sklearn.decomposition.PCA`` is fit on ``X_train`` only; the
    resulting components are used to project ``X_test``. This is
    deliberately distinct from the pipeline-level ``PCA`` step in ``pca.py``:
    that step is fit on every genotype before a leave-one-genotype-out (LOGO)
    fold loop runs, so reusing it here would leak the held-out genotype's
    position into the component loadings.

    Args:
        X_train: Training-fold data, shape ``(n_train, n_traits)``.
        X_test: Held-out data to project, shape ``(n_test, n_traits)``.
        n_components: Number of principal components to fit/project onto.

    Returns:
        ``X_test`` projected onto the ``n_components`` components fit from
        ``X_train``, shape ``(n_test, n_components)``.

    Raises:
        ValueError: If ``X_train`` has fewer traits (columns) than
            ``n_components``.
    """
    if X_train.shape[1] < n_components:
        raise ValueError(
            f"X_train has {X_train.shape[1]} traits, fewer than "
            f"n_components={n_components}."
        )
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    return pca.transform(X_test)


@dataclass
class LOGOCVResult:
    """Leave-one-genotype-out cross-validation result for one prediction target.

    Attributes:
        genotypes: Genotype labels, in the same order as ``y_true``/``y_pred``.
        y_true: Observed target values, one per genotype.
        y_pred: Leave-one-genotype-out predicted values, one per genotype
            (``y_pred[i]`` is the prediction for genotype ``i`` when genotype
            ``i`` was held out).
        r2: Aggregate R^2 over the concatenated leave-one-out predictions.
        rmse: Aggregate Root Mean Squared Error over the same predictions.
            Not comparable across ``LOGOCVResult`` instances built from
            differently-scaled traits.
        spearman_rho: Aggregate Spearman rank correlation over the same
            predictions.
        spearman_p: ``spearman_rho``'s p-value. An asymptotic approximation
            (``scipy.stats.spearmanr``'s default), imprecise below n~=20-30 --
            descriptive, not hypothesis-test-grade, at this program's n~=19.
    """

    genotypes: list
    y_true: np.ndarray
    y_pred: np.ndarray
    r2: float
    rmse: float
    spearman_rho: float
    spearman_p: float


def logo_cv_predict(
    X: pd.DataFrame,
    y: np.ndarray,
    genotypes: Sequence[str],
    reduction_method: str = "pls_latent",
    representative_names: Optional[Sequence[str]] = None,
) -> LOGOCVResult:
    """Predict each genotype's target value via leave-one-genotype-out CV.

    Implements the CV-hygiene contract: a fresh ``sklearn.pipeline.Pipeline``
    is instantiated and fit inside each fold, so no step ever sees the
    held-out genotype during fit. Aggregate R^2, RMSE, and Spearman rho are
    computed once, over the concatenated leave-one-out predictions across all
    folds (not as a separate score per single held-out genotype, which is
    statistically undefined for R^2/rho at n_test=1).

    Precondition (not verifiable from ``X`` alone): ``X``'s columns must never
    include the target trait's own values -- callers are responsible for
    excluding the prediction target from the predictor matrix.

    Args:
        X: Predictor matrix, shape ``(n_genotypes, n_traits)``, columns named
            by trait, index by genotype label.
        y: Target values, one per genotype, same order as ``X``'s rows.
        genotypes: Genotype labels, same order as ``X``'s rows.
        reduction_method: One of ``"pls_latent"`` (default) -- a
            ``StandardScaler`` + ``PLSRegression(n_components=1)`` pipeline
            fit directly on the full trait matrix; ``"representatives"`` --
            ``X`` reduced to ``representative_names`` before a
            ``StandardScaler`` + ``Ridge()`` (default ``alpha=1.0``, an
            accepted but undiscussed choice) pipeline; ``"pc1"`` -- ``X``
            reduced to a single principal-component score computed per fold
            via :func:`fit_pca_on_fold`, before a ``StandardScaler`` +
            ``Ridge()`` pipeline.
        representative_names: Trait (column) names to select when
            ``reduction_method="representatives"``. Required (non-``None``)
            for that method; ignored otherwise.

    Returns:
        A :class:`LOGOCVResult` with per-genotype predictions and aggregate
        R^2/RMSE/Spearman rho.

    Raises:
        ValueError: If ``X``, ``y``, and ``genotypes`` have mismatched
            lengths; if ``reduction_method`` is not one of the three valid
            values; if ``reduction_method="representatives"`` and
            ``representative_names`` is ``None``; if fewer than 2 genotypes
            are provided; or if ``X`` contains any ``NaN`` value.
    """
    if reduction_method not in _VALID_REDUCTION_METHODS:
        raise ValueError(
            f"reduction_method must be one of {_VALID_REDUCTION_METHODS}, "
            f"got {reduction_method!r}"
        )
    if len(X) != len(y) or len(X) != len(genotypes):
        raise ValueError(
            "X, y, and genotypes must have the same length: "
            f"got {len(X)}, {len(y)}, {len(genotypes)}"
        )
    if len(genotypes) < 2:
        raise ValueError(
            "logo_cv_predict requires at least 2 genotypes for "
            "leave-one-genotype-out cross-validation"
        )
    if reduction_method == "representatives" and representative_names is None:
        raise ValueError(
            "representative_names is required when "
            "reduction_method='representatives'"
        )

    X_values = X.to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    if np.isnan(X_values).any():
        raise ValueError("X contains NaN values")

    n = len(genotypes)
    y_pred = np.full(n, np.nan)
    loo = LeaveOneOut()

    if reduction_method == "representatives":
        rep_values = X[list(representative_names)].to_numpy(dtype=float)

    for train_idx, test_idx in loo.split(X_values):
        y_train = y[train_idx]

        if reduction_method == "pls_latent":
            X_train, X_test = X_values[train_idx], X_values[test_idx]
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", PLSRegression(n_components=1)),
                ]
            )
            pipe.fit(X_train, y_train)
            y_pred[test_idx] = pipe.predict(X_test).ravel()
        elif reduction_method == "representatives":
            X_train, X_test = rep_values[train_idx], rep_values[test_idx]
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            pipe.fit(X_train, y_train)
            y_pred[test_idx] = pipe.predict(X_test).ravel()
        else:  # pc1
            X_train_full, X_test_full = X_values[train_idx], X_values[test_idx]
            X_train_reduced = fit_pca_on_fold(
                X_train_full, X_train_full, n_components=1
            )
            X_test_reduced = fit_pca_on_fold(X_train_full, X_test_full, n_components=1)
            pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
            pipe.fit(X_train_reduced, y_train)
            y_pred[test_idx] = pipe.predict(X_test_reduced).ravel()

    r2 = float(r2_score(y, y_pred))
    rmse = float(root_mean_squared_error(y, y_pred))
    spearman_rho, spearman_p = spearmanr(y, y_pred)

    return LOGOCVResult(
        genotypes=list(genotypes),
        y_true=y,
        y_pred=y_pred,
        r2=r2,
        rmse=rmse,
        spearman_rho=float(spearman_rho),
        spearman_p=float(spearman_p),
    )
