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

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


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
