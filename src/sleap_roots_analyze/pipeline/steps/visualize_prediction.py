"""Step: Visualize cross-platform prediction via permutation null + figure (Tier 4, #200)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.cross_platform_prediction import permutation_test
from sleap_roots_analyze.pca import fit_pca
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult


class VisualizePredictionStep(BaseStep):
    """Compute a permutation-null significance test and a summary figure per pair.

    Optional 7th step on ``CrossPlatformPipeline``, entirely absent from
    ``create_tasks()`` when ``config.prediction.visualize=False``.
    ``depends_on=["06_predict_cross_platform"]`` for both data *and*
    ordering (unlike task 6's own ordering-only second dependency on task 5,
    Tier 3.5 Decision 15) -- this step genuinely reads task 6's
    ``predictor_matrices`` and observed results.

    Reuses task 6's already-computed ``source_clean``/``target_clean``
    matrices and ``source_representative_names``/``target_representatives``
    (``StepResult.data["predictor_matrices"]``, Tier 4 Decision 6) rather
    than rebuilding BLUP-loading/NaN-dropping/alignment logic a second time.
    The PC1 target's ground-truth values are recomputed identically to task
    6's own computation (a fixed, deterministic ``fit_pca(...,
    random_state=42)`` call on the same ``target_clean`` matrix) --
    cross-checked by this step's own wiring tests, which assert this
    recomputation exactly reproduces task 6's reported PC1 result.
    """

    def __init__(self) -> None:
        """Initialize VisualizePredictionStep."""
        super().__init__(
            step_name="VisualizePrediction",
            description="Permutation-null significance test + prediction figure",
        )

    def execute(
        self,
        data: Any,
        config: Any,
        run_dir: Path,
        prev_result: Optional[StepResult] = None,
    ) -> StepResult:
        """Execute the visualize-prediction step.

        Args:
            data: Task 6's (``PredictCrossPlatformStep``) result data dict --
                ``{method: {...}, "predictor_matrices": {...}}``.
            config: CrossPlatformConfig with a populated ``prediction`` field
                (``visualize=True``).
            run_dir: Directory to save permutation JSON + figure outputs.
            prev_result: Task 6's StepResult -- read for data and ordering
                (this step's dependency, unlike task 6's own, is not
                ordering-only).

        Returns:
            StepResult holding one ``PermutationTestResult`` per
            ``(method, target_name)`` combination (Sections 7b/7c extend this
            into the final JSON/figure output).
        """
        pcfg = config.prediction
        predictor_matrices = data["predictor_matrices"]
        source_clean = predictor_matrices["source_clean"]
        target_clean = predictor_matrices["target_clean"]
        source_representative_names = list(
            predictor_matrices["source_representative_names"]
        )
        target_representatives = list(predictor_matrices["target_representatives"])

        genotypes = list(source_clean.index)

        # PC1-as-target: recomputed identically to task 6's own computation
        # (whole-dataset ground truth, not a per-fold predictor reduction --
        # Tier 3.5 Decision 6/12). Cross-checked by this step's own wiring
        # tests against task 6's already-reported PC1 result.
        _, pc1_transformed = fit_pca(
            StandardScaler().fit_transform(target_clean.to_numpy()),
            n_components=1,
            random_state=42,
        )
        pc1_values = pc1_transformed.ravel()

        # Canonical enumeration order: methods first ([reduction_method] +
        # comparison_methods), then target_names in task 6's own
        # CrossPlatformPredictionResult.predictions order (representative
        # traits, then "PC1" last).
        target_names = list(target_representatives) + ["PC1"]
        target_y = {
            name: target_clean[name].to_numpy() for name in target_representatives
        }
        target_y["PC1"] = pc1_values

        methods = [pcfg.reduction_method] + list(pcfg.comparison_methods)

        results_by_method: Dict[str, Dict[str, Any]] = {
            method: {} for method in methods
        }
        for method in methods:
            for target_name in target_names:
                results_by_method[method][target_name] = permutation_test(
                    X=source_clean,
                    y=target_y[target_name],
                    genotypes=genotypes,
                    reduction_method=method,
                    representative_names=(
                        source_representative_names
                        if method == "representatives"
                        else None
                    ),
                    n_permutations=pcfg.n_permutations,
                    random_state=pcfg.permutation_random_state,
                )

        return StepResult(data=results_by_method, metadata={}, files_generated=[])
