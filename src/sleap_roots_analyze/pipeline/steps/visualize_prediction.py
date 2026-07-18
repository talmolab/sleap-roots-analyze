"""Step: Visualize cross-platform prediction via permutation null + figure (Tier 4, #200).

Naming note: this module shares its basename with
``sleap_roots_analyze.visualize_prediction`` (a different module, in a
different subpackage -- that one holds ``create_prediction_figure()`` and
the other pure plotting helpers this step calls; this module holds only the
pipeline-step wiring/orchestration). Full import paths never collide. See
that module's own docstring for the same cross-reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from sleap_roots_analyze.cross_platform_prediction import permutation_test
from sleap_roots_analyze.pca import fit_pca
from sleap_roots_analyze.pipeline.core import BaseStep, StepResult
from sleap_roots_analyze.result_types import (
    CrossPlatformPermutationResult,
    TargetPrediction,
)
from sleap_roots_analyze.visualize_prediction import create_prediction_figure


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
        combinations = [
            (method, target_name) for method in methods for target_name in target_names
        ]

        # Independent seed per (target, method) combination (design.md
        # Decision 4, found during round 1's review): reusing one shared
        # seed for every combination would correlate, not independently
        # sample, the null draws the pooled violin panel (Section 6) later
        # combines across targets.
        seed_sequence = np.random.SeedSequence(pcfg.permutation_random_state)
        child_seeds = seed_sequence.spawn(len(combinations))

        def _run_unit(method: str, target_name: str, seed: np.random.SeedSequence):
            result = permutation_test(
                X=source_clean,
                y=target_y[target_name],
                genotypes=genotypes,
                reduction_method=method,
                representative_names=(
                    source_representative_names if method == "representatives" else None
                ),
                n_permutations=pcfg.n_permutations,
                random_state=seed,
            )
            return method, target_name, result

        # Parallelizes across independent (target, method) units, not across
        # individual permutation calls -- empirically measured slower than
        # serial at this workload's per-call cost (design.md Decision 4).
        # joblib.Parallel(backend="loky") fails fast on the first worker
        # exception, so collecting every result here before writing any
        # output file below is a correct, simple all-or-nothing
        # partial-failure contract -- no additional engineering needed.
        dispatched = Parallel(n_jobs=pcfg.permutation_n_jobs, backend="loky")(
            delayed(_run_unit)(method, target_name, seed)
            for (method, target_name), seed in zip(combinations, child_seeds)
        )

        permutation_test_results_by_method: Dict[str, Dict[str, Any]] = {
            method: {} for method in methods
        }
        for method, target_name, result in dispatched:
            permutation_test_results_by_method[method][target_name] = result

        source_platform = prev_result.metadata["source_platform"]
        target_platform = prev_result.metadata["target_platform"]

        files_generated: List[Path] = []
        results_by_method: Dict[str, Any] = {}
        cp_results_by_method: Dict[str, CrossPlatformPermutationResult] = {}
        for method in methods:
            cp_result = CrossPlatformPermutationResult.from_permutation_test_results(
                source_platform=source_platform,
                target_platform=target_platform,
                reduction_method=method,
                permutation_test_results=permutation_test_results_by_method[method],
            )
            output_path = run_dir / f"07_permutation_{method}.json"
            output_path.write_text(cp_result.to_json(indent=2))
            files_generated.append(output_path)
            results_by_method[method] = cp_result.to_dict()
            cp_results_by_method[method] = cp_result

        # Figure uses only the primary reduction_method's results (design.md
        # Decision 9), built from task 6's observed CrossPlatformPredictionResult
        # (for the PC1 obs-vs-pred scatter) and this step's own permutation
        # results for the same method (for the R^2 violin / top-quartile bar).
        primary_method = pcfg.reduction_method
        target_predictions = [
            TargetPrediction(**target_dict)
            for target_dict in data[primary_method]["predictions"]
        ]
        fig = create_prediction_figure(
            target_predictions=target_predictions,
            permutation_results=cp_results_by_method[primary_method].predictions,
        )
        figure_path = run_dir / "07_prediction_figure.png"
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        files_generated.append(figure_path)

        return StepResult(
            data=results_by_method, metadata={}, files_generated=files_generated
        )
