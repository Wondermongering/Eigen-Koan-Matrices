"""Distributed experiment runner for Eigen-Koan Matrices."""

from __future__ import annotations

import os
import json
from typing import Any, Dict, Callable, List

import ray

from eigen_koan_matrix import EigenKoanMatrix
from ekm_stack import EKMExperiment

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def aggregate_partial_results(partials: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Merge a list of partial traversal results."""
    aggregated: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for result in partials:
        matrix_id = result.get("matrix_id")
        model = result.get("model")
        if not matrix_id or not model:
            continue
        aggregated.setdefault(matrix_id, {}).setdefault(model, []).append(result)
    return aggregated

# ---------------------------------------------------------------------------
# Ray-based distributed execution
# ---------------------------------------------------------------------------

@ray.remote(max_retries=2)
def _traverse_remote(matrix_json: str,
                     model_name: str,
                     path: List[int],
                     include_meta: bool,
                     model_runner: Callable[[str], str]) -> Dict[str, Any]:
    """Remote helper to traverse a single path."""
    matrix = EigenKoanMatrix.from_json(matrix_json)
    result = matrix.traverse(model_runner, path=path, include_metacommentary=include_meta)
    result["model"] = model_name
    return result


def run_distributed_experiment(experiment: EKMExperiment,
                               model_runners: Dict[str, Callable[[str], str]],
                               ray_address: str = "auto",
                               include_metacommentary: bool = True) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Run an EKMExperiment across a Ray cluster."""
    ray.init(address=ray_address, ignore_reinit_error=True)

    futures = []
    for matrix_id, matrix in experiment.matrices.items():
        if matrix_id not in experiment.paths:
            continue
        matrix_json = matrix.to_json()
        for model in experiment.models:
            runner = model_runners[model]
            for path in experiment.paths[matrix_id]:
                futures.append(
                    _traverse_remote.remote(
                        matrix_json,
                        model,
                        path,
                        include_metacommentary,
                        runner,
                    )
                )

    partials = ray.get(futures)
    results = aggregate_partial_results(partials)

    # Persist combined results
    results_file = os.path.join(experiment.experiment_dir, "distributed_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    return results
