# meta_ekm.py - Meta-level EKM experimentation framework
# -----------------------------------------------------

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any

from ekm_generator import EKMGenerator
from ekm_stack import EKMExperiment
from eigen_koan_matrix import EigenKoanMatrix


def _random_path(matrix: EigenKoanMatrix) -> List[int]:
    return [random.randint(0, matrix.size - 1) for _ in range(matrix.size)]


def _merge_constraint_counts(counts: Dict[str, Dict[str, int]]) -> Dict[int, int]:
    total: Dict[int, int] = {}
    for model_counts in counts.values():
        for idx, val in model_counts.items():
            i = int(idx)
            total[i] = total.get(i, 0) + val
    return total


@dataclass
class MetaEKMSystem:
    """Generate and test EKMs for a set of research questions."""

    research_questions: List[str]
    models: List[str]
    generator: EKMGenerator = field(default_factory=EKMGenerator)
    experiment_cls: type = EKMExperiment
    matrices: Dict[str, EigenKoanMatrix] = field(init=False, default_factory=dict)

    def generate_matrices(self) -> Dict[str, EigenKoanMatrix]:
        """Generate one EKM per research question."""
        self.matrices = {}
        for rq in self.research_questions:
            matrix = self.generator.generate_ekm(theme=rq)
            self.matrices[matrix.id] = matrix
        return self.matrices

    def _adapt_paths(self, analysis: Dict[str, Any], paths: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        """Create new paths emphasizing underused constraints."""
        new_paths: Dict[str, List[List[int]]] = {}
        for matrix_id, matrix_analysis in analysis.items():
            matrix = self.matrices[matrix_id]
            counts = _merge_constraint_counts(matrix_analysis.get("constraint_preservation", {}))
            if counts:
                least_used = min(counts, key=counts.get)
            else:
                least_used = 0
            focus_path = [least_used for _ in range(matrix.size)]
            new_paths[matrix_id] = paths.get(matrix_id, []) + [focus_path]
        return new_paths

    def run_adaptive_experiment(
        self,
        model_runners: Dict[str, Callable[[str], str]],
        iterations: int = 2,
        paths_per_matrix: int = 2,
    ) -> List[Dict[str, Any]]:
        """Run adaptive experiment across all research questions."""
        if not self.matrices:
            self.generate_matrices()

        paths = {
            mid: [_random_path(m) for _ in range(paths_per_matrix)]
            for mid, m in self.matrices.items()
        }

        history: List[Dict[str, Any]] = []

        for i in range(iterations):
            exp = self.experiment_cls(
                name=f"meta_iter_{i+1}",
                description="Meta-level adaptive experiment",
                matrices=self.matrices,
                models=self.models,
                paths=paths,
                results_dir=f"./meta_results_iter_{i+1}"
            )
            results = exp.run(model_runners)
            analysis = exp.analyze(results)
            history.append({"iteration": i + 1, "results": results, "analysis": analysis})
            paths = self._adapt_paths(analysis, paths)
        return history
