# adaptive_sequence.py - Adaptive testing sequence generation
"""Create new EKMs based on previous experiment analysis."""

from __future__ import annotations

from typing import Dict, Any

from eigen_koan_matrix import EigenKoanMatrix
from ekm_generator import EKMGenerator


def _merge_constraint_counts(counts: Dict[str, Dict[str, int]]) -> Dict[int, int]:
    """Merge constraint usage counts across models."""
    total: Dict[int, int] = {}
    for model_counts in counts.values():
        for idx, val in model_counts.items():
            i = int(idx)
            total[i] = total.get(i, 0) + val
    return total


class AdaptiveTestingSequence:
    """Generate follow-up matrices emphasizing interesting constraints."""

    def __init__(self, generator: EKMGenerator | None = None) -> None:
        self.generator = generator or EKMGenerator()

    def generate_from_analysis(
        self,
        base_matrices: Dict[str, EigenKoanMatrix],
        analysis_results: Dict[str, Any],
    ) -> Dict[str, EigenKoanMatrix]:
        """Generate new matrices from analysis output."""
        new_matrices: Dict[str, EigenKoanMatrix] = {}
        for matrix_id, matrix_analysis in analysis_results.items():
            if matrix_id not in base_matrices:
                continue
            base_matrix = base_matrices[matrix_id]
            counts = _merge_constraint_counts(
                matrix_analysis.get("constraint_preservation", {})
            )
            if counts:
                focus_idx = max(counts, key=counts.get)
            else:
                focus_idx = 0
            focus_constraint = base_matrix.constraint_cols[int(focus_idx)]
            new_matrix = self.generator.generate_ekm(
                size=base_matrix.size,
                theme=f"follow-up {base_matrix.name}",
            )
            new_matrix.constraint_cols[0] = focus_constraint
            new_matrices[new_matrix.id] = new_matrix
        return new_matrices
