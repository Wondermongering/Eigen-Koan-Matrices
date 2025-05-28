# adaptive_sequence.py - Adaptive testing sequence generation
"""Create new EKMs based on previous experiment analysis."""

from __future__ import annotations

from typing import Dict, Any

from eigen_koan_matrix import EigenKoanMatrix
from ekm_generator import EKMGenerator


def _merge_constraint_counts(counts: Dict[str, Dict[str, int]]) -> Dict[int, int]:
    """Merge constraint usage counts from multiple models into a total count per index.

    Args:
        counts: A dictionary where keys are model identifiers and values are
                dictionaries mapping constraint index (as string) to its usage count.
                Example: {"model_A": {"0": 10, "1": 5}, "model_B": {"0": 8, "2": 12}}

    Returns:
        A dictionary mapping constraint index (as integer) to its total usage count.
        Example: {0: 18, 1: 5, 2: 12}
    """
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
        analysis_results: Dict[str, Any],  # Typically output from EKMAnalyzer
    ) -> Dict[str, EigenKoanMatrix]:
        """Generate new EigenKoanMatrices based on the analysis of previous ones.

        This method identifies "interesting" constraints from the analysis results
        (e.g., those frequently preserved or violated) and generates new matrices
        that focus on these constraints.

        Args:
            base_matrices: A dictionary of original EigenKoanMatrices, keyed by their ID.
            analysis_results: A dictionary containing analysis data for each matrix,
                              typically from an EKMAnalyzer. It's expected to have a
                              "constraint_preservation" key for each matrix analysis,
                              which maps model names to constraint index counts.

        Returns:
            A dictionary of newly generated EigenKoanMatrices, keyed by their new ID.
        """
        new_matrices: Dict[str, EigenKoanMatrix] = {}
        for matrix_id, matrix_analysis in analysis_results.items():
            if matrix_id not in base_matrices:
                continue  # Skip if the analyzed matrix is not in the provided base matrices

            base_matrix = base_matrices[matrix_id]
            constraint_preservation_data = matrix_analysis.get("constraint_preservation", {})
            
            # Ensure constraint_preservation_data is not empty and is properly structured
            if not isinstance(constraint_preservation_data, dict) or not constraint_preservation_data:
                # Fallback: if no valid constraint data, perhaps default to the first constraint or skip
                focus_idx = 0 
            else:
                counts = _merge_constraint_counts(constraint_preservation_data)
                if counts:
                    # Find the constraint index with the highest combined count
                    focus_idx = max(counts, key=counts.get) 
                else:
                    # Fallback if counts are empty (e.g. no models preserved any constraints)
                    focus_idx = 0 
            
            # Ensure focus_idx is an integer and a valid index
            focus_idx = int(focus_idx)
            if not (0 <= focus_idx < len(base_matrix.constraint_cols)):
                # Fallback if focus_idx is somehow out of bounds
                focus_idx = 0 # Default to the first constraint
                if not base_matrix.constraint_cols: # If no constraints, cannot proceed for this matrix
                    continue


            focus_constraint = base_matrix.constraint_cols[focus_idx]
            new_matrix = self.generator.generate_ekm(
                size=base_matrix.size,
                theme=f"follow-up {base_matrix.name}",
            )
            new_matrix.constraint_cols[0] = focus_constraint
            new_matrices[new_matrix.id] = new_matrix
        return new_matrices
