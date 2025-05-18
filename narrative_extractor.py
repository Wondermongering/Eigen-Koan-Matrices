# narrative_extractor.py - Convert matrix traversal patterns into narrative explanations
# -----------------------------------------------------------------------------
"""Tools for deriving human-readable narratives from Eigen-Koan Matrix traversals."""

from typing import Dict, List

from eigen_koan_matrix import EigenKoanMatrix


def traversal_to_narrative(matrix: EigenKoanMatrix, result: Dict) -> str:
    """Return a short narrative explaining a single traversal.

    Parameters
    ----------
    matrix:
        The Eigen-Koan Matrix the traversal was performed on.
    result:
        A traversal result produced by :meth:`EigenKoanMatrix.traverse` or loaded
        from saved experiment data.

    Returns
    -------
    str
        A multi-line narrative describing the traversal and basic metrics.
    """
    path: List[int] = result.get("path", [])
    tasks = matrix.get_path_tasks()
    constraints = matrix.get_path_constraints(path)

    analysis = matrix.analyze_path_paradox(path)

    lines: List[str] = []
    lines.append(f"Traversal path {path}:")
    for i, (task, constraint) in enumerate(zip(tasks, constraints)):
        lines.append(f"  Step {i+1}: {task} while {constraint}")

    main_strength = result.get("main_diagonal_strength", 0.0)
    anti_strength = result.get("anti_diagonal_strength", 0.0)
    lines.append(
        f"Main diagonal strength ({matrix.main_diagonal.name}): {main_strength:.2f}"
    )
    lines.append(
        f"Anti-diagonal strength ({matrix.anti_diagonal.name}): {anti_strength:.2f}"
    )
    tension_count = analysis.get("tension_count", 0)
    lines.append(f"Tension points detected: {tension_count}")

    # Qualitative interpretation
    if tension_count > 0:
        lines.append("The path contains conflicting constraints, indicating tension.")
    else:
        lines.append("Constraints align smoothly with minimal tension.")

    if main_strength > anti_strength:
        lines.append(
            f"Overall the traversal leans toward the {matrix.main_diagonal.name} affect."
        )
    elif main_strength < anti_strength:
        lines.append(
            f"Overall the traversal leans toward the {matrix.anti_diagonal.name} affect."
        )
    else:
        lines.append("The traversal balances both diagonal affects equally.")

    response = result.get("response", "").strip()
    if response:
        snippet = response.splitlines()[0][:80]
        lines.append(f'Model response begins: "{snippet}..."')

    return "\n".join(lines)


def narratives_from_results(matrix: EigenKoanMatrix, results: List[Dict]) -> List[str]:
    """Generate narratives for a list of traversal results."""
    return [traversal_to_narrative(matrix, r) for r in results]

