# adversarial_generator.py - Generate matrices to challenge models

import random
from typing import List, Callable, Dict, Tuple

from eigen_koan_matrix import EigenKoanMatrix
from ekm_generator import EKMGenerator


class AdversarialEKMGenerator(EKMGenerator):
    """Generator that searches for matrices which are difficult for a model."""

    def score_matrix(
        self,
        matrix: EigenKoanMatrix,
        model_fn: Callable[[str], str],
        num_paths: int = 3,
    ) -> float:
        """Evaluate a matrix by querying the model on random paths."""
        total = 0
        for _ in range(num_paths):
            result = matrix.traverse(model_fn, include_metacommentary=False)
            total += len(result["response"])  # simple length based score
        return total / max(1, num_paths)

    def mutate_matrix(self, matrix: EigenKoanMatrix) -> EigenKoanMatrix:
        """Return a slightly modified copy of the matrix."""
        new_tasks = matrix.task_rows[:]
        new_constraints = matrix.constraint_cols[:]
        if random.random() < 0.5:
            i, j = random.sample(range(matrix.size), 2)
            new_tasks[i], new_tasks[j] = new_tasks[j], new_tasks[i]
        else:
            i, j = random.sample(range(matrix.size), 2)
            new_constraints[i], new_constraints[j] = (
                new_constraints[j],
                new_constraints[i],
            )
        return EigenKoanMatrix(
            size=matrix.size,
            task_rows=new_tasks,
            constraint_cols=new_constraints,
            main_diagonal=matrix.main_diagonal,
            anti_diagonal=matrix.anti_diagonal,
            cells=[row[:] for row in matrix.cells],
            name=matrix.name,
            description=matrix.description,
        )

    def optimize_against_model(
        self,
        model_fn: Callable[[str], str],
        size: int = 4,
        population: int = 4,
        generations: int = 3,
        num_paths: int = 2,
    ) -> EigenKoanMatrix:
        """Search for an adversarial matrix for a given model."""
        pop = [self.generate_ekm(size=size) for _ in range(population)]
        for _ in range(generations):
            scored = [
                (self.score_matrix(m, model_fn, num_paths=num_paths), m) for m in pop
            ]
            scored.sort(key=lambda x: x[0])
            survivors = [m for _, m in scored[: max(1, population // 2)]]
            pop = survivors[:]
            while len(pop) < population:
                parent = random.choice(survivors)
                pop.append(self.mutate_matrix(parent))
        final_scores = [
            (self.score_matrix(m, model_fn, num_paths=num_paths), m) for m in pop
        ]
        final_scores.sort(key=lambda x: x[0])
        return final_scores[0][1]


class MatrixModelTournament:
    """Run matrices against models in iterative rounds."""

    def __init__(self, models: Dict[str, Callable[[str], str]], generator: AdversarialEKMGenerator):
        self.models = models
        self.generator = generator
        self.history: List[Dict[str, Tuple[float, EigenKoanMatrix]]] = []

    def run(self, rounds: int = 3, **gen_kwargs):
        """Execute the tournament."""
        for _ in range(rounds):
            round_results: Dict[str, Tuple[float, EigenKoanMatrix]] = {}
            for name, fn in self.models.items():
                matrix = self.generator.optimize_against_model(fn, **gen_kwargs)
                score = self.generator.score_matrix(matrix, fn, num_paths=gen_kwargs.get("num_paths", 2))
                round_results[name] = (score, matrix)
            self.history.append(round_results)
        return self.history

