# recursive_ekm.py - Implementing nested Eigen-Koan Matrices

# ekm_generator.py - Automated Eigen-Koan Matrix generation
# --------------------------------------------------------

import json
import random
from typing import List, Dict, Tuple, Optional
from rich.console import Console

from ekm_generator import EKMGenerator

from eigen_koan_matrix import (
    EigenKoanMatrix,
    DiagonalAffect,
    create_philosophical_ekm,
    create_creative_writing_ekm,
    create_scientific_explanation_ekm,
)

console = Console()


class RecursiveEKM:
    """A container for a root Eigen-Koan Matrix with optional nested matrices."""

    def __init__(self,
                 root_matrix: EigenKoanMatrix,
                 name: str = "Recursive EKM",
                 description: str = ""):
        self.root_matrix = root_matrix
        self.name = name
        self.description = description
        # Mapping of (row, col) -> EigenKoanMatrix
        self.sub_matrices: Dict[Tuple[int, int], EigenKoanMatrix] = {}

    def add_sub_matrix(self, row: int, col: int, sub_matrix: EigenKoanMatrix):
        """Attach a sub-matrix to a specific cell in the root matrix."""
        self.sub_matrices[(row, col)] = sub_matrix

    def generate_multi_level_prompt(
        self,
        primary_path: List[int],
        sub_paths: Optional[Dict[Tuple[int, int], List[int]]] = None,
        include_metacommentary: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a prompt that traverses the root and any nested matrices."""
        rng = random.Random(seed) if seed is not None else random
        prompt = self.root_matrix.generate_micro_prompt(primary_path, include_metacommentary)

        for (row, col), matrix in self.sub_matrices.items():
            path = None
            if sub_paths and (row, col) in sub_paths:
                path = sub_paths[(row, col)]
            else:
                path = [rng.randint(0, matrix.size - 1) for _ in range(matrix.size)]

            sub_prompt = matrix.generate_micro_prompt(path, include_metacommentary)
            prompt += (
                f"\n\n[Sub-matrix {matrix.name} at ({row},{col})]\n" + sub_prompt
            )

        return prompt

    def traverse(
        self,
        model_fn: callable,
        primary_path: Optional[List[int]] = None,
        sub_paths: Optional[Dict[Tuple[int, int], List[int]]] = None,
        include_metacommentary: bool = True,
        seed: Optional[int] = None,
    ) -> Dict:
        """Generate a multi-level prompt and query a model."""
        rng = random.Random(seed) if seed is not None else random
        if primary_path is None:
            primary_path = [
                rng.randint(0, self.root_matrix.size - 1)
                for _ in range(self.root_matrix.size)
            ]

        prompt = self.generate_multi_level_prompt(
            primary_path, sub_paths, include_metacommentary, seed=seed
        )

        try:
            response = model_fn(prompt)
        except Exception as e:
            response = f"Error querying model: {e}"

        return {
            "matrix_name": self.name,
            "primary_path": primary_path,
            "prompt": prompt,
            "response": response,
        }

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "description": self.description,
            "root_matrix": json.loads(self.root_matrix.to_json()),
            "sub_matrices": {
                f"{row},{col}": json.loads(matrix.to_json())
                for (row, col), matrix in self.sub_matrices.items()
            },
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "RecursiveEKM":
        data = json.loads(json_str)
        root = EigenKoanMatrix.from_json(json.dumps(data["root_matrix"]))
        obj = cls(root_matrix=root, name=data.get("name", "Recursive EKM"), description=data.get("description", ""))
        for key, matrix_data in data.get("sub_matrices", {}).items():
            row, col = map(int, key.split(","))
            matrix = EigenKoanMatrix.from_json(json.dumps(matrix_data))
            obj.add_sub_matrix(row, col, matrix)
        return obj

    def visualize(self):
        """Visualize the root matrix and list nested matrices."""
        console.print(f"[bold underline]{self.name}[/bold underline]")
        self.root_matrix.visualize()
        if self.sub_matrices:
            console.print("\n[bold]Sub-matrices:[/bold]")
            for (row, col), matrix in self.sub_matrices.items():
                console.print(f"- Cell ({row}, {col}): {matrix.name} ({matrix.size}x{matrix.size})")

def create_example_recursive_ekm() -> RecursiveEKM:
    """Create a small example recursive matrix for demonstration."""
    root = create_philosophical_ekm()
    creative = create_creative_writing_ekm()
    scientific = create_scientific_explanation_ekm()

    rekm = RecursiveEKM(root_matrix=root, name="Example Recursive EKM")
    rekm.add_sub_matrix(0, 0, creative)
    rekm.add_sub_matrix(2, 3, scientific)

    return rekm


# Example usage
def example_generator_usage():
    """Demonstrate the EKM generator."""
    # Initialize generator
    generator = EKMGenerator()
    
    console.print("[bold]1. Generating a single EKM[/bold]")
    ekm = generator.generate_ekm(
        size=4,
        theme="consciousness",
        balancing_emotions=("wonder", "melancholy")
    )
    ekm.visualize()
    
    console.print("\n[bold]2. Generating themed matrices[/bold]")
    themes = ["ethics", "creativity", "science", "time"]
    themed_matrices = generator.generate_themed_matrices(themes)
    
    for theme, matrix in themed_matrices.items():
        console.print(f"\n[bold]Theme: {theme}[/bold]")
        matrix.visualize()
    
    console.print("\n[bold]3. Generating a matrix family[/bold]")
    base_theme = "Consciousness"
    variations = [
        ("Wonder/Dread", ("wonder", "dread")),
        ("Curiosity/Confusion", ("curiosity", "confusion")),
        ("Serenity/Anxiety", ("serenity", "anxiety")),
        ("Hope/Despair", ("hope", "despair"))
    ]
    
    family = generator.generate_matrix_family(base_theme, variations)
    
    for name, matrix in family.items():
        console.print(f"\n[bold]Variation: {name}[/bold]")
        matrix.visualize()
    
    return ekm, themed_matrices, family


if __name__ == "__main__":
    example_generator_usage()
