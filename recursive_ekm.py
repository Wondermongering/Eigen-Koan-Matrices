# recursive_ekm.py - Implementing nested Eigen-Koan Matrices

import json
import datetime
from typing import Dict, Tuple, List, Optional, Callable
from rich.console import Console

from eigen_koan_matrix import EigenKoanMatrix, create_philosophical_ekm, \
    create_creative_writing_ekm, create_scientific_explanation_ekm
from ekm_generator import EKMGenerator

console = Console()

class RecursiveEKM:
    """Represents a nested Eigen-Koan Matrix structure."""

    def __init__(self, root_matrix: EigenKoanMatrix, name: str = "Recursive EKM"):
        self.root_matrix = root_matrix
        self.name = name
        self.sub_matrices: Dict[Tuple[int, int], EigenKoanMatrix] = {}

    def add_sub_matrix(self, row: int, col: int, matrix: EigenKoanMatrix):
        """Attach a sub-matrix to a cell in the root matrix."""
        self.sub_matrices[(row, col)] = matrix

    def generate_multi_level_prompt(
        self,
        primary_path: List[int],
        sub_paths: Optional[Dict[Tuple[int, int], List[int]]] = None,
        include_metacommentary: bool = True,
    ) -> str:
        """Generate a prompt that includes traversals of sub-matrices."""
        sub_paths = sub_paths or {}
        sections = [
            f"# Root Matrix: {self.root_matrix.name}",
            self.root_matrix.generate_micro_prompt(primary_path, include_metacommentary),
        ]

        for row, col in enumerate(primary_path):
            key = (row, col)
            if key in self.sub_matrices:
                sub_matrix = self.sub_matrices[key]
                path = sub_paths.get(key, list(range(sub_matrix.size)))
                sections.append(
                    f"\n# Sub Matrix at ({row},{col}): {sub_matrix.name}"
                )
                sections.append(sub_matrix.generate_micro_prompt(path, include_metacommentary))
        return "\n\n".join(sections)

    def traverse(
        self,
        model_fn: Callable[[str], str],
        primary_path: Optional[List[int]] = None,
        sub_paths: Optional[Dict[Tuple[int, int], List[int]]] = None,
        include_metacommentary: bool = True,
    ) -> Dict:
        """Traverse the recursive structure using a model function."""
        if primary_path is None:
            primary_path = list(range(self.root_matrix.size))
        prompt = self.generate_multi_level_prompt(primary_path, sub_paths, include_metacommentary)
        try:
            response = model_fn(prompt)
        except Exception as e:
            response = f"Error querying model: {str(e)}"
        return {
            "prompt": prompt,
            "response": response,
            "primary_path": primary_path,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    def visualize(self):
        """Visualize the root matrix and list sub-matrices."""
        console.print(f"[bold]Recursive EKM: {self.name}[/bold]")
        self.root_matrix.visualize()
        if self.sub_matrices:
            console.print("\n[bold]Sub-matrices:[/bold]")
            for (row, col), matrix in self.sub_matrices.items():
                console.print(f"({row}, {col}): {matrix.name}")

    def to_json(self) -> str:
        """Serialize the recursive matrix to JSON."""
        data = {
            "name": self.name,
            "root_matrix": json.loads(self.root_matrix.to_json()),
            "sub_matrices": {
                f"{r},{c}": json.loads(m.to_json())
                for (r, c), m in self.sub_matrices.items()
            },
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "RecursiveEKM":
        """Create a RecursiveEKM from a JSON string."""
        data = json.loads(json_str)
        root = EigenKoanMatrix.from_json(json.dumps(data["root_matrix"]))
        obj = cls(root_matrix=root, name=data.get("name", "Recursive EKM"))
        for key, mdata in data.get("sub_matrices", {}).items():
            r, c = map(int, key.split(","))
            obj.sub_matrices[(r, c)] = EigenKoanMatrix.from_json(json.dumps(mdata))
        return obj


def create_example_recursive_ekm() -> RecursiveEKM:
    """Create a simple recursive matrix for demonstration."""
    root = create_philosophical_ekm()
    rec = RecursiveEKM(root_matrix=root, name="Nested Philosophical Inquiry")
    rec.add_sub_matrix(0, 0, create_creative_writing_ekm())
    rec.add_sub_matrix(2, 3, create_scientific_explanation_ekm())
    return rec

if __name__ == "__main__":
    generator = EKMGenerator()
    recursive_ekm = create_example_recursive_ekm()
    recursive_ekm.visualize()
    prompt = recursive_ekm.generate_multi_level_prompt(list(range(recursive_ekm.root_matrix.size)))
    console.print(prompt)
