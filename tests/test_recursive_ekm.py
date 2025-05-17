from eigen_koan_matrix import create_random_ekm
from recursive_ekm import RecursiveEKM


def test_recursive_ekm_traverse_basic():
    root = create_random_ekm(2)
    sub = create_random_ekm(2)
    rec = RecursiveEKM(root_matrix=root, name="Root")
    rec.add_sub_matrix(0, 0, sub)

    def dummy_model(prompt: str) -> str:
        return "dummy-response"

    result = rec.traverse(
        dummy_model,
        primary_path=[0, 1],
        sub_paths={(0, 0): [1, 0]},
        include_metacommentary=False,
    )
    assert result["matrix_name"] == "Root"
    assert "Sub-matrix" in result["prompt"]
    assert result["response"] == "dummy-response"

def test_recursive_ekm_traverse_fields():
    root = create_random_ekm(2)
    rec = RecursiveEKM(root_matrix=root, name="Root2")

    def dummy(prompt: str) -> str:
        return "ok"

    result = rec.traverse(dummy, primary_path=[0, 1], include_metacommentary=False)
    assert set(result.keys()) == {"matrix_name", "primary_path", "prompt", "response"}
    assert result["matrix_name"] == "Root2"
