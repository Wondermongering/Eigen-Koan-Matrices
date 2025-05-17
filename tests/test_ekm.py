from tests import pytest

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect, create_random_ekm


def test_generate_micro_prompt_valid_path():
    size = 3
    ekm = create_random_ekm(size)
    path = [0, 1, 2]
    prompt = ekm.generate_micro_prompt(path)
    assert isinstance(prompt, str)
    # Ensure that each task and selected constraint appears in the prompt
    for row, col in enumerate(path):
        assert ekm.task_rows[row] in prompt
        assert ekm.constraint_cols[col] in prompt


def test_generate_micro_prompt_invalid_path_length():
    ekm = create_random_ekm(3)
    # Path shorter than matrix size
    with pytest.raises(ValueError):
        ekm.generate_micro_prompt([0, 1])


def test_generate_micro_prompt_invalid_column_index():
    ekm = create_random_ekm(3)
    # Column index out of bounds
    with pytest.raises(ValueError):
        ekm.generate_micro_prompt([0, 1, 3])


def test_generate_micro_prompt_with_metacommentary():
    ekm = create_random_ekm(2)
    path = [0, 1]
    prompt = ekm.generate_micro_prompt(path, include_metacommentary=True)
    assert "After completing this task" in prompt


def test_traverse_deterministic_with_seed():
    ekm = create_random_ekm(3)

    def dummy_model(prompt: str) -> str:
        return "ok"

    result1 = ekm.traverse(dummy_model, include_metacommentary=False, seed=123)
    result2 = ekm.traverse(dummy_model, include_metacommentary=False, seed=123)

    assert result1["path"] == result2["path"]
    assert result1["prompt"] == result2["prompt"]


def test_multi_traverse_deterministic_with_seed():
    ekm = create_random_ekm(3)

    def dummy_model(prompt: str) -> str:
        return "ok"

    runs1 = ekm.multi_traverse(dummy_model, num_paths=3, include_metacommentary=False, seed=42)
    runs2 = ekm.multi_traverse(dummy_model, num_paths=3, include_metacommentary=False, seed=42)

    paths1 = [r["path"] for r in runs1]
    paths2 = [r["path"] for r in runs2]

    assert paths1 == paths2
