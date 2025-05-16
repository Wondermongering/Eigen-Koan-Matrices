import pytest

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
