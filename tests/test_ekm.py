from tests import pytest, patch_external_libs


def test_generate_micro_prompt_valid_path():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        size = 3
        ekm = create_random_ekm(size)
        path = [0, 1, 2]
        prompt = ekm.generate_micro_prompt(path)
        assert isinstance(prompt, str)
        for row, col in enumerate(path):
            assert ekm.task_rows[row] in prompt
            assert ekm.constraint_cols[col] in prompt


def test_generate_micro_prompt_invalid_path_length():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(3)
        with pytest.raises(ValueError):
            ekm.generate_micro_prompt([0, 1])


def test_generate_micro_prompt_invalid_column_index():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(3)
        with pytest.raises(ValueError):
            ekm.generate_micro_prompt([0, 1, 3])


def test_generate_micro_prompt_with_metacommentary():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(2)
        path = [0, 1]
        prompt = ekm.generate_micro_prompt(path, include_metacommentary=True)
        assert "After completing this task" in prompt


def test_traverse_deterministic_with_seed():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(3)

        def dummy_model(prompt: str) -> str:
            return "ok"

        result1 = ekm.traverse(dummy_model, include_metacommentary=False, seed=123)
        result2 = ekm.traverse(dummy_model, include_metacommentary=False, seed=123)
        assert result1["path"] == result2["path"]
        assert result1["prompt"] == result2["prompt"]


def test_multi_traverse_deterministic_with_seed():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(3)

        def dummy_model(prompt: str) -> str:
            return "ok"

        runs1 = ekm.multi_traverse(dummy_model, num_paths=3, include_metacommentary=False, seed=42)
        runs2 = ekm.multi_traverse(dummy_model, num_paths=3, include_metacommentary=False, seed=42)
        paths1 = [r["path"] for r in runs1]
        paths2 = [r["path"] for r in runs2]
        assert paths1 == paths2


def test_reality_blurring_path_and_prompt():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(4)
        path = ekm.generate_reality_blurring_path(seed=1)
        assert len(path) == ekm.size
        assert all(0 <= p < ekm.size for p in path)

        prompt = ekm.generate_reality_blurring_prompt(path)
        assert "[MODEL-GUESS]" in prompt


def test_traverse_reality_blur():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        ekm = create_random_ekm(3)

        def dummy_model(prompt: str) -> str:
            return "[FACT] info [MODEL-GUESS] guess"

        result = ekm.traverse_reality_blur(dummy_model, seed=0)
        assert result["fact_mentions"] == 1
        assert result["guess_mentions"] == 1
