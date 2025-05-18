from tests import patch_external_libs


def test_traversal_to_narrative_basic():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from narrative_extractor import traversal_to_narrative

        ekm = create_random_ekm(2)

        def dummy_model(prompt: str) -> str:
            return "ok"

        result = ekm.traverse(dummy_model, path=[0, 1], include_metacommentary=False)
        text = traversal_to_narrative(ekm, result)

        assert isinstance(text, str)
        assert ekm.task_rows[0] in text
        assert ekm.constraint_cols[1] in text
        assert ekm.main_diagonal.name in text
        assert ekm.anti_diagonal.name in text
