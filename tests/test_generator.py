from tests import patch_external_libs


def test_generate_ekm_basic():
    with patch_external_libs():
        from ekm_generator import EKMGenerator
        from eigen_koan_matrix import EigenKoanMatrix

        class DummyGenerator(EKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]

            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]

            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        gen = DummyGenerator()
        ekm = gen.generate_ekm(size=2)
        assert isinstance(ekm, EigenKoanMatrix)
        assert ekm.size == 2
        assert len(ekm.task_rows) == 2
        assert len(ekm.constraint_cols) == 2
        for i in range(2):
            assert ekm.cells[i][i] != "{NULL}"
            assert ekm.cells[i][1 - i] != "{NULL}"
        assert ekm.name == "Generated EKM 2x2"


def test_generate_themed_matrices_basic():
    with patch_external_libs():
        from ekm_generator import EKMGenerator
        from eigen_koan_matrix import EigenKoanMatrix

        class DummyGenerator(EKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]

            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]

            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        gen = DummyGenerator()
        themes = ["ethics", "custom"]
        mats = gen.generate_themed_matrices(themes, size=2)
        assert set(mats.keys()) == set(themes)
        for matrix in mats.values():
            assert isinstance(matrix, EigenKoanMatrix)
            assert matrix.size == 2


def test_create_reality_blurring_matrix():
    with patch_external_libs():
        from research_questions import create_reality_blurring_matrix
        from eigen_koan_matrix import EigenKoanMatrix

        matrix = create_reality_blurring_matrix()
        assert isinstance(matrix, EigenKoanMatrix)
        assert matrix.size == 4
