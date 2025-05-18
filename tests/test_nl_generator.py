from tests import patch_external_libs


def test_generate_from_question_basic():
    with patch_external_libs():
        from natural_language_generator import NaturalLanguageEKMGenerator
        from eigen_koan_matrix import EigenKoanMatrix

        gen = NaturalLanguageEKMGenerator()
        question = "How do language models handle conflicting ethical principles?"
        ekm = gen.generate_from_question(question, size=3)
        assert isinstance(ekm, EigenKoanMatrix)
        assert ekm.size == 3
        assert len(ekm.task_rows) == 3
        assert len(ekm.constraint_cols) == 3
        # ensure some keyword from question appears in tasks or constraints
        keywords = [w for w in question.lower().split() if w not in gen._STOPWORDS]
        assert any(any(k in t.lower() for k in keywords) for t in ekm.task_rows)

