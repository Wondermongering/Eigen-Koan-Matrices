from tests import patch_external_libs


def test_adversarial_generator_basic():
    with patch_external_libs():
        from adversarial_generator import AdversarialEKMGenerator
        from eigen_koan_matrix import EigenKoanMatrix

        class DummyGenerator(AdversarialEKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]

            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]

            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        def dummy_model(prompt: str) -> str:
            return "response"

        gen = DummyGenerator()
        matrix = gen.optimize_against_model(dummy_model, size=2, population=2, generations=1, num_paths=1)
        assert isinstance(matrix, EigenKoanMatrix)
        score = gen.score_matrix(matrix, dummy_model, num_paths=1)
        assert isinstance(score, (int, float))


def test_tournament_basic():
    with patch_external_libs():
        from adversarial_generator import AdversarialEKMGenerator, MatrixModelTournament

        class DummyGenerator(AdversarialEKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]

            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]

            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        def model_a(prompt: str) -> str:
            return "a"

        def model_b(prompt: str) -> str:
            return "b"

        gen = DummyGenerator()
        tour = MatrixModelTournament({"A": model_a, "B": model_b}, gen)
        history = tour.run(rounds=1, size=2, population=2, generations=1, num_paths=1)
        assert isinstance(history, list)
        assert len(history) == 1
        assert set(history[0].keys()) == {"A", "B"}
