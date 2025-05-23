from tests import patch_external_libs


def test_meta_generate_matrices():
    with patch_external_libs():
        from meta_ekm import MetaEKMSystem
        from ekm_generator import EKMGenerator
        from ekm_stack import EKMExperiment

        class DummyGen(EKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]
            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]
            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        class DummyExp(EKMExperiment):
            def run(self, model_runners):
                return {m_id: {model: [] for model in self.models} for m_id in self.matrices}
            def analyze(self, results=None):
                return {m_id: {"constraint_preservation": {}} for m_id in self.matrices}

        system = MetaEKMSystem(research_questions=["RQ1", "RQ2"], models=["m"], generator=DummyGen(), experiment_cls=DummyExp)
        mats = system.generate_matrices()
        assert len(mats) == 2
        for matrix in mats.values():
            assert matrix.size > 0


def test_meta_adaptive_experiment_basic():
    with patch_external_libs():
        from meta_ekm import MetaEKMSystem
        from ekm_generator import EKMGenerator
        from ekm_stack import EKMExperiment

        class DummyGen(EKMGenerator):
            def _select_diverse_elements(self, elements, count, embedding_key=None):
                return elements[:count]
            def _find_contrastive_pair(self, elements):
                return elements[0], elements[1]
            def _select_emotion_tokens(self, emotion_name, count, excluded_tokens=None):
                return [f"{emotion_name}_{i}" for i in range(count)]

        class DummyExp(EKMExperiment):
            def run(self, model_runners):
                return {m_id: {model: [] for model in self.models} for m_id in self.matrices}
            def analyze(self, results=None):
                return {m_id: {"constraint_preservation": {}} for m_id in self.matrices}

        def dummy(prompt: str) -> str:
            return "ok"

        system = MetaEKMSystem(research_questions=["RQ"], models=["m"], generator=DummyGen(), experiment_cls=DummyExp)
        runners = {"m": dummy}
        history = system.run_adaptive_experiment(runners, iterations=1, paths_per_matrix=1)
        assert len(history) == 1
        assert "analysis" in history[0]


def test_create_instruction_hierarchy_matrix():
    with patch_external_libs():
        from research_questions import create_instruction_hierarchy_matrix
        from eigen_koan_matrix import EigenKoanMatrix

        matrix = create_instruction_hierarchy_matrix()
        assert isinstance(matrix, EigenKoanMatrix)
        assert matrix.size == 4
