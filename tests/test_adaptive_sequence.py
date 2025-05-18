from tests import patch_external_libs


def test_generate_from_analysis():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from adaptive_sequence import AdaptiveTestingSequence
        from ekm_generator import EKMGenerator

        class DummyGen(EKMGenerator):
            def generate_ekm(self, size=2, theme="", balancing_emotions=None, name=None, description=None):
                return create_random_ekm(size)

        base = create_random_ekm(2)
        analysis = {
            base.id: {
                "constraint_preservation": {
                    "model": {0: 1, 1: 5}
                }
            }
        }

        seq = AdaptiveTestingSequence(generator=DummyGen())
        mats = seq.generate_from_analysis({base.id: base}, analysis)
        assert len(mats) == 1
        new_matrix = list(mats.values())[0]
        assert new_matrix.size == base.size
        assert new_matrix.constraint_cols[0] == base.constraint_cols[1]
