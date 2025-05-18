from tests import patch_external_libs


def test_adaptive_matrix_updates():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from adaptive_matrix import AdaptiveEigenKoanMatrix

        ekm = create_random_ekm(2)
        adaptive = AdaptiveEigenKoanMatrix.from_ekm(ekm)

        def dummy_model(prompt: str) -> str:
            return prompt

        def reward_fn(prompt: str, response: str) -> float:
            return float(len(response))

        q_before = [row[:] for row in adaptive.q_values]
        adaptive.run_episode(dummy_model, reward_fn, epsilon=0.0)
        changed = any(
            adaptive.q_values[r][c] != q_before[r][c]
            for r in range(2)
            for c in range(2)
        )
        assert changed


def test_adaptation_env_runs():
    with patch_external_libs():
        from eigen_koan_matrix import create_random_ekm
        from adaptive_matrix import AdaptiveEigenKoanMatrix, AdaptationEnv

        ekm = create_random_ekm(2)
        adaptive = AdaptiveEigenKoanMatrix.from_ekm(ekm)

        def dummy_model(prompt: str) -> str:
            return "x"

        def reward_fn(prompt: str, response: str) -> float:
            return 1.0

        env = AdaptationEnv(adaptive, dummy_model, reward_fn)
        results = env.run(episodes=3, epsilon=0.0)
        assert len(results) == 3

