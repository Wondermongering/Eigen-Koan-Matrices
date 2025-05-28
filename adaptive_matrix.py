# adaptive_matrix.py - Dynamic Adaptive Eigen-Koan Matrices
# ---------------------------------------------------------

"""Implement dynamic matrices that adapt based on model feedback.

This module introduces ``AdaptiveEigenKoanMatrix`` which wraps an existing
``EigenKoanMatrix`` and applies a very small reinforcement learning style
update to guide future traversal choices. It also provides a simple
``AdaptationEnv`` for running adaptation simulations.
"""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Dict

from eigen_koan_matrix import EigenKoanMatrix


class AdaptiveEigenKoanMatrix:
    """A wrapper around :class:`EigenKoanMatrix` with adaptive behaviour."""

    def __init__(self, matrix: EigenKoanMatrix):
        self.matrix = matrix
        # Q-values for selecting column at each row
        self.q_values: List[List[float]] = [
            [0.0 for _ in range(matrix.size)] for _ in range(matrix.size)
        ]

    @classmethod
    def from_ekm(cls, matrix: EigenKoanMatrix) -> "AdaptiveEigenKoanMatrix":
        """Create an adaptive matrix from an existing EKM."""
        return cls(matrix)

    def select_path(self, epsilon: float = 0.1, rng: Optional[random.Random] = None) -> List[int]:
        """Select a path using epsilon-greedy strategy based on Q-values.

        Args:
            epsilon: The probability of choosing a random action (exploration).
                     A value of 0 means greedy selection, 1 means fully random.
            rng: An optional random number generator for reproducible path selection.
                 If None, the global `random` module is used.

        Returns:
            A list of integers representing the selected column indices for each row.
        """
        rng = rng or random
        path: List[int] = []
        for row in range(self.matrix.size):
            if rng.random() < epsilon:
                # Explore: select a random column
                col = rng.randint(0, self.matrix.size - 1)
            else:
                # Exploit: select the column with the highest Q-value
                # If multiple columns have the same max Q-value, the one with the lowest index is chosen.
                col = self.q_values[row].index(max(self.q_values[row]))
            path.append(col)
        return path

    def update_q_values(
        self,
        path: List[int],
        reward: float,
        alpha: float = 0.1,  # Learning rate
        gamma: float = 0.9,  # Discount factor
    ) -> None:
        """Update Q-values for the chosen path using the Q-learning rule.

        Args:
            path: The path (list of column indices) taken through the matrix.
            reward: The reward received for taking that path.
            alpha: The learning rate, determining how much new information overrides old information.
            gamma: The discount factor, determining the importance of future rewards.
        """
        for row, col in enumerate(path):
            current = self.q_values[row][col]
            future_estimate = max(self.q_values[row])
            self.q_values[row][col] = current + alpha * (
                reward + gamma * future_estimate - current
            )

    def run_episode(
        self,
        model_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
        *,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.9,
        include_metacommentary: bool = True,
        seed: Optional[int] = None,
    ) -> Dict:
        """Run a single adaptive traversal episode."""
        rng = random.Random(seed) if seed is not None else random
        path = self.select_path(epsilon=epsilon, rng=rng)
        prompt = self.matrix.generate_micro_prompt(path, include_metacommentary)
        response = model_fn(prompt)
        reward = reward_fn(prompt, response)
        self.update_q_values(path, reward, alpha=alpha, gamma=gamma)
        return {
            "path": path,
            "prompt": prompt,
            "response": response,
            "reward": reward,
        }


class AdaptationEnv:
    """A lightweight environment for running adaptation simulations."""

    def __init__(
        self,
        matrix: AdaptiveEigenKoanMatrix,
        model_fn: Callable[[str], str],
        reward_fn: Callable[[str, str], float],
    ) -> None:
        self.matrix = matrix
        self.model_fn = model_fn
        self.reward_fn = reward_fn

    def run(self, episodes: int = 10, *, epsilon: float = 0.1) -> List[Dict]:
        """Run multiple episodes and return the collected results."""
        results = []
        for _ in range(episodes):
            result = self.matrix.run_episode(
                self.model_fn,
                self.reward_fn,
                epsilon=epsilon,
            )
            results.append(result)
        return results

