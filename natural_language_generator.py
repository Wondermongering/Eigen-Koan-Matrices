# natural_language_generator.py - Generate EKMs from natural language questions
"""Utility to translate research questions into Eigen-Koan Matrix designs."""

import random
import re
from typing import List, Tuple, Optional

from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect
from ekm_generator import EKMGenerator


class NaturalLanguageEKMGenerator(EKMGenerator):
    """Generate EKMs from free-form research questions."""

    _STOPWORDS = {
        "the", "a", "an", "of", "in", "on", "under", "about", "for", "to",
        "with", "and", "or", "how", "what", "why", "when", "where", "is",
        "are", "do", "does", "models", "model", "llms", "large", "language",
    }

    _TASK_VERBS = [
        "Analyze", "Explain", "Explore", "Evaluate", "Discuss", "Contrast",
        "Map", "Reconcile", "Investigate", "Synthesize",
    ]

    _CONSTRAINT_TEMPLATES = [
        "while emphasizing {}",
        "through the lens of {}",
        "contrasting {} and other factors",
        "despite {} considerations",
        "in light of {}",
    ]

    def _extract_keywords(self, question: str) -> List[str]:
        words = re.findall(r"[A-Za-z']+", question.lower())
        return [w for w in words if w not in self._STOPWORDS]

    def _generate_tasks(self, keywords: List[str], size: int) -> List[str]:
        tasks: List[str] = []
        for i in range(size):
            verb = self._TASK_VERBS[i % len(self._TASK_VERBS)]
            kw = keywords[i % len(keywords)] if keywords else self.word_banks["domain_words"][i]
            tasks.append(f"{verb} {kw}")
        return tasks

    def _generate_constraints(self, keywords: List[str], size: int) -> List[str]:
        constraints: List[str] = []
        for i in range(size):
            template = self._CONSTRAINT_TEMPLATES[i % len(self._CONSTRAINT_TEMPLATES)]
            kw = keywords[(i + 1) % len(keywords)] if keywords else self.word_banks["domain_words"][i + 1]
            constraints.append(template.format(kw))
        return constraints

    def _find_contrastive_pair(self, elements: List[str]) -> Tuple[str, str]:
        """Return a simple contrastive pair without heavy computation."""
        if len(elements) < 2:
            raise ValueError("Need at least 2 elements to find a contrastive pair")
        return elements[0], elements[1]

    def _select_emotion_tokens(
        self, emotion_name: str, count: int, excluded_tokens: Optional[set] = None
    ) -> List[str]:
        """Simplified emotion token selection avoiding heavy math."""
        excluded_tokens = excluded_tokens or set()
        available = [t for t in self.word_banks["emotional_tokens"] if t not in excluded_tokens]
        if len(available) < count:
            available.extend([w for w in self.word_banks["domain_words"] if w not in excluded_tokens])
        random.shuffle(available)
        return available[:count]

    def generate_from_question(
        self,
        question: str,
        size: int = 4,
        balancing_emotions: Optional[Tuple[str, str]] = None,
    ) -> EigenKoanMatrix:
        """Create an EKM tailored to a research question."""
        keywords = self._extract_keywords(question)
        if not keywords:
            # Fallback to generic keywords if extraction fails
            keywords = [w for w in self.word_banks["domain_words"][:size]]

        tasks = self._generate_tasks(keywords, size)
        constraints = self._generate_constraints(keywords, size)

        name = f"NL-EKM: {question}"[:60]
        description = f"Matrix generated from research question: {question}"

        if balancing_emotions is None:
            emotion_names = list(self.emotion_space.keys())
            main_emotion, anti_emotion = self._find_contrastive_pair(emotion_names)
        else:
            main_emotion, anti_emotion = balancing_emotions

        main_tokens = self._select_emotion_tokens(main_emotion, size)
        anti_tokens = self._select_emotion_tokens(
            anti_emotion, size, excluded_tokens=set(main_tokens)
        )

        main_valence, main_arousal = self.emotion_space.get(main_emotion, (0.5, 0.5))
        anti_valence, anti_arousal = self.emotion_space.get(anti_emotion, (-0.5, 0.5))

        main_diag = DiagonalAffect(
            name=main_emotion.title(),
            tokens=main_tokens,
            description=f"Emotional quality of {main_emotion}",
            valence=main_valence,
            arousal=main_arousal,
        )

        anti_diag = DiagonalAffect(
            name=anti_emotion.title(),
            tokens=anti_tokens,
            description=f"Emotional quality of {anti_emotion}",
            valence=anti_valence,
            arousal=anti_arousal,
        )

        cells = [["{NULL}" for _ in range(size)] for _ in range(size)]
        for i in range(size):
            cells[i][i] = main_tokens[i]
            cells[i][size - 1 - i] = anti_tokens[i]

        return EigenKoanMatrix(
            size=size,
            task_rows=tasks,
            constraint_cols=constraints,
            main_diagonal=main_diag,
            anti_diagonal=anti_diag,
            cells=cells,
            name=name,
            description=description,
        )

