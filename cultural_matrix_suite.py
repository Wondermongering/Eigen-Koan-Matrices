"""Cultural Matrix Suite
=======================

This module defines Eigen-Koan Matrices (EKMs) focused on
cross-cultural contexts and multilingual communication.
Each matrix explores how language models adapt phrasing and
emotional framing when operating within different cultural
norms.
"""

from typing import Dict
from eigen_koan_matrix import EigenKoanMatrix, DiagonalAffect


def create_japanese_politeness_matrix() -> EigenKoanMatrix:
    """Matrix exploring levels of politeness in Japanese communication."""
    tasks = [
        "Request assistance from an elder",
        "Apologize for an inconvenience",
        "Offer a business proposal",
        "Decline an invitation",
    ]

    constraints = [
        "using honorific keigo",
        "in casual plain form",
        "with indirect hints",
        "in humble language",
    ]

    respect = DiagonalAffect(
        name="Respectful Distance",
        tokens=["keigo", "polite", "esteem", "reverence"],
        description="Maintaining social harmony through formal respect",
        valence=0.7,
        arousal=0.5,
    )

    warmth = DiagonalAffect(
        name="Warm Familiarity",
        tokens=["uchi", "friend", "close", "ease"],
        description="Emphasizing closeness and comfort",
        valence=0.6,
        arousal=0.4,
    )

    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=respect,
        anti_diagonal=warmth,
        name="Japanese Politeness Matrix",
        description="How phrasing shifts across politeness levels in Japanese",
    )


def create_arabic_hospitality_matrix() -> EigenKoanMatrix:
    """Matrix examining hospitality rituals in Arabic culture."""
    tasks = [
        "Welcome a guest to your home",
        "Offer food to a traveler",
        "Introduce a respected elder",
        "Say farewell after a visit",
    ]

    constraints = [
        "with traditional greetings",
        "emphasizing generosity",
        "referencing local customs",
        "using religious blessings",
    ]

    generosity = DiagonalAffect(
        name="Heartfelt Generosity",
        tokens=["karam", "welcome", "abundance", "open arms"],
        description="The joy of offering hospitality",
        valence=0.8,
        arousal=0.6,
    )

    formality = DiagonalAffect(
        name="Traditional Etiquette",
        tokens=["honor", "respect", "ritual", "custom"],
        description="Observing formal codes of conduct",
        valence=0.5,
        arousal=0.4,
    )

    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=generosity,
        anti_diagonal=formality,
        name="Arabic Hospitality Matrix",
        description="Explores ritual expressions of welcome and respect",
    )


def create_spanish_formality_matrix() -> EigenKoanMatrix:
    """Matrix contrasting formal and informal Spanish usage."""
    tasks = [
        "Greet a coworker",
        "Request a favor",
        "Explain a mistake",
        "Offer congratulations",
    ]

    constraints = [
        "using formal usted forms",
        "using informal tu forms",
        "with regional idioms",
        "avoiding colloquialisms",
    ]

    deference = DiagonalAffect(
        name="Formal Deference",
        tokens=["respeto", "distancia", "cortesía", "seriedad"],
        description="Maintaining courteous distance",
        valence=0.6,
        arousal=0.4,
    )

    camaraderie = DiagonalAffect(
        name="Friendly Camaraderie",
        tokens=["amistad", "confianza", "calidez", "cercanía"],
        description="Informal warmth among peers",
        valence=0.7,
        arousal=0.5,
    )

    return EigenKoanMatrix(
        size=4,
        task_rows=tasks,
        constraint_cols=constraints,
        main_diagonal=deference,
        anti_diagonal=camaraderie,
        name="Spanish Formality Matrix",
        description="How Spanish shifts between formal and informal registers",
    )


def create_cultural_matrices() -> Dict[str, EigenKoanMatrix]:
    """Return all cultural EKMs in a dictionary."""
    return {
        "jp_politeness": create_japanese_politeness_matrix(),
        "arabic_hospitality": create_arabic_hospitality_matrix(),
        "spanish_formality": create_spanish_formality_matrix(),
    }

