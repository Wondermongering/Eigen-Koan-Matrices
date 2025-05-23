"""Top-level package for the Eigen-Koan Matrices framework."""

from .eigen_koan_matrix import (
    EigenKoanMatrix,
    DiagonalAffect,
    create_random_ekm,
    create_philosophical_ekm,
    create_creative_writing_ekm,
    create_scientific_explanation_ekm,
)

from .recursive_ekm import RecursiveEKM
from .ekm_stack import EKMExperiment
from .ekm_db import EKMDatabase
from .ekm_distributed_runner import run_distributed_experiment
from .ekm_generator import EKMGenerator
from .adaptive_matrix import AdaptiveEigenKoanMatrix, AdaptationEnv
from .adaptive_sequence import AdaptiveTestingSequence
from .ekm_repository import EKMRepository
from .meta_ekm import MetaEKMSystem
from .narrative_extractor import traversal_to_narrative, narratives_from_results
from .natural_language_generator import NaturalLanguageEKMGenerator
from .explanation_generator import generate_explanation
from .research_questions import create_specialized_matrices
from .standard_suite_definitions import (
    PrometheusEKMRegistry,
    load_prometheus_suite_for_experimentation,
)

__all__ = [
    "EigenKoanMatrix",
    "DiagonalAffect",
    "create_random_ekm",
    "create_philosophical_ekm",
    "create_creative_writing_ekm",
    "create_scientific_explanation_ekm",
    "RecursiveEKM",
    "EKMExperiment",
    "EKMDatabase",
    "run_distributed_experiment",
    "EKMGenerator",
    "AdaptiveEigenKoanMatrix",
    "AdaptationEnv",
    "AdaptiveTestingSequence",
    "EKMRepository",
    "MetaEKMSystem",
    "traversal_to_narrative",
    "narratives_from_results",
    "NaturalLanguageEKMGenerator",
    "generate_explanation",
    "create_specialized_matrices",
