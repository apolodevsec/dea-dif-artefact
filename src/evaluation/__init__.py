"""Módulo de Avaliação Experimental e Baselines (Módulo 5)."""

from .partitioner import DataPartitioner
from .evaluator import ExperimentalEvaluator
from .hybrid import HybridFusionDetector
from .baselines import BaselineRunner
from .sensitivity import OFATSensitivityAnalyzer

__all__ = [
    "DataPartitioner",
    "ExperimentalEvaluator",
    "HybridFusionDetector",
    "BaselineRunner",
    "OFATSensitivityAnalyzer",
]

