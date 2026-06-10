"""
bot_evaluator

A framework-agnostic package for evaluating conversational AI systems.
"""

__version__ = "0.1.0"

from .evaluator import RAGEvaluator
from .models import EvaluationReport, TestCase

__all__ = ["RAGEvaluator", "EvaluationReport", "TestCase"]
