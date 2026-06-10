from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class TestCase:
    question: str
    ground_truth_answer: str
    ground_truth_sources: List[str]
    relevant_chunks: List[str]

@dataclass
class EvaluationReport:
    aggregate: Dict[str, Any]
    per_question: List[Dict[str, Any]]
    user_id: str
    timestamp: Optional[str] = None
