import json
from typing import Callable

from bot_evaluator.metrics import (
    context_precision,
    context_recall,
    answer_faithfulness,
    answer_relevance,
    source_accuracy,
    answer_f1
)

class RAGEvaluator:
    """
    A generic evaluator class for RAG pipelines.
    """

    def __init__(self, rag_function: Callable, user_id: str = "u001"):
        """
        :param rag_function: A callable that takes a string (question) and keyword args 
                             (like user_id) and returns a dict with the evaluation outputs.
        :param user_id: The ID of the user performing the evaluation.
        """
        self.rag_function = rag_function
        self.user_id = user_id

    def evaluate(self, eval_set_path: str) -> dict:
        """
        Loads golden dataset, runs rag_function on every question,
        computes all six metrics, returns aggregate + per-question results.
        """
        with open(eval_set_path, 'r', encoding='utf-8') as f:
            eval_set = json.load(f)
            
        results = {
            "aggregate": {},
            "per_question": [],
            "user_id": self.user_id
        }
        
        metrics_sum = {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "source_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        }
        
        count = len(eval_set)
        if count == 0:
            return results
        
        for entry in eval_set:
            entry_result = self.evaluate_single(entry)
            results["per_question"].append(entry_result)
            
            for key in metrics_sum:
                if key in ["precision", "recall", "f1"]:
                    metrics_sum[key] += entry_result["answer_f1"][key]
                else:
                    metrics_sum[key] += entry_result[key]
                    
        results["aggregate"] = {
            "context_precision": metrics_sum["context_precision"] / count,
            "context_recall": metrics_sum["context_recall"] / count,
            "faithfulness": metrics_sum["faithfulness"] / count,
            "answer_relevance": metrics_sum["answer_relevance"] / count,
            "source_accuracy": metrics_sum["source_accuracy"] / count,
            "answer_f1": {
                "precision": metrics_sum["precision"] / count,
                "recall": metrics_sum["recall"] / count,
                "f1": metrics_sum["f1"] / count
            }
        }
            
        return results

    def evaluate_single(self, entry: dict) -> dict:
        question = entry.get("question", "")
        gt_answer = entry.get("ground_truth_answer", "")
        gt_sources = entry.get("ground_truth_sources", [])
        gt_chunks = entry.get("relevant_chunks", [])
        
        # Call the generic RAG function
        # It's expected to return a dictionary containing keys like:
        # 'response', 'sources', 'retrieved_chunks'
        response_dict = self.rag_function(question, user_id=self.user_id)
        
        generated_answer = response_dict.get("response", "")
        predicted_sources = response_dict.get("sources", [])
        retrieved_chunks = response_dict.get("retrieved_chunks", [])
        
        cp = context_precision(retrieved_chunks, gt_chunks)
        cr = context_recall(retrieved_chunks, gt_chunks)
        af = answer_faithfulness(generated_answer, retrieved_chunks)
        ar = answer_relevance(question, generated_answer)
        sa = source_accuracy(predicted_sources, gt_sources)
        f1_scores = answer_f1(generated_answer, gt_answer)
        
        return {
            "question": question,
            "context_precision": cp,
            "context_recall": cr,
            "faithfulness": af,
            "answer_relevance": ar,
            "source_accuracy": sa,
            "answer_f1": f1_scores
        }
