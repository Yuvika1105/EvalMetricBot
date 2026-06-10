import json
from datetime import datetime

def get_interpretation(metric_name: str, score: float) -> str:
    if metric_name == "Context Precision" or metric_name == "Context Recall":
        if score >= 0.8: return "Excellent"
        if score >= 0.5: return "Acceptable"
        return "Investigate"
    if metric_name == "Answer Faithfulness":
        if score >= 0.9: return "Excellent"
        if score >= 0.7: return "Acceptable"
        return "Poor"
    if metric_name == "Answer Relevance":
        if score >= 0.7: return "Excellent"
        if score >= 0.4: return "Acceptable"
        return "Poor"
    if metric_name == "Source Accuracy":
        if score >= 1.0: return "Excellent"
        if score >= 0.5: return "Acceptable"
        return "Poor"
    if metric_name == "Answer F1":
        if score >= 0.8: return "Excellent"
        if score >= 0.5: return "Partially correct"
        if score >= 0.3: return "Investigate"
        return "Poor"
    return ""

def save_report(results: dict, output_path: str) -> None:
    now = datetime.now()
    results["timestamp"] = now.isoformat()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    agg = results.get("aggregate", {})
    cp = agg.get("context_precision", 0.0)
    cr = agg.get("context_recall", 0.0)
    af = agg.get("faithfulness", 0.0)
    ar = agg.get("answer_relevance", 0.0)
    sa = agg.get("source_accuracy", 0.0)
    f1_dict = agg.get("answer_f1", {"precision": 0.0, "recall": 0.0, "f1": 0.0})
    f1 = f1_dict.get("f1", 0.0)
    f1_p = f1_dict.get("precision", 0.0)
    f1_r = f1_dict.get("recall", 0.0)
    
    q_count = len(results.get("per_question", []))
    user_id = results.get("user_id", "u001")
    
    print("Evaluation Summary")
    print("=" * 58)
    print(f" {'Metric':<26} {'Score':<8} {'Interpretation'}")
    print("-" * 58)
    print(f" {'Context Precision':<26} {cp:.2f}    {get_interpretation('Context Precision', cp)}")
    print(f" {'Context Recall':<26} {cr:.2f}    {get_interpretation('Context Recall', cr)}")
    print(f" {'Answer Faithfulness':<26} {af:.2f}    {get_interpretation('Answer Faithfulness', af)}")
    print(f" {'Answer Relevance':<26} {ar:.2f}    {get_interpretation('Answer Relevance', ar)}")
    print(f" {'Source Accuracy':<26} {sa:.2f}    {get_interpretation('Source Accuracy', sa)}")
    print(f" {'Answer F1':<26} {f1:.2f}    {get_interpretation('Answer F1', f1)}")
    print(f"   |- {'Precision':<21} {f1_p:.2f}")
    print(f"   |- {'Recall':<21} {f1_r:.2f}")
    print("=" * 58)
    print(f" Questions evaluated: {q_count}   |   User: {user_id}")
    print(f" Report saved: {output_path}")
