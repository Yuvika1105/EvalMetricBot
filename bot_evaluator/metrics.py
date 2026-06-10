import string
import re

def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, ' ')
    tokens = text.split()
    stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was", "were", "to", "in", "for", "with", "on", "of", "by", "at", "it", "this", "that"}
    return {t for t in tokens if t not in stop_words}

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a.intersection(b))
    union = len(a.union(b))
    return intersection / union if union > 0 else 0.0

def context_precision(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    if not retrieved_chunks:
        return 1.0 if not relevant_chunks else 0.0
    if not relevant_chunks:
        return 0.0
    
    relevant_count = 0
    tokenized_relevant = [_tokenize(chunk) for chunk in relevant_chunks]
    
    for r_chunk in retrieved_chunks:
        r_tokens = _tokenize(r_chunk)
        is_relevant = False
        for rel_tokens in tokenized_relevant:
            if _jaccard(r_tokens, rel_tokens) >= 0.3:
                is_relevant = True
                break
        if is_relevant:
            relevant_count += 1
            
    return relevant_count / len(retrieved_chunks)

def context_recall(retrieved_chunks: list[str], relevant_chunks: list[str]) -> float:
    if not relevant_chunks:
        return 1.0
    
    tokenized_retrieved = [_tokenize(chunk) for chunk in retrieved_chunks]
    retrieved_count = 0
    
    for rel_chunk in relevant_chunks:
        rel_tokens = _tokenize(rel_chunk)
        is_retrieved = False
        for r_tokens in tokenized_retrieved:
            if _jaccard(r_tokens, rel_tokens) >= 0.3:
                is_retrieved = True
                break
        if is_retrieved:
            retrieved_count += 1
            
    return retrieved_count / len(relevant_chunks)

def answer_faithfulness(answer: str, retrieved_chunks: list[str], overlap_threshold: float = 0.4) -> float:
    answer_lower = answer.lower()
    refusal_phrases = ["don't have enough information", "do not have enough information", "cannot answer", "not enough information"]
    for phrase in refusal_phrases:
        if phrase in answer_lower:
            return 1.0
            
    sentences = [s.strip() for s in answer.split('.') if s.strip()]
    if not sentences:
        return 1.0
        
    supported_claims = 0
    total_claims = 0
    tokenized_retrieved = [_tokenize(chunk) for chunk in retrieved_chunks]
    
    for sentence in sentences:
        s_tokens = _tokenize(sentence)
        if not s_tokens:
            continue
        total_claims += 1
            
        is_supported = False
        for r_tokens in tokenized_retrieved:
            if _jaccard(s_tokens, r_tokens) >= overlap_threshold:
                is_supported = True
                break
        if is_supported:
            supported_claims += 1
            
    if total_claims == 0:
        return 1.0
        
    return supported_claims / total_claims

def answer_relevance(question: str, answer: str) -> float:
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    
    a_lower = answer.lower()
    found_count = 0
    for q_token in q_tokens:
        if q_token in a_lower:
            found_count += 1
            
    return found_count / len(q_tokens)

def source_accuracy(predicted_sources: list[str], ground_truth_sources: list[str]) -> float:
    if not ground_truth_sources:
        return 1.0 if not predicted_sources else 0.0
        
    pred_set = set(predicted_sources)
    gt_set = set(ground_truth_sources)
    
    intersection = len(pred_set.intersection(gt_set))
    return intersection / len(gt_set)

def answer_f1(generated_answer: str, ground_truth_answer: str) -> dict:
    gen_tokens = _tokenize(generated_answer)
    gt_tokens = _tokenize(ground_truth_answer)
    
    if not gen_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gen_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    intersection = len(gen_tokens.intersection(gt_tokens))
    
    precision = intersection / len(gen_tokens)
    recall = intersection / len(gt_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return {"precision": precision, "recall": recall, "f1": f1}
