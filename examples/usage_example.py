import json
import os
from bot_evaluator import RAGEvaluator
from bot_evaluator.eval_report import save_report

# 1. Create a dummy golden dataset
dummy_eval_set = [
    {
        "question": "When are performance reviews conducted?",
        "ground_truth_answer": "Performance reviews are conducted annually in December.",
        "ground_truth_sources": ["hr_policy.txt"],
        "relevant_chunks": ["- Performance reviews are conducted annually in December."]
    },
    {
        "question": "Can I carry over unused annual leave?",
        "ground_truth_answer": "Yes, up to 5 days with manager approval.",
        "ground_truth_sources": ["leave_policy.txt"],
        "relevant_chunks": ["Up to 5 unused days may be carried into the following year with manager approval."]
    }
]

with open('dummy_eval_set.json', 'w') as f:
    json.dump(dummy_eval_set, f)

# 2. Define your bot's query function
# This function should accept a 'question' (and optional kwargs like 'user_id') 
# and return a dict with 'response', 'sources', and 'retrieved_chunks'.
def my_dummy_rag_bot(question: str, **kwargs) -> dict:
    """
    This is where you would normally call your actual RAG application.
    For example: 
       return secure_rag_query(question, user_id=kwargs.get('user_id'))
    """
    print(f"Bot is processing question: {question}")
    
    # Simulate bot behavior for the examples
    if "performance reviews" in question.lower():
        return {
            "response": "Performance reviews happen in December.",
            "sources": ["hr_policy.txt"],
            "retrieved_chunks": ["- Performance reviews are conducted annually in December."]
        }
    elif "annual leave" in question.lower():
        return {
            "response": "You can carry over 5 days.",
            "sources": ["leave_policy.txt"],
            "retrieved_chunks": ["Up to 5 unused days may be carried into the following year with manager approval."]
        }
    else:
        return {
            "response": "I don't know.",
            "sources": [],
            "retrieved_chunks": []
        }

# 3. Initialize the Evaluator with your bot function
evaluator = RAGEvaluator(rag_function=my_dummy_rag_bot, user_id="u001")

# 4. Run the evaluation
print("Starting evaluation...")
results = evaluator.evaluate('dummy_eval_set.json')

# 5. Save and view the report
save_report(results, 'evaluation_report.json')

# Cleanup dummy files
os.remove('dummy_eval_set.json')
