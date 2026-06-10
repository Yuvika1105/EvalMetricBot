# Bot Evaluator

A framework-agnostic, production-ready Python package to evaluate any conversational AI system, including RAG chatbots, SQL chatbots, and general LLM assistants.

## Features

- **Framework Agnostic**: Evaluate standard Python functions or API-based chatbots.
- **Metrics**: Accuracy, Precision, Recall, F1 Score, Semantic Similarity.
- **RAG Evaluation**: Faithfulness, Context Relevance, Precision, Recall (via Ragas).
- **SQL Evaluation**: Syntax validation and query comparison.
- **Safety & Reliability**: Check for prompt injection, data leakage, and consistency across multiple requests.
- **Rich Reporting**: Console, JSON, and comprehensive HTML dashboard generation.

## Installation

```bash
pip install bot-evaluator
```

For advanced features:
```bash
pip install bot-evaluator[rag]   # RAG metrics
pip install bot-evaluator[sql]   # SQL metrics
pip install bot-evaluator[llm]   # LLM-as-a-Judge
pip install bot-evaluator[all]   # All optional dependencies
```

## Quick Start

```python
from bot_evaluator import Evaluator

def my_chatbot(question: str) -> str:
    # Your bot logic here
    return "Hello world"

evaluator = Evaluator()

report = evaluator.evaluate(
    chatbot=my_chatbot,
    dataset="examples/data/test_cases.csv"
)

# Print metrics
print(report.metrics)

# Generate HTML Dashboard
report.to_html("report.html")
```

For API-based chatbots:

```python
report = evaluator.evaluate_api(
    endpoint="https://api.example.com/chat",
    dataset="test.csv"
)
```
