"""
Classifier Agent — First node in the Text2SQL pipeline.

Responsibilities:
- Determine if query is structured (can be answered via SQL)
  or unstructured (general knowledge question)
- Detect if query is a follow-up to previous query
- Identify intent — analytics, lookup, or summary
- Pass classification results to supervisor for routing
"""

import json
import time
from typing import Optional


# ── Schema for classifier output ─────────────────────────────────────────────

CLASSIFIER_SCHEMA = {
    "query_type": "structured or unstructured",
    "is_followup": "true or false",
    "intent": "analytics or lookup or summary or unknown",
    "confidence": "float between 0 and 1",
    "reasoning": "brief explanation of classification"
}


# ── Few-shot examples to guide the LLM ───────────────────────────────────────

CLASSIFICATION_EXAMPLES = """
Example 1:
Query: "Show me total fraud transactions last month"
Output: {"query_type": "structured", "is_followup": false, 
         "intent": "analytics", "confidence": 0.97,
         "reasoning": "Requires aggregation query on transactions table"}

Example 2:
Query: "What is fraud?"
Output: {"query_type": "unstructured", "is_followup": false,
         "intent": "unknown", "confidence": 0.95,
         "reasoning": "General knowledge question, no SQL needed"}

Example 3:
Query: "Now filter by Amazon only"
Output: {"query_type": "structured", "is_followup": true,
         "intent": "lookup", "confidence": 0.93,
         "reasoning": "Follow-up filter on previous query"}

Example 4:
Query: "Which users had the highest spend this quarter?"
Output: {"query_type": "structured", "is_followup": false,
         "intent": "analytics", "confidence": 0.96,
         "reasoning": "Requires aggregation and ranking query"}

Example 5:
Query: "Summarize the fraud trends"
Output: {"query_type": "structured", "is_followup": false,
         "intent": "summary", "confidence": 0.91,
         "reasoning": "Needs data retrieval then natural language summary"}
"""


# ── System prompt ─────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for an enterprise 
analytics system with access to a transactions database.

Your job is to classify incoming user queries into:
1. query_type: structured (can be answered via SQL) or unstructured (general knowledge)
2. is_followup: whether this query refers to a previous query
3. intent: analytics (aggregations/trends), lookup (specific records), 
           summary (narrative summary), or unknown
4. confidence: how confident you are (0 to 1)
5. reasoning: brief explanation

Database contains: transactions, users, merchants, fraud_alerts tables.

Return ONLY a valid JSON object. No explanation, no markdown, no extra text.
"""


# ── Classifier Agent Function ─────────────────────────────────────────────────

def classify_query(state: dict) -> dict:
    """
    Classifies the user query and updates pipeline state.

    In production this calls Claude API.
    For portfolio demo, uses mock classification logic
    that demonstrates the same decision patterns.

    Args:
        state: Current pipeline state (QueryState)

    Returns:
        Updated state with classification results
    """
    start_time = time.time()
    user_query = state["user_query"]
    previous_query = state.get("previous_query")

    print(f"\n[ClassifierAgent] Processing: '{user_query}'")

    try:
        # ── Mock classification for demo purposes ─────────────────────────
        # In production: replace with actual Claude API call
        # client = anthropic.Anthropic()
        # response = client.messages.create(
        # model="claude-sonnet-4-20250514",
        # max_tokens=200,
        # temperature=0.0,
        # system=CLASSIFIER_SYSTEM_PROMPT,
        # messages=[{
        # "role": "user",
        # "content": f"Previous query: {previous_query}\n"
        # f"Current query: {user_query}\n"
        # f"Examples:\n{CLASSIFICATION_EXAMPLES}"
        # }]
        # )
        # result = json.loads(response.content[0].text)

        result = _mock_classify(user_query, previous_query)

        # ── Validate result has required fields ───────────────────────────
        required_fields = [
            "query_type", "is_followup", 
            "intent", "confidence"
        ]
        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing field in classifier output: {field}")

        # ── Update state ──────────────────────────────────────────────────
        elapsed = (time.time() - start_time) * 1000

        latency = state.get("latency_ms", {})
        latency["classifier"] = round(elapsed, 2)

        completed = state.get("completed_agents", [])
        completed.append("classifier")

        print(f"[ClassifierAgent] Done — type={result['query_type']}, "
              f"intent={result['intent']}, "
              f"confidence={result['confidence']}, "
              f"latency={elapsed:.0f}ms")

        return {
            **state,
            "query_type": result["query_type"],
            "is_followup": result["is_followup"],
            "intent": result["intent"],
            "classification_confidence": result["confidence"],
            "current_agent": "supervisor",
            "hop_count": state["hop_count"] + 1,
            "completed_agents": completed,
            "latency_ms": latency
        }

    except Exception as e:
        print(f"[ClassifierAgent] ERROR: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "classifier",
            "current_agent": "supervisor"
        }


# ── Mock classification logic ─────────────────────────────────────────────────

def _mock_classify(
    query: str,
    previous_query: Optional[str] = None
) -> dict:
    """
    Mock classifier that simulates LLM classification.
    Demonstrates routing logic without API calls.
    Replace with real LLM call in production.
    """
    query_lower = query.lower().strip()

    # ── Detect follow-up ─────────────────────────────────────────────────
    followup_indicators = [
        "now", "also", "filter", "only", "instead",
        "what about", "and also", "show only", "exclude"
    ]
    is_followup = (
        previous_query is not None and
        any(word in query_lower for word in followup_indicators)
    )

    # ── Detect unstructured ──────────────────────────────────────────────
    unstructured_indicators = [
        "what is", "what are", "explain", "define",
        "how does", "tell me about", "describe"
    ]
    if any(query_lower.startswith(w) for w in unstructured_indicators):
        return {
            "query_type": "unstructured",
            "is_followup": False,
            "intent": "unknown",
            "confidence": 0.92,
            "reasoning": "General knowledge question detected"
        }

    # ── Detect intent ────────────────────────────────────────────────────
    analytics_keywords = [
        "total", "sum", "count", "average", "trend",
        "compare", "highest", "lowest", "top", "most",
        "percentage", "rate", "growth", "decline"
    ]
    summary_keywords = [
        "summarize", "summary", "overview",
        "report", "insights", "analysis"
    ]

    if any(word in query_lower for word in summary_keywords):
        intent = "summary"
        confidence = 0.91
    elif any(word in query_lower for word in analytics_keywords):
        intent = "analytics"
        confidence = 0.94
    else:
        intent = "lookup"
        confidence = 0.88

    return {
        "query_type": "structured",
        "is_followup": is_followup,
        "intent": intent,
        "confidence": confidence,
        "reasoning": f"Structured query detected with intent: {intent}"
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.state import create_initial_state

    test_queries = [
        "Show me total fraud transactions last month",
        "What is fraud detection?",
        "Now filter by Amazon only",
        "Which users had highest spend this quarter?",
        "Summarize fraud trends",
    ]

    print("=" * 60)
    print("CLASSIFIER AGENT — TEST RUN")
    print("=" * 60)

    previous = None
    for query in test_queries:
        state = create_initial_state(
            user_query=query,
            previous_query=previous
        )
        result = classify_query(state)
        print(f"\nQuery: {query}")
        print(f"Type: {result['query_type']}")
        print(f"Intent: {result['intent']}")
        print(f"Follow-up: {result['is_followup']}")
        print(f"Confidence: {result['classification_confidence']}")
        print("-" * 40)
        previous = query