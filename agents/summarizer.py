"""
Summarizer Agent — Final node in the Text2SQL pipeline.

Responsibilities:
- Convert SQL results into natural language answer
- Format response based on intent (analytics/lookup/summary)
- Add business context to raw numbers
- Generate final response for the user
"""

import time
from typing import Optional


# ── Response templates ────────────────────────────────────────────────────────

RESPONSE_TEMPLATES = {
    "no_results": (
        "No results found for your query. "
        "This could mean no data matches your filters, "
        "or the time period has no transactions."
    ),
    "error": (
        "I encountered an issue processing your query. "
        "Please try rephrasing or contact support."
    ),
    "unstructured": (
        "This question requires general knowledge rather than "
        "database lookup. Please ask about specific transactions, "
        "users, merchants, or fraud data."
    )
}


# ── Summarizer Agent ──────────────────────────────────────────────────────────

def summarize_results(state: dict) -> dict:
    """
    Converts SQL execution results into natural language answer.

    Args:
        state: Current pipeline state with SQL results

    Returns:
        Updated state with final natural language answer
    """
    start_time = time.time()
    user_query = state["user_query"]
    intent = state.get("intent", "analytics")
    query_type = state.get("query_type", "structured")
    execution_result = state.get("sql_execution_result", {})
    error = state.get("error")

    print(f"\n[SummarizerAgent] Summarizing results for: '{user_query}'")

    try:
        # ── Handle error state ────────────────────────────────────────────
        if error:
            answer = RESPONSE_TEMPLATES["error"]
            print(f"[SummarizerAgent] Error state detected: {error}")

        # ── Handle unstructured query ─────────────────────────────────────
        elif query_type == "unstructured":
            answer = RESPONSE_TEMPLATES["unstructured"]

        # ── Handle empty results ──────────────────────────────────────────
        elif (not execution_result or
              not execution_result.get("success") or
              execution_result.get("row_count", 0) == 0):
            answer = RESPONSE_TEMPLATES["no_results"]

        # ── Generate natural language summary ─────────────────────────────
        else:
            # In production: replace with Claude API call
            # client = anthropic.Anthropic()
            # response = client.messages.create(
            # model="claude-sonnet-4-20250514",
            # max_tokens=500,
            # temperature=0.3,
            # system=_build_summarizer_prompt(),
            # messages=[{
            # "role": "user",
            # "content": f"Query: {user_query}\n"
            # f"Results: {execution_result}"
            # }]
            # )
            # answer = response.content[0].text

            answer = _mock_summarize(
                user_query,
                intent,
                execution_result
            )

        # ── Update state ──────────────────────────────────────────────────
        elapsed = (time.time() - start_time) * 1000

        latency = state.get("latency_ms", {})
        latency["summarizer"] = round(elapsed, 2)

        completed = state.get("completed_agents", [])
        completed.append("summarizer")

        print(f"[SummarizerAgent] Done — latency={elapsed:.0f}ms")
        print(f"[SummarizerAgent] Answer: {answer[:100]}...")

        return {
            **state,
            "final_answer": answer,
            "answer_confidence": 0.90,
            "current_agent": "complete",
            "hop_count": state["hop_count"] + 1,
            "completed_agents": completed,
            "latency_ms": latency
        }

    except Exception as e:
        print(f"[SummarizerAgent] ERROR: {e}")
        return {
            **state,
            "final_answer": RESPONSE_TEMPLATES["error"],
            "error": str(e),
            "error_agent": "summarizer",
            "current_agent": "complete"
        }


# ── Mock summarization ────────────────────────────────────────────────────────

def _mock_summarize(
    query: str,
    intent: str,
    execution_result: dict
) -> str:
    """
    Generates natural language summary from SQL results.
    Simulates what Claude would produce.
    Replace with real LLM call in production.
    """
    columns = execution_result.get("columns", [])
    rows = execution_result.get("rows", [])
    row_count = execution_result.get("row_count", 0)

    if not rows:
        return RESPONSE_TEMPLATES["no_results"]

    # ── Summary intent ────────────────────────────────────────────────────
    if intent == "summary" and rows:
        row = rows[0]
        data = dict(zip(columns, row))

        total = data.get("total_transactions", 0)
        fraud_count = data.get("fraud_count", 0)
        fraud_rate = data.get("fraud_rate_pct", 0)
        fraud_amount = data.get("fraud_amount", 0)

        return (
            f"Fraud Summary: Out of {total:,} total transactions, "
            f"{fraud_count:,} were flagged as fraudulent "
            f"({fraud_rate:.2f}% fraud rate). "
            f"Total fraudulent amount: "
            f"${fraud_amount:,.2f}. "
            f"This is based on your current transaction database."
        )

    # ── Analytics intent ──────────────────────────────────────────────────
    elif intent == "analytics":
        if "total_spend" in columns or "total_amount" in columns:
            spend_col = (
                "total_spend"
                if "total_spend" in columns
                else "total_amount"
            )
            top_row = dict(zip(columns, rows[0]))
            top_value = top_row.get(spend_col, 0)
            top_user = top_row.get(
                "user_id",
                top_row.get("date", "N/A")
            )

            return (
                f"Found {row_count:,} results. "
                f"Top result: {top_user} with "
                f"${top_value:,.2f}. "
                f"Showing top {min(row_count, 20)} "
                f"records ordered by amount."
            )

        elif "fraud_count" in columns:
            total_fraud = sum(
                row[columns.index("fraud_count")]
                for row in rows
                if row[columns.index("fraud_count")]
            )
            return (
                f"Found {row_count} days with fraud activity. "
                f"Total fraud transactions: {total_fraud:,}. "
                f"Data shown for requested time period."
            )

    # ── Lookup intent ─────────────────────────────────────────────────────
    elif intent == "lookup":
        merchant_col = (
            "merchant"
            if "merchant" in columns
            else None
        )
        sample = dict(zip(columns, rows[0]))
        merchant_info = (
            f" from {sample.get('merchant', 'various merchants')}"
            if merchant_col
            else ""
        )

        return (
            f"Found {row_count:,} transactions{merchant_info}. "
            f"Most recent transaction: "
            f"${sample.get('amount', 0):,.2f} "
            f"on {str(sample.get('timestamp', 'N/A'))[:10]}. "
            f"Results ordered by most recent first."
        )

    # ── Default ───────────────────────────────────────────────────────────
    return (
        f"Query completed successfully. "
        f"Found {row_count:,} matching records. "
        f"Columns returned: {', '.join(columns)}."
    )


def _build_summarizer_prompt() -> str:
    """
    System prompt for LLM-based summarization.
    Used when switching from mock to real Claude calls.
    """
    return """You are a financial analytics assistant summarizing 
database query results for business users.

Convert the SQL results into a clear, concise natural language answer.
- Lead with the key number or finding
- Add business context
- Keep it to 2-3 sentences maximum
- Use $ formatting for amounts
- Use comma formatting for large numbers
- Do not mention SQL or technical details

Return only the summary text. No markdown."""


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.state import create_initial_state
    from agents.classifier import classify_query
    from agents.enrichment import enrich_query
    from agents.sql_generator import generate_sql

    test_queries = [
        "Show me total fraud transactions last month",
        "Which users had highest spend above $500?",
        "Show me suspicious transactions from Netflix",
        "Summarize fraud trends",
        "What is fraud detection?",
    ]

    print("=" * 60)
    print("SUMMARIZER AGENT — FULL PIPELINE TEST")
    print("=" * 60)

    for query in test_queries:
        state = create_initial_state(user_query=query)
        state = classify_query(state)
        state = enrich_query(state)
        state = generate_sql(state)
        result = summarize_results(state)

        print(f"\n{'='*50}")
        print(f"QUERY: {query}")
        print(f"ANSWER: {result.get('final_answer')}")
        print(f"AGENTS: {result.get('completed_agents')}")
        print(f"HOPS: {result.get('hop_count')}")
        total_latency = sum(
            result.get('latency_ms', {}).values()
        )
        print(f"TOTAL LATENCY: {total_latency:.0f}ms")
