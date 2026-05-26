
"""
Enrichment Agent — Second node in the Text2SQL pipeline.

Responsibilities:
- Extract key terms from the user query
- Map terms to actual database column names
- Extract date ranges and filters
- Handle follow-up queries by merging with previous context
- Pass enriched context to SQL generation agent
"""

import time
import re
from typing import Optional


# ── Database schema context ───────────────────────────────────────────────────
# In production this comes from Azure AI Search or a schema registry
# Here we define it statically for demo purposes

SCHEMA_CONTEXT = {
    "tables": {
        "transactions": {
            "columns": [
                "transaction_id", "user_id", "amount",
                "merchant", "merchant_category", "timestamp",
                "is_fraud", "is_flagged", "currency",
                "country", "payment_method"
            ],
            "description": "All payment transactions"
        },
        "users": {
            "columns": [
                "user_id", "name", "email",
                "country", "account_tier", "created_at"
            ],
            "description": "User account information"
        },
        "merchants": {
            "columns": [
                "merchant_id", "merchant_name",
                "category", "country", "risk_score"
            ],
            "description": "Merchant details and risk scores"
        },
        "fraud_alerts": {
            "columns": [
                "alert_id", "transaction_id", "alert_type",
                "severity", "created_at", "resolved_at",
                "resolution_status"
            ],
            "description": "Fraud detection alerts"
        }
    }
}

# ── Business glossary ─────────────────────────────────────────────────────────
# Maps user terminology to actual database terms
# Critical for Text2SQL accuracy

BUSINESS_GLOSSARY = {
    # Fraud related
    "fraud": "is_fraud = TRUE",
    "fraudulent": "is_fraud = TRUE",
    "suspicious": "is_flagged = TRUE",
    "flagged": "is_flagged = TRUE",
    "high risk": "risk_score > 0.7",

    # Amount related
    "high value": "amount > 1000",
    "large transactions": "amount > 5000",
    "small transactions": "amount < 100",

    # Time related
    "last month": "DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
    "this month": "DATE_TRUNC('month', CURRENT_DATE)",
    "last quarter": "DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')",
    "this quarter": "DATE_TRUNC('quarter', CURRENT_DATE)",
    "last week": "CURRENT_DATE - INTERVAL '7 days'",
    "today": "CURRENT_DATE",
    "yesterday": "CURRENT_DATE - INTERVAL '1 day'",
    "last year": "DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year')",

    # User related
    "customers": "users",
    "clients": "users",
    "accounts": "users",

    # Transaction related
    "payments": "transactions",
    "purchases": "transactions",
    "orders": "transactions",
    "chargebacks": "is_fraud = TRUE",
    "reversed": "is_fraud = TRUE",
    "declined": "status = 'declined'"
}


# ── Enrichment Agent Function ─────────────────────────────────────────────────

def enrich_query(state: dict) -> dict:
    """
    Enriches the user query with schema context and term mapping.

    In production this calls Claude API for intelligent extraction.
    For portfolio demo, uses rule-based extraction that demonstrates
    the same enrichment patterns.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with extracted terms, column mappings, filters
    """
    start_time = time.time()
    user_query = state["user_query"]
    is_followup = state.get("is_followup", False)
    previous_query = state.get("previous_query")

    print(f"\n[EnrichmentAgent] Processing: '{user_query}'")

    try:
        # ── If follow-up — merge with previous query ───────────────────────
        effective_query = user_query
        if is_followup and previous_query:
            effective_query = _merge_followup(user_query, previous_query)
            print(f"[EnrichmentAgent] Follow-up detected — "
                  f"merged query: '{effective_query}'")

        # ── Extract terms and map to schema ───────────────────────────────
        # In production: replace with Claude API call
        # client = anthropic.Anthropic()
        # response = client.messages.create(
        # model="claude-sonnet-4-20250514",
        # max_tokens=500,
        # temperature=0.0,
        # system=_build_enrichment_prompt(),
        # messages=[{"role": "user",
        # "content": effective_query}]
        # )
        # result = json.loads(response.content[0].text)

        result = _mock_enrich(effective_query)

        # ── Update state ──────────────────────────────────────────────────
        elapsed = (time.time() - start_time) * 1000

        latency = state.get("latency_ms", {})
        latency["enrichment"] = round(elapsed, 2)

        completed = state.get("completed_agents", [])
        completed.append("enrichment")

        print(f"[EnrichmentAgent] Done — "
              f"terms={result['extracted_terms']}, "
              f"confidence={result['confidence']}, "
              f"latency={elapsed:.0f}ms")

        return {
            **state,
            "extracted_terms": result["extracted_terms"],
            "mapped_columns": result["mapped_columns"],
            "date_range": result["date_range"],
            "filters": result["filters"],
            "enrichment_confidence": result["confidence"],
            "current_agent": "supervisor",
            "hop_count": state["hop_count"] + 1,
            "completed_agents": completed,
            "latency_ms": latency
        }

    except Exception as e:
        print(f"[EnrichmentAgent] ERROR: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "enrichment",
            "current_agent": "supervisor"
        }


# ── Mock enrichment logic ─────────────────────────────────────────────────────

def _mock_enrich(query: str) -> dict:
    """
    Mock enrichment that simulates LLM term extraction.
    Demonstrates column mapping and filter extraction.
    Replace with real LLM call in production.
    """
    query_lower = query.lower()

    extracted_terms = []
    mapped_columns = {}
    filters = {}
    date_range = {}

    # ── Extract and map business terms ────────────────────────────────────
    for term, mapping in BUSINESS_GLOSSARY.items():
        if term in query_lower:
            extracted_terms.append(term)
            mapped_columns[term] = mapping

    # ── Extract table references ──────────────────────────────────────────
    for table, info in SCHEMA_CONTEXT["tables"].items():
        for col in info["columns"]:
            if col.replace("_", " ") in query_lower:
                extracted_terms.append(col)

    # ── Extract date range ────────────────────────────────────────────────
    date_patterns = {
        "last month": {
            "start": "DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')",
            "end": "DATE_TRUNC('month', CURRENT_DATE)"
        },
        "this month": {
            "start": "DATE_TRUNC('month', CURRENT_DATE)",
            "end": "CURRENT_DATE"
        },
        "last quarter": {
            "start": "DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')",
            "end": "DATE_TRUNC('quarter', CURRENT_DATE)"
        },
        "this quarter": {
            "start": "DATE_TRUNC('quarter', CURRENT_DATE)",
            "end": "CURRENT_DATE"
        },
        "last week": {
            "start": "CURRENT_DATE - INTERVAL '7 days'",
            "end": "CURRENT_DATE"
        },
        "last year": {
            "start": "DATE_TRUNC('year', CURRENT_DATE - INTERVAL '1 year')",
            "end": "DATE_TRUNC('year', CURRENT_DATE)"
        }
    }

    for pattern, range_val in date_patterns.items():
        if pattern in query_lower:
            date_range = range_val
            break

    # ── Extract amount filters ─────────────────────────────────────────────
    amount_match = re.search(r'\$?([\d,]+)', query_lower)
    if amount_match:
        amount = float(amount_match.group(1).replace(',', ''))
        if any(w in query_lower for w in ["above", "over", "more than", "greater"]):
            filters["amount"] = f"amount > {amount}"
        elif any(w in query_lower for w in ["below", "under", "less than"]):
            filters["amount"] = f"amount < {amount}"

    # ── Extract merchant filter ────────────────────────────────────────────
    common_merchants = [
        "amazon", "uber", "netflix", "paypal",
        "google", "apple", "walmart", "target"
    ]
    for merchant in common_merchants:
        if merchant in query_lower:
            filters["merchant"] = f"merchant = '{merchant.capitalize()}'"
            extracted_terms.append(f"merchant:{merchant}")

    # ── Determine primary table ────────────────────────────────────────────
    if any(w in query_lower for w in
           ["transaction", "payment", "fraud", "amount", "merchant"]):
        mapped_columns["primary_table"] = "transactions"
    elif any(w in query_lower for w in ["user", "customer", "account"]):
        mapped_columns["primary_table"] = "users"
    else:
        mapped_columns["primary_table"] = "transactions"

    # ── Calculate confidence ───────────────────────────────────────────────
    confidence = 0.95 if extracted_terms else 0.70

    return {
        "extracted_terms": extracted_terms,
        "mapped_columns": mapped_columns,
        "date_range": date_range,
        "filters": filters,
        "confidence": confidence
    }


def _merge_followup(current_query: str, previous_query: str) -> str:
    """
    Merges a follow-up query with the previous query context.
    Ensures follow-up filters are applied to the right base query.
    """
    merge_prompt = f"{previous_query} AND {current_query}"
    return merge_prompt


def _build_enrichment_prompt() -> str:
    """
    Builds the system prompt for LLM-based enrichment.
    Used when switching from mock to real LLM calls.
    """
    schema_str = "\n".join([
        f"Table: {table}\nColumns: {', '.join(info['columns'])}"
        for table, info in SCHEMA_CONTEXT["tables"].items()
    ])

    glossary_str = "\n".join([
        f"{term} → {mapping}"
        for term, mapping in list(BUSINESS_GLOSSARY.items())[:10]
    ])

    return f"""You are a query enrichment agent for an analytics database.

Database Schema:
{schema_str}

Business Glossary (user terms → SQL):
{glossary_str}

Extract from the query:
1. extracted_terms: list of key terms found
2. mapped_columns: dict mapping terms to SQL expressions
3. date_range: dict with start/end if date mentioned
4. filters: dict of additional filters
5. confidence: float 0-1

Return ONLY valid JSON. No markdown. No explanation."""


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.state import create_initial_state
    from agents.classifier import classify_query

    test_queries = [
        ("Show me total fraud transactions last month", None),
        ("Now filter by Amazon only",
         "Show me total fraud transactions last month"),
        ("Which users had highest spend above $5000 this quarter?", None),
        ("Show me suspicious transactions from Netflix", None),
    ]

    print("=" * 60)
    print("ENRICHMENT AGENT — TEST RUN")
    print("=" * 60)

    for query, prev_query in test_queries:
        state = create_initial_state(
            user_query=query,
            previous_query=prev_query
        )
        state = classify_query(state)
        result = enrich_query(state)

        print(f"\nQuery: {query}")
        print(f"Terms: {result['extracted_terms']}")
        print(f"Mapped columns: {result['mapped_columns']}")
        print(f"Date range: {result['date_range']}")
        print(f"Filters: {result['filters']}")
        print(f"Confidence: {result['enrichment_confidence']}")
        print("-" * 40)
