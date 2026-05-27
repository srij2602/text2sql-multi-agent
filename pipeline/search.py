"""
Azure AI Search integration for schema and example retrieval.
Replaces static SCHEMA_CONTEXT with dynamic hybrid search.

In local development: uses mock search
In production: connects to Azure AI Search
"""

# ── Mock search for local development ────────────────────────────────────────

INDEXED_SCHEMAS = [
    {
        "id": "schema_transactions",
        "title": "transactions table",
        "content": """Table: transactions
Columns: transaction_id, user_id, amount, merchant, 
         merchant_category, timestamp, is_fraud, 
         is_flagged, currency, country, payment_method
Use for: fraud analysis, spend analysis, merchant lookup,
         payment history, transaction counts""",
        "table": "transactions",
        "keywords": ["fraud", "transaction", "payment",
                     "merchant", "amount", "spend"]
    },
    {
        "id": "schema_users",
        "title": "users table",
        "content": """Table: users
Columns: user_id, name, email, country, 
         account_tier, created_at
Use for: user profiles, account details,
         customer segmentation""",
        "table": "users",
        "keywords": ["user", "customer", "account",
                     "profile", "email", "country"]
    }
]

INDEXED_SQL_EXAMPLES = [
    {
        "id": "example_fraud_monthly",
        "query": "total fraud transactions last month",
        "sql": """SELECT COUNT(*) as fraud_count, 
       SUM(amount) as total_amount
FROM transactions 
WHERE is_fraud = TRUE
AND timestamp >= date('now', 'start of month', '-1 month')
AND timestamp < date('now', 'start of month')""",
        "keywords": ["fraud", "total", "last month", "count"]
    },
    {
        "id": "example_top_spenders",
        "query": "users with highest spend",
        "sql": """SELECT user_id, SUM(amount) as total_spend
FROM transactions
GROUP BY user_id
ORDER BY total_spend DESC
LIMIT 10""",
        "keywords": ["spend", "highest", "top", "users"]
    },
    {
        "id": "example_merchant_lookup",
        "query": "transactions by merchant",
        "sql": """SELECT transaction_id, user_id, amount, timestamp
FROM transactions
WHERE merchant = 'MerchantName'
ORDER BY timestamp DESC
LIMIT 50""",
        "keywords": ["merchant", "transactions", "lookup"]
    }
]


def search_schema(query: str, top_k: int = 2) -> list:
    """
    Searches for relevant schema context.

    Local: keyword matching mock search
    Production: Azure AI Search hybrid search

    In production replace with:
    from azure.search.documents import SearchClient
    client = SearchClient(endpoint, index_name, credential)
    results = client.search(
        search_text=query,
        vector_queries=[VectorizedQuery(
            vector=embed(query),
            fields="embedding"
        )],
        top=top_k
    )
    """
    query_lower = query.lower()
    scored = []

    for schema in INDEXED_SCHEMAS:
        score = sum(
            1 for kw in schema["keywords"]
            if kw in query_lower
        )
        if score > 0:
            scored.append((score, schema))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:top_k]]


def search_sql_examples(query: str, top_k: int = 2) -> list:
    """
    Searches for similar SQL examples to use as few-shot context.

    Local: keyword matching
    Production: Azure AI Search vector search on
                pre-embedded SQL examples
    """
    query_lower = query.lower()
    scored = []

    for example in INDEXED_SQL_EXAMPLES:
        score = sum(
            1 for kw in example["keywords"]
            if kw in query_lower
        )
        if score > 0:
            scored.append((score, example))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e[1] for e in scored[:top_k]]


def get_context_for_query(query: str) -> dict:
    """
    Main function called by enrichment agent.
    Returns schema + examples for a query.
    """
    schemas = search_schema(query)
    examples = search_sql_examples(query)

    return {
        "relevant_schemas": schemas,
        "sql_examples": examples,
        "schema_text": "\n\n".join(
            s["content"] for s in schemas
        ),
        "examples_text": "\n\n".join(
            f"Query: {e['query']}\nSQL: {e['sql']}"
            for e in examples
        )
    }
