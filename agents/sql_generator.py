"""
SQL Generator Agent — Third node in the Text2SQL pipeline.

Responsibilities:
- Generate SQL from enriched query context
- Validate SQL against schema
- Check for dangerous keywords
- Execute against SQLite demo database
- Return results to supervisor
"""

import re
import time
import sqlite3
import os
from typing import Optional


# ── SQL Generation Templates ──────────────────────────────────────────────────
# Pre-built patterns for common query types
# In production LLM generates these dynamically

SQL_TEMPLATES = {
    "analytics_fraud": """
        SELECT 
            DATE_TRUNC('day', timestamp) as date,
            COUNT(*) as fraud_count,
            SUM(amount) as total_amount,
            AVG(amount) as avg_amount
        FROM transactions
        WHERE is_fraud = TRUE
        {date_filter}
        {extra_filters}
        GROUP BY DATE_TRUNC('day', timestamp)
        ORDER BY date DESC
        LIMIT 100
    """,

    "analytics_spend": """
        SELECT 
            user_id,
            COUNT(*) as transaction_count,
            SUM(amount) as total_spend,
            AVG(amount) as avg_spend
        FROM transactions
        WHERE 1=1
        {date_filter}
        {extra_filters}
        GROUP BY user_id
        ORDER BY total_spend DESC
        LIMIT 20
    """,

    "lookup_merchant": """
        SELECT 
            transaction_id,
            user_id,
            amount,
            merchant,
            timestamp,
            is_fraud
        FROM transactions
        WHERE 1=1
        {merchant_filter}
        {date_filter}
        {extra_filters}
        ORDER BY timestamp DESC
        LIMIT 50
    """,

    "summary_fraud": """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(CASE WHEN is_fraud = TRUE THEN 1 ELSE 0 END) as fraud_count,
            ROUND(
                SUM(CASE WHEN is_fraud = TRUE THEN 1.0 ELSE 0 END) / 
                COUNT(*) * 100, 2
            ) as fraud_rate_pct,
            SUM(CASE WHEN is_fraud = TRUE THEN amount ELSE 0 END) as fraud_amount
        FROM transactions
        WHERE 1=1
        {date_filter}
    """
}

# ── Dangerous SQL keywords ────────────────────────────────────────────────────
DANGEROUS_KEYWORDS = [
    'DROP', 'DELETE', 'TRUNCATE', 'INSERT',
    'UPDATE', 'ALTER', 'CREATE', 'EXEC', 'EXECUTE'
]


# ── SQL Generator Agent ───────────────────────────────────────────────────────

def generate_sql(state: dict) -> dict:
    """
    Generates and validates SQL from enriched query context.

    Args:
        state: Current pipeline state with enrichment results

    Returns:
        Updated state with generated SQL and execution results
    """
    start_time = time.time()
    user_query = state["user_query"]
    intent = state.get("intent", "analytics")
    mapped_columns = state.get("mapped_columns", {})
    date_range = state.get("date_range", {})
    filters = state.get("filters", {})
    extracted_terms = state.get("extracted_terms", [])

    print(f"\n[SQLGeneratorAgent] Generating SQL for: '{user_query}'")

    try:
        # ── Generate SQL ──────────────────────────────────────────────────
        # In production: replace with Claude API call
        # client = anthropic.Anthropic()
        # response = client.messages.create(
        # model="claude-sonnet-4-20250514",
        # max_tokens=500,
        # temperature=0.0,
        # system=_build_sql_prompt(mapped_columns),
        # messages=[{"role": "user",
        # "content": user_query}]
        # )
        # generated_sql = response.content[0].text.strip()

        generated_sql = _mock_generate_sql(
            user_query, intent, mapped_columns,
            date_range, filters, extracted_terms
        )

        # ── Validate SQL ──────────────────────────────────────────────────
        validation = _validate_sql(generated_sql)

        if not validation["valid"]:
            print(f"[SQLGeneratorAgent] Validation failed: "
                  f"{validation['reason']}")
            return {
                **state,
                "error": f"SQL validation failed: {validation['reason']}",
                "error_agent": "sql_generator",
                "sql_valid": False,
                "current_agent": "supervisor"
            }

        # ── Execute SQL against demo database ─────────────────────────────
        execution_result = _execute_sql(generated_sql)

        # ── Update state ──────────────────────────────────────────────────
        elapsed = (time.time() - start_time) * 1000

        latency = state.get("latency_ms", {})
        latency["sql_generator"] = round(elapsed, 2)

        completed = state.get("completed_agents", [])
        completed.append("sql_generator")

        tables_used = re.findall(
            r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)',
            generated_sql,
            re.IGNORECASE
        )
        tables_flat = [t for pair in tables_used for t in pair if t]

        print(f"[SQLGeneratorAgent] Done — "
              f"rows={execution_result.get('row_count', 0)}, "
              f"latency={elapsed:.0f}ms")
        print(f"[SQLGeneratorAgent] SQL:\n{generated_sql.strip()}")

        return {
            **state,
            "generated_sql": generated_sql,
            "sql_tables_used": tables_flat,
            "sql_confidence": 0.92,
            "sql_valid": True,
            "sql_execution_result": execution_result,
            "current_agent": "supervisor",
            "hop_count": state["hop_count"] + 1,
            "completed_agents": completed,
            "latency_ms": latency
        }

    except Exception as e:
        print(f"[SQLGeneratorAgent] ERROR: {e}")
        return {
            **state,
            "error": str(e),
            "error_agent": "sql_generator",
            "current_agent": "supervisor"
        }


# ── Mock SQL generation ───────────────────────────────────────────────────────

def _mock_generate_sql(
    query: str,
    intent: str,
    mapped_columns: dict,
    date_range: dict,
    filters: dict,
    extracted_terms: list
) -> str:
    """
    Generates SQL using templates based on enrichment context.
    Simulates what the LLM would produce.
    Replace with real LLM call in production.
    """
    query_lower = query.lower()

    # Build date filter
    date_filter = ""
    if date_range:
        date_filter = (
            f"AND timestamp >= {date_range.get('start', 'CURRENT_DATE')}"
            f" AND timestamp < {date_range.get('end', 'CURRENT_DATE')}"
        )

    # Build extra filters
    extra_filters = ""
    for key, filter_val in filters.items():
        if key != "merchant":
            extra_filters += f"AND {filter_val}\n"

    # Build merchant filter
    merchant_filter = ""
    if "merchant" in filters:
        merchant_filter = f"AND {filters['merchant']}"

    # Detect fraud filter
    fraud_filter = ""
    if "fraud" in extracted_terms or "is_fraud = TRUE" in str(mapped_columns):
        fraud_filter = "AND is_fraud = TRUE"

    # ── Select template based on intent and terms ─────────────────────────

    # Summary intent
    if intent == "summary":
        sql = SQL_TEMPLATES["summary_fraud"].format(
            date_filter=date_filter
        )

    # Fraud analytics
    elif intent == "analytics" and (
        "fraud" in query_lower or
        "fraud" in extracted_terms
    ):
        sql = SQL_TEMPLATES["analytics_fraud"].format(
            date_filter=date_filter,
            extra_filters=extra_filters
        )

    # Merchant lookup
    elif "merchant" in filters or any(
        "merchant" in t for t in extracted_terms
    ):
        sql = SQL_TEMPLATES["lookup_merchant"].format(
            merchant_filter=merchant_filter,
            date_filter=date_filter,
            extra_filters=extra_filters
        )

    # General spend analytics
    elif intent == "analytics":
        sql = SQL_TEMPLATES["analytics_spend"].format(
            date_filter=date_filter,
            extra_filters=f"{extra_filters} {fraud_filter}"
        )

    # Default — general lookup
    else:
        sql = f"""
            SELECT *
            FROM transactions
            WHERE 1=1
            {fraud_filter}
            {date_filter}
            {extra_filters}
            ORDER BY timestamp DESC
            LIMIT 50
        """

    return sql.strip()


# ── SQL Validation ────────────────────────────────────────────────────────────

def _validate_sql(sql: str) -> dict:
    """
    Validates generated SQL for safety and correctness.
    """
    if not sql or len(sql.strip()) < 10:
        return {"valid": False, "reason": "Empty SQL generated"}

    sql_upper = sql.upper()

    # Check SELECT exists
    if "SELECT" not in sql_upper:
        return {"valid": False,
                "reason": "SQL must contain SELECT statement"}

    # Check dangerous keywords
    for keyword in DANGEROUS_KEYWORDS:
        if re.search(rf'\b{keyword}\b', sql_upper):
            return {"valid": False,
                    "reason": f"Dangerous keyword detected: {keyword}"}

    # Check FROM exists
    if "FROM" not in sql_upper:
        return {"valid": False,
                "reason": "SQL must contain FROM clause"}

    return {"valid": True, "reason": "SQL passed validation"}


# ── SQL Execution ─────────────────────────────────────────────────────────────

def _execute_sql(sql: str) -> dict:
    """
    Executes SQL against SQLite demo database.
    Creates demo database with sample data if not exists.
    """
    db_path = "data/transactions.db"
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect(db_path)

    try:
        # Create demo table if not exists
        _create_demo_database(conn)

        # Adapt SQL for SQLite
        # SQLite doesn't support DATE_TRUNC — replace with strftime
        sqlite_sql = _adapt_sql_for_sqlite(sql)

        cursor = conn.execute(sqlite_sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        return {
            "columns": columns,
            "rows": rows[:10], # return first 10 rows
            "row_count": len(rows),
            "success": True
        }

    except Exception as e:
        return {
            "columns": [],
            "rows": [],
            "row_count": 0,
            "success": False,
            "error": str(e)
        }
    finally:
        conn.close()


def _adapt_sql_for_sqlite(sql: str) -> str:
    """
    Adapts PostgreSQL-style SQL to SQLite for demo purposes.
    Production system uses actual PostgreSQL/Azure SQL.
    """
    # Remove DATE_TRUNC — not supported in SQLite
    sql = re.sub(
        r"DATE_TRUNC\('[^']+',\s*[^)]+\)",
        "date('now')",
        sql
    )
    # Remove INTERVAL expressions
    sql = re.sub(
        r"CURRENT_DATE\s*-\s*INTERVAL\s*'[^']+'\s*",
        "date('now', '-30 days')",
        sql
    )
    sql = sql.replace("CURRENT_DATE", "date('now')")

    # Fix ROUND for SQLite
    sql = re.sub(r'ROUND\(([^,]+),\s*(\d+)\)', r'ROUND(\1, \2)', sql)

    return sql


def _create_demo_database(conn: sqlite3.Connection):
    """
    Creates SQLite demo database with sample transaction data.
    Simulates PayPal/FMCG transaction dataset.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id TEXT PRIMARY KEY,
            user_id TEXT,
            amount REAL,
            merchant TEXT,
            merchant_category TEXT,
            timestamp TEXT,
            is_fraud INTEGER,
            is_flagged INTEGER,
            currency TEXT,
            country TEXT
        )
    """)

    # Check if data already exists
    count = conn.execute(
        "SELECT COUNT(*) FROM transactions"
    ).fetchone()[0]

    if count == 0:
        import random
        import uuid
        from datetime import datetime, timedelta

        merchants = [
            "Amazon", "Uber", "Netflix",
            "PayPal", "Google", "Apple"
        ]
        categories = [
            "shopping", "transport",
            "entertainment", "digital", "food"
        ]
        users = [f"U{i:03d}" for i in range(1, 21)]

        sample_data = []
        base_date = datetime.now() - timedelta(days=90)

        for i in range(500):
            days_offset = random.randint(0, 90)
            tx_date = base_date + timedelta(
                days=days_offset,
                hours=random.randint(0, 23)
            )
            is_fraud = 1 if random.random() < 0.08 else 0

            sample_data.append((
                str(uuid.uuid4()),
                random.choice(users),
                round(random.uniform(5, 2000), 2),
                random.choice(merchants),
                random.choice(categories),
                tx_date.isoformat(),
                is_fraud,
                1 if random.random() < 0.12 else 0,
                "USD",
                random.choice(["US", "UK", "IN", "SG"])
            ))

        conn.executemany("""
            INSERT INTO transactions VALUES 
            (?,?,?,?,?,?,?,?,?,?)
        """, sample_data)
        conn.commit()
        print("[SQLGeneratorAgent] Demo database created with 500 rows")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from pipeline.state import create_initial_state
    from agents.classifier import classify_query
    from agents.enrichment import enrich_query

    test_queries = [
        "Show me total fraud transactions last month",
        "Which users had highest spend above $500?",
        "Show me suspicious transactions from Netflix",
        "Summarize fraud trends",
    ]

    print("=" * 60)
    print("SQL GENERATOR AGENT — TEST RUN")
    print("=" * 60)

    for query in test_queries:
        state = create_initial_state(user_query=query)
        state = classify_query(state)
        state = enrich_query(state)
        result = generate_sql(state)

        print(f"\nQuery: {query}")
        print(f"SQL Valid: {result.get('sql_valid')}")
        print(f"Rows returned: "
              f"{result.get('sql_execution_result', {}).get('row_count', 0)}")
        print(f"Columns: "
              f"{result.get('sql_execution_result', {}).get('columns', [])}")
        print("-" * 40)