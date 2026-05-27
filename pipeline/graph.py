"""
LangGraph Supervisor Pipeline — Wires all agents together.

Architecture:
- Supervisor routes between agents based on state
- Each agent reads and writes to shared QueryState
- Conditional edges handle routing logic
- Max hop limit prevents infinite loops
"""

from typing import Literal
from pipeline.state import QueryState, create_initial_state
from agents.classifier import classify_query
from agents.enrichment import enrich_query
from agents.sql_generator import generate_sql
from agents.summarizer import summarize_results


# ── Supervisor routing logic ──────────────────────────────────────────────────

def supervisor_route(state: QueryState) -> Literal[
    "classifier",
    "enrichment",
    "sql_generator",
    "summarizer",
    "end"
]:
    """
    Central supervisor that decides which agent runs next.
    Called after every agent completes.

    Routing logic:
    1. If error or max hops reached → end
    2. If not classified yet → classifier
    3. If unstructured query → summarizer directly
    4. If not enriched yet → enrichment
    5. If no SQL yet → sql_generator
    6. If SQL done → summarizer
    7. Default → end
    """
    completed = state.get("completed_agents", [])
    error = state.get("error")
    hop_count = state.get("hop_count", 0)
    max_hops = state.get("max_hops", 6)
    query_type = state.get("query_type")

    print(f"\n[Supervisor] Routing — hops={hop_count}, "
          f"completed={completed}, error={error}")

    # ── Safety checks ─────────────────────────────────────────────────────
    if error:
        print("[Supervisor] Error detected → ending pipeline")
        return "end"

    if hop_count >= max_hops:
        print("[Supervisor] Max hops reached → ending pipeline")
        return "end"

    if state.get("current_agent") == "complete":
        print("[Supervisor] Pipeline complete → ending")
        return "end"

    # ── Routing logic ─────────────────────────────────────────────────────
    if "classifier" not in completed:
        print("[Supervisor] → classifier")
        return "classifier"

    if query_type == "unstructured":
        if "summarizer" not in completed:
            print("[Supervisor] Unstructured query → summarizer")
            return "summarizer"
        return "end"

    if "enrichment" not in completed:
        print("[Supervisor] → enrichment")
        return "enrichment"

    if "sql_generator" not in completed:
        print("[Supervisor] → sql_generator")
        return "sql_generator"

    if "summarizer" not in completed:
        print("[Supervisor] → summarizer")
        return "summarizer"

    print("[Supervisor] All agents complete → end")
    return "end"


# ── Build LangGraph pipeline ──────────────────────────────────────────────────

def build_pipeline():
    """
    Builds and returns the LangGraph StateGraph pipeline.
    Uses conditional edges for dynamic agent routing.
    """
    try:
        from langgraph.graph import StateGraph, END

        # Initialize graph with state schema
        graph = StateGraph(QueryState)

        # ── Add agent nodes ───────────────────────────────────────────────
        graph.add_node("supervisor", lambda s: s)
        graph.add_node("classifier", classify_query)
        graph.add_node("enrichment", enrich_query)
        graph.add_node("sql_generator", generate_sql)
        graph.add_node("summarizer", summarize_results)

        # ── Set entry point ───────────────────────────────────────────────
        graph.set_entry_point("supervisor")

        # ── Add conditional edges from supervisor ─────────────────────────
        graph.add_conditional_edges(
            "supervisor",
            supervisor_route,
            {
                "classifier": "classifier",
                "enrichment": "enrichment",
                "sql_generator": "sql_generator",
                "summarizer": "summarizer",
                "end": END
            }
        )

        # ── All agents route back to supervisor ───────────────────────────
        graph.add_edge("classifier", "supervisor")
        graph.add_edge("enrichment", "supervisor")
        graph.add_edge("sql_generator", "supervisor")
        graph.add_edge("summarizer", "supervisor")

        # ── Compile graph ─────────────────────────────────────────────────
        # compiled = graph.compile()
        # print("[Pipeline] LangGraph pipeline built successfully")
        # return compiled, "langgraph"
       
        # ── Add checkpointer for conversation memory ──────────────────────
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            import sqlite3
            import os
            os.makedirs("data", exist_ok = True)
            conn=sqlite3.connect("data/checkpoints.db", check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            compiled = graph.compile(checkpointer=checkpointer)
            print("[Pipeline] LangGraph pipeline built with SQLite checkpointer")
            return compiled, "langgraph"

        except Exception as e:
            #Checkpointer not available - compile without
            print(f"[Pipeline] Checkpointer unavailable: {e} building without")
            compiled = graph.compile()
            print("[Pipeline] LangGraph pipeline built without checkpointer")
            return compiled, "langgraph"

    except ImportError:
        print("[Pipeline] LangGraph not installed — "
              "using sequential fallback pipeline")
        return None, "sequential"


# ── Sequential fallback pipeline ──────────────────────────────────────────────

def run_sequential_pipeline(state: QueryState) -> QueryState:
    """
    Sequential fallback when LangGraph is not installed.
    Runs agents in fixed order with supervisor routing logic.
    Demonstrates same logic as LangGraph version.
    """
    print("\n[Pipeline] Running sequential pipeline")
    max_iterations = state["max_hops"]

    for iteration in range(max_iterations):
        next_agent = supervisor_route(state)

        if next_agent == "end":
            break

        if next_agent == "classifier":
            state = classify_query(state)
        elif next_agent == "enrichment":
            state = enrich_query(state)
        elif next_agent == "sql_generator":
            state = generate_sql(state)
        elif next_agent == "summarizer":
            state = summarize_results(state)

        if state.get("error"):
            print(f"[Pipeline] Error in {state.get('error_agent')}"
                  f" — stopping")
            break

    return state


# ── Main pipeline runner ──────────────────────────────────────────────────────

def run_pipeline(
    user_query: str,
    user_id: str = "default_user",
    previous_query: str = None
) -> dict:
    """
    Main entry point for the Text2SQL pipeline.
    Automatically uses LangGraph if available,
    falls back to sequential execution otherwise.

    Args:
        user_query: Natural language question
        user_id: User identifier for access control
        previous_query: Previous query for follow-up detection

    Returns:
        Final pipeline state with answer and metadata
    """
    print("\n" + "=" * 60)
    print(f"[Pipeline] Starting — query: '{user_query}'")
    print("=" * 60)

    # Create initial state
    state = create_initial_state(
        user_query=user_query,
        user_id=user_id,
        previous_query=previous_query
    )

    # Try LangGraph first, fall back to sequential
    compiled_graph, mode = build_pipeline()

    if mode == "langgraph" and compiled_graph:
        # Run with LangGraph
        config = {
            "configurable":{
                "thread_id":state["session_id"]
            }
        }
        final_state = compiled_graph.invoke(state, config=config)
    else:
        # Run sequential fallback
        final_state = run_sequential_pipeline(state)

    # ── Format final response ─────────────────────────────────────────────
    response = {
        "query": user_query,
        "answer": final_state.get(
            "final_answer",
            "Could not generate answer."
        ),
        "sql": final_state.get("generated_sql", ""),
        "row_count": (
            (final_state.get("sql_execution_result") or {}).get("row_count", 0)
        ),
        "agents_used": final_state.get("completed_agents", []),
        "total_hops": final_state.get("hop_count", 0),
        "latency_ms": final_state.get("latency_ms", {}),
        "total_latency_ms": sum(
            final_state.get("latency_ms", {}).values()
        ),
        "error": final_state.get("error"),
        "mode": mode
    }

    print("\n" + "=" * 60)
    print("[Pipeline] COMPLETE")
    print(f"Answer: {response['answer']}")
    print(f"Agents: {response['agents_used']}")
    print(f"Hops: {response['total_hops']}")
    print(f"Latency: {response['total_latency_ms']:.0f}ms")
    print("=" * 60)

    return response


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    test_cases = [
        {
            "query": "Show me total fraud transactions last month",
            "prev": None
        },
        {
            "query": "Now filter by Amazon only",
            "prev": "Show me total fraud transactions last month"
        },
        {
            "query": "Which users had highest spend above $1000?",
            "prev": None
        },
        {
            "query": "Summarize fraud trends",
            "prev": None
        },
        {
            "query": "What is machine learning?",
            "prev": None
        },
    ]

    print("\n" + "*" * 20)
    print("TEXT2SQL MULTI-AGENT PIPELINE — FULL TEST")
    print("*" * 20)

    for i, case in enumerate(test_cases, 1):
        print(f"\n\n{'─'*60}")
        print(f"TEST CASE {i}/{len(test_cases)}")
        print(f"{'─'*60}")

        result = run_pipeline(
            user_query=case["query"],
            user_id="test_user",
            previous_query=case["prev"]
        )

        print(f"\n RESULT {i}:")
        print(f" Query : {result['query']}")
        print(f" Answer : {result['answer']}")
        print(f" SQL rows: {result['row_count']}")
        print(f" Latency : {result['total_latency_ms']:.0f}ms")
        print(f" Mode : {result['mode']}")
        if result['error']:
            print(f" Error : {result['error']}")
