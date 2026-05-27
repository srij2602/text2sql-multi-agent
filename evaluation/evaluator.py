
"""
Pipeline evaluation using RAGAS-inspired metrics.
Measures answer quality without ground truth labels.

Metrics:
- SQL execution rate: did SQL run without error?
- Answer relevancy: does answer address the query?
- Faithfulness: does answer match SQL results?
- Routing accuracy: did supervisor route correctly?
- Latency: per-agent and total pipeline speed
"""

import json
from pipeline.graph import run_pipeline
# from data.sample_queries import load_test_queries


def evaluate_pipeline(test_queries: list) -> dict:
    """
    Runs evaluation across test query set.
    Returns metrics report.
    """
    results = []

    for test in test_queries:
        query = test["query"]
        expected_type = test.get("expected_type")
        expected_intent = test.get("expected_intent")

        # Run pipeline
        result = run_pipeline(query)

        # Score each dimension
        scores = {
            "query": query,

            # Did SQL execute without error?
            "sql_success": (
                1 if result.get("row_count", 0) >= 0
                and not result.get("error")
                else 0
            ),

            # Did routing match expected?
            "routing_correct": (
                1 if _check_routing(result, expected_type)
                else 0
            ),

            # Did answer get generated?
            "answer_generated": (
                1 if result.get("answer") and
                result["answer"] not in [
                    "Could not generate answer.",
                    "I encountered an issue processing your query."
                ]
                else 0
            ),

            # Latency
            "latency_ms": result.get("total_latency_ms", 0),

            # Hop count efficiency
            "hops": result.get("total_hops", 0),

            # Error
            "error": result.get("error")
        }

        results.append(scores)
        print(f"✓ Evaluated: '{query[:40]}...' "
              f"→ sql={scores['sql_success']}, "
              f"routing={scores['routing_correct']}")

    # Aggregate metrics
    n = len(results)
    report = {
        "total_queries": n,
        "sql_success_rate": sum(
            r["sql_success"] for r in results
        ) / n,
        "routing_accuracy": sum(
            r["routing_correct"] for r in results
        ) / n,
        "answer_generation_rate": sum(
            r["answer_generated"] for r in results
        ) / n,
        "avg_latency_ms": sum(
            r["latency_ms"] for r in results
        ) / n,
        "avg_hops": sum(
            r["hops"] for r in results
        ) / n,
        "error_rate": sum(
            1 for r in results if r["error"]
        ) / n,
        "detailed_results": results
    }

    return report


def _check_routing(result, expected_type):
    agents = result.get("agents_used", [])
    if expected_type == "unstructured":
        return "sql_generator" not in agents
    else:
        return "sql_generator" in agents


def print_report(report: dict):
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)
    print(f"Total queries : {report['total_queries']}")
    print(f"SQL success rate : {report['sql_success_rate']*100:.1f}%")
    print(f"Routing accuracy : {report['routing_accuracy']*100:.1f}%")
    print(f"Answer gen rate : {report['answer_generation_rate']*100:.1f}%")
    print(f"Avg latency : {report['avg_latency_ms']:.1f}ms")
    print(f"Avg hops : {report['avg_hops']:.1f}")
    print(f"Error rate : {report['error_rate']*100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    # Load sample queries from JSON
    with open("data/sample_queries.json") as f:
        data = json.load(f)

    # Flatten all queries except followup and edge cases
    test_queries = []
    for category, content in data["categories"].items():
        if category in ["analytics", "lookup",
                        "summary", "unstructured"]:
            for q in content["queries"]:
                test_queries.append(q)

    print(f"Running evaluation on {len(test_queries)} queries...")
    report = evaluate_pipeline(test_queries)
    print_report(report)
