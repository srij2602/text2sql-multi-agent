"""
FastAPI REST API for Text2SQL Multi-Agent Pipeline.

Endpoints:
- POST /query — Run pipeline on a natural language query
- POST /query/session — Run query with session memory (follow-ups)
- GET /health — Health check
- GET /metrics — Pipeline performance metrics
- GET /queries/sample — Get sample queries for testing
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import time
import json
import uuid
from datetime import datetime


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Text2SQL Multi-Agent API",
    description="""
Production-grade Text2SQL pipeline using LangGraph supervisor architecture.

Built by Srijita Acharyya — Senior Data Scientist & LLM Engineer

## Features
- Multi-agent pipeline: Classifier → Enrichment → SQL Generator → Summarizer
- Follow-up query detection and context merging
- Session memory via LangGraph checkpointing
- SQL validation and safety layer
- Real-time performance metrics
    """,
    version="1.0.0",
    contact={
        "name": "Srijita Acharyya",
        "url": "https://github.com/srij2602"
    }
)


# ── Request and Response Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Natural language question",
        example="Show me total fraud transactions last month"
    )
    user_id: str = Field(
        default="default_user",
        description="User identifier for access control"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation memory. "
                    "Same session_id enables follow-up queries."
    )
    previous_query: Optional[str] = Field(
        default=None,
        description="Previous query for follow-up detection"
    )


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    sql: Optional[str]
    row_count: int
    agents_used: list
    total_hops: int
    latency_ms: dict
    total_latency_ms: float
    session_id: str
    error: Optional[str]
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    pipeline_mode: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    avg_hops: float
    success_rate: float
    uptime_seconds: float


# ── In-memory metrics store ───────────────────────────────────────────────────
# In production use Redis or Azure Monitor

class MetricsStore:
    def __init__(self):
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency = 0.0
        self.total_hops = 0
        self.start_time = time.time()

    def record(self, success: bool, latency: float, hops: int):
        self.total_requests += 1
        self.total_latency += latency
        self.total_hops += hops
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

    def get_metrics(self) -> dict:
        n = self.total_requests or 1
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": round(self.total_latency / n, 2),
            "avg_hops": round(self.total_hops / n, 2),
            "success_rate": round(
                self.successful_requests / n * 100, 2
            ),
            "uptime_seconds": round(
                time.time() - self.start_time, 2
            )
        }


metrics = MetricsStore()


# ── Pipeline import ───────────────────────────────────────────────────────────

from pipeline.graph import run_pipeline


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Health check endpoint.
    Use this to verify the API is running.
    """
    from pipeline.graph import build_pipeline
    _, mode = build_pipeline()

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
        pipeline_mode=mode
    )


@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    """
    Run the Text2SQL pipeline on a natural language query.

    Returns the SQL generated, query results, and
    a natural language summary.

    Example queries:
    - "Show me total fraud transactions last month"
    - "Which users had highest spend above $1000?"
    - "Summarize fraud trends"
    - "Show me suspicious transactions from Netflix"
    """
    request_id = str(uuid.uuid4())[:8]
    session_id = request.session_id or str(uuid.uuid4())

    print(f"\n[API] Request {request_id}: '{request.query}'")

    try:
        # Run pipeline
        result = run_pipeline(
            user_query=request.query,
            user_id=request.user_id,
            previous_query=request.previous_query
        )

        # Record metrics
        metrics.record(
            success=result.get("error") is None,
            latency=result.get("total_latency_ms", 0),
            hops=result.get("total_hops", 0)
        )

        return QueryResponse(
            request_id=request_id,
            query=request.query,
            answer=result.get("answer", "No answer generated"),
            sql=result.get("sql", ""),
            row_count=result.get("row_count", 0),
            agents_used=result.get("agents_used", []),
            total_hops=result.get("total_hops", 0),
            latency_ms=result.get("latency_ms", {}),
            total_latency_ms=result.get("total_latency_ms", 0),
            session_id=session_id,
            error=result.get("error"),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        metrics.record(success=False, latency=0, hops=0)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.post("/query/session", response_model=QueryResponse)
def run_query_with_session(request: QueryRequest):
    """
    Run query with session memory for follow-up support.

    Use the same session_id across multiple requests
    to enable follow-up queries.

    Example flow:
    1. POST /query/session with session_id="abc"
       query="Show me fraud transactions last month"

    2. POST /query/session with session_id="abc"
       query="Now filter by Amazon only"
       (Agent remembers previous query automatically)
    """
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())
    request_id = str(uuid.uuid4())[:8]

    print(f"\n[API] Session request {request_id}: "
          f"session={session_id}, query='{request.query}'")

    try:
        result = run_pipeline(
            user_query=request.query,
            user_id=request.user_id,
            previous_query=request.previous_query
        )

        metrics.record(
            success=result.get("error") is None,
            latency=result.get("total_latency_ms", 0),
            hops=result.get("total_hops", 0)
        )

        return QueryResponse(
            request_id=request_id,
            query=request.query,
            answer=result.get("answer", "No answer generated"),
            sql=result.get("sql", ""),
            row_count=result.get("row_count", 0),
            agents_used=result.get("agents_used", []),
            total_hops=result.get("total_hops", 0),
            latency_ms=result.get("latency_ms", {}),
            total_latency_ms=result.get("total_latency_ms", 0),
            session_id=session_id,
            error=result.get("error"),
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        metrics.record(success=False, latency=0, hops=0)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """
    Get pipeline performance metrics.

    Tracks: total requests, success rate,
    average latency, average hops.
    """
    m = metrics.get_metrics()
    return MetricsResponse(**m)


@app.get("/queries/sample")
def get_sample_queries():
    """
    Returns sample queries for API testing.
    Loaded from data/sample_queries.json.
    """
    try:
        with open("data/sample_queries.json") as f:
            data = json.load(f)
        return {
            "status": "success",
            "sample_queries": data
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "sample_queries.json not found"
        }


@app.get("/")
def root():
    """Root endpoint — API information."""
    return {
        "name": "Text2SQL Multi-Agent API",
        "version": "1.0.0",
        "author": "Srijita Acharyya",
        "github": "https://github.com/srij2602/text2sql-multi-agent",
        "docs": "/docs",
        "endpoints": {
            "POST /query": "Run pipeline on natural language query",
            "POST /query/session": "Run with session memory",
            "GET /health": "Health check",
            "GET /metrics": "Performance metrics",
            "GET /queries/sample": "Sample test queries"
        }
    }