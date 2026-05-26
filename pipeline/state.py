"""
State schema for the Text2SQL multi-agent pipeline.
Shared across all agents via LangGraph state management.
"""

from typing import TypedDict, Optional, List, Annotated
from datetime import datetime

class QueryState(TypedDict):
    """
    Central state object passed between all agents.
    Each agent reads relevant fields and writes its outputs.
    """

    # Input -----------------------------------------------
    user_query:str #Original user question
    user_id:str #For RBAC and audit logging
    session_id:str #Unique session identifier
    timestamp:str #Query timestamp

    # Classification --------------------------------------
    query_type:Optional[str] #structured/unstructured
    is_followup:Optional[bool] #follow-up or new query
    previous_query:Optional[str] #previous query if follow-up
    intent:Optional[str] #analytics/lookup/summary
    classification_confidence:Optional[float]

    # Enrichment ------------------------------------------
    extracted_terms:Optional[List[str]] #key terms extracted
    mapped_columns:Optional[dict] #terms mapped to columns
    date_range:Optional[dict] #extracted date filters
    filters:Optional[dict] #additional filters
    enrichment_confidence:Optional[float]

    # SQL Generation --------------------------------------
    generated_sql:Optional[str] #raw SQL query
    sql_tables_used:Optional[List[str]] #tables referenced
    sql_confidence:Optional[float] #generation confidence
    sql_valid:Optional[bool] #passed validation?
    sql_execution_result:Optional[dict] #query results

    # Summarization --------------------------------------
    final_answer:Optional[str] #natural language answer
    answer_confidence:Optional[float]

    # Pipeline Control -----------------------------------
    current_agent:Optional[str] #which agent is running
    hop_count:int #number of agent hops
    max_hops:int #maximum allowed hops
    completed_agents: Optional[List[str]] #agents already run
    needs_clarification: Optional[bool] #ask user for more info
    clarification_question: Optional[str] #what to ask user

    # Error Handling -------------------------------------
    error:Optional[str] #error message if any
    error_agent:Optional[str] #which agent failed
    retry_count:int #number of retries

    # Audit & Monitoring ---------------------------------
    mlflow_run_id:Optional[str] #MLflow tracking
    latency_ms:Optional[dict] #per-agent latency

def create_initial_state(
        user_query:str,
        user_id:str = "default_user",
        session_id:str=None,
        previous_query:str=None
)->QueryState:
    """
    Creates a fresh state object for a new query.

    Args:
        user_query : Natural lamguage question from user
        user_id : User identifier for access control
        session_id : Session identifier for conversation memory 
        previous_query : Previous query if this is a follow-up

    Returns:
        Initialized QueryState with defaults
    """
    import uuid

    return QueryState(
        #Input
        user_query=user_query,
        user_id=user_id,
        session_id=session_id or str(uuid.uuid4()),
        timestamp=datetime.now().isoformat(),

        #Classification - to be filled by classifier agent
        query_type=None,
        is_followup=previous_query is not None,
        previous_query=previous_query,
        intent=None,
        classification_confidence=None,

        #Enrichment - to be filled by enrichment agent
        extracted_terms=None,
        mapped_columns=None,
        date_range=None,
        filters=None,
        enrichment_confidence=None,

        #SQL Generation - to be filled by sql agent
        generated_sql=None,
        sql_tables_used=None,
        sql_confidence=None,
        sql_valid=None,
        sql_execution_result=None,

        #Sumarization - to be filled by summarizer agent
        final_answer=None,
        answer_confidence=None,

        #Pipeline control
        current_agent='supervisor',
        hop_count = 0,
        max_hops = 6,
        completed_agents=[],
        needs_clarification=False,
        clarification_question=None,

        #Error handling
        error=None,
        error_agent=None,
        retry_count=0,

        #Monitoring
        mlflow_run_id=None,
        latency_ms={}
    )



