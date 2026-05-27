# Text2SQL Multi-Agent System
A production-grade multi-agent AI pipeline that converts natural language queries into SQL, executes them, and returns natural language summaries.

Built with LangGraph supervisor architecture, inspired by real enterprise deployments in FMCG risk analytics.

# Architecture
User Query
↓
Supervisor(LangGraph)
↓
Classifier -> Enrichment -> SQL Gen -> Summarizer

**Supervisor** - Routes between agents using conditional edges.
Tracks state, detects errors, enforces max hop limits.

**Classifier Agent** - Detects query type (structured/unstructured), intent (analytics/lookup/summary), and follow-up queries.

**Enrichment Agent** - Extracts key terms, maps to database columns using a business glossary, extracts date ranges and filters.

**SQL Generator Agent** - Generates validated SQL, executes against database, returns structured results.

**Summarizer Agent** - Converts SQL results into natural language business summaries.

--

## Features

-LangGraph supervisor with conditional routing
-Follow-up query detection and content marging
-Business glossary mapping user terms to SQL expressions
-SQL validation - blocks dangerous keywords
-SQLite demo database with 500 sample transactions
-Sequential fallback pipeline when LangGraph unavailable
-Per-agent latency tracking
-Typed state schema shared across all agents

--

## Project Structure
text2sql-multi-agent/
|----agents/
| | ----classifier.py #Query classification
agent
| | ----enrichment.py #Term extraction and mapping
mapping
| | ----sql_generator.py #SQL generation and execution
execution
| | ----summarizer.py #Natural language summarization
|----pipeline/
| | ----state.py #Shared QueryState
schema
| | ----graph.py #LangGraph supervisor
pipeline
|----data/
| | ----transactions.db #Auto-generated
SQLite demo database
| ----env.example #Environment
variables template
| ----requirements.txt #Dependencies

## Quick start
```bash
#Clone repository
git clone https://github.com/srij2602/text2sql-multi-agent.git
cd text2sql-multi-agent

#Create virtual environment
python -m venv venv
venv\Scripts\activate #Windows
source venv/bin/activate #Mac/Linux

#Install dependencies
pip install -r requirements.txt

#Run full pipeline.txt
python -m pipeline.graph
```

# Example Queries
```bash
from pipeline.graph import run_pipeline

#Analytics query
result = run_pipeline("Shoq me total fraud transactions last month")

#Follow-up query
result = run_pipeline("Now filter by Amazon only", previous_query = "Show me total fraud transactions last month")

#Summary query
result = run_pipeline("Summarize fraud trends")

#Lookup query
result = run_pipeline("Show suspicious transactions from Uber")
```

## Example Output
Query: Which users had highest spend above $500?
Answer: Found 20 results. Top result: U0006 with $30,796.31.
Agents: ['classifier', 'enrichment', 'sql_generator', 'summarizer']
Hops: 4
Latency: 3ms

## Production Extensions

To deploy with real LLM calls:

1. Copy .env.example to .env
2. Add your ANTHROPIC_API_KEY
3. Uncomment the Claude API blocks in each agent file
4. Replace mock functions with real LLM calls
5. Connect to PostGreSQL or Azure SQL instead of SQLite

## Teck Stack

1. LangGraph - Multi-Agent orchestration
2. LangChain - LLM integration layer
3. Claude(Anthropic) - LLM for production calls
4. SQLite - Demo database
5. MLflow - Production tracing (configured)
6. Python 3.12

## Related Work

This project is based on production exprerience building a multi-agent RAG & DBGenie system for FMCG risk analytics, deployed on Microsoft Azure with:

1. Azure OpenAI service for LLM calls
2. Azure AI Search for hybrid vector retrieval
3. MLflow for agent-level tracing
4. LangGraph for supervisor orchestration

Built by Srijita Acharyya -- Senior Data Scientist & LLM Engineer