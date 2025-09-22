from typing import Dict, Any, List, Tuple, Optional
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA
from app.rag.retriever import get_relevant_docs, get_vectorstore
from app.models.llm import Agent
from app.utils import config
from app.utils.logger import logger
import tiktoken
import json

llm = Agent()


def _doc_to_source(doc) -> Dict[str, Any]:
    meta = getattr(doc, 'metadata', {}) or {}
    return {
        "source": meta.get('source', 'desconocido'),
        "chunk": meta.get('chunk', None),
        "score": meta.get('score', None),
        "offsets": meta.get('offsets', None),
    }


def create_react_agent(k: int = None, chain_type: str = "stuff", max_iterations: int = 3, budget: Optional[Dict[str, int]] = None, verbose: bool = False) -> Tuple[Any, Any, Any]:

    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k or config.DEFAULT_TOP_K})

    llm_client = llm._client
    retrieval_qa = RetrievalQA.from_chain_type(llm=llm_client, retriever=retriever, chain_type=chain_type)

    METRICS = getattr(create_react_agent, "METRICS", {"total_queries": 0, "total_tool_calls": 0})
    create_react_agent.METRICS = METRICS

    TOOL_STATE = {"calls": 0, "tokens": 0}
    create_react_agent.TOOL_STATE = TOOL_STATE

    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)

    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text))

    max_calls = budget.get('max_calls') if budget and 'max_calls' in budget else getattr(config, 'BUDGET_CALLS_PER_QUERY', 5)
    token_budget = budget.get('token_budget') if budget and 'token_budget' in budget else getattr(config, 'TOKEN_BUDGET_PER_QUERY', None)

    def rag_tool(query: str) -> Dict[str, Any]:
        METRICS['total_tool_calls'] = METRICS.get('total_tool_calls', 0) + 1
        TOOL_STATE['calls'] += 1

        if TOOL_STATE['calls'] > max_calls:
            return {"answer": "Límite de llamadas alcanzado.", "sources": [], "calls_used": TOOL_STATE['calls'], "tokens_used": TOOL_STATE['tokens']}

        est = estimate_tokens(query)
        TOOL_STATE['tokens'] += est
        if token_budget and TOOL_STATE['tokens'] > token_budget:
            return {"answer": "Límite de tokens alcanzado.", "sources": [], "calls_used": TOOL_STATE['calls'], "tokens_used": TOOL_STATE['tokens']}

        answer_text = retrieval_qa.run(query)
        docs = retriever.invoke(query)
        sources = []
        for d in (docs or [])[:5]:
            s = _doc_to_source(d)
            sources.append(s)

        tokens_used = est
        out = {"answer": answer_text, "sources": sources, "calls_used": TOOL_STATE['calls'], "tokens_used": tokens_used}
        return json.dumps(out, ensure_ascii=False)

    tools = [Tool(name="RAG_Search", func=rag_tool, description="Búsqueda RAG que retorna JSON con sources y métricas.")]

    agent_executor = initialize_agent(
        tools,
        llm_client,
        agent="zero-shot-react-description",
        verbose=verbose,
        agent_kwargs={"max_iterations": max_iterations},
    )

    return agent_executor, retrieval_qa, retriever


def run_react_agent(question: str, k: int = None, chain_type: str = "stuff", max_iterations: int = 3, budget: Optional[Dict[str, int]] = None, verbose: bool = False) -> Dict[str, Any]:
    agent_executor, retrieval_qa, retriever = create_react_agent(k=k, chain_type=chain_type, max_iterations=max_iterations, budget=budget, verbose=verbose)

    result = agent_executor.run(question)

    parsed = None
    try:
        parsed = json.loads(result)
    except Exception:
        parsed = None

    docs = get_relevant_docs(question, k=k)
    sources = [_doc_to_source(d) for d in docs]

    calls_used = getattr(create_react_agent, 'TOOL_STATE', {}).get('calls', 0)
    tokens_used = getattr(create_react_agent, 'TOOL_STATE', {}).get('tokens', 0)

    if parsed and isinstance(parsed, dict) and parsed.get('answer'):
        parsed_sources = parsed.get('sources', [])
        return {"answer": parsed.get('answer'), "sources": parsed_sources or sources, "calls_used": parsed.get('calls_used', calls_used), "tokens_used": parsed.get('tokens_used', tokens_used)}

    return {"answer": result, "sources": sources, "calls_used": calls_used, "tokens_used": tokens_used}

