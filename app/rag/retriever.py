from app.models.embeddings import EmbeddingClient
from app.utils import config
from app.utils.logger import logger
from typing import List
from langchain.schema import Document
from langchain_chroma import Chroma
from math import sqrt

def get_vectorstore():
    emb = EmbeddingClient()
    vectordb = Chroma(persist_directory=config.CHROMA_PERSIST_DIR, embedding_function=emb._client)
    return vectordb

def get_relevant_docs(query: str, k: int = None) -> List[Document]:
    k = k or config.DEFAULT_TOP_K
    vectordb = get_vectorstore()
    top_n = config.RERANK_TOP_K if config.RERANK_ENABLED else k
    retriever = vectordb.as_retriever(search_kwargs={"k": top_n})

    if hasattr(retriever, "invoke"):
        candidates = retriever.invoke(query)
    elif hasattr(retriever, "get_relevant_documents"):
        candidates = retriever.get_relevant_documents(query)
    else:
        candidates = [d for d, _ in retriever.similarity_search_with_relevance_scores(query, k=top_n)]

    if not candidates:
        logger.info(f"No documents retrieved for query: {query}")
        return []

    if not config.RERANK_ENABLED or k >= top_n:
        selected = candidates[:k]
        logger.info(f"Returning top {len(selected)} candidates without rerank for query: {query}")
        return selected

    scored = []

    if len(candidates) > 0 and isinstance(candidates[0], tuple) and len(candidates[0]) == 2:
        for doc, score in candidates:
            scored.append((score, doc))
    else:
        emb_client = EmbeddingClient()
        q_emb = emb_client.embed([query])
        if isinstance(q_emb, list) and len(q_emb) == 1:
            q_emb = q_emb[0]

        collection = getattr(vectordb, '_collection', None) or getattr(vectordb, 'client', None)

        for doc in candidates:
            meta = getattr(doc, 'metadata', {}) or {}
            d_emb = None

            if meta.get('embedding'):
                d_emb = meta.get('embedding')

            if d_emb is None:
                doc_id = meta.get('id') or meta.get('source') or None
                if collection is not None and hasattr(collection, 'get'):
                    resp = collection.get(ids=[doc_id], include=['embeddings', 'metadatas'])
                    if resp and 'embeddings' in resp and len(resp['embeddings']) > 0:
                        d_emb = resp['embeddings'][0]

            if d_emb is None:
                doc_text = getattr(doc, 'page_content', '')
                d_emb = emb_client.embed([doc_text])
                if isinstance(d_emb, list) and len(d_emb) == 1:
                    d_emb = d_emb[0]

            dot = sum(a * b for a, b in zip(q_emb, d_emb))
            norm_q = sqrt(sum(a * a for a in q_emb))
            norm_d = sqrt(sum(a * a for a in d_emb))
            score = dot / (norm_q * norm_d) if norm_q and norm_d else 0.0
            scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [doc for _, doc in scored[:k]]
    logger.info(f"Reranked and returning top {len(selected)} docs for query: {query}")
    return selected
