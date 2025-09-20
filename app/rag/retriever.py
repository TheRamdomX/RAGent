from langchain_chroma import Chroma
from app.models.embeddings import EmbeddingClient
from app.utils import config
from app.utils.logger import logger
from typing import List
from langchain.schema import Document

def get_vectorstore():
    emb = EmbeddingClient()
    vectordb = Chroma(persist_directory=config.CHROMA_PERSIST_DIR, embedding_function=emb._client)
    return vectordb

def get_relevant_docs(query: str, k: int = None) -> List[Document]:
    k = k or config.DEFAULT_TOP_K
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    # Use invoke method to avoid deprecation warning
    docs = retriever.invoke(query)
    return docs
