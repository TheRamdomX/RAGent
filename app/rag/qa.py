from app.rag.retriever import get_relevant_docs
from app.models.llm import GPT5Nano
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any, Tuple

llm = GPT5Nano()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente pedagógico. Usa el contexto recuperado para responder precisa y concisamente. "
    "Si no hay información en el contexto relevante, indica que no lo sabes en vez de inventar."
)

def build_prompt(context_docs, question: str) -> str:
    context_texts = []
    for i, d in enumerate(context_docs):
        meta = d.metadata if hasattr(d, "metadata") else {}
        header = f"[Fuente: {meta.get('source','desconocido')} | chunk={meta.get('chunk', i)}]"
        context_texts.append(f"{header}\n{d.page_content}")
    context_block = "\n\n---\n\n".join(context_texts) if context_texts else ""
    prompt = f"{SYSTEM_INSTRUCTIONS}\n\nContexto recuperado:\n{context_block}\n\nPregunta del usuario:\n{question}\n\nRespuesta (en español, con ejemplos si aplica):"
    return prompt

def answer_with_rag(question: str, k: int = None) -> Dict[str, Any]:
    docs = get_relevant_docs(question, k=k)
    prompt = build_prompt(docs, question)
    logger.debug("Sending prompt to LLM (length approx %d chars)" % len(prompt))
    answer = llm.generate(prompt)
    return {"answer": answer, "source_documents": docs}
