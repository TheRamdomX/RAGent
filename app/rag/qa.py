from app.rag.retriever import get_relevant_docs, get_vectorstore
from app.models.llm import Agent
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any, Tuple
import tiktoken

llm = Agent()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente pedagógico. Usa el contexto recuperado para responder precisa y concisamente. "
    "Si no hay información en el contexto, indica explícitamente 'No disponible en el contexto' y no inventes detalles."
    "Puedes hacer inferencias simples solo si están claramente apoyadas por el contexto; marca cualquier información adicional como 'INFO_ADICIONAL'."
)

def build_prompt(context_docs, question: str) -> str:
    max_model_tokens = config.MAX_MODEL_TOKENS
    reserved = config.RESERVED_RESPONSE_TOKENS

    context_texts = []

    base_suffix = f"\n\nPregunta del usuario:\n{question}\n\nRespuesta (en español, con ejemplos si aplica):"
    base_prefix = SYSTEM_INSTRUCTIONS + "\n\nContexto recuperado:\n"

    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)

    base_tokens = len(encoding.encode(SYSTEM_INSTRUCTIONS + base_suffix))
    allowed_tokens_for_context = max_model_tokens - reserved - base_tokens
    if allowed_tokens_for_context <= 0:
        allowed_tokens_for_context = max_model_tokens // 4

    used_tokens = 0
    for i, d in enumerate(context_docs):
        meta = d.metadata if hasattr(d, 'metadata') else {}
        header = f"[Fuente: {meta.get('source','desconocido')} | chunk={meta.get('chunk', i)}]\n"
        content = d.page_content or ""
        tok_count = len(encoding.encode(header + content))

        if used_tokens + tok_count > allowed_tokens_for_context:
            remaining = allowed_tokens_for_context - used_tokens
            if remaining <= 0:
                break
            lo, hi = 0, len(content)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if len(encoding.encode(header + content[:mid])) <= remaining:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best > 0:
                truncated = content[:best]
                context_texts.append(f"{header}{truncated}")
                used_tokens += len(encoding.encode(header + truncated))
            break
        else:
            context_texts.append(f"{header}{content}")
            used_tokens += tok_count

    context_block = "\n\n---\n\n".join(context_texts) if context_texts else ""


    formatting = (
        "Instrucciones de formato:\n"
        "Si la información no está en el contexto, responde exactamente: 'No disponible en el contexto'.\n"
        "Al final, incluye una sección 'FUENTES' con la lista deduplicada de referencias utilizadas.\n"
        "Mantén la respuesta breve y directa; si se requieren pasos, numéralos.\n"
    )

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\nContexto recuperado:\n{context_block}\n\n{formatting}\nPregunta del usuario:\n{question}\n\nRespuesta (en español, con ejemplos si aplica):"
    )
    return prompt

def answer_with_rag(question: str, k: int = None, collection_name: str = None) -> Dict[str, Any]:
    docs = get_relevant_docs(question, k=k, collection_name=collection_name)
    prompt = build_prompt(docs, question)
    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)
    tokens_used = len(encoding.encode(prompt))

    answer = llm.generate(prompt)
    return {"answer": answer, "source_documents": docs, "tokens_used": tokens_used}

