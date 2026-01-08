from app.rag.retriever import get_relevant_docs, get_vectorstore
from app.models.llm import Agent
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any, Tuple, List, Set
import tiktoken
import re

llm = Agent()

SYSTEM_INSTRUCTIONS = (
    "Eres un asistente pedagógico. Usa el contexto recuperado para responder precisa y concisamente. "
    "Si no hay información en el contexto, indica explícitamente 'No disponible en el contexto' y no inventes detalles."
    "Puedes hacer inferencias simples solo si están claramente apoyadas por el contexto; marca cualquier información adicional como 'INFO_ADICIONAL'."
)

# Extensión de archivo soportada para detectar en la pregunta del usuario
FILE_EXTENSIONS = {".pdf"}


def extract_mentioned_files(question: str, available_sources: Set[str]) -> List[str]:
    """
    Detecta si el usuario menciona explícitamente archivos en su pregunta.
    Busca patrones como:
    - Nombres de archivo con extensión (ej: "homl.pdf")
    - Nombres parciales que coincidan con fuentes disponibles
    
    Args:
        question: La pregunta del usuario
        available_sources: Set de nombres de fuentes disponibles en la collection
    
    Returns:
        Lista de nombres de archivos detectados (normalizados a minúsculas)
    """
    mentioned = []
    question_lower = question.lower()
    
    # Patrón 1: Buscar nombres de archivo con extensión explícita
    # Matches: homl.pdf, mi_documento.pdf, archivo-v2.pdf, etc.
    file_pattern = r'[\w\-\.]+(?:' + '|'.join(re.escape(ext) for ext in FILE_EXTENSIONS) + r')'
    explicit_files = re.findall(file_pattern, question_lower)
    mentioned.extend(explicit_files)
    
    # Patrón 2: Buscar coincidencias parciales con fuentes disponibles
    # Si el usuario dice "en el libro homl" y existe "homl.pdf", lo detectamos
    for source in available_sources:
        source_lower = source.lower()
        # Obtener nombre base sin extensión
        base_name = re.sub(r'\.[^.]+$', '', source_lower)
        
        # Buscar si el nombre base aparece en la pregunta
        # Usamos word boundaries para evitar falsos positivos
        if base_name and len(base_name) > 2:  # Ignorar nombres muy cortos
            pattern = r'\b' + re.escape(base_name) + r'\b'
            if re.search(pattern, question_lower):
                mentioned.append(source_lower)
    
    # Patrón 3: Detectar frases como "en el archivo X", "del documento X", "según X"
    context_patterns = [
        r'(?:en|del|según|from|in)\s+(?:el\s+)?(?:archivo|documento|libro|pdf|file)\s+["\']?([\w\-\.]+)["\']?',
        r'(?:archivo|documento|libro|pdf|file)\s+["\']?([\w\-\.]+)["\']?',
    ]
    for pattern in context_patterns:
        matches = re.findall(pattern, question_lower)
        for match in matches:
            # Verificar si coincide con alguna fuente disponible
            for source in available_sources:
                if match in source.lower() or source.lower().startswith(match):
                    mentioned.append(source.lower())
                    break
    
    # Eliminar duplicados manteniendo orden
    seen = set()
    unique_mentioned = []
    for f in mentioned:
        if f not in seen:
            seen.add(f)
            unique_mentioned.append(f)
    
    return unique_mentioned


def prioritize_docs_by_source(context_docs: List, files_focus: List[str]) -> List:
    """
    Reordena los documentos priorizando aquellos cuya fuente está en files_focus.
    Los documentos prioritarios van primero, seguidos del resto.
    
    Args:
        context_docs: Lista de documentos recuperados
        files_focus: Lista de nombres de archivo a priorizar (en minúsculas)
    
    Returns:
        Lista de documentos reordenada
    """
    if not files_focus:
        return context_docs
    
    focus_set = set(f.lower() for f in files_focus)
    prioritized = []
    others = []
    
    for d in context_docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        src = (meta.get("source") or "").lower()
        # Clasificar: si la fuente está en el conjunto de foco, va a prioritized
        if src in focus_set:
            prioritized.append(d)
        else:
            others.append(d)
    
    logger.info(f"Priorización de documentos: {len(prioritized)} prioritarios, {len(others)} otros")
    return prioritized + others

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
    # Obtener documentos relevantes
    docs = get_relevant_docs(question, k=k, collection_name=collection_name)
    
    # Extraer las fuentes disponibles de los documentos recuperados
    available_sources = set()
    for d in docs:
        meta = d.metadata if hasattr(d, "metadata") else {}
        src = meta.get("source")
        if src:
            available_sources.add(src)
    
    # Detectar si el usuario menciona archivos específicos
    mentioned_files = extract_mentioned_files(question, available_sources)
    
    if mentioned_files:
        logger.info(f"Archivos mencionados detectados: {mentioned_files}")
        # Priorizar documentos de los archivos mencionados
        docs = prioritize_docs_by_source(docs, mentioned_files)
    
    prompt = build_prompt(docs, question)
    encoding = tiktoken.encoding_for_model(config.LLM_MODEL)
    tokens_used = len(encoding.encode(prompt))

    answer = llm.generate(prompt)
    return {"answer": answer, "source_documents": docs, "tokens_used": tokens_used, "files_focus": mentioned_files}

