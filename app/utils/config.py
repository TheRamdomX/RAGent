import os
from dotenv import load_dotenv

load_dotenv()

# Clave de API para OpenAI. Variable de entorno: OPENAI_API_KEY
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Directorio donde se persiste la base de vectores Chroma.
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Nombre del modelo de embeddings a usar.
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Modelo LLM por defecto para generación. Variable de entorno: LLM_MODEL
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-nano")

# Número por defecto de documentos a devolver en búsquedas (top-k). Variable de entorno: DEFAULT_TOP_K
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "15"))

# Tamaño aproximado en caracteres para dividir el texto en chunks (chunks grandes ~1000 chars)
MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))  # chars approximation

# Número de candidatos a recuperar antes de aplicar reranking (si está habilitado).
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "20"))

# Habilita o deshabilita la etapa de reranking. Variable de entorno: RERANK_ENABLED
RERANK_ENABLED: bool = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")

# Límite aproximado de tokens que el modelo puede manejar. Ajustable vía MAX_MODEL_TOKENS
MAX_MODEL_TOKENS: int = int(os.getenv("MAX_MODEL_TOKENS", "300000"))

# Reservar tokens para la respuesta del modelo (para evitar exceder límites).
RESERVED_RESPONSE_TOKENS: int = int(os.getenv("RESERVED_RESPONSE_TOKENS", "2048"))

# Estimación de cuántos caracteres corresponden a un token (usado para estimaciones rápidas).
TOKENS_PER_CHARS: int = int(os.getenv("TOKENS_PER_CHARS", "4"))

# Si se establece a true, forzar el uso de Marker OCR para PDFs cuando sea aplicable.
FORCE_MARKER_OCR: bool = os.getenv("FORCE_MARKER_OCR", "false").lower() in ("1", "true", "yes")

# Si el texto extraído tiene menos caracteres que este valor y FORCE_MARKER_OCR=True, se invocará Marker OCR.
MARKER_OCR_THRESHOLD: int = int(os.getenv("MARKER_OCR_THRESHOLD", "500"))

# Nombre de colección por defecto para la ingestión de documentos.
DEFAULT_COLLECTION_NAME: str = os.getenv("DEFAULT_COLLECTION_NAME", "study_collection")

# Controla si, por defecto, los documentos se persisten en la base de vectores en disco.
DEFAULT_PERSIST: bool = os.getenv("DEFAULT_PERSIST", "true").lower() in ("1", "true", "yes")

# Umbral de deduplicación por similitud (0..1). Si la similitud >= DEDUP_SIM_THRESHOLD se considera duplicado.
DEDUP_SIM_THRESHOLD: float = float(os.getenv("DEDUP_SIM_THRESHOLD", "0.9"))

# Tamaño de chunk por defecto y solapamiento (en caracteres) para chunking.
# Tamaño por defecto de chunk y solapamiento en caracteres.
# Variables de entorno: CHUNK_DEFAULT_SIZE, CHUNK_DEFAULT_OVERLAP
CHUNK_DEFAULT_SIZE: int = int(os.getenv("CHUNK_DEFAULT_SIZE", str(MAX_CHUNK_SIZE)))
CHUNK_DEFAULT_OVERLAP: int = int(os.getenv("CHUNK_DEFAULT_OVERLAP", "200"))

# Longitud mínima en caracteres para aceptar un chunk extraído.
MIN_CHUNK_CHARS: int = int(os.getenv("MIN_CHUNK_CHARS", "50"))


# Controla la aleatoriedad (0.0 más determinista, 1.0 más variado).
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# Limita el tamaño de la respuesta generada por el LLM.
LLM_MAX_COMPLETION_TOKENS: int = int(os.getenv("LLM_MAX_COMPLETION_TOKENS", "2048"))


# Limita el número de llamadas a herramientas por consulta(ReAct).
BUDGET_CALLS_PER_QUERY: int = int(os.getenv("BUDGET_CALLS_PER_QUERY", "5"))

# Limita tokens consumidos por herramientas en una consulta (0 = sin límite).
TOKEN_BUDGET_PER_QUERY: int = int(os.getenv("TOKEN_BUDGET_PER_QUERY", "0"))
