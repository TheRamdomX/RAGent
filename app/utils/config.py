import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "3"))
MAX_CHUNK_SIZE: int = int(os.getenv("MAX_CHUNK_SIZE", "1000"))  # chars approximation
