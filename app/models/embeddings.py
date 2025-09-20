from langchain_openai import OpenAIEmbeddings
from app.utils import config
from app.utils.logger import logger
import os

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

class EmbeddingClient:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._client = OpenAIEmbeddings(model=self.model_name)

    def embed(self, texts):
        """
        texts: str or list[str]
        returns list[vector] or vector
        """
        return self._client.embed_documents(texts) if isinstance(texts, list) else self._client.embed_query(texts)
