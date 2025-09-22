from langchain_openai import ChatOpenAI
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any

class Agent:
    def __init__(self, model_name: str = config.LLM_MODEL, temperature: float = None, max_completion_tokens: int = None):
        self.model_name = model_name
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.max_completion_tokens = config.LLM_MAX_COMPLETION_TOKENS if max_completion_tokens is None else max_completion_tokens
        self._client = ChatOpenAI(model=self.model_name, temperature=self.temperature, max_completion_tokens=self.max_completion_tokens)

    def generate(self, prompt: str, **kwargs) -> str:

        response = self._client.invoke(prompt)
        if isinstance(response, str):
            return response
        try:
            return response.content  # type: ignore
        except Exception:
            return str(response)