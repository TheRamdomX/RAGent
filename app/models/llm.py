from langchain_openai import ChatOpenAI
from app.utils import config
from app.utils.logger import logger
from typing import Dict, Any

class GPT5Nano:
    def __init__(self, model_name: str = config.LLM_MODEL, temperature: float = 1.0, max_completion_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self._client = ChatOpenAI(model=self.model_name, temperature=self.temperature, max_completion_tokens=self.max_completion_tokens)
        logger.debug(f"GPT5Nano client initialized with {self.model_name}")

    def generate(self, prompt: str, **kwargs) -> str:
        """
        prompt: full prompt text (system+context+user)
        returns: generated text
        """
        # ChatOpenAI expects messages; we'll pass as a simple user message
        response = self._client.call_as_llm(prompt) if hasattr(self._client, "call_as_llm") else self._client.predict(prompt)
        # Above chooses a likely interface; adapt according to langchain version.
        # Fallback:
        if isinstance(response, str):
            return response
        # If response is an object with .content
        try:
            return response.content  # type: ignore
        except Exception:
            # try text field
            return str(response)
