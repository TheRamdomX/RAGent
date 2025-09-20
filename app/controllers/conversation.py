from typing import Dict, Any, List
from app.rag.qa import answer_with_rag
from app.utils.logger import logger

class ConversationManager:
    def __init__(self, use_rag: bool = True):
        self.use_rag = use_rag
        self.history: List[Dict[str, str]] = []  # simple turn history

    def handle_query(self, query: str, use_rag_override: bool = None):
        use_rag = self.use_rag if use_rag_override is None else use_rag_override
        # logger.info(f"Handling query. use_rag={use_rag}")
        self.history.append({"role": "user", "text": query})
        if use_rag:
            res = answer_with_rag(query)
            self.history.append({"role": "assistant", "text": res["answer"]})
            return res
        else:
            # if not RAG, we could call LLM directly (simple)
            from app.models.llm import GPT5Nano
            llm = GPT5Nano()
            prompt = f"Eres un asistente. Responde: {query}"
            answer = llm.generate(prompt)
            self.history.append({"role": "assistant", "text": answer})
            return {"answer": answer, "source_documents": []}
