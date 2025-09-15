from app.controllers.conversation import ConversationManager
from typing import Dict, Any

class Chatbot:
    def __init__(self, use_rag: bool = True):
        self.manager = ConversationManager(use_rag=use_rag)

    def ask(self, query: str, use_rag_override: bool = None) -> Dict[str, Any]:
        """
        Devuelve un dict con keys: 'answer' y 'source_documents'
        """
        return self.manager.handle_query(query, use_rag_override=use_rag_override)
