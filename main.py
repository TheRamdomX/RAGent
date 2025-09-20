import os
import typer
from app.chatbot import Chatbot
from app.data.ingestion import ingest_files
from app.utils.logger import logger
from app.utils import config

app = typer.Typer()

@app.command()
def ingest(paths: list[str], collection: str = "study_collection"):
        """
        Ingesta archivos a la base de conocimiento.
        Ejemplo:
            python main.py ingest docs/algebra.pdf docs/notes.docx
        """
        ingest_files(paths, collection_name=collection)

@app.command()
def chat(use_rag: bool = True):
    """
    Entra a modo chat CLI.
    """
    bot = Chatbot(use_rag=use_rag)
    print("📚 Modo chat. Escribe 'exit' para salir.")
    while True:
        q = input("Tú> ").strip()
        if q.lower() in ("exit", "quit", "salir"):
            break
        res = bot.ask(q)
        print("\n🤖 Respuesta:")
        print(res["answer"])
        if res.get("source_documents"):
            # Mostrar todas las fuentes únicas usadas
            sources = set()
            for d in res["source_documents"]:
                src = d.metadata.get("source", "desconocido") if hasattr(d, "metadata") else "desconocido"
                sources.add(src)
            if sources:
                print("\n📎 Fuentes:")
                for src in sorted(sources):
                    print(f" - {src}")
        print("\n---\n")

@app.command()
def run(paths: list[str] = typer.Argument(None), collection: str = "study_collection", use_rag: bool = True):
    """
    Ingesta (si se pasan archivos) y luego abre el chat.
    Ejemplo:
      python main.py run docs/algebra.pdf docs/notes.docx
    """
    if paths:
        ingest_files(paths, collection_name=collection)
    else:
        if not os.path.exists(config.CHROMA_PERSIST_DIR):
            print("No existe base de datos y no se entregaron archivos. Ingresa archivos primero con 'ingest'.")
            raise typer.Exit(code=1)

    chat(use_rag=use_rag)

if __name__ == "__main__":
    app()
