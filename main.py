import os
import typer
from app.chatbot import Chatbot
from app.data.ingestion import ingest_files
from app.utils.logger import logger
from app.utils import config
from app.rag.retriever import get_vectorstore
import os

app = typer.Typer()

@app.command()
def ingest(paths: list[str], collection: str = "study_collection", dry_run: bool = False):

        docs = ingest_files(paths, collection_name=collection, dry_run=dry_run)
        if dry_run:
            print(f"Dry-run: {len(docs)} chunks would be created/added from provided paths")

@app.command()
def chat(use_rag: bool = True):
    
    bot = Chatbot(use_rag=use_rag)
    print("Modo chat. Escribe 'exit' para salir.")
    while True:
        q = input("Tú> ").strip()
        if q.lower() in ("exit", "quit", "salir"):
            break
        res = bot.ask(q)
        print("\nRespuesta:")
        print(res["answer"])
        if res.get("source_documents"):
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
def run(paths: list[str] = typer.Argument(None), collection: str = "study_collection", use_rag: bool = True, dry_run: bool = False):

    if paths:
        docs = ingest_files(paths, collection_name=collection, dry_run=dry_run)
        if dry_run:
            print(f"Dry-run: {len(docs)} chunks would be created/added from provided paths")
    else:
        if not os.path.exists(config.CHROMA_PERSIST_DIR):
            print("No existe base de datos y no se entregaron archivos. Ingresa archivos primero con 'ingest'.")
            raise typer.Exit(code=1)

    chat(use_rag=use_rag)


@app.command()
def delete(targets: list[str] = typer.Argument(..., help="Paths to source files (e.g. files/maze.pdf) or document ids to delete"), ids: str = typer.Option(None, help="Comma-separated document ids to delete")):

    vs = get_vectorstore()

    ids_to_delete = []

    if ids:
        for _id in ids.split(','):
            _id = _id.strip()
            if _id:
                ids_to_delete.append(_id)

    for t in targets:
        if os.path.sep not in t and len(t) > 15 and ids is None:
            ids_to_delete.append(t)
            continue
        base = os.path.basename(t)
        data = vs.get()
        for _id, md in zip(data.get('ids', []), data.get('metadatas', [])):
            if not md:
                continue
            if md.get('source') == base or md.get('source') == t:
                ids_to_delete.append(_id)

    if not ids_to_delete:
        print('No se encontraron documentos para eliminar con los targets/ids proporcionados.')
        raise typer.Exit()

    ids_to_delete = sorted(set(ids_to_delete))

    print(f'Se eliminarán {len(ids_to_delete)} documentos. Primeros IDs: {ids_to_delete[:5]}')
    if not typer.confirm('¿Confirmas la eliminación? Esto es irreversible'):
        print('Cancelado')
        raise typer.Exit()

    try:
        vs.delete(ids=ids_to_delete)
        logger.info(f'Eliminados {len(ids_to_delete)} documentos')
        print(f'Eliminados {len(ids_to_delete)} documentos.')
    except Exception as e:
        logger.exception(f'Error al eliminar documentos: {e}')
        print(f'Error al eliminar documentos: {e}')


@app.command("list")
def list_files(a: bool = typer.Option(False, help="Show first ids per source")):
    vs = get_vectorstore()
    data = vs.get()
    ids = data.get('ids', []) or []
    metadatas = data.get('metadatas', []) or []

    counts = {}
    samples = {}
    for _id, md in zip(ids, metadatas):
        if not md:
            continue
        src = md.get('source', 'desconocido')
        counts[src] = counts.get(src, 0) + 1
        samples.setdefault(src, []).append(_id)

    if not counts:
        print('No hay documentos en la base de datos.')
        raise typer.Exit()

    print('Fuentes en la base de datos:')
    for src, cnt in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        line = f" - {src}: {cnt} chunks"
        if a:
            s = samples.get(src, [])[:5]
            line += f" | sample ids: {s}"
        print(line)

if __name__ == "__main__":
    app()
