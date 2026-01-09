import os
import typer
from app.chatbot import Chatbot
from app.data.ingestion import ingest_files
from app.utils.logger import logger
from app.utils import config
from app.rag.retriever import get_vectorstore

app = typer.Typer()

@app.command()
def ingest(
    paths: list[str],
    collection: str = typer.Option("study_collection", help="Nombre de la colección"),
    dry_run: bool = typer.Option(False, help="Simular sin guardar"),
    force_ocr: bool = typer.Option(None, help="Forzar OCR para PDFs"),
    extract_graph: bool = typer.Option(False, "--graph", "-g", help="Extraer relaciones al grafo de conocimiento")
):
    """Ingiere archivos PDF en ChromaDB y opcionalmente extrae relaciones al grafo."""
    print(f"Ingiriendo {len(paths)} archivo(s) en colección '{collection}'...")
    if extract_graph:
        print("Extracción de grafo de conocimiento habilitada")
    
    docs = ingest_files(
        paths,
        collection_name=collection,
        dry_run=dry_run,
        force_ocr=force_ocr,
        extract_graph=extract_graph
    )
    
    if dry_run:
        print(f"Dry-run: {len(docs)} chunks would be created/added from provided paths")
    else:
        print(f"Ingesta completada: {len(docs)} chunks agregados")
        if extract_graph:
            print("Relaciones extraídas al grafo de conocimiento")

@app.command()
def chat(use_rag: bool = True, collection: str = "study_collection"):
    
    bot = Chatbot(use_rag=use_rag, collection_name=collection)
    print(f"Modo chat (collection: {collection}). Escribe 'exit' para salir.")
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
def run(paths: list[str] = typer.Argument(None), collection: str = "study_collection", use_rag: bool = True, dry_run: bool = False, force_ocr: bool = None):

    if paths:
        docs = ingest_files(paths, collection_name=collection, dry_run=dry_run, force_ocr=force_ocr)
        if dry_run:
            print(f"Dry-run: {len(docs)} chunks would be created/added from provided paths")
    else:
        if not os.path.exists(config.CHROMA_PERSIST_DIR):
            print("No existe base de datos y no se entregaron archivos. Ingresa archivos primero con 'ingest'.")
            raise typer.Exit(code=1)

    chat(use_rag=use_rag, collection=collection)


@app.command()
def delete(targets: list[str] = typer.Argument(..., help="Paths to source files (e.g. files/maze.pdf) or document ids to delete"), ids: str = typer.Option(None, help="Comma-separated document ids to delete"), collection: str = typer.Option("study_collection", help="Collection name to delete from")):

    vs = get_vectorstore(collection_name=collection)

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
def list_files(a: bool = typer.Option(False, help="Show first ids per source"), collection: str = typer.Option("study_collection", help="Collection name to list")):
    vs = get_vectorstore(collection_name=collection)
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


# =============================================================================
# Comandos del Grafo de Conocimiento
# =============================================================================

@app.command("graph-status")
def graph_status():
    """Muestra el estado del grafo de conocimiento."""
    try:
        from app.knowledge_graph.graph_store import GraphStore
        from app.knowledge_graph.sparql_queries import SPARQLQueryCatalog
        
        store = GraphStore()
        store.open()
        
        count = store.count_triples()
        print(f"📊 Grafo de conocimiento:")
        print(f"   Tipo: {store.store_type}")
        print(f"   Ruta: {store.local_path if store.store_type == 'local' else store.fuseki_url}")
        print(f"   Tripletas: {count}")
        
        # Mostrar resumen por tipo de entidad
        catalog = SPARQLQueryCatalog(store)
        resumen = catalog.get_resumen_grafo()
        if resumen.success and resumen.results:
            print(f"\n{resumen.formatted}")
        
        store.close()
    except ImportError as e:
        print(f"❌ Módulo de grafo no disponible: {e}")
    except Exception as e:
        print(f"❌ Error accediendo al grafo: {e}")


@app.command("graph-query")
def graph_query(
    query: str = typer.Argument(..., help="Pregunta estructural o consulta SPARQL"),
    raw: bool = typer.Option(False, help="Interpretar query como SPARQL directo")
):
    """Ejecuta una consulta en el grafo de conocimiento."""
    try:
        from app.knowledge_graph.graph_store import GraphStore
        from app.knowledge_graph.sparql_queries import SPARQLQueryCatalog
        
        store = GraphStore()
        store.open()
        catalog = SPARQLQueryCatalog(store)
        
        if raw:
            # Ejecutar SPARQL directo
            results = store.query(query)
            print(f"Resultados ({len(results)}):")
            for r in results[:20]:
                print(f"  {r}")
        else:
            # Detectar intención y ejecutar
            result = catalog.execute_structural_query(query)
            if result:
                print(result.formatted)
                print(f"\n[Query type: {result.query_type}]")
            else:
                # Intentar búsqueda general
                result = catalog.search_entities(query)
                print(result.formatted)
        
        store.close()
    except ImportError as e:
        print(f"❌ Módulo de grafo no disponible: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


@app.command("graph-extract")
def graph_extract(
    paths: list[str] = typer.Argument(..., help="PDFs de los que extraer relaciones"),
    collection: str = typer.Option("study_collection", help="Collection de ChromaDB"),
    force: bool = typer.Option(False, help="Forzar re-extracción aunque ya existan en el grafo")
):
    """Extrae relaciones de documentos ya ingestados al grafo de conocimiento."""
    try:
        from app.knowledge_graph.graph_store import GraphStore
        from app.knowledge_graph.extract_relations import RelationExtractor
        
        # Obtener chunks de ChromaDB
        vs = get_vectorstore(collection_name=collection)
        data = vs.get(include=["documents", "metadatas"])
        
        chunks_to_process = []
        sources_filter = set(os.path.basename(p) for p in paths)
        
        for _id, doc, meta in zip(data.get('ids', []), data.get('documents', []), data.get('metadatas', [])):
            if not meta:
                continue
            source = meta.get('source', '')
            if source in sources_filter or not paths:
                chunks_to_process.append({
                    "text": doc,
                    "chunk_id": meta.get('chunk_id', _id),
                    "source": source,
                    "page": meta.get('page_start')
                })
        
        if not chunks_to_process:
            print("No se encontraron chunks para procesar.")
            raise typer.Exit()
        
        print(f"Procesando {len(chunks_to_process)} chunks...")
        
        store = GraphStore()
        store.open()
        extractor = RelationExtractor()
        
        processed, triples = extractor.extract_batch(chunks_to_process, store=store)
        
        store.close()
        
        print(f"✅ Extracción completada: {processed} chunks → {triples} tripletas")
        
    except ImportError as e:
        print(f"❌ Módulo de grafo no disponible: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


@app.command("graph-clear")
def graph_clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirmar eliminación")
):
    """Elimina todas las tripletas del grafo de conocimiento."""
    if not confirm:
        if not typer.confirm("⚠️ Esto eliminará TODO el grafo de conocimiento. ¿Continuar?"):
            print("Cancelado")
            raise typer.Exit()
    
    try:
        from app.knowledge_graph.graph_store import GraphStore
        
        store = GraphStore()
        store.open()
        count_before = store.count_triples()
        store.clear(confirm=True)
        store.close()
        
        print(f"✅ Grafo limpiado. Se eliminaron {count_before} tripletas.")
        
    except ImportError as e:
        print(f"❌ Módulo de grafo no disponible: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    app()
