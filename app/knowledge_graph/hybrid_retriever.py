"""
hybrid_retriever.py - Recuperación híbrida (Grafo + Vector)

Combina búsqueda estructural SPARQL con búsqueda semántica ChromaDB.
Flujo durante consulta:
1. Detectar intención estructural → SPARQL
2. Recuperar chunks relevantes → ChromaDB
3. Combinar resultados → Prompt híbrido
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain.schema import Document

from app.knowledge_graph.graph_store import GraphStore
from app.knowledge_graph.sparql_queries import SPARQLQueryCatalog, QueryResult
from app.rag.retriever import get_relevant_docs, get_vectorstore
from app.utils import config
from app.utils.logger import logger


@dataclass
class HybridResult:
    """Resultado de recuperación híbrida."""
    # Resultados estructurales (SPARQL)
    structural_result: Optional[QueryResult]
    has_structural: bool
    
    # Resultados vectoriales (ChromaDB)
    vector_docs: List[Document]
    
    # Chunks enriquecidos con información del grafo
    enriched_docs: List[Document]
    
    # Contexto combinado para el prompt
    combined_context: str
    
    # Metadatos
    query_type: str
    sources_used: List[str]


class HybridRetriever:
    """
    Recuperador híbrido que combina:
    - Búsqueda estructural en grafo RDF (hechos, relaciones)
    - Búsqueda semántica en ChromaDB (texto similar)
    """
    
    def __init__(
        self,
        graph_store: GraphStore = None,
        collection_name: str = None
    ):
        """
        Inicializa el recuperador híbrido.
        
        Args:
            graph_store: GraphStore para consultas SPARQL.
            collection_name: Nombre de la colección ChromaDB.
        """
        self._store = graph_store
        self._query_catalog = SPARQLQueryCatalog(graph_store)
        self._collection_name = collection_name or config.DEFAULT_COLLECTION_NAME
    
    def set_graph_store(self, store: GraphStore):
        """Establece el GraphStore."""
        self._store = store
        self._query_catalog.set_store(store)
    
    def retrieve(
        self,
        question: str,
        k: int = None,
        use_structural: bool = True,
        use_vector: bool = True,
        enrich_with_graph: bool = True
    ) -> HybridResult:
        """
        Realiza recuperación híbrida.
        
        Args:
            question: Pregunta del usuario.
            k: Número de documentos a recuperar de ChromaDB.
            use_structural: Si intentar búsqueda estructural SPARQL.
            use_vector: Si usar búsqueda vectorial ChromaDB.
            enrich_with_graph: Si enriquecer docs con info del grafo.
        
        Returns:
            HybridResult con toda la información recuperada.
        """
        k = k or config.DEFAULT_TOP_K
        structural_result = None
        has_structural = False
        vector_docs = []
        enriched_docs = []
        sources_used = []
        query_type = "vector_only"
        
        # 1. Intentar búsqueda estructural
        if use_structural and self._store:
            structural_result = self._query_catalog.execute_structural_query(question)
            if structural_result and structural_result.success:
                has_structural = True
                query_type = f"hybrid_{structural_result.query_type}"
                sources_used.append("knowledge_graph")
                logger.info(f"Búsqueda estructural exitosa: {structural_result.query_type}")
        
        # 2. Búsqueda vectorial
        if use_vector:
            vector_docs = get_relevant_docs(
                question,
                k=k,
                collection_name=self._collection_name
            )
            if vector_docs:
                sources_used.append("chroma_db")
                logger.info(f"Recuperados {len(vector_docs)} docs de ChromaDB")
        
        # 3. Enriquecer documentos con información del grafo
        if enrich_with_graph and self._store and vector_docs:
            enriched_docs = self._enrich_documents(vector_docs)
        else:
            enriched_docs = vector_docs
        
        # 4. Combinar contexto
        combined_context = self._build_combined_context(
            structural_result,
            enriched_docs,
            question
        )
        
        return HybridResult(
            structural_result=structural_result,
            has_structural=has_structural,
            vector_docs=vector_docs,
            enriched_docs=enriched_docs,
            combined_context=combined_context,
            query_type=query_type,
            sources_used=sources_used
        )
    
    def _enrich_documents(self, docs: List[Document]) -> List[Document]:
        """
        Enriquece documentos con información del grafo.
        Agrega entidades mencionadas, relaciones, etc.
        """
        if not self._store:
            return docs
        
        enriched = []
        for doc in docs:
            meta = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            
            # Intentar encontrar el chunk en el grafo
            chunk_id = meta.get("chunk_id") or f"chunk_{meta.get('chunk', '')}"
            source = meta.get("source", "")
            
            # Buscar entidades mencionadas por este chunk
            entities = self._get_chunk_entities(chunk_id, source)
            if entities:
                meta["graph_entities"] = entities
            
            # Crear documento enriquecido
            enriched.append(Document(
                page_content=doc.page_content,
                metadata=meta
            ))
        
        return enriched
    
    def _get_chunk_entities(self, chunk_id: str, source: str) -> List[Dict[str, str]]:
        """Obtiene las entidades que menciona un chunk."""
        # Sanitizar IDs
        safe_chunk_id = chunk_id.replace(" ", "_").replace("/", "_")
        
        sparql = f"""
        PREFIX rag: <http://ragent.local/ontology#>
        PREFIX ragent: <http://ragent.local/data#>
        
        SELECT DISTINCT ?entidad ?nombre ?tipo WHERE {{
            {{
                ragent:{safe_chunk_id} rag:menciona ?entidad .
            }} UNION {{
                ?chunk rag:perteneceA ?doc .
                ?doc rag:fuente "{source}" .
                ?chunk rag:menciona ?entidad .
            }}
            OPTIONAL {{ ?entidad rag:nombre ?nombre . }}
            OPTIONAL {{ ?entidad rdf:type ?tipo . }}
        }}
        LIMIT 10
        """
        
        results = self._store.query(sparql)
        return [
            {
                "id": r.get("entidad", "").split("#")[-1],
                "nombre": r.get("nombre", ""),
                "tipo": r.get("tipo", "").split("#")[-1]
            }
            for r in results
        ]
    
    def _build_combined_context(
        self,
        structural: Optional[QueryResult],
        docs: List[Document],
        question: str
    ) -> str:
        """
        Construye el contexto combinado para el prompt.
        
        Estructura:
        1. Hechos estructurales (si hay)
        2. Contexto textual de documentos
        3. Entidades relacionadas (si hay)
        """
        sections = []
        
        # Sección 1: Hechos estructurales del grafo
        if structural and structural.success and structural.formatted:
            sections.append(
                "## Información estructurada (del grafo de conocimiento)\n\n"
                f"{structural.formatted}"
            )
        
        # Sección 2: Contexto textual
        if docs:
            text_parts = []
            for i, doc in enumerate(docs):
                meta = doc.metadata if hasattr(doc, 'metadata') else {}
                source = meta.get("source", "desconocido")
                chunk_num = meta.get("chunk", i)
                page = meta.get("page_start", meta.get("page", "?"))
                
                # Header del chunk
                header = f"[Fuente: {source} | Chunk {chunk_num} | Pág. {page}]"
                
                # Agregar entidades del grafo si las hay
                entities = meta.get("graph_entities", [])
                if entities:
                    entity_names = [e.get("nombre") or e.get("id") for e in entities[:5]]
                    header += f"\nEntidades relacionadas: {', '.join(entity_names)}"
                
                text_parts.append(f"{header}\n{doc.page_content}")
            
            sections.append(
                "## Contexto textual (de documentos)\n\n"
                + "\n\n---\n\n".join(text_parts)
            )
        
        # Combinar todo
        if not sections:
            return "No se encontró contexto relevante."
        
        return "\n\n".join(sections)
    
    def get_related_chunks(self, entity_name: str, k: int = 5) -> List[Document]:
        """
        Obtiene chunks relacionados con una entidad del grafo.
        Útil para expandir contexto basado en relaciones.
        """
        if not self._store:
            return []
        
        # Buscar chunks que mencionan la entidad
        result = self._query_catalog.get_chunks_que_mencionan(entity_name)
        
        if not result.success:
            return []
        
        # Obtener los chunks de ChromaDB
        vectordb = get_vectorstore(collection_name=self._collection_name)
        chunk_ids = [r.get("chunk", "") for r in result.results[:k]]
        
        # Por ahora, hacer búsqueda por la entidad
        return get_relevant_docs(
            entity_name,
            k=k,
            collection_name=self._collection_name
        )
    
    def explain_entity(self, entity_name: str) -> Dict[str, Any]:
        """
        Genera una explicación estructurada de una entidad.
        Combina información del grafo con texto de soporte.
        """
        if not self._store:
            return {"error": "No hay grafo configurado"}
        
        # Buscar entidad en el grafo
        search_result = self._query_catalog.search_entities(entity_name)
        
        if not search_result.success:
            return {"error": f"No se encontró '{entity_name}'"}
        
        # Obtener primera coincidencia
        entity = search_result.results[0] if search_result.results else {}
        entity_id = entity.get("entidad", "").split("#")[-1]
        
        # Obtener relaciones
        relations = self._store.get_chunk_relations(entity_id)
        
        # Obtener chunks de soporte
        support_docs = self.get_related_chunks(entity_name, k=3)
        
        return {
            "entity": entity,
            "relations": relations,
            "support_text": [doc.page_content[:500] for doc in support_docs],
            "formatted": search_result.formatted
        }


# Instancia global para uso conveniente
_hybrid_retriever: Optional[HybridRetriever] = None


def get_hybrid_retriever(
    graph_store: GraphStore = None,
    collection_name: str = None
) -> HybridRetriever:
    """
    Obtiene o crea una instancia del recuperador híbrido.
    """
    global _hybrid_retriever
    
    if _hybrid_retriever is None or graph_store is not None:
        _hybrid_retriever = HybridRetriever(
            graph_store=graph_store,
            collection_name=collection_name
        )
    
    return _hybrid_retriever
