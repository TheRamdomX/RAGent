"""
graph_store.py - Capa de persistencia para el grafo RDF

Proporciona acceso a Apache Jena TDB2 (via Fuseki HTTP o RDFLib local).
Responsabilidades:
- Abrir/cerrar conexión al grafo
- Insertar tripletas
- Ejecutar consultas SPARQL
- NO contiene lógica de negocio ni parsing de texto
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, OWL
import requests

from app.utils import config
from app.utils.logger import logger


# Namespace del proyecto
RAG = Namespace("http://ragent.local/ontology#")
RAGENT = Namespace("http://ragent.local/data#")


@dataclass
class Triple:
    """Representa una tripleta RDF."""
    subject: str
    predicate: str
    object: str
    object_is_literal: bool = False
    literal_datatype: Optional[str] = None


class GraphStore:
    """
    Almacén de grafos RDF con soporte para:
    - RDFLib local (archivo .ttl)
    - Apache Jena Fuseki (endpoint SPARQL remoto)
    """
    
    def __init__(
        self,
        store_type: str = None,
        local_path: str = None,
        fuseki_url: str = None,
        dataset: str = None
    ):
        self.store_type = store_type or getattr(config, 'GRAPH_STORE_TYPE', 'local')
        self.local_path = local_path or getattr(config, 'GRAPH_LOCAL_PATH', './knowledge_graph.ttl')
        self.fuseki_url = fuseki_url or getattr(config, 'FUSEKI_URL', 'http://localhost:3030')
        self.dataset = dataset or getattr(config, 'FUSEKI_DATASET', 'ragent')
        
        self._graph: Optional[Graph] = None
        self._ontology_loaded = False
        
    def _get_ontology_path(self) -> str:
        """Retorna la ruta al archivo de ontología."""
        return os.path.join(os.path.dirname(__file__), 'ontology.ttl')
    
    def open(self) -> "GraphStore":
        """Abre la conexión al grafo."""
        if self.store_type == "local":
            self._graph = Graph()
            self._graph.bind("rag", RAG)
            self._graph.bind("ragent", RAGENT)
            self._graph.bind("rdf", RDF)
            self._graph.bind("rdfs", RDFS)
            self._graph.bind("xsd", XSD)
            self._graph.bind("owl", OWL)
            
            # Cargar datos existentes si el archivo existe
            if os.path.exists(self.local_path):
                self._graph.parse(self.local_path, format="turtle")
                logger.info(f"Grafo cargado desde {self.local_path} ({len(self._graph)} tripletas)")
            
            # Cargar ontología
            if not self._ontology_loaded:
                ontology_path = self._get_ontology_path()
                if os.path.exists(ontology_path):
                    self._graph.parse(ontology_path, format="turtle")
                    self._ontology_loaded = True
                    logger.info(f"Ontología cargada desde {ontology_path}")
        
        elif self.store_type == "fuseki":
            resp = requests.get(f"{self.fuseki_url}/$/ping", timeout=5)
            if resp.status_code == 200:
                logger.info(f"Conectado a Fuseki en {self.fuseki_url}")
        
        return self
    
    def close(self):
        """Cierra la conexión y persiste cambios."""
        if self.store_type == "local" and self._graph is not None:
            self._persist_local()
        self._graph = None
    
    def _persist_local(self):
        """Guarda el grafo local a disco."""
        if self._graph is None:
            return
        os.makedirs(os.path.dirname(self.local_path) or ".", exist_ok=True)
        self._graph.serialize(destination=self.local_path, format="turtle")
        logger.info(f"Grafo persistido en {self.local_path} ({len(self._graph)} tripletas)")
    
    def _make_uri(self, value: str, namespace: Namespace = RAGENT) -> URIRef:
        """Crea una URI a partir de un string."""
        safe_value = value.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe_value = "".join(c for c in safe_value if c.isalnum() or c in "_-.")
        return namespace[safe_value]
    
    def _resolve_predicate(self, predicate: str) -> URIRef:
        """Resuelve un predicado a su URI completa."""
        predicate_map = {
            "menciona": RAG.menciona,
            "pertenece_a": RAG.perteneceA,
            "perteneceA": RAG.perteneceA,
            "evalua": RAG.evalua,
            "evalúa": RAG.evalua,
            "es_parte_de": RAG.esParteDe,
            "esParteDe": RAG.esParteDe,
            "requiere": RAG.requiere,
            "relacionado_con": RAG.relacionadoCon,
            "relacionadoCon": RAG.relacionadoCon,
            "tipo": RAG.tipo,
            "nombre": RAG.nombre,
            "ponderacion": RAG.ponderacion,
            "ponderación": RAG.ponderacion,
            "texto": RAG.texto,
            "chunkId": RAG.chunkId,
            "chunk_id": RAG.chunkId,
            "fuente": RAG.fuente,
            "pagina": RAG.pagina,
            "página": RAG.pagina,
            "autor": RAG.autor,
            "profesor": RAG.profesor,
            "tiene_fecha": RAG.tieneFecha,
            "tieneFecha": RAG.tieneFecha,
            "fecha_valor": RAG.fechaValor,
            "fechaValor": RAG.fechaValor,
            "codigo": RAG.codigo,
            "código": RAG.codigo,
            "descripcion": RAG.descripcion,
            "descripción": RAG.descripcion,
            "rdf:type": RDF.type,
            "type": RDF.type,
            "a": RDF.type,
        }
        
        pred_lower = predicate.lower().strip()
        if predicate in predicate_map:
            return predicate_map[predicate]
        if pred_lower in predicate_map:
            return predicate_map[pred_lower]
        
        return self._make_uri(predicate, RAG)
    
    def _resolve_class(self, class_name: str) -> URIRef:
        """Resuelve un nombre de clase a su URI."""
        class_map = {
            "chunk": RAG.Chunk,
            "documento": RAG.Documento,
            "ramo": RAG.Ramo,
            "evaluacion": RAG.Evaluacion,
            "evaluación": RAG.Evaluacion,
            "concepto": RAG.Concepto,
            "persona": RAG.Persona,
            "fecha": RAG.Fecha,
            "prueba": RAG.Prueba,
            "tarea": RAG.Tarea,
            "proyecto": RAG.Proyecto,
            "examen": RAG.Examen,
        }
        return class_map.get(class_name.lower(), self._make_uri(class_name, RAG))
    
    def add_triple(self, triple: Triple) -> bool:
        """Agrega una tripleta al grafo."""
        if self.store_type == "local":
            return self._add_triple_local(triple)
        else:
            return self._add_triple_fuseki(triple)
    
    def _add_triple_local(self, triple: Triple) -> bool:
        """Agrega tripleta al grafo local RDFLib."""
        if self._graph is None:
            self.open()
        
        subj = self._make_uri(triple.subject)
        pred = self._resolve_predicate(triple.predicate)
        
        if triple.object_is_literal:
            if triple.literal_datatype:
                dt_map = {
                    "decimal": XSD.decimal,
                    "integer": XSD.integer,
                    "float": XSD.float,
                    "date": XSD.date,
                    "datetime": XSD.dateTime,
                    "boolean": XSD.boolean,
                }
                datatype = dt_map.get(triple.literal_datatype.lower(), XSD.string)
                obj = Literal(triple.object, datatype=datatype)
            else:
                obj = Literal(triple.object)
        elif pred == RDF.type:
            obj = self._resolve_class(triple.object)
        else:
            obj = self._make_uri(triple.object)
        
        self._graph.add((subj, pred, obj))
        return True
    
    def _add_triple_fuseki(self, triple: Triple) -> bool:
        """Agrega tripleta via SPARQL UPDATE a Fuseki."""
        subj = f"<{RAGENT}{triple.subject}>"
        pred = f"<{self._resolve_predicate(triple.predicate)}>"
        
        if triple.object_is_literal:
            if triple.literal_datatype:
                obj = f'"{triple.object}"^^<{XSD}{triple.literal_datatype}>'
            else:
                obj = f'"{triple.object}"'
        else:
            obj = f"<{RAGENT}{triple.object}>"
        
        update_query = f"INSERT DATA {{ {subj} {pred} {obj} . }}"
        
        resp = requests.post(
            f"{self.fuseki_url}/{self.dataset}/update",
            data={"update": update_query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=10
        )
        return resp.status_code in (200, 204)
    
    def add_triples(self, triples: List[Triple]) -> int:
        """Agrega múltiples tripletas."""
        count = 0
        for t in triples:
            if self.add_triple(t):
                count += 1
        
        if self.store_type == "local":
            self._persist_local()
        
        logger.info(f"Agregadas {count}/{len(triples)} tripletas al grafo")
        return count
    
    def query(self, sparql: str) -> List[Dict[str, Any]]:
        """Ejecuta una consulta SPARQL."""
        if self.store_type == "local":
            return self._query_local(sparql)
        else:
            return self._query_fuseki(sparql)
    
    def _query_local(self, sparql: str) -> List[Dict[str, Any]]:
        """Ejecuta SPARQL en grafo local."""
        if self._graph is None:
            self.open()
        
        results = []
        qres = self._graph.query(sparql)
        for row in qres:
            result = {}
            for var in qres.vars:
                val = getattr(row, str(var), None)
                if val is not None:
                    result[str(var)] = str(val)
            if result:
                results.append(result)
        
        return results
    
    def _query_fuseki(self, sparql: str) -> List[Dict[str, Any]]:
        """Ejecuta SPARQL contra Fuseki."""
        resp = requests.post(
            f"{self.fuseki_url}/{self.dataset}/query",
            data={"query": sparql},
            headers={"Accept": "application/sparql-results+json"},
            timeout=30
        )
        
        if resp.status_code == 200:
            data = resp.json()
            results = []
            for binding in data.get("results", {}).get("bindings", []):
                result = {}
                for var, val in binding.items():
                    result[var] = val.get("value", "")
                results.append(result)
            return results
        return []
    
    def get_chunk_relations(self, chunk_id: str) -> List[Dict[str, str]]:
        """Obtiene todas las relaciones de un chunk específico."""
        sparql = f"""
        PREFIX rag: <http://ragent.local/ontology#>
        PREFIX ragent: <http://ragent.local/data#>
        
        SELECT ?predicate ?object ?objectType WHERE {{
            ragent:{chunk_id} ?predicate ?object .
            OPTIONAL {{ ?object a ?objectType . }}
        }}
        """
        return self.query(sparql)
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, str]]:
        """Obtiene todas las entidades de un tipo específico."""
        type_uri = self._resolve_class(entity_type)
        sparql = f"""
        PREFIX rag: <http://ragent.local/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?entity ?nombre ?descripcion WHERE {{
            ?entity rdf:type <{type_uri}> .
            OPTIONAL {{ ?entity rag:nombre ?nombre . }}
            OPTIONAL {{ ?entity rag:descripcion ?descripcion . }}
        }}
        """
        return self.query(sparql)
    
    def entity_exists(self, entity_id: str) -> bool:
        """Verifica si una entidad existe en el grafo."""
        sparql = f"""
        PREFIX ragent: <http://ragent.local/data#>
        ASK {{ ragent:{entity_id} ?p ?o . }}
        """
        if self.store_type == "local" and self._graph:
            return bool(self._graph.query(sparql))
        return False
    
    def count_triples(self) -> int:
        """Retorna el número total de tripletas en el grafo."""
        if self.store_type == "local" and self._graph:
            return len(self._graph)
        
        sparql = "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o . }"
        results = self.query(sparql)
        if results:
            return int(results[0].get("count", 0))
        return 0
    
    def clear(self, confirm: bool = False):
        """Elimina todas las tripletas del grafo."""
        if not confirm:
            return
        
        if self.store_type == "local":
            self._graph = Graph()
            self._graph.bind("rag", RAG)
            self._graph.bind("ragent", RAGENT)
            self._ontology_loaded = False
            self._persist_local()
            logger.info("Grafo local limpiado")
        else:
            sparql = "DELETE WHERE { ?s ?p ?o . }"
            requests.post(
                f"{self.fuseki_url}/{self.dataset}/update",
                data={"update": sparql},
                timeout=30
            )
            logger.info("Grafo Fuseki limpiado")
    
    def __enter__(self):
        return self.open()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
