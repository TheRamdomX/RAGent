"""
sparql_queries.py - Catálogo de consultas SPARQL semánticas

Proporciona consultas pre-construidas para preguntas estructurales comunes.
Responsabilidades:
- Definir consultas SPARQL reutilizables
- Detectar intención estructural en preguntas
- Generar SPARQL dinámico basado en parámetros

NO contiene lógica LLM.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

from app.knowledge_graph.graph_store import GraphStore
from app.utils.logger import logger


# Prefijos SPARQL estándar
SPARQL_PREFIXES = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rag: <http://ragent.local/ontology#>
PREFIX ragent: <http://ragent.local/data#>
"""


@dataclass
class QueryResult:
    """Resultado de una consulta SPARQL."""
    query_type: str
    sparql: str
    results: List[Dict[str, Any]]
    formatted: str
    success: bool
    error: Optional[str] = None


class SPARQLQueryCatalog:
    """
    Catálogo de consultas SPARQL para el grafo de conocimiento.
    Proporciona métodos para consultas estructurales comunes.
    """
    
    def __init__(self, store: GraphStore = None):
        """
        Inicializa el catálogo.
        
        Args:
            store: GraphStore a usar para ejecutar consultas.
        """
        self._store = store
    
    def set_store(self, store: GraphStore):
        """Establece el GraphStore a usar."""
        self._store = store
    
    def _execute(self, sparql: str) -> List[Dict[str, Any]]:
        """Ejecuta una consulta SPARQL."""
        if not self._store:
            logger.error("No hay GraphStore configurado")
            return []
        return self._store.query(sparql)
    
    # =========================================================================
    # Consultas sobre Evaluaciones
    # =========================================================================
    
    def get_evaluaciones_de_ramo(self, ramo: str) -> QueryResult:
        """
        Obtiene todas las evaluaciones de un ramo específico.
        
        Args:
            ramo: Nombre o ID del ramo.
        
        Returns:
            QueryResult con las evaluaciones encontradas.
        """
        ramo_filter = self._build_name_filter("ramo", ramo)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?eval ?evalNombre ?tipo ?ponderacion ?fecha WHERE {{
            ?ramo rdf:type rag:Ramo .
            {ramo_filter}
            ?eval rag:esParteDe ?ramo .
            ?eval rdf:type ?evalType .
            FILTER(?evalType IN (rag:Evaluacion, rag:Prueba, rag:Tarea, rag:Proyecto, rag:Examen))
            OPTIONAL {{ ?eval rag:nombre ?evalNombre . }}
            OPTIONAL {{ ?eval rag:tipo ?tipo . }}
            OPTIONAL {{ ?eval rag:ponderacion ?ponderacion . }}
            OPTIONAL {{ 
                ?eval rag:tieneFecha ?fechaEntity .
                ?fechaEntity rag:fechaValor ?fecha .
            }}
        }}
        ORDER BY ?evalNombre
        """
        
        results = self._execute(sparql)
        formatted = self._format_evaluaciones(results)
        
        return QueryResult(
            query_type="evaluaciones_de_ramo",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    def get_como_se_evalua(self, tema: str) -> QueryResult:
        """
        Obtiene cómo se evalúa un tema/concepto específico.
        
        Args:
            tema: Nombre del tema o concepto.
        
        Returns:
            QueryResult con las evaluaciones que cubren ese tema.
        """
        tema_filter = self._build_name_filter("concepto", tema)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?eval ?evalNombre ?tipo ?ponderacion ?ramo ?ramoNombre WHERE {{
            ?concepto rdf:type rag:Concepto .
            {tema_filter}
            ?eval rag:evalua ?concepto .
            OPTIONAL {{ ?eval rag:nombre ?evalNombre . }}
            OPTIONAL {{ ?eval rag:tipo ?tipo . }}
            OPTIONAL {{ ?eval rag:ponderacion ?ponderacion . }}
            OPTIONAL {{ 
                ?eval rag:esParteDe ?ramo .
                ?ramo rag:nombre ?ramoNombre .
            }}
        }}
        """
        
        results = self._execute(sparql)
        formatted = self._format_como_evalua(results, tema)
        
        return QueryResult(
            query_type="como_se_evalua",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    def get_ponderaciones(self, ramo: str = None) -> QueryResult:
        """
        Obtiene las ponderaciones de evaluaciones.
        
        Args:
            ramo: Filtrar por ramo específico (opcional).
        """
        ramo_clause = ""
        if ramo:
            ramo_filter = self._build_name_filter("ramo", ramo)
            ramo_clause = f"""
            ?eval rag:esParteDe ?ramo .
            ?ramo rdf:type rag:Ramo .
            {ramo_filter}
            """
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT ?eval ?evalNombre ?tipo ?ponderacion ?ramoNombre WHERE {{
            ?eval rag:ponderacion ?ponderacion .
            OPTIONAL {{ ?eval rag:nombre ?evalNombre . }}
            OPTIONAL {{ ?eval rag:tipo ?tipo . }}
            {ramo_clause}
            OPTIONAL {{
                ?eval rag:esParteDe ?r .
                ?r rag:nombre ?ramoNombre .
            }}
        }}
        ORDER BY DESC(?ponderacion)
        """
        
        results = self._execute(sparql)
        formatted = self._format_ponderaciones(results)
        
        return QueryResult(
            query_type="ponderaciones",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    # =========================================================================
    # Consultas sobre Conceptos
    # =========================================================================
    
    def get_conceptos_de_documento(self, documento: str) -> QueryResult:
        """
        Obtiene todos los conceptos mencionados en un documento.
        
        Args:
            documento: Nombre del documento/archivo.
        """
        doc_filter = self._build_name_filter("doc", documento, use_fuente=True)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?concepto ?nombre ?descripcion WHERE {{
            ?doc rdf:type rag:Documento .
            {doc_filter}
            ?chunk rag:perteneceA ?doc .
            ?chunk rag:menciona ?concepto .
            ?concepto rdf:type rag:Concepto .
            OPTIONAL {{ ?concepto rag:nombre ?nombre . }}
            OPTIONAL {{ ?concepto rag:descripcion ?descripcion . }}
        }}
        ORDER BY ?nombre
        """
        
        results = self._execute(sparql)
        formatted = self._format_conceptos(results, documento)
        
        return QueryResult(
            query_type="conceptos_de_documento",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    def get_prerequisitos(self, concepto: str) -> QueryResult:
        """
        Obtiene los prerequisitos de un concepto.
        
        Args:
            concepto: Nombre del concepto.
        """
        concepto_filter = self._build_name_filter("c", concepto)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?prereq ?prereqNombre ?prereqDesc WHERE {{
            ?c rdf:type rag:Concepto .
            {concepto_filter}
            ?c rag:requiere ?prereq .
            OPTIONAL {{ ?prereq rag:nombre ?prereqNombre . }}
            OPTIONAL {{ ?prereq rag:descripcion ?prereqDesc . }}
        }}
        """
        
        results = self._execute(sparql)
        formatted = self._format_prerequisitos(results, concepto)
        
        return QueryResult(
            query_type="prerequisitos",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    def get_conceptos_relacionados(self, concepto: str) -> QueryResult:
        """
        Obtiene conceptos relacionados con otro concepto.
        """
        concepto_filter = self._build_name_filter("c", concepto)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?relacionado ?relNombre ?relacion WHERE {{
            ?c rdf:type rag:Concepto .
            {concepto_filter}
            {{
                ?c rag:relacionadoCon ?relacionado .
                BIND("relacionado" AS ?relacion)
            }} UNION {{
                ?relacionado rag:relacionadoCon ?c .
                BIND("relacionado" AS ?relacion)
            }} UNION {{
                ?c rag:requiere ?relacionado .
                BIND("requiere" AS ?relacion)
            }} UNION {{
                ?relacionado rag:requiere ?c .
                BIND("requerido_por" AS ?relacion)
            }}
            ?relacionado rdf:type rag:Concepto .
            OPTIONAL {{ ?relacionado rag:nombre ?relNombre . }}
        }}
        """
        
        results = self._execute(sparql)
        formatted = self._format_relacionados(results, concepto)
        
        return QueryResult(
            query_type="conceptos_relacionados",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    # =========================================================================
    # Consultas sobre Chunks
    # =========================================================================
    
    def get_chunks_que_mencionan(self, entidad: str) -> QueryResult:
        """
        Obtiene los chunks que mencionan una entidad específica.
        
        Args:
            entidad: Nombre de la entidad (concepto, evaluación, etc.)
        """
        entidad_filter = self._build_name_filter("entidad", entidad)
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?chunk ?doc ?fuente ?pagina WHERE {{
            {entidad_filter}
            ?chunk rag:menciona ?entidad .
            ?chunk rdf:type rag:Chunk .
            OPTIONAL {{ 
                ?chunk rag:perteneceA ?doc .
                ?doc rag:fuente ?fuente .
            }}
            OPTIONAL {{ ?chunk rag:pagina ?pagina . }}
        }}
        ORDER BY ?fuente ?pagina
        """
        
        results = self._execute(sparql)
        formatted = self._format_chunks(results, entidad)
        
        return QueryResult(
            query_type="chunks_que_mencionan",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    # =========================================================================
    # Consultas generales
    # =========================================================================
    
    def get_resumen_grafo(self) -> QueryResult:
        """Obtiene un resumen del contenido del grafo."""
        sparql = f"""{SPARQL_PREFIXES}
        SELECT ?tipo (COUNT(?entidad) AS ?cantidad) WHERE {{
            ?entidad rdf:type ?tipo .
            FILTER(?tipo IN (
                rag:Chunk, rag:Documento, rag:Ramo, 
                rag:Evaluacion, rag:Prueba, rag:Tarea, rag:Proyecto, rag:Examen,
                rag:Concepto, rag:Persona, rag:Fecha
            ))
        }}
        GROUP BY ?tipo
        ORDER BY DESC(?cantidad)
        """
        
        results = self._execute(sparql)
        formatted = self._format_resumen(results)
        
        return QueryResult(
            query_type="resumen_grafo",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=True
        )
    
    def search_entities(self, query: str, entity_type: str = None) -> QueryResult:
        """
        Búsqueda general de entidades por nombre.
        
        Args:
            query: Término de búsqueda.
            entity_type: Tipo de entidad a buscar (opcional).
        """
        type_filter = ""
        if entity_type:
            type_uri = f"rag:{entity_type}"
            type_filter = f"?entidad rdf:type {type_uri} ."
        
        # Escapar caracteres especiales en la búsqueda
        safe_query = query.replace("\\", "\\\\").replace('"', '\\"')
        
        sparql = f"""{SPARQL_PREFIXES}
        SELECT DISTINCT ?entidad ?nombre ?tipo ?descripcion WHERE {{
            ?entidad rag:nombre ?nombre .
            {type_filter}
            OPTIONAL {{ ?entidad rdf:type ?tipo . }}
            OPTIONAL {{ ?entidad rag:descripcion ?descripcion . }}
            FILTER(CONTAINS(LCASE(?nombre), LCASE("{safe_query}")))
        }}
        LIMIT 20
        """
        
        results = self._execute(sparql)
        formatted = self._format_search(results, query)
        
        return QueryResult(
            query_type="search_entities",
            sparql=sparql,
            results=results,
            formatted=formatted,
            success=len(results) > 0
        )
    
    # =========================================================================
    # Detección de intención estructural
    # =========================================================================
    
    def detect_structural_intent(self, question: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Detecta si una pregunta tiene intención estructural.
        
        Args:
            question: Pregunta del usuario.
        
        Returns:
            Tupla (tipo_consulta, parámetros) o None si no es estructural.
        """
        q = question.lower().strip()
        
        # Patrones para evaluaciones
        eval_patterns = [
            (r"(?:cómo|como)\s+(?:se\s+)?(?:evalúa|evalua|evaluará)\s+(.+)", "como_se_evalua"),
            (r"(?:qué|que|cuáles|cuales)\s+(?:son\s+las\s+)?evaluaciones?\s+(?:de|del|tiene)\s+(.+)", "evaluaciones_de_ramo"),
            (r"(?:cuánto|cuanto)\s+(?:vale|pondera|pesa)\s+(.+)", "ponderaciones"),
            (r"ponderaci[óo]n(?:es)?\s+(?:de|del)?\s*(.+)?", "ponderaciones"),
        ]
        
        # Patrones para conceptos
        concept_patterns = [
            (r"(?:qué|que)\s+(?:conceptos?|temas?)\s+(?:se\s+)?(?:mencionan?|tratan?|cubren?)\s+(?:en\s+)?(.+)", "conceptos_de_documento"),
            (r"(?:qué|que)\s+(?:necesito|debo)\s+saber\s+(?:antes\s+de|para)\s+(.+)", "prerequisitos"),
            (r"prerequisitos?\s+(?:de|para)\s+(.+)", "prerequisitos"),
            (r"(?:qué|que)\s+(?:está|esta)\s+relacionado\s+con\s+(.+)", "conceptos_relacionados"),
        ]
        
        # Patrones para chunks/fuentes
        source_patterns = [
            (r"(?:dónde|donde)\s+(?:se\s+)?(?:menciona|habla\s+de|explica)\s+(.+)", "chunks_que_mencionan"),
            (r"(?:en\s+qué|en\s+que)\s+(?:parte|sección|documento)\s+(?:se\s+)?(?:menciona|explica)\s+(.+)", "chunks_que_mencionan"),
        ]
        
        # Probar patrones de evaluaciones
        for pattern, query_type in eval_patterns:
            match = re.search(pattern, q)
            if match:
                param = match.group(1).strip() if match.lastindex else ""
                if query_type == "ponderaciones" and not param:
                    return (query_type, {})
                return (query_type, {"tema": param, "ramo": param})
        
        # Probar patrones de conceptos
        for pattern, query_type in concept_patterns:
            match = re.search(pattern, q)
            if match:
                param = match.group(1).strip()
                return (query_type, {"documento": param, "concepto": param})
        
        # Probar patrones de fuentes
        for pattern, query_type in source_patterns:
            match = re.search(pattern, q)
            if match:
                param = match.group(1).strip()
                return (query_type, {"entidad": param})
        
        return None
    
    def execute_structural_query(self, question: str) -> Optional[QueryResult]:
        """
        Ejecuta una consulta estructural basada en la pregunta.
        
        Args:
            question: Pregunta del usuario.
        
        Returns:
            QueryResult si se detectó intención estructural, None si no.
        """
        intent = self.detect_structural_intent(question)
        if not intent:
            return None
        
        query_type, params = intent
        logger.info(f"Intención estructural detectada: {query_type} con params {params}")
        
        # Mapear a métodos
        if query_type == "evaluaciones_de_ramo":
            return self.get_evaluaciones_de_ramo(params.get("ramo", ""))
        elif query_type == "como_se_evalua":
            return self.get_como_se_evalua(params.get("tema", ""))
        elif query_type == "ponderaciones":
            return self.get_ponderaciones(params.get("ramo"))
        elif query_type == "conceptos_de_documento":
            return self.get_conceptos_de_documento(params.get("documento", ""))
        elif query_type == "prerequisitos":
            return self.get_prerequisitos(params.get("concepto", ""))
        elif query_type == "conceptos_relacionados":
            return self.get_conceptos_relacionados(params.get("concepto", ""))
        elif query_type == "chunks_que_mencionan":
            return self.get_chunks_que_mencionan(params.get("entidad", ""))
        
        return None
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _build_name_filter(self, var: str, value: str, use_fuente: bool = False) -> str:
        """Construye un filtro SPARQL por nombre."""
        safe_value = value.replace("\\", "\\\\").replace('"', '\\"').lower()
        
        if use_fuente:
            return f"""
            {{
                ?{var} rag:fuente ?fuente .
                FILTER(CONTAINS(LCASE(?fuente), "{safe_value}"))
            }} UNION {{
                ?{var} rag:nombre ?nombre .
                FILTER(CONTAINS(LCASE(?nombre), "{safe_value}"))
            }}
            """
        
        return f"""
        {{
            ?{var} rag:nombre ?nombre .
            FILTER(CONTAINS(LCASE(?nombre), "{safe_value}"))
        }} UNION {{
            FILTER(CONTAINS(LCASE(STR(?{var})), "{safe_value}"))
        }}
        """
    
    def _format_evaluaciones(self, results: List[Dict]) -> str:
        """Formatea resultados de evaluaciones."""
        if not results:
            return "No se encontraron evaluaciones."
        
        lines = ["**Evaluaciones encontradas:**\n"]
        for r in results:
            nombre = r.get("evalNombre", r.get("eval", "Sin nombre"))
            tipo = r.get("tipo", "")
            pond = r.get("ponderacion", "")
            fecha = r.get("fecha", "")
            
            line = f"- {nombre}"
            if tipo:
                line += f" ({tipo})"
            if pond:
                line += f" - Ponderación: {float(pond)*100:.0f}%"
            if fecha:
                line += f" - Fecha: {fecha}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_como_evalua(self, results: List[Dict], tema: str) -> str:
        """Formatea resultados de cómo se evalúa."""
        if not results:
            return f"No se encontró información sobre cómo se evalúa '{tema}'."
        
        lines = [f"**Evaluaciones que cubren '{tema}':**\n"]
        for r in results:
            nombre = r.get("evalNombre", "Sin nombre")
            tipo = r.get("tipo", "")
            pond = r.get("ponderacion", "")
            ramo = r.get("ramoNombre", "")
            
            line = f"- {nombre}"
            if tipo:
                line += f" ({tipo})"
            if pond:
                line += f" - {float(pond)*100:.0f}%"
            if ramo:
                line += f" en {ramo}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_ponderaciones(self, results: List[Dict]) -> str:
        """Formatea resultados de ponderaciones."""
        if not results:
            return "No se encontraron ponderaciones."
        
        lines = ["**Ponderaciones:**\n"]
        total = 0.0
        for r in results:
            nombre = r.get("evalNombre", r.get("eval", "Sin nombre"))
            tipo = r.get("tipo", "")
            pond = float(r.get("ponderacion", 0))
            ramo = r.get("ramoNombre", "")
            
            total += pond
            line = f"- {nombre}: {pond*100:.0f}%"
            if tipo:
                line += f" ({tipo})"
            if ramo:
                line += f" - {ramo}"
            lines.append(line)
        
        if len(results) > 1:
            lines.append(f"\n**Total: {total*100:.0f}%**")
        
        return "\n".join(lines)
    
    def _format_conceptos(self, results: List[Dict], documento: str) -> str:
        """Formatea lista de conceptos."""
        if not results:
            return f"No se encontraron conceptos en '{documento}'."
        
        lines = [f"**Conceptos en '{documento}':**\n"]
        for r in results:
            nombre = r.get("nombre", r.get("concepto", "Sin nombre"))
            desc = r.get("descripcion", "")
            
            line = f"- {nombre}"
            if desc:
                line += f": {desc[:100]}..."
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_prerequisitos(self, results: List[Dict], concepto: str) -> str:
        """Formatea prerequisitos."""
        if not results:
            return f"No se encontraron prerequisitos para '{concepto}'."
        
        lines = [f"**Prerequisitos para '{concepto}':**\n"]
        for r in results:
            nombre = r.get("prereqNombre", r.get("prereq", "Sin nombre"))
            desc = r.get("prereqDesc", "")
            
            line = f"- {nombre}"
            if desc:
                line += f": {desc[:80]}..."
            lines.append(line)
        
        return "\n".join(lines)
    
    def _format_relacionados(self, results: List[Dict], concepto: str) -> str:
        """Formatea conceptos relacionados."""
        if not results:
            return f"No se encontraron conceptos relacionados con '{concepto}'."
        
        lines = [f"**Conceptos relacionados con '{concepto}':**\n"]
        for r in results:
            nombre = r.get("relNombre", r.get("relacionado", "Sin nombre"))
            relacion = r.get("relacion", "relacionado")
            
            lines.append(f"- {nombre} ({relacion})")
        
        return "\n".join(lines)
    
    def _format_chunks(self, results: List[Dict], entidad: str) -> str:
        """Formatea chunks que mencionan una entidad."""
        if not results:
            return f"No se encontraron referencias a '{entidad}'."
        
        lines = [f"**Referencias a '{entidad}':**\n"]
        for r in results:
            fuente = r.get("fuente", "Desconocido")
            pagina = r.get("pagina", "?")
            chunk = r.get("chunk", "")
            
            lines.append(f"- {fuente}, página {pagina}")
        
        return "\n".join(lines)
    
    def _format_resumen(self, results: List[Dict]) -> str:
        """Formatea resumen del grafo."""
        if not results:
            return "El grafo está vacío."
        
        lines = ["**Resumen del grafo de conocimiento:**\n"]
        for r in results:
            tipo = r.get("tipo", "").split("#")[-1]
            cantidad = r.get("cantidad", 0)
            lines.append(f"- {tipo}: {cantidad}")
        
        return "\n".join(lines)
    
    def _format_search(self, results: List[Dict], query: str) -> str:
        """Formatea resultados de búsqueda."""
        if not results:
            return f"No se encontraron resultados para '{query}'."
        
        lines = [f"**Resultados para '{query}':**\n"]
        for r in results:
            nombre = r.get("nombre", "Sin nombre")
            tipo = r.get("tipo", "").split("#")[-1]
            desc = r.get("descripcion", "")
            
            line = f"- {nombre}"
            if tipo:
                line += f" [{tipo}]"
            if desc:
                line += f": {desc[:60]}..."
            lines.append(line)
        
        return "\n".join(lines)
