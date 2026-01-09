"""
extract_relations.py - LLM Estructurador

Convierte texto de chunks en tripletas RDF usando un LLM.
Responsabilidades:
- Recibir un chunk de texto
- Llamar al LLM con prompt estructurado + ontología
- Validar y parsear la salida
- Devolver tripletas limpias

IMPORTANTE: Este LLM CREA hechos, NO responde preguntas.
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.models.llm import Agent
from app.utils.logger import logger
from app.knowledge_graph.graph_store import Triple, GraphStore


# Ontología resumida para el prompt del LLM
ONTOLOGY_CONTEXT = """
## Ontología RAGent (clases y relaciones válidas)

### Clases de entidades:
- Chunk: Fragmento de texto de un documento
- Documento: Archivo PDF fuente
- Ramo: Asignatura o curso académico
- Evaluacion: Instancia de evaluación (Prueba, Tarea, Proyecto, Examen)
- Concepto: Tema, idea o conocimiento específico
- Persona: Autor, profesor o persona mencionada
- Fecha: Fecha relevante

### Relaciones válidas (predicados):
- menciona: Chunk → Concepto/Evaluacion/Ramo/Persona
- perteneceA: Chunk → Documento
- evalua: Evaluacion → Concepto
- esParteDe: Evaluacion → Ramo
- requiere: Concepto → Concepto (prerequisito)
- relacionadoCon: Entidad → Entidad (relación genérica)
- autor: Documento → Persona
- profesor: Ramo → Persona
- tieneFecha: Evaluacion → Fecha
- tipo: Entidad → string (tipo específico, ej: "Prueba escrita")
- nombre: Entidad → string
- ponderacion: Evaluacion → decimal (0.0-1.0)
- codigo: Ramo → string
- descripcion: Entidad → string
"""

EXTRACTION_PROMPT = """Eres un extractor de conocimiento estructurado. Tu tarea es analizar el texto y extraer tripletas RDF (sujeto, predicado, objeto).

{ontology}

## Reglas de extracción:
1. Identifica entidades nombradas (conceptos, evaluaciones, ramos, personas, fechas)
2. Genera un ID único para cada entidad (snake_case, sin espacios ni caracteres especiales)
3. Establece relaciones usando SOLO los predicados de la ontología
4. Para valores literales (texto, números, fechas), indica el tipo de dato
5. Sé conservador: solo extrae información explícita en el texto
6. El chunk actual tiene ID: {chunk_id}

## Formato de salida (JSON):
{{
  "entities": [
    {{"id": "entity_id", "type": "Clase", "name": "Nombre legible"}},
    ...
  ],
  "triples": [
    {{"subject": "id_sujeto", "predicate": "predicado", "object": "id_objeto", "is_literal": false}},
    {{"subject": "id_sujeto", "predicate": "predicado", "object": "valor", "is_literal": true, "datatype": "string|decimal|integer|date"}},
    ...
  ]
}}

## Texto a analizar:
---
{text}
---

## Metadatos del chunk:
- ID: {chunk_id}
- Fuente: {source}
- Página: {page}

Extrae las tripletas en formato JSON:"""


@dataclass
class ExtractionResult:
    """Resultado de la extracción de relaciones."""
    entities: List[Dict[str, str]]
    triples: List[Triple]
    raw_response: str
    chunk_id: str
    success: bool
    error: Optional[str] = None


class RelationExtractor:
    """
    Extractor de relaciones usando LLM como estructurador.
    Convierte texto → tripletas RDF.
    """
    
    def __init__(self, llm_agent: Agent = None):
        """
        Inicializa el extractor.
        
        Args:
            llm_agent: Agente LLM a usar. Si no se provee, crea uno nuevo.
        """
        self._llm = llm_agent or Agent()
        self._extraction_cache: Dict[str, ExtractionResult] = {}
    
    def extract_from_chunk(
        self,
        text: str,
        chunk_id: str,
        source: str = "unknown",
        page: int = None,
        use_cache: bool = True
    ) -> ExtractionResult:
        """
        Extrae entidades y relaciones de un chunk de texto.
        
        Args:
            text: Contenido del chunk.
            chunk_id: Identificador único del chunk.
            source: Nombre del archivo fuente.
            page: Número de página (opcional).
            use_cache: Si usar caché de extracciones previas.
        
        Returns:
            ExtractionResult con entidades y tripletas.
        """
        # Verificar caché
        cache_key = f"{chunk_id}:{hash(text)}"
        if use_cache and cache_key in self._extraction_cache:
            logger.debug(f"Usando extracción cacheada para chunk {chunk_id}")
            return self._extraction_cache[cache_key]
        
        # Construir prompt
        prompt = EXTRACTION_PROMPT.format(
            ontology=ONTOLOGY_CONTEXT,
            text=text[:3000],  # Limitar texto para no exceder contexto
            chunk_id=chunk_id,
            source=source,
            page=page if page is not None else "N/A"
        )
        
        # Llamar al LLM
        response = self._llm.generate(prompt)
        
        # Parsear respuesta
        result = self._parse_response(response, chunk_id, source, page)
        
        # Cachear resultado
        if use_cache:
            self._extraction_cache[cache_key] = result
        
        return result
    
    def _parse_response(
        self,
        response: str,
        chunk_id: str,
        source: str,
        page: int
    ) -> ExtractionResult:
        """Parsea la respuesta del LLM a entidades y tripletas."""
        
        # Intentar extraer JSON de la respuesta
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning(f"No se encontró JSON en respuesta para chunk {chunk_id}")
            return ExtractionResult(
                entities=[],
                triples=[],
                raw_response=response,
                chunk_id=chunk_id,
                success=False,
                error="No JSON found in response"
            )
        
        data = json.loads(json_match.group())
        
        # Extraer entidades
        entities = data.get("entities", [])
        
        # Convertir tripletas al formato interno
        triples = []
        
        # Agregar tripleta base: chunk pertenece a documento
        triples.append(Triple(
            subject=chunk_id,
            predicate="type",
            object="Chunk",
            object_is_literal=False
        ))
        
        triples.append(Triple(
            subject=chunk_id,
            predicate="perteneceA",
            object=self._sanitize_id(source),
            object_is_literal=False
        ))
        
        if page is not None:
            triples.append(Triple(
                subject=chunk_id,
                predicate="pagina",
                object=str(page),
                object_is_literal=True,
                literal_datatype="integer"
            ))
        
        # Agregar documento si no existe
        doc_id = self._sanitize_id(source)
        triples.append(Triple(
            subject=doc_id,
            predicate="type",
            object="Documento",
            object_is_literal=False
        ))
        triples.append(Triple(
            subject=doc_id,
            predicate="fuente",
            object=source,
            object_is_literal=True
        ))
        
        # Procesar entidades extraídas
        for entity in entities:
            entity_id = self._sanitize_id(entity.get("id", ""))
            entity_type = entity.get("type", "Concepto")
            entity_name = entity.get("name", entity_id)
            
            if not entity_id:
                continue
            
            # Tipo de entidad
            triples.append(Triple(
                subject=entity_id,
                predicate="type",
                object=entity_type,
                object_is_literal=False
            ))
            
            # Nombre de entidad
            triples.append(Triple(
                subject=entity_id,
                predicate="nombre",
                object=entity_name,
                object_is_literal=True
            ))
            
            # El chunk menciona esta entidad
            triples.append(Triple(
                subject=chunk_id,
                predicate="menciona",
                object=entity_id,
                object_is_literal=False
            ))
        
        # Procesar tripletas adicionales del LLM
        for triple_data in data.get("triples", []):
            subj = self._sanitize_id(triple_data.get("subject", ""))
            pred = triple_data.get("predicate", "")
            obj = triple_data.get("object", "")
            is_literal = triple_data.get("is_literal", False)
            datatype = triple_data.get("datatype", None)
            
            if not all([subj, pred, obj]):
                continue
            
            # Validar predicado
            if not self._is_valid_predicate(pred):
                logger.debug(f"Predicado inválido ignorado: {pred}")
                continue
            
            triples.append(Triple(
                subject=subj,
                predicate=pred,
                object=obj if is_literal else self._sanitize_id(obj),
                object_is_literal=is_literal,
                literal_datatype=datatype
            ))
        
        logger.info(f"Extraídas {len(entities)} entidades y {len(triples)} tripletas de chunk {chunk_id}")
        
        return ExtractionResult(
            entities=entities,
            triples=triples,
            raw_response=response,
            chunk_id=chunk_id,
            success=True
        )
    
    def _sanitize_id(self, value: str) -> str:
        """Sanitiza un valor para usarlo como ID."""
        if not value:
            return ""
        # Reemplazar caracteres no válidos
        safe = value.replace(" ", "_").replace("/", "_").replace("\\", "_")
        safe = re.sub(r'[^a-zA-Z0-9_\-.]', '', safe)
        # Asegurar que no empiece con número
        if safe and safe[0].isdigit():
            safe = "e_" + safe
        return safe[:100]  # Limitar longitud
    
    def _is_valid_predicate(self, predicate: str) -> bool:
        """Verifica si un predicado es válido según la ontología."""
        valid_predicates = {
            "menciona", "pertenecea", "pertenece_a",
            "evalua", "evalúa", "esparte_de", "espartede",
            "requiere", "relacionadocon", "relacionado_con",
            "autor", "profesor", "tienefecha", "tiene_fecha",
            "tipo", "nombre", "ponderacion", "ponderación",
            "codigo", "código", "descripcion", "descripción",
            "fuente", "pagina", "página", "texto", "chunkid", "chunk_id",
            "type", "a", "rdf:type"
        }
        return predicate.lower().replace("-", "_") in valid_predicates
    
    def extract_batch(
        self,
        chunks: List[Dict[str, Any]],
        store: GraphStore = None
    ) -> Tuple[int, int]:
        """
        Extrae relaciones de múltiples chunks y opcionalmente las guarda.
        
        Args:
            chunks: Lista de chunks con 'text', 'chunk_id', 'source', 'page'.
            store: GraphStore donde guardar las tripletas (opcional).
        
        Returns:
            Tupla (chunks_procesados, tripletas_totales).
        """
        total_triples = 0
        processed = 0
        
        for chunk in chunks:
            text = chunk.get("text") or chunk.get("page_content", "")
            chunk_id = chunk.get("chunk_id") or chunk.get("id") or f"chunk_{processed}"
            source = chunk.get("source", "unknown")
            page = chunk.get("page") or chunk.get("page_start")
            
            if not text or len(text.strip()) < 50:
                continue
            
            result = self.extract_from_chunk(
                text=text,
                chunk_id=chunk_id,
                source=source,
                page=page
            )
            
            if result.success and result.triples:
                total_triples += len(result.triples)
                
                if store:
                    store.add_triples(result.triples)
            
            processed += 1
        
        logger.info(f"Batch completado: {processed} chunks, {total_triples} tripletas")
        return processed, total_triples
    
    def clear_cache(self):
        """Limpia la caché de extracciones."""
        self._extraction_cache.clear()
