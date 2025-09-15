import subprocess
import tempfile
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from app.utils.logger import logger

def extract_text_with_marker(pdf_path: str, force_ocr: bool = False, langs: str = "en,es") -> str:
    """
    Usa Marker para extraer texto del PDF.
    Si force_ocr=True, fuerza OCR en todas las páginas.
    langs: idiomas para OCR (Marker lo permite).
    Retorna texto plano obtenido de Marker (Markdown/JSON → extraído).
    """
    logger.info(f"Ejecutando Marker sobre {pdf_path}, force_ocr={force_ocr}, langs={langs}")
    # Crear modelo + renderer
    artifact_dict = create_model_dict()

    converter = PdfConverter(
        artifact_dict=artifact_dict,
        languages=langs.split(","),
        force_ocr=force_ocr
    )

    rendered = converter(pdf_path)  # esto ejecuta Marker
    # rendered tiene distintos atributos, depende si output default es markdown/json
    # Usamos text_from_rendered para extraer texto limpio
    text, _, images = text_from_rendered(rendered)
    # text es el contenido en formato markdown. Podemos convertir markdown → texto plano si queremos
    # Aquí simplificamos: devolvemos markdown o limpio
    return text
