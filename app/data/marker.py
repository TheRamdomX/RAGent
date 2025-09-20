import subprocess
import tempfile
import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from app.utils.logger import logger

def extract_text_with_marker(pdf_path: str, force_ocr: bool = False, langs: str = "en,es") -> str:

    artifact_dict = create_model_dict()

    converter = PdfConverter(
        artifact_dict=artifact_dict,
        languages=langs.split(","),
        force_ocr=force_ocr
    )

    rendered = converter(pdf_path)  
    text, _, images = text_from_rendered(rendered)

    return text
