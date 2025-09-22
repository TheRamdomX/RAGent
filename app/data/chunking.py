from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.utils import config
from typing import List, Union, Dict
import unicodedata
import re

def normalize_text(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = s.replace('\t', ' ')
    s = '\n'.join(re.sub(r' {2,}', ' ', line).strip() for line in s.split('\n'))
    return s.strip()


def chunk_text(text: Union[str, List[str]], chunk_size_chars: int = None, chunk_overlap: int = None) -> List[Dict]:

    chunk_size_chars = chunk_size_chars or config.CHUNK_DEFAULT_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_DEFAULT_OVERLAP
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap)

    chunks_out: List[Dict] = []

    MIN_CHARS = config.MIN_CHUNK_CHARS

    if isinstance(text, list):
        sep = "\n"
        normalized_pages = [normalize_text(p or "") for p in text]
        page_starts = []
        pos = 0
        for i, p in enumerate(normalized_pages):
            page_starts.append(pos)
            pos += len(p)
            if i != len(normalized_pages) - 1:
                pos += len(sep)

        full = sep.join(normalized_pages)

        for page_idx, page_text in enumerate(normalized_pages):
            if not page_text:
                continue
            raw_chunks = splitter.split_text(page_text)
            p_start_abs = page_starts[page_idx]
            cur = 0
            for ch in raw_chunks:
                start_in_page = page_text.find(ch, cur)
                if start_in_page == -1:
                    start_in_page = cur
                end_in_page = start_in_page + len(ch)

                start_abs = p_start_abs + start_in_page
                end_abs = p_start_abs + end_in_page

                chunk_txt = full[start_abs:end_abs]

                if not chunk_txt or len(chunk_txt.strip()) < MIN_CHARS:
                    cur = end_in_page
                    continue

                chunks_out.append({
                    "page_start": page_idx,
                    "page_end": page_idx,
                    "char_start": start_abs,
                    "char_end": end_abs,
                    "text": chunk_txt,
                })
                cur = end_in_page
    else:
        normalized = normalize_text(text or "")
        raw_chunks = splitter.split_text(normalized)
        cur = 0
        for ch in raw_chunks:
            start = normalized.find(ch, cur)
            if start == -1:
                start = cur
            end = start + len(ch)
            chunk_text = normalized[start:end]
            if not chunk_text or len(chunk_text.strip()) < MIN_CHARS:
                cur = end
                continue
            chunks_out.append({"page_start": -1, "page_end": -1, "char_start": start, "char_end": end, "text": chunk_text})
            cur = end

    return chunks_out
