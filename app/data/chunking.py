from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def chunk_text(text: str, chunk_size_chars: int = 1000, chunk_overlap: int = 200) -> List[str]:

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
