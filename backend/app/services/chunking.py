from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import(CHUNK_SIZE, CHUNK_OVERLAP)

def chunk_text(page_texts: list[str], source_name: str, paper_title: str):

    documents = []
    for page_number, page_text in enumerate(page_texts, start=1):
        documents.append(
            Document(page_content=page_text, 
                metadata={
                    "source": source_name,
                    "title": paper_title,
                    "page" : page_number
                }
            )
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = index

    return chunks