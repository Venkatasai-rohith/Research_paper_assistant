import os
from core.config import HYBRID_CANDIDATE_K
from core.config import VECTORSTORE_PATH
from exceptions import VectorStoreCreationError
from exceptions import VectorStoreLoadError
from exceptions import VectorStoreSaveError
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever


def create_vectorstore(documents: list[Document], embedding_model: Embeddings) -> VectorStore:

    try:

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model)
        
        return vectorstore

    except Exception as e:

        raise VectorStoreCreationError(
            "Unable to create the vector store from the provided documents"
        ) from e

def save_vectorstore(vectorstore: FAISS) -> None:
    try:
        vectorstore.save_local(VECTORSTORE_PATH)

    except Exception as e:
       raise VectorStoreSaveError(
            "Unable to save vector store"
        ) from e

def load_vectorstore(embedding_model: Embeddings) -> FAISS:
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    
    try:
        vectorstore = FAISS.load_local(
            folder_path=VECTORSTORE_PATH,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True)
        return vectorstore

    except Exception as e:
        raise VectorStoreLoadError(
            "Unable to load vector store"
        ) from e

def get_retriever(vectorstore: VectorStore) -> BaseRetriever:

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": HYBRID_CANDIDATE_K,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )