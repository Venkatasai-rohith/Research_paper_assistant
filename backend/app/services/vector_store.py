from core.config import HYBRID_CANDIDATE_K
from exceptions import VectorStoreCreationError
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

def get_retriever(vectorstore: VectorStore) -> BaseRetriever:

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": HYBRID_CANDIDATE_K,
            "fetch_k": 20,
            "lambda_mult": 0.5
        }
    )
