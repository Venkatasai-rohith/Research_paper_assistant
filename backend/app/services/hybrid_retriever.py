from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from core.config import RRF_CONSTANT, HYBRID_FINAL_K

def get_document_id(doc: Document) -> str:

    source = doc.metadata.get("source", "")
    chunk_index = doc.metadata.get("chunk_index", "")

    return f"{source}:{chunk_index}"

def hybrid_retriever(question: str, dense_retriever: BaseRetriever, bm25_retriever: BaseRetriever) -> list[Document]:

    dense_documents = dense_retriever.invoke(question)

    bm25_documents = bm25_retriever.invoke(question)

    scores = {}
    documents = {}

    for rank, doc in enumerate(dense_documents, start=1):
        document_id = get_document_id(doc)
        scores[document_id] = scores.get(document_id, 0) + (1 / (RRF_CONSTANT + rank))
        documents[document_id] = doc

    for rank, doc in enumerate(bm25_documents, start=1):
        document_id = get_document_id(doc)
        scores[document_id] = scores.get(document_id, 0) + (1 / (RRF_CONSTANT + rank))
        documents[document_id] = doc

    ranked_document_ids = sorted(scores, key=scores.get, reverse=True)

    return [
        documents[document_id]
        for document_id in ranked_document_ids[:HYBRID_FINAL_K]
            ]

def multi_paper_hybrid_retriever(
    question: str,
    retriever_pairs: list[tuple[BaseRetriever, BaseRetriever]]
) -> list[Document]:

    scores = {}
    documents = {}

    for dense_retriever, bm25_retriever in retriever_pairs:

        dense_documents = dense_retriever.invoke(question)
        bm25_documents = bm25_retriever.invoke(question)

        # Dense ranking
        for rank, doc in enumerate(dense_documents, start=1):

            document_id = get_document_id(doc)

            scores[document_id] = scores.get(document_id, 0) + (
                1 / (RRF_CONSTANT + rank)
            )

            documents[document_id] = doc

        # BM25 ranking
        for rank, doc in enumerate(bm25_documents, start=1):

            document_id = get_document_id(doc)

            scores[document_id] = scores.get(document_id, 0) + (
                1 / (RRF_CONSTANT + rank)
            )

            documents[document_id] = doc

    ranked_document_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[document_id]
        for document_id in ranked_document_ids[:HYBRID_FINAL_K]
    ]