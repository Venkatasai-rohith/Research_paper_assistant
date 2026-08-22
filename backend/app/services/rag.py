from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from services.citation_builder import build_citations
from services.hybrid_retriever import multi_paper_hybrid_retriever
from core.streaming import sse_event, StreamEvent
from prompt import QA_PROMPT
from prompt import COMPARISON_PROMPT

def _prepare_rag(question: str, retriever_pairs):

    documents = multi_paper_hybrid_retriever(question, retriever_pairs)

    if not documents:
        return None, None, None

    print("\n========== MULTI-PAPER HYBRID RESULTS ==========")

    for i, doc in enumerate(documents, start=1):

        print(f"\nChunk {i}")
        print(f"Source : {doc.metadata.get('source')}")
        print(f"Page   : {doc.metadata.get('page')}")
        print(f"Index  : {doc.metadata.get('chunk_index')}")
        print("-" * 60)
        print(doc.page_content[:1000])

        
    context = "\n\n".join(
        doc.page_content for doc in documents
    )

    citations = build_citations(documents)

    return context, citations, documents


def generate_answer(
    question: str,
    retriever_pairs: list[tuple[BaseRetriever, BaseRetriever]] | list[list[BaseRetriever]],
    llm: BaseChatModel
) -> dict:

    context, citations, documents = _prepare_rag(question, retriever_pairs)

    if context is None:
        return {
            "answer": "I couldn't find relevant information in the uploaded document.",
            "citations": [],
        }
    
    chain = QA_PROMPT | llm

    response = chain.invoke(
        {
        "context": context,
        "question": question
        }
    )

    return {
        "answer": response.content,
        "citations" : citations,
    }

def stream_generate_answer(
    question: str,
    retriever_pairs,
    llm: BaseChatModel,
):

    context, citations, documents = _prepare_rag(question, retriever_pairs)

    if context is None:
        yield sse_event(
            StreamEvent.ERROR,
            {
                "message": "I couldn't find relevant information in the uploaded document."
            }
        )
        return

    chain = QA_PROMPT | llm

    yield sse_event(
        StreamEvent.START,
        {
            "intent": "qa"
        }
    )

    for chunk in chain.stream(
        {
            "context": context,
            "question": question
        }
    ):

        if chunk.content:

            yield sse_event(
                StreamEvent.TOKEN,
                {
                    "text": chunk.content
                }
            )

    yield sse_event(
        StreamEvent.CITATIONS,
        citations
    )

    yield sse_event(
        StreamEvent.END,
        {}
    )

def stream_generate_comparison(
    question: str,
    context: str,
    llm: BaseChatModel,
    citations: list
):
    prompt = COMPARISON_PROMPT.format(question=question, context=context)

    yield sse_event(
        StreamEvent.START,
        {
            "intent": "comparison"
        }
    )

    for chunk in llm.stream(prompt):

        if chunk.content:

            yield sse_event(
                StreamEvent.TOKEN,
                {
                    "text": chunk.content
                }
            )
    yield sse_event(
        StreamEvent.CITATIONS,
        citations
    )

    yield sse_event(
        StreamEvent.END,
        {}
    )