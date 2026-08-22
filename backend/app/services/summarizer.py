from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from services.citation_builder import build_citations
from core.summary_config import SUMMARY_INSTRUCTIONS
from core.config import SUMMARY_GROUP_WORDS
from prompt import PAPER_SUMMARY_PROMPT
from schemas import SummaryType
import time

def get_documents_from_vectorstore(vectorstore: FAISS)-> list[Document]:
    if vectorstore is None:
        raise ValueError("Vectorstore is not available")

    documents = list(vectorstore.docstore._dict.values())

    documents.sort(
        key=lambda doc: doc.metadata.get("chunk_index", 0)
    )

    return documents

SKIP_HEADINGS = [
    "references",
    "bibliography",
    "acknowledgements",
    "acknowledgments",
    "appendix",
    "author contributions",
    "funding",
    "conflict of interest",
]

def filter_documents(documents: list[Document]) -> list[Document]:

    filtered = []

    for doc in documents:

        text = doc.page_content.lower()

        if any(
            heading in text
            for heading in SKIP_HEADINGS
        ):
            continue

        filtered.append(doc)

    return filtered

def group_documents(
    documents: list[Document],
    max_words: int
) -> list[list[Document]]:

    groups = []
    current_group = []
    current_words = 0

    for document in documents:

        document_words = len(document.page_content.split())

        if (
            current_group
            and current_words + document_words > max_words
        ):
            groups.append(current_group)

            current_group = []
            current_words = 0

        current_group.append(document)
        current_words += document_words

    if current_group:
        groups.append(current_group)

    return groups

def summarize_group(index: int, group: list[Document], chain, instruction: str) -> tuple[int, str]:

    group_text = "\n\n".join(
        doc.page_content for doc in group
    )

    print(
        f"Processing group {index + 1} | "
        f"Characters: {len(group_text)} | "
        f"Words: {len(group_text.split())}"
    )

    try:

        response = chain.invoke(
            {
                "document": group_text,
                "instruction": instruction
            }
        )

    except Exception:

        print(
            f"FAILED MAP GROUP {index + 1} | "
            f"Characters: {len(group_text)} | "
            f"Words: {len(group_text.split())}"
        )

        raise

    return index, response.content

def stream_single_group_summary(
    group: list[Document],
    llm: BaseChatModel,
    summary_type: SummaryType
):
    instruction = SUMMARY_INSTRUCTIONS[summary_type]
    chain = PAPER_SUMMARY_PROMPT | llm
    group_text = "\n\n".join(doc.page_content for doc in group)

    for chunk in chain.stream(
        {
            "document": group_text,
            "instruction": instruction
        }
    ):
        if chunk.content:
            yield chunk.content


def generate_partial_summaries(grouped_documents: list[list[Document]], llm: BaseChatModel, summary_type: SummaryType) -> list[str]:

    instruction = SUMMARY_INSTRUCTIONS[summary_type]
    chain = PAPER_SUMMARY_PROMPT | llm

    results = [None] * len(grouped_documents)
    workers = min(len(grouped_documents), 8) or 1

    with ThreadPoolExecutor(max_workers=workers) as executor:

        futures = {
            executor.submit(
                summarize_group,
                i,
                group,
                chain,
                instruction
            ): i
            for i, group in enumerate(grouped_documents)
        }

        for future in as_completed(futures):

            index, summary = future.result()

            results[index] = summary

            print(
                f"Completed group {index + 1}/"
                f"{len(grouped_documents)}"
            )

    return results

def combine_partial_summaries(partial_summaries: list[str], llm: BaseChatModel, summary_type: SummaryType) -> str:
    instruction = SUMMARY_INSTRUCTIONS[summary_type]
    chain = PAPER_SUMMARY_PROMPT | llm
    final_text = "\n\n".join(partial_summaries)

    response = chain.invoke(
        {
            "document": final_text,
            "instruction": instruction
        }
    )
    return response.content

def stream_combine_partial_summaries(
    partial_summaries: list[str],
    llm: BaseChatModel,
    summary_type: SummaryType
):
    instruction = SUMMARY_INSTRUCTIONS[summary_type]

    chain = PAPER_SUMMARY_PROMPT | llm

    final_text = "\n\n".join(partial_summaries)

    for chunk in chain.stream(
        {
            "document": final_text,
            "instruction": instruction
        }
    ):
        if chunk.content:
            yield chunk.content

def summarize_document(vectorstore: FAISS, llm: BaseChatModel, summary_type: SummaryType) -> dict:

    documents = get_documents_from_vectorstore(vectorstore)
    print(f"Original chunks: {len(documents)}")

    documents = filter_documents(documents)
    print(f"Chunks after filtering: {len(documents)}")

    groups = group_documents(documents, SUMMARY_GROUP_WORDS)

    start = time.perf_counter()

    partial_summaries = generate_partial_summaries(
        groups,
        llm,
        summary_type
    )

    print(f"Map stage took: {time.perf_counter() - start:.2f} seconds")

    start = time.perf_counter()

    final_summary = combine_partial_summaries(
        partial_summaries,
        llm,
        summary_type
    )

    print(f"Reduce stage took: {time.perf_counter() - start:.2f} seconds")

    citations = build_citations(documents)

    return {
        "answer": final_summary,
        "citations": citations
    }
