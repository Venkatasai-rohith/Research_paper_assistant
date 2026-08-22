from langchain_core.documents import Document


def build_citations(documents: list[Document]) -> list[dict]:
    
    seen = set()
    citations = []

    for document in documents:

        title = document.metadata.get("title")
        page = document.metadata.get("page")

        key = (title, page)

        if key not in seen:
            seen.add(key)

            citations.append(
                {
                    "title": title,
                    "page": page
                }
            )

    return citations