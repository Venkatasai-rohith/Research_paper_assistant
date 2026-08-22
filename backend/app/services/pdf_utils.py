import fitz
import re
from exceptions import PDFExtractionError


def extract_text(pdf_path: str) -> list[str]:
    """
    Extract text from each page of a PDF.

    Returns:
        List of page texts.
    """

    try:

        document = fitz.open(pdf_path)

        page_texts = []

        for page in document:

            text = page.get_text("text")

            if text.strip():
                page_texts.append(text)

        document.close()

        return page_texts

    except Exception as e:

        raise PDFExtractionError(
            "Unable to extract text from the uploaded PDF."
        ) from e

import re


def extract_abstract(page_texts: list[str]) -> str:

    full_text = "\n".join(page_texts)

    match = re.search(
        r"\babstract\b",
        full_text,
        re.IGNORECASE
    )

    if not match:
        return ""

    abstract_start = match.end()

    introduction_match = re.search(
        r"\n\s*(?:1[\.\s]+)?introduction\b",
        full_text[abstract_start:],
        re.IGNORECASE
    )

    if introduction_match:
        abstract_end = (
            abstract_start +
            introduction_match.start()
        )
    else:
        abstract_end = abstract_start + 5000

    abstract = full_text[
        abstract_start:abstract_end
    ].strip()

    abstract = re.sub(
        r"\s+",
        " ",
        abstract
    )

    return abstract