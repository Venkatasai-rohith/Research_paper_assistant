import re

def normalize_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = " ".join(text.split())

    return text