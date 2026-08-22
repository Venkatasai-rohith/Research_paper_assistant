from langchain_huggingface import HuggingFaceEmbeddings

from core.config import EMBEDDING_MODEL


def get_embedding_model():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )