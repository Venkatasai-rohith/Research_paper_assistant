from langchain_groq import ChatGroq
from core.config import (GROK_API_KEY, LLM_MODEL, TEMPERATURE,)

def get_llm():

    return ChatGroq(
        api_key=GROK_API_KEY, 
        model=LLM_MODEL, 
        temperature=TEMPERATURE,
        streaming=True,)