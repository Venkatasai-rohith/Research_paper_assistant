from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from current working dir or app dir
load_dotenv()
_app_env = Path(__file__).resolve().parent.parent / ".env"
if _app_env.exists():
    load_dotenv(dotenv_path=_app_env)

GROK_API_KEY = os.getenv("GROK_API_KEY")

if not GROK_API_KEY:
    raise ValueError("GROK_API_KEY not found in environment variables.")

LLM_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 8
TEMPERATURE = 0.2

VECTORSTORE_PATH = "storage/faiss_index"

SUMMARY_GROUP_WORDS = 3500

HYBRID_CANDIDATE_K = 8
HYBRID_FINAL_K = 5
RRF_CONSTANT = 60