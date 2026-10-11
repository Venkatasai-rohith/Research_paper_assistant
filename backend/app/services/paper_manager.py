import json
import shutil
from pathlib import Path
from typing import Dict
from utils.text_utils import normalize_text
from core.config import HYBRID_CANDIDATE_K
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

PAPERS_DIR = Path(__file__).resolve().parent.parent / "storage" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

class PaperManager:

    def __init__(self):
        self._papers: Dict[str, dict] = {}

    def _paper_directory(self, paper_id: str) -> Path:
        return PAPERS_DIR / paper_id

    def add_paper(self, paper_id: str, title: str, abstract: str, vectorstore, retriever):

        documents = list(vectorstore.docstore._dict.values())

        bm25_retriever = BM25Retriever.from_documents(documents)

        bm25_retriever.k = HYBRID_CANDIDATE_K

        self._papers[paper_id] = {
            "paper_id": paper_id,
            "title": title,
            "normalized_title": normalize_text(title),
            "abstract": abstract,
            "vectorstore": vectorstore,
            "retriever": retriever,
            "bm25_retriever": bm25_retriever
        }
        self.save_paper(paper_id)

    def save_paper(self, paper_id: str):

        paper = self.get_paper(paper_id)

        if paper is None:
            raise ValueError(f"Paper '{paper_id}' not found.")

        paper_dir = self._paper_directory(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        paper["vectorstore"].save_local(str(paper_dir))

        metadata = {
            "paper_id": paper["paper_id"],
            "title": paper["title"],
            "normalized_title": paper["normalized_title"],
            "abstract": paper.get("abstract", "")
        }

        with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f,indent=4, ensure_ascii=False)

    def get_paper(self, paper_id: str):
        return self._papers.get(paper_id)

    def list_papers(self):
        return [
            {
                "paper_id": paper["paper_id"],
                "title": paper["title"]
            }
            for paper in self._papers.values()
        ]

    def remove_paper(self, paper_id: str):
        return self._papers.pop(paper_id, None)

    def clear(self):
        self._papers.clear()

    def get_all_papers(self):
        return self._papers

    def exists(self, paper_id: str):
        return paper_id in self._papers

    def find_target_papers(self, question: str, candidate_paper_ids: list[str]) -> list[str]:
        question = normalize_text(question)

        matched = []

        for paper_id in candidate_paper_ids:

            paper = self.get_paper(paper_id)

            if paper is None:
                continue

            title = paper["normalized_title"]

            if title in question:
                matched.append(paper_id)
                continue

            title_words = set(title.split())
            question_words = set(question.split())

            overlap = len(title_words & question_words)

            required = max(2, len(title_words) // 2)

            if overlap >= required:
                matched.append(paper_id)

        return matched

    def load_saved_papers(self, embedding_model):

        self.clear()

        if not PAPERS_DIR.exists():
            return

        for paper_dir in PAPERS_DIR.iterdir():

            if not paper_dir.is_dir():
                continue

            metadata_file = paper_dir / "metadata.json"

            if not metadata_file.exists():
                continue

            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            vectorstore = FAISS.load_local(
                str(paper_dir),
                embedding_model,
                allow_dangerous_deserialization=True
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": HYBRID_CANDIDATE_K}
            )

            documents = list(vectorstore.docstore._dict.values())

            bm25_retriever = BM25Retriever.from_documents(documents)

            bm25_retriever.k = HYBRID_CANDIDATE_K

            self._papers[metadata["paper_id"]] = {
                "paper_id": metadata["paper_id"],
                "title": metadata["title"],
                "normalized_title": metadata.get("normalized_title", normalize_text(metadata["title"])),
                "abstract": metadata.get("abstract", ""),
                "vectorstore": vectorstore,
                "retriever": retriever,
                "bm25_retriever": bm25_retriever,
            }


    def delete_paper(self, paper_id: str):

        paper = self.remove_paper(paper_id)

        if paper is None:
            return None

        paper_dir = self._paper_directory(paper_id)

        if paper_dir.exists():
            shutil.rmtree(paper_dir)

        return paper