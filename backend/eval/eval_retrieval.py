"""
RAG Evaluation Suite — Retrieval + Planner metrics.

Covers:
    - Planner accuracy (intent classification, paper routing)
    - Retrieval quality (Precision@K, Hit@K, MRR)

Run from the backend/ directory:
    cd backend
    python -m eval.eval_retrieval
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Make the backend and app packages importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

sys.modules.setdefault("tensorflow", None)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from app.services.embeddings import get_embedding_model
from app.services.llm import get_llm
from app.services.paper_manager import PaperManager
from app.services.research_planner import ResearchPlanner
from app.services.hybrid_retriever import multi_paper_hybrid_retriever
from eval.test_cases import QA_CASES, PLANNER_CASES
from eval.rate_limit import call_with_retry, configure_eval, limit_cases, pause_after_case


# Setup — shared across all eval functions

def build_env():
    """Load models and paper manager (same as app startup)."""
    configure_eval()
    print("Loading models...")
    embedding_model = get_embedding_model()
    llm = get_llm()
    paper_manager = PaperManager()
    paper_manager.load_saved_papers(embedding_model)
    planner = ResearchPlanner(llm)
    print(f"Loaded {len(paper_manager.get_all_papers())} papers.\n")
    return embedding_model, llm, paper_manager, planner


# 1. Planner Evaluation

def eval_planner(planner: ResearchPlanner, paper_manager: PaperManager):
    """
    Evaluates intent classification and paper routing accuracy.
    """
    print("=" * 60)
    print("PLANNER EVALUATION")
    print("=" * 60)

    results = []
    correct_intent = 0
    correct_papers = 0
    total = len(PLANNER_CASES)

    planner_cases = limit_cases(PLANNER_CASES)

    for case in planner_cases:
        papers = [
            paper_manager.get_paper(pid)
            for pid in case["paper_ids"]
            if paper_manager.get_paper(pid) is not None
        ]

        if not papers:
            print(f"[SKIP] {case['id']} — no papers loaded for paper_ids {case['paper_ids']}")
            continue

        output, stop = call_with_retry(
            lambda: planner.create_plan(question=case["question"], papers=papers),
            label=f"planner [{case['id']}]",
            stop_on_daily_quota=True,
        )
        if stop == "daily_quota":
            print("Stopping planner eval — daily Groq quota exhausted.")
            break
        if output is None:
            print(f"[SKIP] {case['id']} — planner call failed.")
            continue

        predicted_intent = output.intent.value
        expected_intent = case["expected_intent"]
        intent_correct = predicted_intent == expected_intent

        predicted_papers = set(output.target_papers)
        expected_papers = set(case.get("expected_target_paper_ids", []))
        papers_correct = predicted_papers == expected_papers

        if intent_correct:
            correct_intent += 1
        if papers_correct:
            correct_papers += 1

        status = "[PASS]" if intent_correct else "[FAIL]"
        print(
            f"{status} [{case['id']}] Intent: {predicted_intent!r} "
            f"(expected {expected_intent!r}) | "
            f"Papers match: {'[MATCH]' if papers_correct else '[MISMATCH]'} "
            f"predicted={sorted(predicted_papers)} expected={sorted(expected_papers)}"
        )

        results.append({
            "id": case["id"],
            "question": case["question"],
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": intent_correct,
            "expected_papers": sorted(expected_papers),
            "predicted_papers": sorted(predicted_papers),
            "papers_correct": papers_correct,
        })
        pause_after_case("planner")

    evaluated = len(results)
    print(f"\nPlanner Results ({evaluated} evaluated / {total} total):")
    if evaluated > 0:
        print(f"   Intent Accuracy:        {correct_intent}/{evaluated} = {correct_intent/evaluated*100:.1f}%")
        print(f"   Paper Routing Accuracy: {correct_papers}/{evaluated} = {correct_papers/evaluated*100:.1f}%\n")
    else:
        print("   No test cases evaluated. Check eval/test_cases.py.\n")

    return results


# 2. Retrieval Evaluation

def _chunk_contains_keyword(chunk_text: str, keywords: list[str]) -> bool:
    """Returns True if any keyword appears in the chunk text (case-insensitive)."""
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def eval_retrieval(paper_manager: PaperManager):
    """
    Evaluates hybrid retrieval quality using keyword-based relevance.

    Metrics:
        - Precision@K: fraction of retrieved chunks matching relevance keywords
        - Hit@K: fraction of test cases with at least one matching chunk
        - MRR: Mean Reciprocal Rank of first keyword-matching chunk
    """
    print("=" * 60)
    print("RETRIEVAL EVALUATION")
    print("=" * 60)

    cases_with_keywords = limit_cases(
        [c for c in QA_CASES if c.get("relevant_chunk_keywords")]
    )

    if not cases_with_keywords:
        print("No QA cases with 'relevant_chunk_keywords' found. Skipping retrieval eval.")
        return []

    precision_scores = []
    hit_scores = []
    mrr_scores = []
    results = []

    for case in cases_with_keywords:
        retriever_pairs = []
        for paper_id in case["paper_ids"]:
            paper = paper_manager.get_paper(paper_id)
            if paper is None:
                continue
            retriever_pairs.append([paper["retriever"], paper["bm25_retriever"]])

        if not retriever_pairs:
            print(f"[SKIP] {case['id']} — no papers loaded.")
            continue

        retrieved_docs = multi_paper_hybrid_retriever(
            question=case["question"],
            retriever_pairs=retriever_pairs
        )

        keywords = case["relevant_chunk_keywords"]
        relevance_flags = [
            _chunk_contains_keyword(doc.page_content, keywords)
            for doc in retrieved_docs
        ]

        k = len(retrieved_docs)
        relevant_retrieved = sum(relevance_flags)

        precision = relevant_retrieved / k if k > 0 else 0.0

        # Hit@K: 1 if at least one relevant chunk was retrieved, else 0.
        hit_at_k = 1.0 if relevant_retrieved > 0 else 0.0

        # MRR: reciprocal rank of first hit
        mrr = 0.0
        for rank, is_relevant in enumerate(relevance_flags, start=1):
            if is_relevant:
                mrr = 1.0 / rank
                break

        precision_scores.append(precision)
        hit_scores.append(hit_at_k)
        mrr_scores.append(mrr)

        status = "[HIT]" if relevant_retrieved > 0 else "[MISS]"
        print(
            f"{status} [{case['id']}] Retrieved={k} | "
            f"Relevant={relevant_retrieved} | "
            f"P@{k}={precision:.2f} | MRR={mrr:.2f}"
        )

        results.append({
            "id": case["id"],
            "question": case["question"],
            "retrieved_k": k,
            "relevant_retrieved": relevant_retrieved,
            "precision": precision,
            "hit_at_k": hit_at_k,
            "mrr": mrr,
        })

    n = len(results)
    if n:
        print(f"\nRetrieval Results ({n} cases):")
        print(f"   Mean Precision@K: {sum(precision_scores)/n:.3f}")
        print(f"   Mean Hit@K:      {sum(hit_scores)/n:.3f}")
        print(f"   MRR:              {sum(mrr_scores)/n:.3f}\n")
    else:
        print("\nNo retrieval cases evaluated. Check eval/test_cases.py.\n")

    return results


# Main

if __name__ == "__main__":
    _, _, paper_manager, planner = build_env()

    planner_results = eval_planner(planner, paper_manager)
    retrieval_results = eval_retrieval(paper_manager)
