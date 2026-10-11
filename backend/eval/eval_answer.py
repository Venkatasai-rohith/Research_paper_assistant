"""
RAG Evaluation Suite — Answer Quality (Faithfulness + Relevance).

Uses an LLM-as-judge approach:
    - Faithfulness: is the answer grounded in the retrieved context?
    - Relevance:    does the answer directly address the question?

Also optionally runs RAGAS if the package is installed.

Run from the backend/ directory:
    cd backend
    python -m eval.eval_answer
"""

import sys
import os
import json
import re
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

sys.modules.setdefault("tensorflow", None)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from app.services.embeddings import get_embedding_model
from app.services.llm import get_llm
from app.services.paper_manager import PaperManager
from app.services.hybrid_retriever import multi_paper_hybrid_retriever
from app.services.research_service import ResearchService
from app.core.streaming import StreamEvent
from eval.test_cases import QA_CASES, COMPARISON_CASES
from eval.rate_limit import (
    call_with_retry,
    configure_eval,
    get_config,
    invoke_llm,
    limit_cases,
    pause_after_case,
)


# LLM-as-Judge Prompts

FAITHFULNESS_PROMPT = """You are a strict evaluation judge.

Your task: assess whether the ANSWER is faithful to the CONTEXT.
Faithful means every factual claim in the answer can be found in or directly inferred from the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Rate faithfulness on a scale of 1 to 5:
1 = Answer contains major claims not supported by context (hallucination)
2 = Several unsupported claims
3 = Mostly faithful, minor unsupported details
4 = Almost fully faithful, trivial gaps
5 = Completely faithful, every claim traceable to context

Respond ONLY with a JSON object like this:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""


RELEVANCE_PROMPT = """You are a strict evaluation judge.

Your task: assess whether the ANSWER directly and completely addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Rate relevance on a scale of 1 to 5:
1 = Answer is off-topic or completely misses the question
2 = Partially addresses the question but misses the main point
3 = Addresses the question but incomplete
4 = Mostly complete answer, minor gaps
5 = Directly and completely answers the question

Respond ONLY with a JSON object like this:
{{"score": <1-5>, "reasoning": "<one sentence>"}}"""


# Helpers

def build_env():
    """Load models and paper manager (same as app startup)."""
    configure_eval()
    print("Loading models...")
    embedding_model = get_embedding_model()
    llm = get_llm()
    paper_manager = PaperManager()
    paper_manager.load_saved_papers(embedding_model)
    research_service = ResearchService(paper_manager=paper_manager, llm=llm)
    print(f"Loaded {len(paper_manager.get_all_papers())} papers.\n")
    return llm, paper_manager, research_service


def collect_streamed_answer(stream) -> tuple[str, list]:
    """
    Consume an SSE stream generator and extract the full answer text
    and citations list.
    """
    answer_parts = []
    citations = []

    for event_str in stream:
        # event_str is like: "event: token\ndata: {...}\n\n"
        lines = event_str.strip().split("\n")
        event_type = None
        data_str = None

        for line in lines:
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()

        if not event_type or not data_str:
            continue

        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if event_type == StreamEvent.TOKEN.value:
            answer_parts.append(data.get("text", ""))
        elif event_type == StreamEvent.CITATIONS.value:
            citations = data if isinstance(data, list) else []

    return "".join(answer_parts), citations


def get_context_for_question(question: str, paper_ids: list[str], paper_manager: PaperManager) -> str:
    """Retrieve context chunks for a question (replicates what the QA executor does)."""
    retriever_pairs = []
    for paper_id in paper_ids:
        paper = paper_manager.get_paper(paper_id)
        if paper:
            retriever_pairs.append([paper["retriever"], paper["bm25_retriever"]])

    if not retriever_pairs:
        return ""

    docs = multi_paper_hybrid_retriever(question, retriever_pairs)
    return "\n\n".join(doc.page_content for doc in docs)


def _parse_judge_response(response) -> tuple[float, str]:
    content = response.content

    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict)
        )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("Judge returned an empty response")

    content = content.strip()
    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
        flags=re.IGNORECASE,
    ).strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise ValueError(
                f"No JSON object found in judge response: {content[:200]!r}"
            )
        data = json.loads(match.group(0))

    if not isinstance(data, dict) or "score" not in data:
        raise ValueError("Judge JSON must contain a 'score' field")

    score = float(data["score"])
    if not 1.0 <= score <= 5.0:
        raise ValueError(f"Judge score must be between 1 and 5, got {score}")

    return score, str(data.get("reasoning", ""))


def judge_score(llm, prompt: str) -> tuple[float, str, str | None]:
    """
    Ask the LLM judge for a JSON score and reasoning.

    Returns (score, reason, stop_reason). stop_reason is ``daily_quota`` when
    Groq daily limits are hit. score -1.0 means the judge call failed.
    """
    cfg = get_config()

    for attempt in range(cfg.max_retries):
        response, stop = invoke_llm(llm, prompt, label="judge")
        if stop == "daily_quota":
            return -1.0, "daily quota exhausted", stop
        if response is None:
            return -1.0, "rate limit exceeded", None

        try:
            score, reasoning = _parse_judge_response(response)
            return score, reasoning, None
        except Exception as e:
            print(f"  [Judge parse error] {type(e).__name__}: {e}")
            if attempt < cfg.max_retries - 1:
                time.sleep(cfg.backoff_base_sec * 0.25)
                continue
            return -1.0, "judge response or API error", None

    return -1.0, "evaluation failed", None


# 1. QA Answer Evaluation

def eval_qa_answers(llm, paper_manager: PaperManager, research_service: ResearchService):
    """
    For each QA case:
      1. Run the full pipeline (stream_answer)
      2. Score faithfulness and relevance via LLM-as-judge
    """
    print("=" * 60)
    print("QA ANSWER EVALUATION (LLM-as-Judge)")
    print("=" * 60)

    valid_cases = limit_cases(
        [c for c in QA_CASES if "REPLACE" not in c["paper_ids"][0]]
    )
    if not valid_cases:
        print("No filled-in QA cases found. Edit eval/test_cases.py and replace the placeholder paper_ids.\n")
        return []

    results = []
    faithfulness_scores = []
    relevance_scores = []

    for case in valid_cases:
        print(f"\n[{case['id']}] {case['question'][:80]}")

        def run_pipeline():
            stream = research_service.stream_answer(
                paper_ids=case["paper_ids"],
                question=case["question"],
            )
            return collect_streamed_answer(stream)

        pipeline_result, stop = call_with_retry(
            run_pipeline,
            label=f"QA pipeline [{case['id']}]",
            stop_on_daily_quota=True,
        )
        if stop == "daily_quota":
            print("Stopping QA eval — daily Groq quota exhausted.")
            break
        if pipeline_result is None:
            print(f"  [SKIP] {case['id']} — answer generation failed.")
            continue

        answer, citations = pipeline_result
        if not answer.strip():
            print(f"  [SKIP] {case['id']} — empty answer.")
            continue

        print(f"  Answer preview: {answer[:120].replace(chr(10), ' ')}...")

        faith_score, faith_reason = -1.0, "skipped"
        rel_score, rel_reason = -1.0, "skipped"
        judge_quota_exhausted = False

        if not get_config().skip_judge:
            context = get_context_for_question(
                question=case["question"],
                paper_ids=case["paper_ids"],
                paper_manager=paper_manager,
            )

            faith_score, faith_reason, stop = judge_score(
                llm,
                FAITHFULNESS_PROMPT.format(
                    context=context[:6000],
                    question=case["question"],
                    answer=answer,
                ),
            )

            if stop == "daily_quota":
                print(
                    "Daily Groq quota exhausted during faithfulness judging. "
                    "Keeping the generated answer."
                )
                faith_score = -1.0
                faith_reason = "Not scored: daily Groq quota exhausted"
                judge_quota_exhausted = True

            else:
                rel_score, rel_reason, stop = judge_score(
                    llm,
                    RELEVANCE_PROMPT.format(
                        question=case["question"],
                        answer=answer,
                    ),
                )

                if stop == "daily_quota":
                    print(
                        "Daily Groq quota exhausted during relevance judging. "
                        "Keeping the generated answer and faithfulness score."
                    )
                    rel_score = -1.0
                    rel_reason = "Not scored: daily Groq quota exhausted"
                    judge_quota_exhausted = True

        print(f"  Faithfulness: {faith_score}/5 — {faith_reason}")
        print(f"  Relevance:    {rel_score}/5 — {rel_reason}")
        print(f"  Citations:    {len(citations)}")

        if 1.0 <= faith_score <= 5.0:
            faithfulness_scores.append(faith_score)

        if 1.0 <= rel_score <= 5.0:
            relevance_scores.append(rel_score)

        results.append({
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "faithfulness": faith_score,
            "faithfulness_reason": faith_reason,
            "relevance": rel_score,
            "relevance_reason": rel_reason,
            "citation_count": len(citations),
        })

        if judge_quota_exhausted:
            print("Stopping QA eval after saving the current answer.")
            break

        pause_after_case("QA")

    print(f"\nQA Answer Results ({len(results)} answers generated):")

    if faithfulness_scores:
        mean_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
        print(
            f"   Mean Faithfulness: {mean_faithfulness:.2f}/5 "
            f"({len(faithfulness_scores)} successfully scored)"
        )
    else:
        print("   Mean Faithfulness: unavailable (no valid judge scores)")

    if relevance_scores:
        mean_relevance = sum(relevance_scores) / len(relevance_scores)
        print(
            f"   Mean Relevance:    {mean_relevance:.2f}/5 "
            f"({len(relevance_scores)} successfully scored)\n"
        )
    else:
        print("   Mean Relevance:    unavailable (no valid judge scores)\n")

    return results


# 2. RAGAS Evaluation (optional, requires `pip install ragas datasets`)

def eval_ragas(paper_manager: PaperManager, research_service: ResearchService):
    """
    Runs RAGAS evaluation which computes:
        - faithfulness
        - answer_relevancy
        - context_precision
        - context_recall

    Requires: pip install ragas==0.1.21 langchain-openai datasets
    """
    # Monkey-patch the missing vertexai stub so ragas can import
    # on environments without Google Cloud SDK installed.
    try:
        import sys
        import types
        if "langchain_community.chat_models.vertexai" not in sys.modules:
            stub = types.ModuleType("langchain_community.chat_models.vertexai")
            stub.ChatVertexAI = None
            sys.modules["langchain_community.chat_models.vertexai"] = stub
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset
    except ImportError as e:
        print(f"[SKIP] RAGAS not available: {e}")
        print("       Run: pip install ragas==0.1.21 langchain-openai datasets")
        return

    print("=" * 60)
    print("RAGAS EVALUATION")
    print("=" * 60)

    valid_cases = [
        c for c in QA_CASES
        if "REPLACE" not in c["paper_ids"][0] and c.get("ground_truth") and "REPLACE" not in c["ground_truth"]
    ]

    valid_cases = limit_cases(valid_cases)

    if not valid_cases:
        print("No filled-in QA cases with ground_truth found. Edit eval/test_cases.py.\n")
        return

    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for case in valid_cases:
        print(f"  Running [{case['id']}]...")

        def run_pipeline():
            stream = research_service.stream_answer(
                paper_ids=case["paper_ids"],
                question=case["question"],
            )
            return collect_streamed_answer(stream)

        pipeline_result, stop = call_with_retry(
            run_pipeline,
            label=f"RAGAS [{case['id']}]",
            stop_on_daily_quota=True,
        )
        if stop == "daily_quota":
            print("Stopping RAGAS — daily Groq quota exhausted.")
            break
        if pipeline_result is None:
            continue
        answer, _ = pipeline_result

        retriever_pairs = []
        for paper_id in case["paper_ids"]:
            paper = paper_manager.get_paper(paper_id)
            if paper:
                retriever_pairs.append([paper["retriever"], paper["bm25_retriever"]])

        from app.services.hybrid_retriever import multi_paper_hybrid_retriever
        docs = multi_paper_hybrid_retriever(case["question"], retriever_pairs)
        contexts = [doc.page_content for doc in docs]

        ragas_data["question"].append(case["question"])
        ragas_data["answer"].append(answer)
        ragas_data["contexts"].append(contexts)
        ragas_data["ground_truth"].append(case["ground_truth"])

    if not ragas_data["question"]:
        print("No data collected for RAGAS.\n")
        return

    dataset = Dataset.from_dict(ragas_data)

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\nRAGAS Results:")
    print(result)
    print()
    return result

# Main

if __name__ == "__main__":
    llm, paper_manager, research_service = build_env()
    qa_results = eval_qa_answers(llm, paper_manager, research_service)
