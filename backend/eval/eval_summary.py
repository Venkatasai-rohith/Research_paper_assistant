"""
RAG Evaluation Suite — Summary Quality.

Evaluates the summary pipeline using LLM-as-judge for:
    - Coverage:    Does the summary mention key contributions/methods/results?
    - Coherence:   Is the summary logically consistent?
    - Conciseness: Is it appropriately brief vs verbose?

Run from the backend/ directory:
    cd backend
    python -m eval.eval_summary
"""
import re
import sys
import os
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

sys.modules.setdefault("tensorflow", None)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

from app.services.embeddings import get_embedding_model
from app.services.llm import get_llm
from app.services.paper_manager import PaperManager
from app.services.research_service import ResearchService
from app.core.streaming import StreamEvent
from eval.test_cases import SUMMARY_CASES
from eval.rate_limit import (
    call_with_retry,
    configure_eval,
    get_config,
    invoke_llm,
    limit_cases,
    pause_after_case,
)


SUMMARY_JUDGE_PROMPT = """You are a strict research paper summary evaluator.

A research paper was summarized using a map-reduce approach.

SUMMARY:
{summary}

EXPECTED TOPICS (key themes that should be covered):
{expected_topics}

Rate the summary on three dimensions (each 1–5):

1. Coverage: Are the key topics present in the summary?
   1=none covered, 5=all covered thoroughly

2. Coherence: Is the summary logically structured and consistent?
   1=disjointed, 5=very well structured

3. Conciseness: Is the length appropriate — not too brief, not padded?
   1=way too long or too short, 5=perfectly concise

Respond ONLY with a JSON object like this:
{{
  "coverage": <1-5>,
  "coherence": <1-5>,
  "conciseness": <1-5>,
  "coverage_reason": "<one sentence>",
  "coherence_reason": "<one sentence>",
  "conciseness_reason": "<one sentence>"
}}"""


def build_env():
    configure_eval()
    print("Loading models...")
    embedding_model = get_embedding_model()
    llm = get_llm()
    paper_manager = PaperManager()
    paper_manager.load_saved_papers(embedding_model)
    research_service = ResearchService(paper_manager=paper_manager, llm=llm)
    print(f"Loaded {len(paper_manager.get_all_papers())} papers.\n")
    return llm, paper_manager, research_service


def collect_streamed_answer(stream) -> str:
    """Consume an SSE stream and return the full answer text."""
    parts = []
    for event_str in stream:
        lines = event_str.strip().split("\n")
        event_type, data_str = None, None
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
            parts.append(data.get("text", ""))
    return "".join(parts)


def eval_summaries(llm, research_service: ResearchService):
    print("=" * 60)
    print("SUMMARY EVALUATION (LLM-as-Judge)")
    print("=" * 60)

    valid_cases = limit_cases(
        [c for c in SUMMARY_CASES if "REPLACE" not in c["paper_ids"][0]]
    )

    if not valid_cases:
        print("No filled-in summary cases. Edit eval/test_cases.py.\n")
        return []

    results = []
    coverage_scores, coherence_scores, conciseness_scores = [], [], []

    for case in valid_cases:
        print(f"\n[{case['id']}] {case['question']}")

        def run_pipeline():
            stream = research_service.stream_answer(
                paper_ids=case["paper_ids"],
                question=case["question"],
            )
            return collect_streamed_answer(stream)

        summary_text, stop = call_with_retry(
            run_pipeline,
            label=f"summary pipeline [{case['id']}]",
            stop_on_daily_quota=True,
        )
        if stop == "daily_quota":
            print("Stopping summary eval — daily Groq quota exhausted.")
            break
        if summary_text is None:
            print(f"  [SKIP] {case['id']} — summary generation failed.")
            continue

        summary = summary_text
        if not summary.strip():
            print("  [WARN] Empty summary returned.")
            continue

        print(f"  Summary length: {len(summary.split())} words")

        cov, coh, con = -1, -1, -1
        scores = {}


        if stop == "daily_quota":
            print(
                "Daily Groq quota exhausted during judging. "
                "Keeping the generated summary with unavailable judge scores."
            )
            scores = {}
        cov, coh, con = -1, -1, -1
        scores = {}
        quota_exhausted = False

        if not get_config().skip_judge:
            topics = case.get("expected_topics", [])
            prompt = SUMMARY_JUDGE_PROMPT.format(
                summary=summary[:8000],
                expected_topics=", ".join(topics) if topics else "general overview",
            )

            response, stop = invoke_llm(
                llm,
                prompt,
                label=f"summary judge [{case['id']}]",
            )

            if stop == "daily_quota":
                print(
                    "Daily Groq quota exhausted during judging. "
                    "Keeping the generated summary with unavailable judge scores."
                )
                quota_exhausted = True

            elif response is None:
                print(
                    "  [WARN] Judge call failed; keeping summary without judge scores."
                )

            else:
                try:
                    content = response.content.strip()
                    content = re.sub(
                        r"^```(?:json)?\s*|\s*```$",
                        "",
                        content,
                        flags=re.IGNORECASE,
                    ).strip()

                    try:
                        scores = json.loads(content)
                    except json.JSONDecodeError:
                        match = re.search(r"\{.*\}", content, re.DOTALL)
                        if not match:
                            raise ValueError(
                                "No JSON object found in judge response"
                            )
                        scores = json.loads(match.group(0))

                    if not isinstance(scores, dict):
                        raise ValueError(
                            "Judge response must be a JSON object"
                        )

                    for metric in (
                        "coverage",
                        "coherence",
                        "conciseness",
                    ):
                        score = float(scores[metric])
                        if not 1.0 <= score <= 5.0:
                            raise ValueError(
                                f"Invalid {metric} score: {score}"
                            )
                        scores[metric] = score

                except (
                    json.JSONDecodeError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as e:
                    print(f"  [judge error] Invalid judge response: {e}")
                    scores = {}

        cov = scores.get("coverage", -1)
        coh = scores.get("coherence", -1)
        con = scores.get("conciseness", -1)

        print(f"  Coverage:    {cov}/5 — {scores.get('coverage_reason', '')}")
        print(f"  Coherence:   {coh}/5 — {scores.get('coherence_reason', '')}")
        print(f"  Conciseness: {con}/5 — {scores.get('conciseness_reason', '')}")

        if cov > 0:
            coverage_scores.append(cov)
        if coh > 0:
            coherence_scores.append(coh)
        if con > 0:
            conciseness_scores.append(con)

        results.append({
            "id": case["id"],
            "paper_ids": case["paper_ids"],
            "word_count": len(summary.split()),
            "coverage": cov,
            "coherence": coh,
            "conciseness": con,
        })

        if quota_exhausted:
            print("Stopping summary eval after saving the current summary.")
            break

        pause_after_case("summary")

    if results:
        n = len(results)
        print(f"\nSummary Results ({n} cases):")
        if coverage_scores:
            print(f"   Mean Coverage:    {sum(coverage_scores)/len(coverage_scores):.2f}/5")
        if coherence_scores:
            print(f"   Mean Coherence:   {sum(coherence_scores)/len(coherence_scores):.2f}/5")
        if conciseness_scores:
            print(f"   Mean Conciseness: {sum(conciseness_scores)/len(conciseness_scores):.2f}/5")
        print()

    return results


if __name__ == "__main__":
    llm, paper_manager, research_service = build_env()
    eval_summaries(llm, research_service)
