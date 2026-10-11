"""
run_eval.py — Master runner for the full RAG evaluation suite.

Runs all evaluators and saves results to eval/results/.

Run from the backend/ directory:
    cd backend
    python eval/run_eval.py

Options:
    --retrieval   Run only retrieval + planner eval
    --answer      Run only answer quality eval
    --summary     Run only summary quality eval
    --ragas       Also run RAGAS (requires: pip install ragas datasets)
    --save        Save results to eval/results/ as JSON + CSV
"""

import sys
import os
import json
import argparse
from datetime import datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

sys.modules.setdefault("tensorflow", None)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument("--retrieval", action="store_true", help="Run retrieval + planner eval")
    parser.add_argument("--answer", action="store_true", help="Run answer quality eval")
    parser.add_argument("--summary", action="store_true", help="Run summary quality eval")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS eval (requires ragas package)")
    parser.add_argument("--save", action="store_true", help="Save results to eval/results/")
    return parser.parse_args()


def save_results(results: dict, run_id: str):
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, f"eval_{run_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] Results saved to: {json_path}")

    try:
        import pandas as pd
        for key, rows in results.items():
            if rows:
                csv_path = os.path.join(results_dir, f"eval_{run_id}_{key}.csv")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
                print(f"   CSV: {csv_path}")
    except ImportError:
        pass

    # Generate Markdown Report
    report_lines = [
        f"# RAG Evaluation Report — {run_id}",
        "",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Metrics",
        "",
        "| Evaluation Category | Metric | Score | Details |",
        "|---|---|:---:|---|",
    ]

    if "planner" in results and results["planner"]:
        r = results["planner"]
        n = len(r)
        intent_acc = sum(1 for x in r if x.get("intent_correct")) / n * 100
        paper_acc = sum(1 for x in r if x.get("papers_correct")) / n * 100
        report_lines.append(f"| **Planner** | Intent Accuracy | **{intent_acc:.1f}%** | {sum(1 for x in r if x.get('intent_correct'))}/{n} correct |")
        report_lines.append(f"| **Planner** | Routing Accuracy | **{paper_acc:.1f}%** | {sum(1 for x in r if x.get('papers_correct'))}/{n} correct |")

    if "retrieval" in results and results["retrieval"]:
        r = results["retrieval"]
        n = len(r)
        mp = sum(x.get("precision", 0) for x in r) / n
        mr = sum(x.get("recall", 0) for x in r) / n
        mmrr = sum(x.get("mrr", 0) for x in r) / n
        report_lines.append(f"| **Hybrid Retrieval** | Mean Precision@K | **{mp:.3f}** | Across {n} test cases |")
        report_lines.append(f"| **Hybrid Retrieval** | Mean Recall@K | **{mr:.3f}** | Across {n} test cases |")
        report_lines.append(f"| **Hybrid Retrieval** | Mean Reciprocal Rank (MRR) | **{mmrr:.3f}** | Across {n} test cases |")

    if "qa_answers" in results and results["qa_answers"]:
        r = results["qa_answers"]
        vf = [x["faithfulness"] for x in r if x.get("faithfulness", -1) > 0]
        vr = [x["relevance"] for x in r if x.get("relevance", -1) > 0]
        if vf:
            report_lines.append(f"| **QA Answers** | Mean Faithfulness | **{sum(vf)/len(vf):.2f} / 5.0** | LLM-as-judge groundedness ({len(vf)} evaluated) |")
        if vr:
            report_lines.append(f"| **QA Answers** | Mean Relevance | **{sum(vr)/len(vr):.2f} / 5.0** | LLM-as-judge relevance ({len(vr)} evaluated) |")

    if "summaries" in results and results["summaries"]:
        r = results["summaries"]
        vcov = [x["coverage"] for x in r if x.get("coverage", -1) > 0]
        vcoh = [x["coherence"] for x in r if x.get("coherence", -1) > 0]
        vcon = [x["conciseness"] for x in r if x.get("conciseness", -1) > 0]
        if vcov:
            report_lines.append(f"| **Summaries** | Mean Coverage | **{sum(vcov)/len(vcov):.2f} / 5.0** | LLM-as-judge ({len(vcov)} evaluated) |")
        if vcoh:
            report_lines.append(f"| **Summaries** | Mean Coherence | **{sum(vcoh)/len(vcoh):.2f} / 5.0** | LLM-as-judge ({len(vcoh)} evaluated) |")
        if vcon:
            report_lines.append(f"| **Summaries** | Mean Conciseness | **{sum(vcon)/len(vcon):.2f} / 5.0** | LLM-as-judge ({len(vcon)} evaluated) |")

    report_lines.append("")

    # Detailed tables
    if "qa_answers" in results and results["qa_answers"]:
        report_lines.extend([
            "## QA Answers Breakdown",
            "",
            "| ID | Question | Faithfulness | Relevance | Citations |",
            "|---|---|:---:|:---:|:---:|",
        ])
        for x in results["qa_answers"]:
            q_clean = x.get("question", "")[:60].replace("|", "/")
            report_lines.append(f"| `{x.get('id')}` | {q_clean} | {x.get('faithfulness')}/5 | {x.get('relevance')}/5 | {x.get('citation_count', 0)} |")
        report_lines.append("")

    if "summaries" in results and results["summaries"]:
        report_lines.extend([
            "## Summaries Breakdown",
            "",
            "| ID | Words | Coverage | Coherence | Conciseness |",
            "|---|:---:|:---:|:---:|:---:|",
        ])
        for x in results["summaries"]:
            report_lines.append(f"| `{x.get('id')}` | {x.get('word_count')} words | {x.get('coverage')}/5 | {x.get('coherence')}/5 | {x.get('conciseness')}/5 |")
        report_lines.append("")

    report_content = "\n".join(report_lines)
    report_path = os.path.join(results_dir, f"eval_{run_id}_report.md")
    latest_path = os.path.join(results_dir, "EVALUATION_REPORT.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"   Report: {report_path}")
    print(f"   Latest: {latest_path}")


def main():
    args = parse_args()
    run_all = not any([args.retrieval, args.answer, args.summary])

    from app.services.embeddings import get_embedding_model
    from app.services.llm import get_llm
    from app.services.paper_manager import PaperManager
    from app.services.research_service import ResearchService
    from app.services.research_planner import ResearchPlanner

    print("=" * 60)
    print("RAG EVALUATION SUITE")
    print("=" * 60)
    print("Loading models and papers...")

    embedding_model = get_embedding_model()
    llm = get_llm()
    paper_manager = PaperManager()
    paper_manager.load_saved_papers(embedding_model)
    research_service = ResearchService(paper_manager=paper_manager, llm=llm)
    planner = ResearchPlanner(llm)

    papers = paper_manager.get_all_papers()
    print(f"Papers loaded: {len(papers)}")
    for pid, p in papers.items():
        print(f"  - {p['title']} [{pid[:8]}...]")
    print()

    all_results = {}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Retrieval + Planner
    if run_all or args.retrieval:
        from eval.eval_retrieval import eval_planner, eval_retrieval
        all_results["planner"] = eval_planner(planner, paper_manager)
        all_results["retrieval"] = eval_retrieval(paper_manager)

    # Answer Quality
    if run_all or args.answer:
        from eval.eval_answer import eval_qa_answers
        all_results["qa_answers"] = eval_qa_answers(llm, paper_manager, research_service)

    # Summary Quality
    if run_all or args.summary:
        from eval.eval_summary import eval_summaries
        all_results["summaries"] = eval_summaries(llm, research_service)

    # RAGAS (optional)
    if args.ragas:
        from eval.eval_answer import eval_ragas
        eval_ragas(paper_manager, research_service)

    # Save
    if args.save:
        save_results(all_results, run_id)

    # Final Summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)

    if "planner" in all_results and all_results["planner"]:
        r = all_results["planner"]
        n = len(r)
        intent_acc = sum(1 for x in r if x["intent_correct"]) / n * 100
        paper_acc = sum(1 for x in r if x["papers_correct"]) / n * 100
        print(f"Planner    → Intent Accuracy: {intent_acc:.1f}% | Paper Routing: {paper_acc:.1f}%")

    if "retrieval" in all_results and all_results["retrieval"]:
        r = all_results["retrieval"]
        n = len(r)
        print(f"Retrieval  → Mean P@K: {sum(x['precision'] for x in r)/n:.3f} | MRR: {sum(x['mrr'] for x in r)/n:.3f}")

    if "qa_answers" in all_results and all_results["qa_answers"]:
        r = all_results["qa_answers"]
        n = len(r)
        valid_faith = [x["faithfulness"] for x in r if x["faithfulness"] > 0]
        valid_rel = [x["relevance"] for x in r if x["relevance"] > 0]
        if valid_faith:
            print(f"QA Answers → Faithfulness: {sum(valid_faith)/len(valid_faith):.2f}/5 | Relevance: {sum(valid_rel)/len(valid_rel):.2f}/5")

    if "summaries" in all_results and all_results["summaries"]:
        r = all_results["summaries"]
        n = len(r)
        cov = [x["coverage"] for x in r if x.get("coverage", -1) > 0]
        coh = [x["coherence"] for x in r if x.get("coherence", -1) > 0]
        if cov:
            print(f"Summaries  → Coverage: {sum(cov)/len(cov):.2f}/5 | Coherence: {sum(coh)/len(coh):.2f}/5")

    print()


if __name__ == "__main__":
    main()
