from services.paper_manager import PaperManager
from services.research_planner import ResearchPlanner
from services.evidence_plan import EvidencePlan, Intent
from executors.qa_executor import QAExecutor
from executors.comparison_executor import ComparisonExecutor
from executors.summary_executor import SummaryExecutor
from core.streaming import sse_event, StreamEvent

class ResearchService:

    def __init__(self, paper_manager, llm):

        self.paper_manager = paper_manager

        self.llm = llm

        self.planner = ResearchPlanner(llm)

        self.qa_executor = QAExecutor(paper_manager,llm)
        self.comparison_executor = ComparisonExecutor(paper_manager, llm)
        self.summary_executor = SummaryExecutor(paper_manager, llm)

    def stream_answer(self, paper_ids: list[str], question):

        papers = []
        for paper_id in paper_ids:
            paper = self.paper_manager.get_paper(paper_id)
            if paper is not None:
                papers.append(paper)
            
        planner_output = self.planner.create_plan(question=question, papers=papers)

        plan = EvidencePlan(intent=planner_output.intent, user_query=question, target_papers=planner_output.target_papers, retrieval_queries=planner_output.retrieval_queries,)

        if plan.intent == Intent.UNANSWERABLE or not plan.target_papers:

            def unanswerable_stream():

                yield sse_event(
                    StreamEvent.START,
                    {
                        "intent": "unanswerable"
                    }
                )

                yield sse_event(
                    StreamEvent.TOKEN,
                    {
                        "text": "I couldn't find an uploaded paper that can answer this question."
                    }
                )

                yield sse_event(
                    StreamEvent.CITATIONS,
                    []
                )

                yield sse_event(
                    StreamEvent.END,
                    {}
                )

            return unanswerable_stream()

        if plan.intent == Intent.QA:
            return self.qa_executor.stream_execute(plan)

        if plan.intent == Intent.COMPARISON:
            return self.comparison_executor.stream_execute(plan)

        if plan.intent == Intent.SUMMARY:
            return self.summary_executor.stream_execute(plan)

        raise ValueError(f"Unsupported intent: {plan.intent}")