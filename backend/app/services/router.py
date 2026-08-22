from services.evidence_plan import EvidencePlan, Intent
from services.paper_manager import PaperManager


class ResearchRouter:

    def __init__(self, paper_manager: PaperManager):
        self.paper_manager = paper_manager

    def create_plan(self, question: str, paper_ids: list[str]) -> EvidencePlan:

        question_lower = question.lower()

        matched_papers = self.paper_manager.find_target_papers(question, paper_ids)

        target_papers = matched_papers if matched_papers else paper_ids

        if any(word in question_lower for word in [
            "summarize",
            "summary",
            "overview"
        ]):
            return EvidencePlan(
                intent=Intent.SUMMARY,
                user_query=question,
                target_papers=target_papers,
                retrieval_queries=[],
                max_chunks=0,
                aggregate=False
            )

        if any(word in question_lower for word in [
            "compare",
            "difference",
            "versus",
            "vs"
        ]):
            return EvidencePlan(
                intent=Intent.COMPARISON,
                user_query=question,
                target_papers=target_papers,
                retrieval_queries=[question],
                max_chunks=6,
                aggregate=True,
            )

        return EvidencePlan(
            intent=Intent.QA,
            user_query=question,
            target_papers=target_papers,
            retrieval_queries=[question],
            max_chunks=4,
            aggregate=False
        )