from services.paper_manager import PaperManager
from services.rag import generate_answer, stream_generate_answer
from services.evidence_plan import EvidencePlan
from core.streaming import sse_event, StreamEvent
from exceptions import PaperNotFoundError

class QAExecutor:

    def __init__(self, paper_manager: PaperManager, llm):
        self.paper_manager = paper_manager
        self.llm = llm

    def execute(self, plan: EvidencePlan):

        if not plan.target_papers:
            return {
                "answer": "I couldn't find an uploaded paper that can answer this question.",
                "citations": []
            }

        retriever_pairs = []
        for paper_id in plan.target_papers:
            paper = self.paper_manager.get_paper(paper_id)
            if paper is None:
                raise PaperNotFoundError(f"Paper '{paper_id}' not found")
            retriever_pairs.append([paper["retriever"], paper["bm25_retriever"]])

        question = plan.retrieval_queries[0] if plan.retrieval_queries else plan.user_query
        return generate_answer(
            question=question,
            retriever_pairs=retriever_pairs,
            llm=self.llm
        )
        

    def stream_execute(self, plan: EvidencePlan):

        if not plan.target_papers:
            def empty_stream():
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
            return empty_stream()

        retriever_pairs = []
        for paper_id in plan.target_papers:

            paper = self.paper_manager.get_paper(paper_id)

            if paper is None:
                raise PaperNotFoundError(f"Paper '{paper_id}' not found")

            retriever_pairs.append([paper["retriever"], paper["bm25_retriever"]])

        question = plan.retrieval_queries[0] if plan.retrieval_queries else plan.user_query

        return stream_generate_answer(
            question=question,
            retriever_pairs=retriever_pairs,
            llm=self.llm
        )