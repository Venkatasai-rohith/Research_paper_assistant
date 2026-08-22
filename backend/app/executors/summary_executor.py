from services.summarizer import (
    get_documents_from_vectorstore,
    filter_documents,
    group_documents,
    generate_partial_summaries,
    stream_single_group_summary,
    stream_combine_partial_summaries,
)
from services.citation_builder import build_citations
from core.config import SUMMARY_GROUP_WORDS
from core.streaming import sse_event, StreamEvent
from exceptions import PaperNotFoundError


class SummaryExecutor:

    def __init__(self, paper_manager, llm):
        self.paper_manager = paper_manager
        self.llm = llm

    def stream_execute(self, plan):
        if not plan.target_papers:
            yield sse_event(
                StreamEvent.START,
                {
                    "intent": "unanswerable"
                }
            )
            yield sse_event(
                StreamEvent.TOKEN,
                {
                    "text": "I couldn't find an uploaded paper to summarize."
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
            return

        all_citations = []

        yield sse_event(
            StreamEvent.START,
            {
                "intent": "summary"
            }
        )

        for paper_id in plan.target_papers:

            paper = self.paper_manager.get_paper(paper_id)

            if paper is None:
                raise PaperNotFoundError(
                    f"Paper '{paper_id}' not found."
                )

            documents = get_documents_from_vectorstore(
                paper["vectorstore"]
            )

            documents = filter_documents(documents)

            groups = group_documents(
                documents,
                SUMMARY_GROUP_WORDS
            )

            # Tell the frontend the paper being summarized
            yield sse_event(
                StreamEvent.TOKEN,
                {
                    "text": f"### {paper['title']}\n\n"
                }
            )

            citations = build_citations(documents)
            all_citations.extend(citations)

            if len(groups) <= 1:
                # Fast path: direct streaming for papers that fit in a single group
                target_group = groups[0] if groups else documents
                for chunk in stream_single_group_summary(
                    target_group,
                    self.llm,
                    plan.summary_type
                ):
                    yield sse_event(
                        StreamEvent.TOKEN,
                        {
                            "text": chunk
                        }
                    )
            else:
                # Multi-group path: parallel map across all groups, then stream reduce
                partial_summaries = generate_partial_summaries(
                    groups,
                    self.llm,
                    plan.summary_type
                )

                for chunk in stream_combine_partial_summaries(
                    partial_summaries,
                    self.llm,
                    plan.summary_type
                ):
                    yield sse_event(
                        StreamEvent.TOKEN,
                        {
                            "text": chunk
                        }
                    )

            yield sse_event(
                StreamEvent.TOKEN,
                {
                    "text": "\n\n"
                }
            )

        yield sse_event(
            StreamEvent.CITATIONS,
            all_citations
        )

        yield sse_event(
            StreamEvent.END,
            {}
        )