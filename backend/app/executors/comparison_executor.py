from services.evidence_plan import EvidencePlan
from exceptions import PaperNotFoundError
from services.hybrid_retriever import hybrid_retriever
from services.citation_builder import build_citations
from services.rag import stream_generate_comparison
from core.streaming import sse_event, StreamEvent
from prompt import COMPARISON_PROMPT

class ComparisonExecutor:

    def __init__(self, paper_manager, llm):
        self.paper_manager = paper_manager
        self.llm = llm

    def execute(self, plan: EvidencePlan):
        if not plan.target_papers:
            return {
                "answer": "I couldn't find relevant uploaded papers to compare.",
                "citations": []
            }

        evidence = {}
        query = plan.retrieval_queries[0] if plan.retrieval_queries else plan.user_query
        
        for paper_id in plan.target_papers:

            paper = self.paper_manager.get_paper(paper_id)

            if paper is None:
                raise PaperNotFoundError(f"Paper with id '{paper_id}' not found.")

            documents = hybrid_retriever(question=query, dense_retriever=paper["retriever"], bm25_retriever=paper["bm25_retriever"])

            evidence[paper_id] = {
                "title": paper["title"],
                "documents": documents
            }


        context = ""

        for paper in evidence.values():
            context += f"Paper: {paper['title']}\n\n"

            for document in paper["documents"]:
                context += (
                    f"[Source: {paper['title']} | Page: {document.metadata['page']}]\n"
                    f"{document.page_content}\n\n"
                )

            context += "-" * 60 + "\n\n"

        print(context)

        prompt = COMPARISON_PROMPT.format(question=plan.user_query, context=context)

        response = self.llm.invoke(prompt)

        all_documents = []

        for paper in evidence.values():
            all_documents.extend(paper["documents"])

        citations = build_citations(all_documents)

        return {
            "answer": response.content,
            "citations": citations
        }
    
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
                        "text": "I couldn't find relevant uploaded papers to compare."
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

        evidence = {}
        query = plan.retrieval_queries[0] if plan.retrieval_queries else plan.user_query

        for paper_id in plan.target_papers:

            paper = self.paper_manager.get_paper(paper_id)

            if paper is None:
                raise PaperNotFoundError(
                    f"Paper with id '{paper_id}' not found."
                )

            documents = hybrid_retriever(
                question=query,
                dense_retriever=paper["retriever"],
                bm25_retriever=paper["bm25_retriever"]
            )

            evidence[paper_id] = {
                "title": paper["title"],
                "documents": documents
            }

        context = ""

        all_documents = []

        for paper in evidence.values():

            context += f"Paper: {paper['title']}\n\n"

            for document in paper["documents"]:

                context += (
                    f"[Source: {paper['title']} | "
                    f"Page: {document.metadata['page']}]\n"
                    f"{document.page_content}\n\n"
                )

                all_documents.append(document)

            context += "-" * 60 + "\n\n"

        citations = build_citations(all_documents)

        return stream_generate_comparison(
            question=plan.user_query,
            context=context,
            llm=self.llm,
            citations=citations
        )