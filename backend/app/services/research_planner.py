from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from services.evidence_plan import Intent


PLANNER_PROMPT = """
You are the research planning component of a research paper assistant.

Your job is to analyze the user's question and create a research plan.

You have access to the following uploaded papers:

{papers}

User question:
{question}

Choose exactly one intent:

1. qa
   - The user wants an answer to a specific question about the uploaded papers.
   - Use this for factual, conceptual, methodological, or explanatory questions relevant to the papers.

2. summary
   - The user wants a summary, overview, key points, or explanation of a paper.

3. comparison
   - The user explicitly wants to compare, contrast, or identify differences/similarities
     between multiple papers or approaches.

4. unanswerable
   - The question cannot reasonably be answered using the uploaded papers (e.g. general knowledge questions, unrelated questions, off-topic queries).

Select the paper IDs that are relevant to answering the question.

Rules for target_papers:
- Select only IDs from the provided papers.
- If the question clearly refers to one paper, select that paper.
- If the question requires information from multiple papers, select all relevant papers.
- If the question is unrelated to all uploaded papers, return an empty list and use intent "unanswerable".
- Do not select papers merely because they are available.

For QA:
- Generate one focused retrieval query that captures the information needed.
- The retrieval query should preserve important technical terms from the user's question.

For comparison:
- Generate a retrieval query that captures the aspects that need to be compared.

For summary:
- Retrieval queries can be empty because the summary executor already operates on the paper contents.

For unanswerable:
- target_papers must be empty.
- retrieval_queries must be empty.

Do not answer the user's question.
Only produce the structured planning output.

Return ONLY a valid JSON object matching this structure:

{{
  "intent": "qa | summary | comparison | unanswerable",
  "target_papers": ["paper_id"],
  "retrieval_queries": ["query"]
}}

Do not return a JSON schema.
Do not return markdown.
Do not include explanations outside the JSON object.
"""

class PlannerOutput(BaseModel):

    intent: Intent

    target_papers: list[str] = Field(
        default_factory=list
    )

    retrieval_queries: list[str] = Field(
        default_factory=list
    )

class ResearchPlanner:

    def __init__(self, llm: BaseChatModel):

        self.llm = llm

        self.structured_llm = llm.with_structured_output(
            PlannerOutput,
            method="json_mode"
        )

    def create_plan(self, question: str, papers: list[dict]) -> PlannerOutput:

        paper_context = "\n\n".join(
            f"Paper ID: {paper['paper_id']}\n"
            f"Title: {paper['title']}"
            f"Abstract: {paper.get('abstract', '')}"
            for paper in papers
        )

        prompt = PLANNER_PROMPT.format(
            papers=paper_context,
            question=question
        )

        return self.structured_llm.invoke(prompt)