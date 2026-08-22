from dataclasses import dataclass, field
from schemas import SummaryType
from enum import Enum


class Intent(Enum):
    QA = "qa"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    UNANSWERABLE = "unanswerable"


@dataclass
class EvidencePlan:

    intent: Intent

    user_query: str

    target_papers: list[str] = field(default_factory=list)

    retrieval_queries: list[str] = field(default_factory=list)

    summary_type: SummaryType = SummaryType.BRIEF

    max_chunks: int = 4

    aggregate: bool = False