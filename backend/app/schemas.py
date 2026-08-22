from pydantic import BaseModel
from enum import Enum

class ChatRequest(BaseModel):
    paper_ids: list[str]
    question: str

class UploadResponse(BaseModel):
    message: str

class Citation(BaseModel):
    title: str
    page: int

class ResearchResponse(BaseModel):
    intent: str
    result: dict

class SummaryType(str, Enum):
    BRIEF = "brief"
    DETAILED = "detailed"
    STUDENT = "student"
    TECHNICAL = "technical"
    BULLET = "bullet"

class PaperSummary(BaseModel):
    paper_id: str
    title: str
    answer: str
    citations: list[Citation]