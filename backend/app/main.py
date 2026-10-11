import sys
import os

sys.modules.setdefault("tensorflow", None)
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import uuid
import tempfile
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from schemas import ChatRequest
from contextlib import asynccontextmanager
from exceptions import PDFExtractionError, VectorStoreCreationError
from services.paper_manager import PaperManager
from services.pdf_utils import extract_text, extract_abstract
from services.chunking import chunk_text
from services.vector_store import create_vectorstore, get_retriever
from services.embeddings import get_embedding_model
from services.llm import get_llm
from services.research_service import ResearchService


@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.embedding_model = get_embedding_model()
    app.state.llm = get_llm()

    app.state.paper_manager = PaperManager()
    app.state.paper_manager.load_saved_papers(app.state.embedding_model)

    app.state.research_service = ResearchService(paper_manager=app.state.paper_manager, llm=app.state.llm)

    yield

    


app = FastAPI(lifespan=lifespan)

allowed_origins_env = os.getenv("ALLOWED_ORIGINS")
allowed_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if allowed_origins_env
    else ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(PDFExtractionError)
async def pdf_exception_handler(
    request: Request,
    exc: PDFExtractionError
):

    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc)
        }
    )



@app.exception_handler(VectorStoreCreationError)
async def vectorstore_exception_handler(
    request: Request,
    exc: VectorStoreCreationError
):

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc)
        }
    )


@app.get("/")
async def root():
    return {
        "message": "Research Assistant API is running"
    }


@app.post("/upload")
async def upload_pdf(files: Annotated[list[UploadFile], File(...)]):
    uploaded_papers = []
    for file in files:
        if (file.content_type != "application/pdf" or not file.filename.lower().endswith(".pdf")):
            raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )
    
        temp_path = None 

        try: 
            contents = await file.read()
    
            with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
            ) as temp_file:
        
                temp_file.write(contents)

                temp_path = temp_file.name
        
            page_texts = extract_text(temp_path)

            abstract = extract_abstract(page_texts)

            paper_title = file.filename

            documents = chunk_text(page_texts=page_texts, source_name=file.filename, paper_title=paper_title)


            vectorstore = create_vectorstore(documents=documents, embedding_model=app.state.embedding_model)

            retriever = get_retriever(vectorstore)

            paper_id = str(uuid.uuid4())

            app.state.paper_manager.add_paper(
                paper_id=paper_id,
                title= paper_title,
                abstract = abstract,
                vectorstore=vectorstore,
                retriever=retriever
            )

            uploaded_papers.append({
                "paper_id": paper_id,
                "title": paper_title
            })
    
        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    return {
                "message": "PDF(s) uploaded Successfully",
                "papers": uploaded_papers
                }

@app.get("/papers")
async def get_papers():

    return {
        "papers": app.state.paper_manager.list_papers()
    }

@app.post("/research")
async def chat(request: ChatRequest):

    for paper_id in request.paper_ids:
        if app.state.paper_manager.get_paper(paper_id) is None:
            raise HTTPException(
                status_code=400,
                detail=f"Paper '{paper_id}' not found"
            )

    return StreamingResponse(app.state.research_service.stream_answer(
        paper_ids=request.paper_ids,
        question=request.question
    ),
    media_type="text/event-stream")


@app.delete("/papers/{paper_id}")
async def delete_paper(paper_id: str):

    paper = app.state.paper_manager.delete_paper(paper_id)

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail=f"Paper '{paper_id}' not found."
        )

    return {
        "message": "Paper deleted successfully.",
        "paper_id": paper["paper_id"],
        "title": paper["title"]
    }