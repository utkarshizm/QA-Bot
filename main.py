from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import os
from datetime import datetime

from rag_pipeline import RAGPipeline

app = FastAPI(
    title="AI Document Q&A — RAG Pipeline API",
    description="Upload PDFs, ask questions, get grounded answers with citations.",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc"     # ReDoc at /redoc
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Init RAG pipeline
rag = RAGPipeline(api_key=os.getenv("GEMINI_API_KEY"))

class QuestionRequest(BaseModel):
    question: str
    k: Optional[int] = 3
    use_reranker: Optional[bool] = False

class AnswerResponse(BaseModel):
    answer: str
    sources: List[dict]
    chunks_retrieved: int
    processing_time_ms: float

@app.post("/upload", summary="Upload PDF documents")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file to build the knowledge base."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files allowed")
    
    content = await file.read()
    rag.index_document(content, filename=file.filename)
    return {"status": "indexed", "filename": file.filename}

@app.post("/ask", response_model=AnswerResponse, summary="Ask a question")
async def ask_question(req: QuestionRequest):
    """Ask a question and get an answer grounded in your documents."""
    start = datetime.now()
    
    answer, sources = rag.query(
        question=req.question,
        k=req.k,
        rerank=req.use_reranker
    )
    
    elapsed = (datetime.now() - start).total_seconds() * 1000
    
    return AnswerResponse(
        answer=answer,
        sources=sources,
        chunks_retrieved=len(sources),
        processing_time_ms=round(elapsed, 2)
    )

@app.get("/health", summary="Health check")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.delete("/clear", summary="Clear knowledge base")
async def clear_kb():
    rag.clear()
    return {"status": "cleared"}
