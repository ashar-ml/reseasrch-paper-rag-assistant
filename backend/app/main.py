# Monkey patch for numpy compatibility with older scipy/sentence-transformers versions
import numpy as np
np.long = int
np.ulong = int

import asyncio
import shutil
from pathlib import Path
from typing import Any, Dict, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
import uvicorn

from app.core.config import settings
from app.api.router import api_router
from app.services.parser import PaperParser
from app.services.vectorstore import VectorStoreService
from app.agent.graph import app_graph

app = FastAPI(
    title="Research Paper RAG Assistant API",
    description="Backend services for parsing papers, hybrid retrieval, agentic QA with citations, and evaluation.",
    version="1.0.0"
)

# CORS configuration for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust/restrict for production deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------
# Pydantic Schemas
# -----------------
class AskRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=500, 
        description="Query text to search and answer from research papers.",
        example="What is self-attention?"
    )

class CitationSchema(BaseModel):
    index: int = Field(..., description="Citation source index index.")
    paper_title: str = Field(..., description="Title of the research paper.")
    page: int = Field(..., description="Page number of the paper chunk.")
    text_snippet: str = Field(..., description="Original cited text snippet.")

class AskResponse(BaseModel):
    query: str = Field(..., description="Original user query.")
    answer: str = Field(..., description="Generated answer with index numbers.")
    citations: List[CitationSchema] = Field(..., description="List of source citations.")

# -----------------
# Endpoints
# -----------------

@app.get("/health", tags=["Status"])
async def health_check():
    """Health status check endpoint."""
    return {
        "status": "healthy",
        "provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL_NAME
    }

@app.post("/upload-pdf", tags=["Ingestion"])
async def upload_pdf(file: UploadFile = File(...)):
    """
    Handles PDF uploading, parsing, metadata extraction, paragraph chunking, and database indexing.
    Runs blocking CPU-bound parsing in a background thread to keep the event loop responsive.
    """
    logger.info(f"Root API received PDF upload: {file.filename}")
    
    # 1. Validation
    if not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Validation failed: File '{file.filename}' is not a PDF.")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_dir = settings.raw_data_path
    file_path = temp_dir / file.filename
    
    try:
        # Save uploaded bytes to file in threadpool
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Async Execution for CPU-bound parser
        logger.info(f"Starting non-blocking PDF parse for: {file.filename}")
        pages = await asyncio.to_thread(PaperParser.parse_pdf, file_path)
        
        if not pages:
            logger.error(f"Ingestion failed: No readable text extracted from {file.filename}.")
            raise HTTPException(status_code=422, detail="Unable to extract text layers from the PDF document.")
            
        # 3. Extract Metadata
        first_page_text = pages[0]["text"]
        metadata = await asyncio.to_thread(PaperParser.extract_metadata, first_page_text, file.filename)
        
        # 4. Generate Semantic Chunks
        chunks = await asyncio.to_thread(PaperParser.chunk_paper, pages, metadata, file.filename)
        
        # 5. Insert documents in Chroma DB
        await asyncio.to_thread(VectorStoreService.add_documents, chunks)
        
        logger.info(f"Ingestion successful for '{file.filename}'. Added {len(chunks)} chunks.")
        return {
            "status": "success",
            "message": f"Successfully parsed and indexed '{file.filename}'.",
            "metadata": {
                "title": metadata.get("title"),
                "authors": metadata.get("authors"),
                "abstract": metadata.get("abstract")[:200] + "..." if metadata.get("abstract") else "N/A",
                "year": metadata.get("year")
            },
            "stats": {
                "total_pages": len(pages),
                "total_chunks": len(chunks)
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error processing PDF '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"PDF ingestion process failed: {str(e)}")

@app.post("/ask", response_model=AskResponse, tags=["Agentic RAG"])
async def ask_assistant(request: AskRequest):
    """
    Queries the Agentic RAG workflow using LangGraph state graph.
    Invokes nodes and routing paths asynchronously.
    """
    logger.info(f"Root API received RAG query: '{request.query}'")
    
    # 1. State Context Initialization
    initial_state = {
        "query": request.query,
        "original_query": request.query,
        "documents": [],
        "answer": "",
        "citations": [],
        "rewrite_count": 0
    }
    
    try:
        # 2. Async Graph Execution
        logger.info("Invoking LangGraph CRAG flow asynchronously...")
        final_state = await app_graph.ainvoke(initial_state)
        
        # 3. Response validation check & output formatting
        return AskResponse(
            query=request.query,
            answer=final_state.get("answer", "No answer could be generated."),
            citations=final_state.get("citations", [])
        )
    except Exception as e:
        logger.error(f"LangGraph execution crashed for query '{request.query}': {e}")
        raise HTTPException(status_code=500, detail=f"Agent RAG workflow crashed: {str(e)}")

# Include legacy routers for backward compatibility
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)
