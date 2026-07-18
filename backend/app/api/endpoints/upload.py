import shutil
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger

from app.core.config import settings
from app.services.parser import PaperParser
from app.services.vectorstore import VectorStoreService

router = APIRouter()

@router.get("")
async def list_uploaded_pdfs():
    """
    Endpoint to list all uploaded research paper PDFs and their basic metadata (e.g. size).
    """
    logger.info("Listing uploaded PDFs...")
    try:
        temp_dir = settings.raw_data_path
        if not temp_dir.exists():
            return []
        
        papers = []
        for file_path in temp_dir.glob("*.pdf"):
            papers.append({
                "filename": file_path.name,
                "size_kb": round(file_path.stat().st_size / 1024, 2)
            })
        return papers
    except Exception as e:
        logger.error(f"Failed to list PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list PDFs: {str(e)}")

@router.post("")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a research paper PDF.
    Triggers parsing, metadata extraction, semantic chunking, and vector indexing.
    """
    logger.info(f"Received PDF file: {file.filename}")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # Save the file temporarily in the raw directory
    temp_dir = settings.raw_data_path
    file_path = temp_dir / file.filename
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 1. Parse pages from PDF
        pages = PaperParser.parse_pdf(file_path)
        if not pages:
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")
            
        # 2. Extract metadata from first page
        first_page_text = pages[0]["text"]
        metadata = PaperParser.extract_metadata(first_page_text, file.filename)
        
        # 3. Create semantic context chunks
        chunks = PaperParser.chunk_paper(pages, metadata, file.filename)
        
        # 4. Insert chunks into ChromaDB
        VectorStoreService.add_documents(chunks)
        
        return {
            "status": "success",
            "message": f"Successfully processed and indexed '{file.filename}'.",
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
        
    except Exception as e:
        logger.error(f"Failed to ingest PDF upload: {e}")
        raise HTTPException(status_code=500, detail=f"PDF Ingestion failed: {str(e)}")
