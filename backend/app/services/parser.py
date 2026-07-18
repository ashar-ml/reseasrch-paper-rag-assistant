import json
import re
from pathlib import Path
from typing import Any, Dict, List
import pdfplumber
from langchain_core.documents import Document
from loguru import logger

from app.core.config import settings

def get_llm():
    """Initializes and returns the configured LLM client."""
    if settings.LLM_PROVIDER.lower() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            groq_api_key=settings.GROQ_API_KEY,
            model_name=settings.GROQ_MODEL,
            temperature=0.0
        )
    else:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.0
        )

class PaperParser:
    @staticmethod
    def parse_pdf(file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF file page by page using pdfplumber.
        Returns a list of dictionaries with page number and text content.
        """
        pages = []
        logger.info(f"Parsing PDF file: {file_path.name}")
        with pdfplumber.open(file_path) as pdf:
            for idx, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page_number": idx + 1,
                    "text": text
                })
        logger.info(f"Parsed {len(pages)} pages from {file_path.name}.")
        return pages

    @staticmethod
    def extract_metadata(first_page_text: str, filename: str) -> Dict[str, Any]:
        """
        Extracts academic metadata using the LLM with fallback heuristics.
        """
        # Heuristic fallback structure
        fallback_metadata = {
            "title": Path(filename).stem,
            "authors": ["Unknown"],
            "abstract": "Abstract extraction failed.",
            "year": None
        }

        if not first_page_text.strip():
            return fallback_metadata

        # Clean check for default keys
        if settings.GROQ_API_KEY == "gsk_your_api_key_here" and settings.LLM_PROVIDER.lower() == "groq":
            logger.warning("Placeholder Groq API Key found. Skipping LLM metadata extraction, using heuristics.")
            return fallback_metadata

        try:
            llm = get_llm()
            prompt = f"""You are an expert academic assistant. Extract metadata from the following text of the first page of a research paper.
Return the output EXACTLY as a valid JSON object matching the schema below.
DO NOT include any markdown code blocks (e.g. ```json), explanations, or trailing characters. ONLY output the raw JSON object.

Schema:
{{
  "title": "Title of the research paper",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Abstract of the paper (if present)",
  "year": 2024
}}

Text:
---
{first_page_text[:3000]}
---
"""
            response = llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean LLM response to remove markdown wrapping if generated
            content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"^```\s*", "", content, flags=re.MULTILINE)
            content = re.sub(r"\s*```$", "", content, flags=re.MULTILINE)
            content = content.strip()

            metadata = json.loads(content)
            logger.info(f"Successfully extracted metadata using LLM: {metadata.get('title')}")
            
            # Validations on LLM response
            if not metadata.get("title") or "Title of the research paper" in metadata.get("title"):
                metadata["title"] = Path(filename).stem
            if not metadata.get("authors"):
                metadata["authors"] = ["Unknown"]
            return metadata
        except Exception as e:
            logger.warning(f"LLM metadata extraction failed ({e}). Falling back to heuristics.")
            
            # Fallback heuristic: clean title from first few lines
            lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
            heuristic_title = Path(filename).stem
            for line in lines[:4]:
                if len(line) > 15 and not line.startswith(("ISSN", "http", "www", "Journal", "Volume")):
                    heuristic_title = line
                    break

            fallback_metadata["title"] = heuristic_title
            return fallback_metadata

    @staticmethod
    def chunk_paper(pages: List[Dict[str, Any]], metadata: Dict[str, Any], filename: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
        """
        Custom page-aware paragraph chunker. Prepend metadata context to improve retrieval relevance.
        """
        documents = []
        title = metadata.get("title", Path(filename).stem)
        authors = ", ".join(metadata.get("authors", ["Unknown"]))
        year = metadata.get("year", "N/A")

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            
            if not text.strip():
                continue
                
            # Paragraph split
            paragraphs = text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                # If chunk exceeds capacity, save it
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    # Inject prepended context header block
                    prepended_text = f"[Paper: {title} | Authors: {authors} | Page: {page_num}]\n{current_chunk.strip()}"
                    
                    doc = Document(
                        page_content=prepended_text,
                        metadata={
                            "source": filename,
                            "page": page_num,
                            "paper_title": title,
                            "authors": authors,
                            "year": year,
                            "original_text": current_chunk.strip()
                        }
                    )
                    documents.append(doc)
                    
                    # Reset chunk with overlap
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:] + "\n" + para
                else:
                    if current_chunk:
                        current_chunk += "\n" + para
                    else:
                        current_chunk = para
                        
            # Save trailing remainder
            if current_chunk:
                prepended_text = f"[Paper: {title} | Authors: {authors} | Page: {page_num}]\n{current_chunk.strip()}"
                doc = Document(
                    page_content=prepended_text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "paper_title": title,
                        "authors": authors,
                        "year": year,
                        "original_text": current_chunk.strip()
                    }
                )
                documents.append(doc)
                
        logger.info(f"Created {len(documents)} chunks for {filename}.")
        return documents
