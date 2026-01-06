"""PDF parsing functionality."""
from pathlib import Path
from typing import List, Dict
from loguru import logger
import pymupdf  # PyMuPDF (fitz)


class PDFParser:
    """Parses PDF documents into text with metadata."""
    
    def __init__(self):
        """Initialize PDF parser."""
        self.logger = logger.bind(name=self.__class__.__name__)
    
    def parse(self, pdf_path: Path) -> Dict:
        """
        Parse a PDF file and extract text with metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with 'text', 'metadata', and 'pages' information
        """
        try:
            self.logger.info(f"Parsing PDF: {pdf_path}")
            doc = pymupdf.open(pdf_path)
            
            text_parts = []
            pages_info = []
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
                    pages_info.append({
                        "page_number": page_num,
                        "text": text,
                    })
            
            full_text = "\n\n".join(text_parts)
            metadata = {
                "filename": pdf_path.name,
                "file_path": str(pdf_path),
                "total_pages": len(pages_info),
            }
            
            doc.close()
            
            self.logger.info(f"Successfully parsed {len(pages_info)} pages from {pdf_path.name}")
            
            return {
                "text": full_text,
                "metadata": metadata,
                "pages": pages_info,
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {pdf_path}: {str(e)}")
            raise







