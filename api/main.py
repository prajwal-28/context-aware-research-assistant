"""FastAPI application."""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
from pathlib import Path
import shutil
import os
from loguru import logger

from config.settings import settings  # This will trigger logger setup via config.__init__
from graph.neo4j_store import Neo4jStore
from retrieval.vector_store import VectorStore
from retrieval.graphrag_retriever import GraphRAGRetriever
from retrieval.query_engine import QueryEngine
from services.document_service import DocumentService
from services.query_service import QueryService

# Initialize FastAPI app
app = FastAPI(
    title="Context-Aware Research Assistant API",
    description="API for PDF document ingestion and graph-based querying",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components (initialized on startup)
neo4j_store: Neo4jStore = None
vector_store: VectorStore = None
document_service: DocumentService = None
query_service: QueryService = None

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Pydantic models
class QueryRequest(BaseModel):
    """Query request model."""
    query: str
    top_k: Optional[int] = 5
    max_hops: Optional[int] = 2


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global neo4j_store, vector_store, document_service, query_service
    
    logger.info("Initializing services...")
    
    try:
        # Initialize stores
        neo4j_store = Neo4jStore()
        vector_store = VectorStore()
        
        # Initialize services
        document_service = DocumentService(neo4j_store, vector_store)
        
        # Initialize retrieval components
        retriever = GraphRAGRetriever(vector_store, neo4j_store)
        query_engine = QueryEngine(retriever)
        query_service = QueryService(retriever, query_engine)
        
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global neo4j_store
    if neo4j_store:
        neo4j_store.close()
    logger.info("Services shut down")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Context-Aware Research Assistant API"}


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest PDF documents.
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        List of ingested document IDs
    """
    if not document_service:
        raise HTTPException(status_code=500, detail="Document service not initialized")
    
    doc_ids = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} is not a PDF"
            )
        
        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Ingest document
            doc_id = document_service.ingest_document(file_path)
            doc_ids.append(doc_id)
            
            logger.info(f"Successfully uploaded and ingested: {file.filename}")
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing {file.filename}: {str(e)}"
            )
    
    return {"document_ids": doc_ids, "message": f"Successfully ingested {len(doc_ids)} document(s)"}


@app.post("/api/query")
async def query_documents(request: QueryRequest):
    """
    Process a query and return answer.
    
    Args:
        request: Query request with query text and optional parameters
        
    Returns:
        Query result with answer, sources, and retrieval information
    """
    if not query_service:
        raise HTTPException(status_code=500, detail="Query service not initialized")
    
    try:
        result = query_service.process_query(
            request.query, 
            top_k=request.top_k, 
            max_hops=request.max_hops
        )
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    health = {
        "status": "ok",
        "neo4j": "unknown",
        "vector_store": "unknown",
    }
    
    try:
        if neo4j_store:
            neo4j_store.driver.verify_connectivity()
            health["neo4j"] = "connected"
    except Exception as e:
        health["neo4j"] = f"error: {str(e)}"
    
    if vector_store:
        health["vector_store"] = "initialized"
    
    return health


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )

