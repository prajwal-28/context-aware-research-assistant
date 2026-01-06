"""Service for document ingestion and processing."""
from pathlib import Path
from typing import List
from loguru import logger
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import DocumentChunker
from graph.neo4j_store import Neo4jStore
from graph.entity_extractor import EntityExtractor
from retrieval.vector_store import VectorStore


class DocumentService:
    """
    Coordinates document ingestion: PDF parsing -> chunking -> graph storage -> vector indexing.
    """
    
    def __init__(self, neo4j_store: Neo4jStore, vector_store: VectorStore):
        """
        Initialize document service.
        
        Args:
            neo4j_store: Neo4j store instance
            vector_store: Vector store instance
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.pdf_parser = PDFParser()
        self.chunker = DocumentChunker()
        self.neo4j_store = neo4j_store
        self.entity_extractor = EntityExtractor()
        self.vector_store = vector_store
    
    def ingest_document(self, pdf_path: Path) -> str:
        """
        Ingest a PDF document: parse, chunk, extract entities, store in graph and vector DB.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Document ID
        """
        try:
            self.logger.info(f"Starting ingestion of {pdf_path}")
            
            # Step 1: Parse PDF
            parsed = self.pdf_parser.parse(pdf_path)
            doc_id = f"doc_{pdf_path.stem}"
            
            # Step 2: Create document node in Neo4j
            self.neo4j_store.create_document_node(
                doc_id=doc_id,
                filename=parsed["metadata"]["filename"],
                metadata=parsed["metadata"],
            )
            
            # Step 3: Chunk document
            chunks = self.chunker.chunk_document(
                text=parsed["text"],
                metadata=parsed["metadata"],
            )
            
            # Step 4: Process each chunk
            all_relationships = []
            
            for chunk in chunks:
                chunk_id = chunk.metadata["chunk_id"]
                chunk_text = chunk.text
                
                # Create chunk node in Neo4j
                self.neo4j_store.create_chunk_node(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata=chunk.metadata,
                    doc_id=doc_id,
                )
                
                # Extract entities and relationships
                entities, relationships = self.entity_extractor.extract(
                    text=chunk_text,
                    chunk_id=chunk_id,
                )
                
                # Create entity nodes for this chunk
                if entities:
                    self.neo4j_store.create_entity_nodes(entities, chunk_id)
                
                all_relationships.extend(relationships)
            
            # Step 5: Create relationships (after all entities exist)
            if all_relationships:
                self.neo4j_store.create_relationships(all_relationships)
            
            # Step 7: Add chunks to vector store
            self.vector_store.add_documents(chunks)
            
            self.logger.info(f"Successfully ingested document {doc_id}")
            return doc_id
            
        except Exception as e:
            self.logger.error(f"Error ingesting document {pdf_path}: {e}")
            raise
    
    def ingest_multiple_documents(self, pdf_paths: List[Path]) -> List[str]:
        """Ingest multiple PDF documents."""
        doc_ids = []
        for pdf_path in pdf_paths:
            try:
                doc_id = self.ingest_document(pdf_path)
                doc_ids.append(doc_id)
            except Exception as e:
                self.logger.error(f"Failed to ingest {pdf_path}: {e}")
                continue
        return doc_ids

