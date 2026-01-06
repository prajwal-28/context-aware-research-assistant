"""Document chunking functionality."""
from typing import List, Dict
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from loguru import logger
from config.settings import settings


class DocumentChunker:
    """
    Chunks documents into semantically meaningful pieces.
    Uses semantic splitting to preserve context across boundaries.
    """
    
    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 200):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model for semantic splitting
        try:
            self.embedding_model = OpenAIEmbedding(
                api_key=settings.openai_api_key,
                model="text-embedding-3-small"
            )
            
            # Use semantic splitter for better context preservation
            self.node_parser = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embedding_model,
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize semantic splitter: {e}. Using simple splitter.")
            from llama_index.core.node_parser import SimpleNodeParser
            self.node_parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
    
    def chunk_document(self, text: str, metadata: Dict) -> List[LlamaDocument]:
        """
        Chunk a document into semantically meaningful pieces.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of LlamaDocument chunks with metadata
        """
        try:
            self.logger.info(f"Chunking document: {metadata.get('filename', 'unknown')}")
            
            # Create LlamaIndex document
            llama_doc = LlamaDocument(
                text=text,
                metadata=metadata,
            )
            
            # Parse into nodes
            nodes = self.node_parser.get_nodes_from_documents([llama_doc])
            
            # Convert nodes back to documents with enriched metadata
            chunks = []
            for idx, node in enumerate(nodes):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": f"{metadata.get('filename', 'doc')}_chunk_{idx}",
                    "chunk_index": idx,
                    "total_chunks": len(nodes),
                }
                
                chunk_doc = LlamaDocument(
                    text=node.text,
                    metadata=chunk_metadata,
                )
                chunks.append(chunk_doc)
            
            self.logger.info(f"Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error chunking document: {str(e)}")
            raise







