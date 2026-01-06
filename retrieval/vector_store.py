"""Vector store for embedding-based retrieval."""
from typing import List, Dict, Optional
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, Document as LlamaDocument, StorageContext
from llama_index.core.vector_stores import VectorStoreQuery, VectorStoreQueryResult
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.config import Settings
from loguru import logger
from config.settings import settings
import os


class VectorStore:
    """
    Manages vector embeddings and similarity search.
    Uses ChromaDB for persistent vector storage.
    """
    
    def __init__(self, collection_name: str = "research_documents"):
        """
        Initialize vector store.
        
        Args:
            collection_name: Name of the Chroma collection
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = OpenAIEmbedding(
            api_key=settings.openai_api_key,
            model="text-embedding-3-small"
        )
        
        # Initialize ChromaDB
        persist_dir = "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        try:
            chroma_collection = chroma_client.get_or_create_collection(collection_name)
        except Exception:
            chroma_collection = chroma_client.create_collection(collection_name)
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embedding_model,
        )
        
        self.logger.info(f"Vector store initialized with collection: {collection_name}")
    
    def add_documents(self, documents: List[LlamaDocument]):
        """Add documents to the vector store."""
        try:
            self.logger.info(f"Adding {len(documents)} documents to vector store")
            for doc in documents:
                self.index.insert(doc)
            self.logger.info("Documents added successfully")
        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of documents with scores and metadata
        """
        try:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            nodes = retriever.retrieve(query)
            
            results = []
            for node in nodes:
                results.append({
                    "text": node.text,
                    "metadata": node.metadata,
                    "score": node.score if hasattr(node, 'score') else None,
                    "node_id": node.node_id,
                })
            
            self.logger.info(f"Retrieved {len(results)} results for query")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    def get_chunk_ids_from_results(self, results: List[Dict]) -> List[str]:
        """Extract chunk IDs from search results."""
        chunk_ids = []
        for result in results:
            metadata = result.get("metadata", {})
            chunk_id = metadata.get("chunk_id")
            if chunk_id:
                chunk_ids.append(chunk_id)
        return chunk_ids







