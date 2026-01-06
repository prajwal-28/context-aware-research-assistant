"""Service for processing queries."""
from typing import Dict
from loguru import logger
from retrieval.graphrag_retriever import GraphRAGRetriever
from retrieval.query_engine import QueryEngine


class QueryService:
    """Service for processing user queries."""
    
    def __init__(self, retriever: GraphRAGRetriever, query_engine: QueryEngine):
        """
        Initialize query service.
        
        Args:
            retriever: GraphRAG retriever instance
            query_engine: Query engine instance
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.query_engine = query_engine
    
    def process_query(self, query: str, top_k: int = 5, max_hops: int = 2) -> Dict:
        """
        Process a user query and return answer.
        
        Args:
            query: User question
            top_k: Number of vector results
            max_hops: Graph traversal depth
            
        Returns:
            Query result with answer and sources
        """
        return self.query_engine.query(query, top_k=top_k, max_hops=max_hops)







