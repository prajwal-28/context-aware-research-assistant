"""GraphRAG-style retrieval combining vector search and graph traversal."""
from typing import List, Dict
from loguru import logger
from .vector_store import VectorStore
from graph.neo4j_store import Neo4jStore


class GraphRAGRetriever:
    """
    Implements GraphRAG retrieval pipeline:
    1. Vector similarity search to find initial relevant chunks
    2. Graph traversal to expand context using relationships
    3. Combines both for comprehensive retrieval
    """
    
    def __init__(self, vector_store: VectorStore, neo4j_store: Neo4jStore):
        """
        Initialize GraphRAG retriever.
        
        Args:
            vector_store: Vector store for similarity search
            neo4j_store: Neo4j store for graph traversal
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.vector_store = vector_store
        self.neo4j_store = neo4j_store
    
    def retrieve(self, query: str, top_k: int = 5, max_hops: int = 2) -> Dict:
        """
        Perform GraphRAG retrieval: vector search + graph traversal.
        
        This method clearly separates:
        - Vector retrieval (semantic similarity)
        - Graph traversal (relationship-based expansion)
        - Result combination
        
        Args:
            query: User query
            top_k: Number of initial vector results
            max_hops: Maximum graph traversal hops
            
        Returns:
            Dictionary with 'vector_results' and 'graph_context'
        """
        # Step 1: Vector similarity search
        # This finds chunks semantically similar to the query
        self.logger.info(f"Step 1: Vector similarity search for query: {query}")
        vector_results = self.vector_store.similarity_search(query, top_k=top_k)
        
        if not vector_results:
            self.logger.warning("No vector results found")
            return {
                "vector_results": [],
                "graph_context": [],
                "combined_context": [],
            }
        
        # Step 2: Extract chunk IDs from vector results
        chunk_ids = self.vector_store.get_chunk_ids_from_results(vector_results)
        self.logger.info(f"Found {len(chunk_ids)} relevant chunks: {chunk_ids}")
        
        # Step 3: Graph traversal (multi-hop exploration)
        # This expands context by following relationships in the graph
        self.logger.info(f"Step 2: Graph traversal from {len(chunk_ids)} chunks (max {max_hops} hops)")
        graph_context = self.neo4j_store.traverse_from_chunks(chunk_ids, max_hops=max_hops)
        self.logger.info(f"Found {len(graph_context)} related nodes via graph traversal")
        
        # Step 4: Get full chunk texts for graph nodes that reference chunks
        graph_chunk_ids = [
            item["id"] for item in graph_context 
            if "Chunk" in item.get("labels", [])
        ]
        if graph_chunk_ids:
            graph_chunks = self.neo4j_store.get_chunks_by_ids(graph_chunk_ids)
        else:
            graph_chunks = []
        
        # Step 5: Combine results
        # Vector results are the primary context, graph provides expansion
        combined_context = []
        
        # Add vector results (primary)
        for result in vector_results:
            combined_context.append({
                "source": "vector",
                "text": result["text"],
                "metadata": result["metadata"],
                "score": result.get("score"),
            })
        
        # Add graph chunks (expansion)
        for chunk in graph_chunks:
            # Avoid duplicates
            if chunk["id"] not in chunk_ids:
                combined_context.append({
                    "source": "graph",
                    "text": chunk.get("text", ""),
                    "metadata": chunk.get("metadata", {}),
                    "document_filename": chunk.get("document_filename"),
                })
        
        # Add other graph entities (Policy, Topic, etc.)
        for item in graph_context:
            if "Chunk" not in item.get("labels", []):
                combined_context.append({
                    "source": "graph_entity",
                    "type": item.get("labels", [])[0] if item.get("labels") else "Unknown",
                    "name": item.get("name", ""),
                    "metadata": item.get("metadata", {}),
                    "text": item.get("text", ""),
                })
        
        self.logger.info(f"Retrieved {len(combined_context)} total context items")
        
        return {
            "vector_results": vector_results,
            "graph_context": graph_context,
            "combined_context": combined_context,
        }







