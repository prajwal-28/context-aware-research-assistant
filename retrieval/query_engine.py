"""Query engine that synthesizes answers using retrieved context."""
from typing import Dict, List
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from loguru import logger
from config.settings import settings
from .graphrag_retriever import GraphRAGRetriever


class QueryEngine:
    """
    Synthesizes answers using GraphRAG retrieved context.
    Uses LLM to reason over combined vector + graph context.
    """
    
    QUERY_PROMPT = """You are a helpful research assistant that answers questions using provided document context.

Your task:
1. Answer the question using the provided context
2. Explain your reasoning briefly
3. Cite which document sections were used

Context from documents:
{context}

User question: {query}

Provide a comprehensive answer that:
- Directly addresses the question
- Explains the reasoning (2-3 sentences)
- Cites specific document sections/filenames used

Answer:"""

    def __init__(self, retriever: GraphRAGRetriever):
        """
        Initialize query engine.
        
        Args:
            retriever: GraphRAG retriever instance
        """
        self.logger = logger.bind(name=self.__class__.__name__)
        self.retriever = retriever
        
        try:
            self.llm = OpenAI(
                api_key=settings.openai_api_key,
                model="gpt-4o-mini",
                temperature=0.1,  # Low temperature for factual responses
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def _format_context(self, combined_context: List[Dict]) -> str:
        """Format retrieved context for the prompt."""
        formatted_parts = []
        
        for idx, item in enumerate(combined_context, 1):
            source = item.get("source", "unknown")
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            filename = metadata.get("filename", item.get("document_filename", "Unknown"))
            
            if text:
                formatted_parts.append(
                    f"[Source {idx}] From: {filename}\n"
                    f"Retrieval method: {source}\n"
                    f"Content: {text[:500]}...\n"
                )
            elif item.get("name"):
                # For entities without text
                entity_type = item.get("type", "Entity")
                formatted_parts.append(
                    f"[Source {idx}] Entity: {entity_type} - {item.get('name')}\n"
                    f"Retrieval method: {source}\n"
                )
        
        return "\n".join(formatted_parts)
    
    def _extract_sources(self, combined_context: List[Dict]) -> List[Dict]:
        """Extract source citations from context."""
        sources = []
        seen_filenames = set()
        
        for item in combined_context:
            metadata = item.get("metadata", {})
            filename = metadata.get("filename", item.get("document_filename", "Unknown"))
            
            if filename not in seen_filenames and filename != "Unknown":
                seen_filenames.add(filename)
                sources.append({
                    "filename": filename,
                    "source_type": item.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index"),
                })
        
        return sources
    
    def query(self, query: str, top_k: int = 5, max_hops: int = 2) -> Dict:
        """
        Answer a query using GraphRAG retrieval.
        
        Args:
            query: User question
            top_k: Number of initial vector results
            max_hops: Graph traversal depth
            
        Returns:
            Dictionary with 'answer', 'sources', and 'retrieval_info'
        """
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Retrieve context using GraphRAG
            retrieval_result = self.retriever.retrieve(query, top_k=top_k, max_hops=max_hops)
            combined_context = retrieval_result.get("combined_context", [])
            
            if not combined_context:
                return {
                    "answer": "I couldn't find relevant information in the documents to answer this question.",
                    "sources": [],
                    "retrieval_info": retrieval_result,
                }
            
            # Step 2: Format context for LLM
            formatted_context = self._format_context(combined_context)
            
            # Step 3: Generate answer using LLM
            prompt = PromptTemplate(self.QUERY_PROMPT)
            formatted_prompt = prompt.format(context=formatted_context, query=query)
            
            self.logger.info("Generating answer with LLM...")
            response = self.llm.complete(formatted_prompt)
            answer = str(response).strip()
            
            # Step 4: Extract sources
            sources = self._extract_sources(combined_context)
            
            self.logger.info(f"Generated answer with {len(sources)} sources")
            
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_info": {
                    "vector_results_count": len(retrieval_result.get("vector_results", [])),
                    "graph_context_count": len(retrieval_result.get("graph_context", [])),
                    "total_context_items": len(combined_context),
                },
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "retrieval_info": {},
            }







