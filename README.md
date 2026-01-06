# Context-Aware Research Assistant

A Python-based research assistant that demonstrates **GraphRAG** (Graph-based Retrieval-Augmented Generation) by combining vector embeddings with knowledge graph traversal. This system ingests PDF documents, extracts entities and relationships, stores them in Neo4j, and answers questions using hybrid retrieval that goes beyond simple semantic search.

## Project Overview

Traditional RAG (Retrieval-Augmented Generation) systems rely solely on vector similarity search to find relevant document chunks. While effective for direct queries, they struggle with complex questions that require understanding relationships across documents—such as "How does policy X affect topic Y?" or "What are the connections between concepts mentioned in different sections?"

**GraphRAG** addresses this limitation by building a knowledge graph that captures not just document content, but also the semantic relationships between entities (policies, topics, concepts, sections). When answering queries, the system first finds relevant chunks via vector search, then traverses the graph to discover related entities and expand context. This multi-hop reasoning enables cross-document understanding that pure vector search cannot achieve.

This project uses **Neo4j** as the graph database because it provides native graph traversal capabilities, relationship querying, and efficient multi-hop exploration. The knowledge graph structure allows the system to answer questions that require understanding how different concepts relate to each other across multiple documents, making it particularly valuable for research scenarios involving policy documents, technical specifications, or interconnected knowledge domains.

## Key Features

- **Multi-PDF Ingestion**: Upload and process multiple PDF documents simultaneously
- **Semantic Chunking**: Intelligent document segmentation that preserves context across boundaries
- **Knowledge Graph Construction**: Automatic extraction of entities (Documents, Policies, Sections, Topics, Concepts) and their relationships, stored in Neo4j
- **Hybrid Retrieval**: Combines vector similarity search (semantic matching) with graph traversal (relationship-based expansion)
- **Cross-Document Reasoning**: Answers questions that require understanding connections across multiple documents
- **Source-Aware Answers**: Every answer includes citations showing which document sections were used
- **GraphRAG Pipeline**: Clear separation between vector retrieval and graph traversal for transparency and debugging

## System Architecture

The system follows a clean, modular architecture that separates concerns and enables clear understanding of the GraphRAG flow:

### Ingestion Pipeline

1. **PDF Parsing** (`ingestion/pdf_parser.py`): Extracts text and metadata from PDF files
2. **Semantic Chunking** (`ingestion/chunker.py`): Splits documents into semantically meaningful chunks using LlamaIndex's semantic splitter
3. **Vector Embedding** (`retrieval/vector_store.py`): Generates embeddings for each chunk and stores them in ChromaDB for similarity search
4. **Entity Extraction** (`graph/entity_extractor.py`): Uses LLM to identify entities (Policy, Section, Topic, Concept) and relationships from each chunk
5. **Graph Storage** (`graph/neo4j_store.py`): Stores entities as nodes and relationships as edges in Neo4j, creating a knowledge graph

### Query Pipeline (GraphRAG)

When a user asks a question, the system executes a three-phase retrieval process:

1. **Vector Similarity Search** (`retrieval/graphrag_retriever.py`):
   - Uses semantic embeddings to find chunks most similar to the query
   - Returns top-k most relevant chunks (configurable, default: 5)

2. **Multi-Hop Graph Traversal** (`graph/neo4j_store.py`):
   - Starts from the vector-retrieved chunks
   - Traverses relationships in Neo4j (e.g., Policy → AFFECTS → Topic, Concept → RELATES_TO → Section)
   - Expands context by following connections up to N hops (configurable, default: 2)
   - Discovers related entities that weren't in the initial vector results

3. **LLM-Based Answer Synthesis** (`retrieval/query_engine.py`):
   - Combines vector results (primary context) with graph results (expanded context)
   - Formats the combined context for the LLM
   - Generates an answer with reasoning and source citations

This architecture clearly separates vector retrieval from graph traversal, making it easy to understand, debug, and tune each component independently.

## Tech Stack

- **Python 3.9+**: Core language
- **FastAPI**: REST API backend for document ingestion and querying
- **Streamlit**: Web UI for document upload and query interface
- **LlamaIndex**: Framework for document processing, embeddings, and LLM integration
- **Neo4j**: Graph database for storing entities and relationships
- **ChromaDB**: Vector database for semantic similarity search
- **OpenAI API**: Provides embeddings (`text-embedding-3-small`) and LLM reasoning (`gpt-4o-mini`)

## Project Structure

```
context-aware-research-assistant/
├── api/                    # FastAPI backend endpoints
│   └── main.py             # REST API for upload and query
├── config/                 # Configuration and logging
│   ├── settings.py         # Environment-based settings
│   └── logger.py           # Logging configuration
├── graph/                  # Neo4j graph operations
│   ├── neo4j_store.py     # Graph database operations
│   └── entity_extractor.py # LLM-based entity extraction
├── ingestion/              # PDF parsing and chunking
│   ├── pdf_parser.py      # PDF text extraction
│   └── chunker.py         # Semantic document chunking
├── retrieval/              # Vector store and GraphRAG
│   ├── vector_store.py    # ChromaDB vector operations
│   ├── graphrag_retriever.py # Hybrid retrieval logic
│   └── query_engine.py    # LLM-based answer synthesis
├── services/               # Business logic coordination
│   ├── document_service.py # Document ingestion orchestration
│   └── query_service.py    # Query processing orchestration
├── ui/                     # Streamlit frontend
│   └── app.py             # Web UI for upload and query
├── main.py                 # API entry point
├── run_api.py             # Convenience script to run API
├── run_ui.py            # Convenience script to run UI
├── requirements.txt       # Python dependencies
├── env.example            # Environment variables template
└── README.md              # This file
```

## Local Setup

### Prerequisites

- Python 3.9 or higher
- Neo4j database (local installation or remote instance)
- OpenAI API key with credits

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd context-aware-research-assistant
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Neo4j**:
   - Install Neo4j: https://neo4j.com/download/
   - Start Neo4j: `neo4j start` (or use Docker)
   - Default connection: `bolt://localhost:7687`
   - Note your username and password

5. **Configure environment variables**:
   ```bash
   cp env.example .env
   ```
   
   Edit `.env` with your credentials:
   ```
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. **Start the FastAPI backend** (Terminal 1):
   ```bash
   python run_api.py
   ```
   Backend runs at http://localhost:8000

7. **Start the Streamlit UI** (Terminal 2):
   ```bash
   python run_ui.py
   ```
   UI runs at http://localhost:8501

### Usage

1. **Upload Documents**: Use the Streamlit UI to upload PDF files. The system will parse, chunk, extract entities, and build the knowledge graph.
2. **Query Documents**: Ask questions in natural language. The system uses GraphRAG to find relevant content and generate answers with source citations.

## Deployment Readiness

This project is structured to be **deployment-ready**, with clear separation of backend and frontend services and environment-based configuration. The architecture supports deployment on platforms like Railway or Render, but deployment was intentionally left optional to avoid exposing API credentials.

### Deployment Considerations

- **Backend (FastAPI)**: The API is built with FastAPI and can be deployed as a standalone service. It uses environment variables for all configuration, making it suitable for cloud deployment.
- **Frontend (Streamlit)**: The Streamlit UI can be deployed separately or alongside the backend. It communicates with the API via HTTP requests.
- **Database Requirements**: Neo4j can be deployed separately (Neo4j Aura, self-hosted, or Docker) and accessed via connection string.
- **Environment Configuration**: All secrets (API keys, database credentials) are loaded from environment variables, following security best practices.

**Note**: This project demonstrates deployment-ready architecture but is not deployed by default. To deploy, you would need to:
1. Set up Neo4j instance (cloud or self-hosted)
2. Configure environment variables in your deployment platform
3. Deploy FastAPI backend as a web service
4. Optionally deploy Streamlit UI as a separate service

## Important Note on API Usage

**OpenAI API Credits Required**: This system uses OpenAI's API for two purposes:
- **Embeddings**: Generating vector embeddings for semantic search (`text-embedding-3-small`)
- **LLM Reasoning**: Entity extraction and answer generation (`gpt-4o-mini`)

Without valid API credentials and credits, the pipeline will not execute. API keys are **never hardcoded** for security reasons—they must be provided via environment variables. The system design is complete and functional, but requires API access to operate.

Cost considerations:
- Embeddings: ~$0.02 per 1M tokens
- LLM (gpt-4o-mini): ~$0.15 per 1M input tokens, $0.60 per 1M output tokens
- Processing 10 PDFs (~100 pages) typically costs $1-5 depending on document complexity

## API Endpoints

### `POST /api/upload`
Upload and process PDF documents.

**Request**: Multipart form data with PDF files

**Response**:
```json
{
  "document_ids": ["doc_1", "doc_2"],
  "message": "Successfully ingested 2 document(s)"
}
```

### `POST /api/query`
Process a natural language query using GraphRAG.

**Request**:
```json
{
  "query": "How does maternity leave policy affect project deadlines?",
  "top_k": 5,
  "max_hops": 2
}
```

**Response**:
```json
{
  "answer": "Based on the documents...",
  "sources": [
    {
      "filename": "policy.pdf",
      "source_type": "vector",
      "chunk_index": 3
    }
  ],
  "retrieval_info": {
    "vector_results_count": 5,
    "graph_context_count": 12,
    "total_context_items": 17
  }
}
```

### `GET /api/health`
Check API and database connectivity status.

## How GraphRAG Works

This implementation clearly separates vector retrieval from graph traversal:

1. **Vector Search Phase**: Semantic embeddings find chunks similar to the query
2. **Graph Traversal Phase**: Starting from vector results, traverse relationships to discover related entities
3. **Context Combination**: Merge vector and graph results, preserving source attribution
4. **Answer Synthesis**: LLM generates answer using combined context

This separation enables:
- Clear debugging of each retrieval phase
- Independent tuning of vector vs. graph retrieval
- Transparent explanation of where answers come from

## Why This Project Matters

This project demonstrates several important concepts in modern AI system design:

- **Beyond Basic RAG**: Shows how GraphRAG extends traditional RAG by adding relationship-aware retrieval
- **Graph Data Modeling**: Illustrates how to model document knowledge as a graph (entities, relationships, properties)
- **Hybrid Retrieval**: Demonstrates combining vector search (semantic) with graph traversal (relational) for comprehensive context
- **Production-Ready Architecture**: Clean separation of concerns, environment-based configuration, and deployment considerations
- **Real-World Considerations**: Handles PDF parsing, entity extraction, metadata management, and source attribution

This is not just a demo—it's a complete, working system that could be extended for production use with proper infrastructure and scaling considerations.

## Limitations & Future Improvements

- **Entity Extraction**: Currently uses LLM-based extraction; could be enhanced with specialized NER models
- **Graph Schema**: Fixed entity types (Policy, Section, Topic, Concept); could be made more flexible
- **Scalability**: ChromaDB is suitable for small-to-medium datasets; could scale to distributed vector DBs (Pinecone, Weaviate)
- **Caching**: No query result caching; could add Redis for performance
- **UI**: Basic Streamlit UI; could be enhanced with React/Next.js for better UX
- **Error Handling**: Basic error handling; could add retry logic and better user feedback

## Troubleshooting

### Neo4j Connection Issues
- Verify Neo4j is running: `neo4j status`
- Check connection string in `.env`
- Ensure firewall allows connection to Neo4j port (7687)

### OpenAI API Errors
- Verify API key is correct and has credits
- Check rate limits if processing many documents
- System uses `gpt-4o-mini` by default for cost efficiency

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`
- Check Python version (3.9+)

### PDF Processing Errors
- Ensure PDFs are not password-protected
- Check file size limits (default: 200MB per file)
- Verify PyMuPDF is installed correctly

## License

MIT License

## Acknowledgments

- Built with [LlamaIndex](https://www.llamaindex.ai/) for document processing and LLM integration
- Graph database: [Neo4j](https://neo4j.com/)
- Inspired by GraphRAG research from Microsoft
- Vector storage: [ChromaDB](https://www.trychroma.com/)
