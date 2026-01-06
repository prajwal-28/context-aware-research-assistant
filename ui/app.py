"""Streamlit UI for Context-Aware Research Assistant."""
import streamlit as st
import requests
from pathlib import Path
import time

# Page configuration
st.set_page_config(
    page_title="Context-Aware Research Assistant",
    page_icon="📚",
    layout="wide",
)

# API endpoint
API_BASE_URL = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Base URL for the FastAPI backend"
)

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def upload_documents(files):
    """Upload documents to the API."""
    try:
        file_data = []
        for file in files:
            file_data.append(("files", (file.name, file.getvalue(), "application/pdf")))
        
        response = requests.post(
            f"{API_BASE_URL}/api/upload",
            files=file_data,
            timeout=300,  # 5 minutes for large files
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def query_documents(query, top_k=5, max_hops=2):
    """Send query to the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"query": query, "top_k": top_k, "max_hops": max_hops},
            timeout=120,
        )
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()
    except Exception as e:
        return False, {"error": str(e)}


# Main UI
st.title("📚 Context-Aware Research Assistant")
st.markdown("Upload PDF documents and ask questions using GraphRAG retrieval")

# Sidebar
st.sidebar.header("Configuration")

# Health check
st.sidebar.subheader("API Status")
health_ok, health_info = check_api_health()
if health_ok:
    st.sidebar.success("✅ API Connected")
    if health_info.get("neo4j") == "connected":
        st.sidebar.success("✅ Neo4j Connected")
    else:
        st.sidebar.warning(f"⚠️ Neo4j: {health_info.get('neo4j')}")
else:
    st.sidebar.error("❌ API Not Available")
    st.sidebar.error(f"Error: {health_info.get('error', 'Unknown')}")

# Query settings
st.sidebar.subheader("Query Settings")
top_k = st.sidebar.slider("Top K (vector results)", 3, 10, 5)
max_hops = st.sidebar.slider("Max Graph Hops", 1, 3, 2)

# Main content area
tab1, tab2 = st.tabs(["📤 Upload Documents", "❓ Query Documents"])

# Tab 1: Document Upload
with tab1:
    st.header("Upload PDF Documents")
    st.markdown("Upload one or more PDF documents to build the knowledge graph")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) selected")
        
        if st.button("Upload and Process", type="primary"):
            with st.spinner("Uploading and processing documents... This may take a while."):
                success, result = upload_documents(uploaded_files)
                
                if success:
                    st.success(f"✅ Successfully uploaded {len(uploaded_files)} document(s)")
                    st.json(result)
                    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])
                else:
                    st.error("❌ Upload failed")
                    st.json(result)

# Tab 2: Query
with tab2:
    st.header("Query Documents")
    st.markdown("Ask questions about your uploaded documents")
    
    # Query input
    user_query = st.text_area(
        "Enter your question",
        height=100,
        placeholder="Example: How does maternity leave policy affect project deadlines?",
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        submit_button = st.button("Submit Query", type="primary")
    
    # Process query
    if submit_button and user_query:
        if not health_ok:
            st.error("⚠️ API is not available. Please check the API URL and ensure the backend is running.")
        else:
            with st.spinner("Processing query... This may take 30-60 seconds."):
                success, result = query_documents(user_query, top_k=top_k, max_hops=max_hops)
                
                if success:
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(result.get("answer", "No answer generated"))
                    
                    # Display sources
                    sources = result.get("sources", [])
                    if sources:
                        st.subheader("Sources")
                        for idx, source in enumerate(sources, 1):
                            with st.expander(f"Source {idx}: {source.get('filename', 'Unknown')}"):
                                st.write(f"**Document:** {source.get('filename')}")
                                st.write(f"**Retrieval Method:** {source.get('source_type', 'unknown')}")
                                if source.get('chunk_index') is not None:
                                    st.write(f"**Chunk Index:** {source.get('chunk_index')}")
                    
                    # Display retrieval info
                    retrieval_info = result.get("retrieval_info", {})
                    if retrieval_info:
                        st.subheader("Retrieval Information")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Vector Results",
                                retrieval_info.get("vector_results_count", 0)
                            )
                        with col2:
                            st.metric(
                                "Graph Context",
                                retrieval_info.get("graph_context_count", 0)
                            )
                        with col3:
                            st.metric(
                                "Total Context",
                                retrieval_info.get("total_context_items", 0)
                            )
                    
                    # Add to query history
                    st.session_state.query_history.append({
                        "query": user_query,
                        "result": result,
                        "timestamp": time.time(),
                    })
                else:
                    st.error("❌ Query failed")
                    st.json(result)
    
    # Query history
    if st.session_state.query_history:
        st.subheader("Query History")
        for idx, item in enumerate(reversed(st.session_state.query_history[-5:]), 1):
            with st.expander(f"Query {idx}: {item['query'][:50]}..."):
                st.markdown(f"**Query:** {item['query']}")
                st.markdown("**Answer:**")
                st.markdown(item['result'].get("answer", "No answer"))
                st.markdown(f"**Sources:** {len(item['result'].get('sources', []))} document(s)")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Context-Aware Research Assistant | Powered by LlamaIndex, Neo4j, and GraphRAG
    </div>
    """,
    unsafe_allow_html=True,
)







