"""Streamlit frontend for the Modular RAG system"""

import streamlit as st
import requests
import os
from pathlib import Path

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Modular RAG System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📚 Modular RAG System</div>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Query settings
    st.subheader("Query Settings")
    top_k = st.slider("Number of sources", min_value=1, max_value=10, value=5)
    similarity_threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    st.markdown("---")
    
    # System info
    st.subheader("📊 System Info")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            st.success("✅ API Connected")
            
            if data.get("vector_store_stats"):
                stats = data["vector_store_stats"]
                st.metric("Total Chunks", stats.get("num_chunks", 0))
                st.metric("Embedding Dimension", stats.get("embedding_dimension", 0))
        else:
            st.error("❌ API Error")
    except Exception as e:
        st.error(f"❌ Cannot connect to API\n\n{str(e)}")
    
    st.markdown("---")
    
    # Document management
    st.subheader("📄 Documents")
    try:
        response = requests.get(f"{API_URL}/documents")
        if response.status_code == 200:
            docs_data = response.json()
            st.metric("Total Documents", docs_data.get("total", 0))
            
            if docs_data.get("documents"):
                st.write("**Uploaded Documents:**")
                for doc in docs_data["documents"]:
                    with st.expander(f"📄 {doc['filename'][:30]}..."):
                        st.write(f"**ID:** {doc['doc_id']}")
                        st.write(f"**Size:** {doc['size']} chars")
                        st.write(f"**Loaded:** {doc['loaded_at'][:19]}")
                        
                        if st.button(f"Delete", key=f"del_{doc['doc_id']}"):
                            del_response = requests.delete(f"{API_URL}/documents/{doc['doc_id']}")
                            if del_response.status_code == 200:
                                st.success("Document deleted!")
                                st.rerun()
                            else:
                                st.error("Error deleting document")
    except Exception as e:
        st.warning("Could not load documents")

# Main content
tab1, tab2 = st.tabs(["🔍 Query", "📤 Upload"])

# Query Tab
with tab1:
    st.header("Ask a Question")
    
    query = st.text_input(
        "Enter your question:",
        placeholder="What would you like to know?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        query_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    if query_button and query:
        with st.spinner("🔄 Processing your query..."):
            try:
                # Send query to API
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "top_k": top_k,
                        "similarity_threshold": similarity_threshold
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display answer
                    st.markdown("### 💡 Answer")
                    st.markdown(f"**{data['answer']}**")
                    
                    st.markdown("---")
                    
                    # Display sources
                    if data.get("sources"):
                        st.markdown(f"### 📚 Sources ({data['num_sources']})")
                        
                        for i, source in enumerate(data["sources"], 1):
                            with st.expander(f"Source {i} - Score: {source['score']:.3f}"):
                                st.markdown(f"**Chunk ID:** `{source['chunk_id']}`")
                                st.markdown(f"**Relevance Score:** {source['score']:.3f}")
                                st.markdown("**Content:**")
                                st.info(source['text'])
                                
                                if source.get('metadata'):
                                    st.markdown("**Metadata:**")
                                    st.json(source['metadata'])
                    else:
                        st.warning("No sources found for this query.")
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
    
    elif query_button and not query:
        st.warning("Please enter a question first.")

# Upload Tab
with tab2:
    st.header("Upload Documents")
    
    st.markdown("""
    Upload documents to add them to the knowledge base. Supported formats:
    - **PDF** (.pdf)
    - **Text** (.txt)
    - **Word** (.docx)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['pdf', 'txt', 'docx'],
        help="Upload a document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
        
        if st.button("📤 Upload and Process", type="primary"):
            with st.spinner("🔄 Uploading and processing document..."):
                try:
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    # Prepare file for upload
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    
                    # Send file to API
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success(f"✅ {data['message']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Document ID", data.get('doc_id', 'N/A')[:20] + "...")
                        with col2:
                            st.metric("Filename", data.get('filename', 'N/A')[:20] + "...")
                        with col3:
                            st.metric("Chunks Created", data.get('num_chunks', 0))
                        
                        st.balloons()
                    else:
                        error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                        st.error(f"Error uploading document: {error_detail}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Modular RAG System v1.0 | Built with FastAPI & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)