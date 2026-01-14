"""FastAPI application for the Modular RAG system"""

import os
import sys
import shutil
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Add src directory to path if not already there
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import modules
from modules.config import config
from modules.ingestion import DocumentLoader
from modules.chunking import TextChunker
from modules.embeddings import EmbeddingModel
from modules.retrieval import VectorStore
from modules.generation import AnswerGenerator
from utils.helpers import validate_file, sanitize_filename
from api.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    DocumentListResponse,
    DeleteResponse,
    HealthResponse,
    DocumentInfo
)

# Setup logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Modular RAG API",
    description="Retrieval-Augmented Generation system with document processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG components (lazy loading)
document_loader = None
text_chunker = None
embedding_model = None
vector_store = None
answer_generator = None


def initialize_components():
    """Initialize RAG components"""
    global document_loader, text_chunker, embedding_model, vector_store, answer_generator
    
    if document_loader is None:
        logger.info("Initializing RAG components...")
        
        # Document loader
        document_loader = DocumentLoader()
        
        # Load existing documents from disk
        documents_dir = Path(config.storage.documents_dir)
        if documents_dir.exists():
            for file_path in documents_dir.glob("*"):
                if file_path.is_file() and file_path.suffix.lower() in {'.pdf', '.txt', '.docx'}:
                    try:
                        logger.info(f"Loading existing document: {file_path.name}")
                        document_loader.load(str(file_path))
                    except Exception as e:
                        logger.warning(f"Could not load {file_path.name}: {e}")
        
        # Text chunker
        text_chunker = TextChunker(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            strategy=config.chunking.strategy
        )
        
        # Embedding model
        embedding_model = EmbeddingModel(
            model_name=config.embedding.model_name,
            device=config.embedding.device,
            batch_size=config.embedding.batch_size
        )
        
        # Vector store
        embedding_dim = embedding_model.get_embedding_dimension()
        vector_store = VectorStore(
            embedding_dimension=embedding_dim,
            index_type="flat",
            persist_dir=config.storage.vector_store_dir
        )
        
        # Answer generator
        answer_generator = AnswerGenerator(
            model_name=config.generation.model_name,
            max_length=config.generation.max_length,
            temperature=config.generation.temperature,
            use_openai=config.generation.use_openai,
            openai_api_key=config.generation.openai_api_key
        )
        
        logger.info("RAG components initialized successfully")
        print("RAG components initialized successfully")
        logger.info(f"Loaded {len(document_loader.loaded_documents)} documents from disk")
        print("Loaded documents from disk")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Save vector store on shutdown"""
    if vector_store is not None:
        vector_store.save(config.storage.vector_store_dir)
        logger.info("Vector store saved")


@app.get('/health', response_model=HealthResponse)
def health():
    """Health check endpoint"""
    stats = None
    if vector_store is not None:
        stats = vector_store.get_stats()
    
    return HealthResponse(
        status='ok',
        vector_store_stats=stats
    )


@app.post('/upload', response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    Args:
        file: Uploaded file
        
    Returns:
        Upload response with document info
    """
    try:
        initialize_components()
        
        # Save uploaded file
        filename = sanitize_filename(file.filename)
        file_path = os.path.join(config.storage.documents_dir, filename)
        
        with open(file_path, 'wb') as f:
            shutil.copyfileobj(file.file, f)
        
        # Validate file
        is_valid, error_msg = validate_file(file_path)
        if not is_valid:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Load document
        logger.info(f"Loading document: {filename}")
        document = document_loader.load(file_path)
        
        # Chunk document
        logger.info(f"Chunking document: {filename}")
        chunks = text_chunker.chunk(
            document.content,
            metadata={'doc_id': document.doc_id, **document.metadata}
        )
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks generated from document")
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = embedding_model.embed_batch(chunk_texts)
        
        # Add to vector store
        logger.info("Adding chunks to vector store")
        vector_store.add_documents(
            chunks=chunk_texts,
            embeddings=embeddings,
            metadata=[chunk.metadata for chunk in chunks],
            chunk_ids=[chunk.chunk_id for chunk in chunks]
        )
        
        # Save vector store
        vector_store.save(config.storage.vector_store_dir)
        
        return UploadResponse(
            success=True,
            message=f"Document '{filename}' uploaded and processed successfully",
            doc_id=document.doc_id,
            filename=filename,
            num_chunks=len(chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/query', response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system
    
    Args:
        request: Query request
        
    Returns:
        Query response with answer and sources
    """
    try:
        initialize_components()
        
        # Generate query embedding
        logger.info(f"Processing query: {request.query}")
        query_embedding = embedding_model.embed(request.query)
        
        # Retrieve relevant chunks
        logger.info(f"Retrieving top {request.top_k} chunks")
        retrieved_chunks = vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )
        
        if not retrieved_chunks:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                query=request.query,
                num_sources=0,
                sources=[]
            )
        
        # Generate answer
        logger.info("Generating answer")
        response = answer_generator.generate(
            query=request.query,
            context_chunks=retrieved_chunks
        )
        
        return QueryResponse(**response)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/documents', response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents
    
    Returns:
        List of documents
    """
    try:
        initialize_components()
        
        documents = document_loader.list_documents()
        
        return DocumentListResponse(
            documents=[DocumentInfo(**doc) for doc in documents],
            total=len(documents)
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/documents/{doc_id}', response_model=DeleteResponse)
async def delete_document(doc_id: str):
    """
    Delete a document and its chunks
    
    Args:
        doc_id: Document ID
        
    Returns:
        Delete response
    """
    try:
        initialize_components()
        
        # Check if document exists
        document = document_loader.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        
        # Delete from vector store
        chunks_deleted = vector_store.delete_document(doc_id)
        
        # Delete from document loader
        if doc_id in document_loader.loaded_documents:
            del document_loader.loaded_documents[doc_id]
        
        # Delete file
        file_path = document.metadata.get('filepath')
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Save vector store
        vector_store.save(config.storage.vector_store_dir)
        
        return DeleteResponse(
            success=True,
            message=f"Document '{doc_id}' deleted successfully",
            chunks_deleted=chunks_deleted
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))