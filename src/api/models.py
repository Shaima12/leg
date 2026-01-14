"""Pydantic models for API requests and responses"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response model for document upload"""
    success: bool
    message: str
    doc_id: Optional[str] = None
    filename: Optional[str] = None
    num_chunks: Optional[int] = None
    

class QueryRequest(BaseModel):
    """Request model for querying"""
    query: str = Field(..., description="User query", min_length=1)
    top_k: int = Field(default=5, description="Number of chunks to retrieve", ge=1, le=20)
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score",
        ge=0.0,
        le=1.0
    )


class Source(BaseModel):
    """Source document chunk"""
    chunk_id: str
    text: str
    score: float
    metadata: Dict


class QueryResponse(BaseModel):
    """Response model for querying"""
    answer: str
    query: str
    num_sources: int
    sources: List[Source]


class DocumentInfo(BaseModel):
    """Document information"""
    doc_id: str
    filename: str
    size: int
    loaded_at: str


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    documents: List[DocumentInfo]
    total: int


class DeleteResponse(BaseModel):
    """Response model for document deletion"""
    success: bool
    message: str
    chunks_deleted: int = 0


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    vector_store_stats: Optional[Dict] = None
