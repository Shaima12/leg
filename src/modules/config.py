"""Configuration settings for the Modular RAG system"""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model"""
    model_name: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        description="Name of the sentence transformer model"
    )
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    device: str = Field(default="cpu", description="Device to run embeddings on (cpu/cuda)")


class ChunkingConfig(BaseModel):
    """Configuration for text chunking"""
    chunk_size: int = Field(default=500, description="Size of text chunks in characters")
    chunk_overlap: int = Field(default=50, description="Overlap between chunks")
    strategy: str = Field(
        default="fixed",
        description="Chunking strategy: fixed, sentence, or semantic"
    )


class RetrievalConfig(BaseModel):
    """Configuration for retrieval"""
    top_k: int = Field(default=5, description="Number of chunks to retrieve")
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity score for retrieval"
    )


class GenerationConfig(BaseModel):
    """Configuration for answer generation"""
    model_name: str = Field(
        default=os.getenv("LLM_MODEL", "google/flan-t5-base"),
        description="Name of the LLM model"
    )
    max_length: int = Field(default=512, description="Maximum length of generated answer")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    use_openai: bool = Field(
        default=False,
        description="Whether to use OpenAI API instead of local model"
    )
    openai_api_key: Optional[str] = Field(
        default=os.getenv("OPENAI_API_KEY"),
        description="OpenAI API key"
    )


class StorageConfig(BaseModel):
    """Configuration for data storage"""
    data_dir: str = Field(default="data", description="Directory for storing data")
    vector_store_dir: str = Field(
        default="data/vector_store",
        description="Directory for vector store persistence"
    )
    documents_dir: str = Field(
        default="data/documents",
        description="Directory for uploaded documents"
    )


class Config(BaseModel):
    """Main configuration class for the RAG system"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def __init__(self, **data):
        super().__init__(**data)
        # Create necessary directories
        os.makedirs(self.storage.data_dir, exist_ok=True)
        os.makedirs(self.storage.vector_store_dir, exist_ok=True)
        os.makedirs(self.storage.documents_dir, exist_ok=True)


# Global config instance
config = Config()