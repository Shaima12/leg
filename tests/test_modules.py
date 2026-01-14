"""Tests for core modules"""

import pytest
from pathlib import Path

from modules.chunking import TextChunker, Chunk
from modules.ingestion import DocumentLoader, Document


class TestTextChunker:
    """Tests for TextChunker"""
    
    def test_fixed_chunking(self):
        """Test fixed-size chunking"""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10, strategy="fixed")
        text = "This is a test document. " * 10
        chunks = chunker.chunk(text, metadata={"doc_id": "test"})
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(len(chunk.text) <= 50 for chunk in chunks)
    
    def test_sentence_chunking(self):
        """Test sentence-based chunking"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, strategy="sentence")
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunker.chunk(text, metadata={"doc_id": "test"})
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_semantic_chunking(self):
        """Test semantic (paragraph) chunking"""
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, strategy="semantic")
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = chunker.chunk(text, metadata={"doc_id": "test"})
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
    
    def test_invalid_strategy(self):
        """Test invalid chunking strategy"""
        with pytest.raises(ValueError):
            TextChunker(strategy="invalid")


class TestDocumentLoader:
    """Tests for DocumentLoader"""
    
    def test_load_txt_file(self):
        """Test loading a text file"""
        # Create a test file
        test_file = Path(__file__).parent / "test_load.txt"
        test_content = "This is test content."
        
        with open(test_file, "w") as f:
            f.write(test_content)
        
        try:
            loader = DocumentLoader()
            doc = loader.load(str(test_file))
            
            assert isinstance(doc, Document)
            assert doc.content == test_content
            assert doc.metadata["filename"] == "test_load.txt"
            assert doc.metadata["extension"] == ".txt"
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist"""
        loader = DocumentLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_file.txt")
    
    def test_unsupported_file_type(self):
        """Test loading an unsupported file type"""
        # Create a test file with unsupported extension
        test_file = Path(__file__).parent / "test.xyz"
        
        with open(test_file, "w") as f:
            f.write("test")
        
        try:
            loader = DocumentLoader()
            
            with pytest.raises(ValueError):
                loader.load(str(test_file))
        finally:
            if test_file.exists():
                test_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
