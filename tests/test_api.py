"""Tests for the FastAPI application"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store_stats" in data


def test_query_endpoint_without_documents():
    """Test query endpoint when no documents are uploaded"""
    response = client.post(
        "/query",
        json={
            "query": "What is the meaning of life?",
            "top_k": 5,
            "similarity_threshold": 0.5
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["num_sources"] == 0


def test_query_endpoint_invalid_request():
    """Test query endpoint with invalid request"""
    response = client.post(
        "/query",
        json={
            "query": "",  # Empty query
            "top_k": 5
        }
    )
    assert response.status_code == 422  # Validation error


def test_list_documents_empty():
    """Test listing documents when none are uploaded"""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)


def test_delete_nonexistent_document():
    """Test deleting a document that doesn't exist"""
    response = client.delete("/documents/nonexistent_id")
    assert response.status_code == 404


def test_upload_endpoint_no_file():
    """Test upload endpoint without file"""
    response = client.post("/upload")
    assert response.status_code == 422  # Validation error


# Integration test (requires actual file)
def test_upload_and_query_flow():
    """Test complete flow: upload document and query it"""
    # Create a test text file
    test_content = "This is a test document about artificial intelligence and machine learning."
    test_file_path = Path(__file__).parent / "test_document.txt"
    
    with open(test_file_path, "w") as f:
        f.write(test_content)
    
    try:
        # Upload document
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/upload",
                files={"file": ("test_document.txt", f, "text/plain")}
            )
        
        assert response.status_code == 200
        upload_data = response.json()
        assert upload_data["success"] is True
        assert "doc_id" in upload_data
        doc_id = upload_data["doc_id"]
        
        # Query the uploaded document
        response = client.post(
            "/query",
            json={
                "query": "What is this document about?",
                "top_k": 3,
                "similarity_threshold": 0.3
            }
        )
        
        assert response.status_code == 200
        query_data = response.json()
        assert "answer" in query_data
        assert query_data["num_sources"] > 0
        
        # List documents
        response = client.get("/documents")
        assert response.status_code == 200
        docs_data = response.json()
        assert docs_data["total"] >= 1
        
        # Delete document
        response = client.delete(f"/documents/{doc_id}")
        assert response.status_code == 200
        delete_data = response.json()
        assert delete_data["success"] is True
        
    finally:
        # Cleanup
        if test_file_path.exists():
            test_file_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])