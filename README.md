# Modular RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with a modular architecture, featuring document processing, semantic search, and AI-powered question answering.

## 🌟 Features

- **Document Processing**: Support for PDF, TXT, and DOCX files
- **Multiple Chunking Strategies**: Fixed-size, sentence-based, and semantic chunking
- **Vector Search**: FAISS-powered similarity search with embeddings
- **AI Generation**: Local LLM (FLAN-T5) or OpenAI API integration
- **REST API**: FastAPI backend with comprehensive endpoints
- **Web Interface**: Beautiful Streamlit frontend
- **Modular Architecture**: Clean separation of concerns for easy maintenance

## 📁 Project Structure

```
Modular RAG/
├── src/
│   ├── modules/           # Core RAG modules
│   │   ├── config.py      # Configuration management
│   │   ├── ingestion.py   # Document loading
│   │   ├── chunking.py    # Text chunking strategies
│   │   ├── embeddings.py  # Embedding generation
│   │   ├── retrieval.py   # Vector store & search
│   │   └── generation.py  # Answer generation
│   ├── api/               # FastAPI application
│   │   ├── main.py        # API endpoints
│   │   └── models.py      # Pydantic models
│   ├── frontend/          # Streamlit interface
│   │   └── app.py         # Web UI
│   └── utils/             # Utilities
│       ├── constants.py   # Application constants
│       ├── helpers.py     # Helper functions
│       └── logging.py     # Logging configuration
├── tests/                 # Test files
├── data/                  # Data storage (created automatically)
│   ├── documents/         # Uploaded documents
│   └── vector_store/      # FAISS index
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

1. **Clone the repository**
   ```bash
   cd "Modular RAG"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

#### 1. Start the API Server

```bash
cd src/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

#### 2. Start the Streamlit Frontend

In a new terminal:

```bash
cd src/frontend
streamlit run app.py
```

The web interface will open automatically at `http://localhost:8501`

## 📖 Usage

### Web Interface

1. **Upload Documents**
   - Go to the "Upload" tab
   - Select a PDF, TXT, or DOCX file
   - Click "Upload and Process"
   - Wait for processing to complete

2. **Query Documents**
   - Go to the "Query" tab
   - Enter your question
   - Adjust settings in the sidebar (number of sources, similarity threshold)
   - Click "Search"
   - View the generated answer and source documents

### API Usage

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

#### Query the System

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of the document?",
    "top_k": 5,
    "similarity_threshold": 0.5
  }'
```

#### List Documents

```bash
curl -X GET "http://localhost:8000/documents"
```

#### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/documents/{doc_id}"
```

#### Health Check

```bash
curl -X GET "http://localhost:8000/health"
```

## ⚙️ Configuration

### Environment Variables

Edit `.env` file:

```bash
# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# LLM Model (for local generation)
LLM_MODEL=google/flan-t5-base

# OpenAI Configuration (optional)
USE_OPENAI=false
OPENAI_API_KEY=your_api_key_here

# API Configuration
API_URL=http://localhost:8000

# Logging
LOG_LEVEL=INFO
```

### Chunking Strategies

The system supports three chunking strategies:

1. **Fixed**: Fixed-size chunks with overlap (default)
2. **Sentence**: Sentence-based chunking
3. **Semantic**: Paragraph-based semantic chunking

Configure in `src/modules/config.py` or via environment variables.

## 🏗️ Architecture

### Core Components

1. **Document Ingestion**: Loads and extracts text from various file formats
2. **Text Chunking**: Splits documents into manageable chunks using different strategies
3. **Embeddings**: Generates vector representations using sentence transformers
4. **Vector Store**: FAISS-based similarity search
5. **Answer Generation**: LLM-powered answer generation with context

### Data Flow

```
Document Upload → Ingestion → Chunking → Embedding → Vector Store
                                                            ↓
User Query → Embedding → Similarity Search → Context Retrieval → Answer Generation
```

## 🧪 Testing

Run tests:

```bash
pytest tests/ -v
```

## 🔧 Development

### Adding New Document Types

1. Add file extension to `SUPPORTED_FILE_EXTENSIONS` in `src/utils/constants.py`
2. Implement loader method in `src/modules/ingestion.py`

### Using Different Models

#### Embedding Models

Change `EMBEDDING_MODEL` in `.env` to any sentence-transformers model:
```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

#### LLM Models

Change `LLM_MODEL` in `.env` to any HuggingFace seq2seq model:
```bash
LLM_MODEL=google/flan-t5-large
```

Or use OpenAI:
```bash
USE_OPENAI=true
OPENAI_API_KEY=your_key_here
```

## 📊 Performance Tips

1. **Use GPU**: Set `device=cuda` in config for faster embeddings and generation
2. **Batch Processing**: Larger batch sizes for embedding generation
3. **Index Type**: Use IVF or HNSW indices for large datasets
4. **Chunk Size**: Optimize chunk size based on your documents (default: 500 chars)

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Cannot connect to API"
- **Solution**: Ensure the API server is running on port 8000

**Issue**: "CUDA out of memory"
- **Solution**: Set `device=cpu` in config or use a smaller model

**Issue**: "File too large"
- **Solution**: Increase `MAX_FILE_SIZE_MB` in `src/utils/constants.py`

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ using FastAPI, Streamlit, and HuggingFace Transformers**