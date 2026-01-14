# Quick Start Guide - Modular RAG System

## Running the Application

### Method 1: From Project Root (Recommended)

**Terminal 1 - Start API:**
```bash
cd "Modular RAG"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd "Modular RAG/src/frontend"
streamlit run app.py
```

### Method 2: From src Directory

**Terminal 1 - Start API:**
```bash
cd "Modular RAG/src"
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Start Frontend:**
```bash
cd "Modular RAG/src/frontend"
streamlit run app.py
```

## Access Points

- **Web Interface**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## Running Tests

```bash
cd "Modular RAG"
pytest tests/ -v
```

## Environment Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to configure:
   - Embedding model
   - LLM model
   - OpenAI API key (optional)

## Common Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run specific test:**
```bash
pytest tests/test_api.py::test_health_endpoint -v
```

**Run tests with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

**Issue**: ModuleNotFoundError when running API
- **Solution**: Always run from project root using `uvicorn src.api.main:app`

**Issue**: Cannot connect to API from frontend
- **Solution**: Ensure API is running on port 8000 first

**Issue**: Tests failing
- **Solution**: Run `pytest` from project root directory
