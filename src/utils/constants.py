"""Application constants"""

# Supported file types
SUPPORTED_FILE_EXTENSIONS = {'.pdf', '.txt', '.docx'}
SUPPORTED_MIME_TYPES = {
    'application/pdf',
    'text/plain',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
}

# Default parameters
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.5

# Model defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-base"

# API settings
API_VERSION = "v1"
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Chunking strategies
CHUNKING_STRATEGIES = ['fixed', 'sentence', 'semantic']