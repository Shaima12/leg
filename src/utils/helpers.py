"""Helper functions and utilities"""

import os
import re
from typing import Optional
from pathlib import Path
from utils.constants import SUPPORTED_FILE_EXTENSIONS, MAX_FILE_SIZE_BYTES


def validate_file(file_path: str) -> tuple[bool, Optional[str]]:
    """
    Validate a file for processing
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        return False, f"File not found: {file_path}"
    
    # Check if it's a file
    if not path.is_file():
        return False, f"Path is not a file: {file_path}"
    
    # Check file extension
    if path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
        return False, f"Unsupported file type: {path.suffix}. Supported: {SUPPORTED_FILE_EXTENSIONS}"
    
    # Check file size
    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File too large: {file_size / (1024*1024):.2f}MB. Max: {MAX_FILE_SIZE_BYTES / (1024*1024)}MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, None


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext
    
    return filename


def extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """
    Extract simple keywords from text
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction (word frequency)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Count frequencies
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Return top keywords
    return [word for word, freq in sorted_words[:max_keywords]]