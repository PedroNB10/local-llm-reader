"""
Configuration management for the RAG system
"""
import os
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass


@dataclass
class Config:
    """Application configuration"""
    
    # Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen:0.5b")
    
    # Database Configuration
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "chroma")
    DATA_PATH: str = os.getenv("DATA_PATH", "data")
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "80"))
    
    # Retrieval Configuration
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    
    def __post_init__(self):
        """Validate configuration"""
        if self.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.CHUNK_OVERLAP < 0:
            raise ValueError("CHUNK_OVERLAP must be non-negative")
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if self.TOP_K_RESULTS <= 0:
            raise ValueError("TOP_K_RESULTS must be positive")


# Global configuration instance
config = Config()
