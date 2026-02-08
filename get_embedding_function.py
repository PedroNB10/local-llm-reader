from langchain_ollama import OllamaEmbeddings
from config import config


def get_embedding_function():
    """
    Get the embedding function configured via environment variables.
    Default model: nomic-embed-text
    
    You can change the model by setting the EMBEDDING_MODEL environment variable:
    export EMBEDDING_MODEL=mxbai-embed-large
    
    Popular embedding models:
    - nomic-embed-text (768 dims, good balance)
    - mxbai-embed-large (1024 dims, better quality)
    - all-minilm (384 dims, faster/smaller)
    """
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    return embeddings