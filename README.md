# Local LLM Reader 📚

A Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using local LLMs via Ollama. Ask questions about your PDFs and Markdown files, and get AI-powered answers based on your content.

## ✨ Features

- 🤖 **Local LLM Integration** - Uses Ollama for complete privacy and offline operation
- 📄 **Multiple Formats** - Supports PDF and Markdown files
- 🔍 **Smart Search** - Vector-based semantic search with ChromaDB
- 💬 **Interactive Chat** - Beautiful terminal interface with animated loading states
- ⚙️ **Flexible Configuration** - Environment-based model selection
- 🎨 **Rich Terminal UI** - Colored output with progress indicators

## 📋 Prerequisites

- **Python 3.8+** (Python 3.13+ recommended)
  - Using [asdf](https://asdf-vm.com/)? Run: `asdf install python 3.13.1`
- **[Ollama](https://ollama.ai/)** - For running local LLMs

## 🚀 Quick Start

### 1. Install Ollama

```bash
# Linux/Mac
curl https://ollama.ai/install.sh | sh

# Start Ollama service
ollama serve
```

### 2. Pull Required Models

```bash
# Embedding model (required)
ollama pull nomic-embed-text

# LLM model for generation (required)
ollama pull qwen:0.5b
```

### 3. Clone and Setup

```bash
git clone <your-repo-url>
cd local-llm-reader

# If using asdf, Python version will be auto-selected from .tool-versions
# Otherwise, ensure you're using Python 3.8+

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env to customize models (optional)
nano .env
```

### 5. Add Your Documents

Place your PDF or Markdown files in the `data/` directory:

```bash
cp your-document.pdf data/
cp your-notes.md data/
```

### 6. Build the Database

```bash
# First time or after adding new documents
python populate_database.py

# To rebuild from scratch
python populate_database.py --reset
```

### 7. Start Chatting!

```bash
# Interactive chat mode (recommended)
python chat.py

# Or single query
python query_data.py "What is this document about?"
```

## 📖 Usage Guide

### Interactive Chat Mode

The interactive chat provides a conversational interface:

```bash
python chat.py
```

Features:
- Type your questions naturally
- Get formatted responses with sources
- Continue conversation without reloading
- Type `exit` or `quit` to leave

### Single Query Mode

For quick one-off questions:

```bash
python query_data.py "Your question here"
```

### Managing Documents

**Adding new documents:**
```bash
# 1. Add files to data/ directory
cp new-document.pdf data/

# 2. Update the database
python populate_database.py
```

**Rebuilding database:**
```bash
# Clear and rebuild everything
python populate_database.py --reset
```

## ⚙️ Configuration

Edit `.env` to customize your setup:

```env
# Embedding Model - Used for document vectorization
# Better embeddings = better search results
EMBEDDING_MODEL=nomic-embed-text

# LLM Model - Used for generating answers
# Larger models = better quality, more RAM
LLM_MODEL=qwen:0.5b

# Retrieval Settings
TOP_K_RESULTS=5          # Number of document chunks to retrieve
CHUNK_SIZE=800           # Size of text chunks (in characters)
CHUNK_OVERLAP=80         # Overlap between chunks

# Paths
CHROMA_PATH=chroma       # Vector database location
DATA_PATH=data           # Your documents location
```

## 🤖 Model Recommendations

### Embedding Models

Choose based on your RAM and quality needs:

| Model | Dimensions | RAM Usage | Quality | Speed |
|-------|-----------|-----------|---------|-------|
| `nomic-embed-text` | 768 | ~800MB | Good | Fast |
| `mxbai-embed-large` | 1024 | ~1.5GB | Better | Medium |
| `all-minilm` | 384 | ~400MB | Fair | Very Fast |

```bash
# Pull your chosen embedding model
ollama pull nomic-embed-text
```

### Generation Models

Choose based on your needs:

| Model | Size | RAM Usage | Quality | Speed | Best For |
|-------|------|-----------|---------|-------|----------|
| `qwen:0.5b` | 0.5B | ~1GB | Fair | Very Fast | Quick queries |
| `llama3.2` | 3B | ~4GB | Good | Medium | Balanced |
| `mistral` | 7B | ~8GB | Better | Slower | Quality answers |
| `llama3.1:8b` | 8B | ~10GB | Best | Slower | Complex queries |

```bash
# Pull your chosen LLM
ollama pull llama3.2
```

### Recommended Combinations

**Low RAM (~4GB):**
```env
EMBEDDING_MODEL=all-minilm
LLM_MODEL=qwen:0.5b
```

**Balanced (~8GB):**
```env
EMBEDDING_MODEL=nomic-embed-text
LLM_MODEL=llama3.2
```

**High Quality (~12GB+):**
```env
EMBEDDING_MODEL=mxbai-embed-large
LLM_MODEL=mistral
```

## 🏗️ Project Structure

```
local-llm-reader/
├── chat.py                 # Interactive chat interface
├── query_data.py          # Single query interface
├── populate_database.py   # Document ingestion
├── config.py              # Configuration management
├── get_embedding_function.py  # Embedding setup
├── requirements.txt       # Python dependencies
├── .env.example          # Example configuration
├── .env                  # Your configuration (git-ignored)
├── data/                 # Your documents (git-ignored)
│   ├── .gitkeep
│   └── README.md
└── chroma/               # Vector database (generated)
```

## 🔧 Troubleshooting

### "No module named 'langchain'"

```bash
pip install -r requirements.txt
```

### "Collection expecting embedding with dimension of X, got Y"

The embedding model changed. Rebuild the database:

```bash
python populate_database.py --reset
```

### "Unable to find matching results"

- Ensure documents are in `data/` directory
- Check database is populated: `python populate_database.py`
- Try lowering `TOP_K_RESULTS` or adjusting `CHUNK_SIZE`

### Ollama connection errors

```bash
# Make sure Ollama is running
ollama serve

# Verify models are installed
ollama list
```

### Slow performance

- Use smaller models (qwen:0.5b, all-minilm)
- Reduce `TOP_K_RESULTS` in .env
- Increase `CHUNK_SIZE` to reduce total chunks

## 🎯 Best Practices

1. **Start Small** - Test with a few documents first
2. **Choose Right Models** - Balance quality vs. speed for your needs
3. **Consistent Embeddings** - Don't change `EMBEDDING_MODEL` without rebuilding database
4. **Chunk Size** - Larger chunks (1000-2000) for narrative docs, smaller (500-800) for technical docs
5. **Monitor RAM** - Use `htop` or Activity Monitor to check resource usage


## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - RAG framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Rich](https://github.com/Textualize/rich) - Terminal UI

## 📚 Learn More

- [RAG Concepts](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Ollama Documentation](https://github.com/ollama/ollama/blob/main/docs/README.md)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)

---


