# Data Directory

Place your documents here for the RAG system to process.

## Supported Formats
- **PDF files** (`.pdf`)
- **Markdown files** (`.md`)

## Usage

1. Add your documents to this directory
2. Run the ingestion script:
   ```bash
   python populate_database.py
   ```
3. Query your documents:
   ```bash
   python chat.py
   ```

## Example Structure

```
data/
├── document1.pdf
├── document2.pdf
└── notes.md
```

## Notes
- Files in this directory are ignored by git (except this README)
- The system will recursively process all supported files
- For best results, use descriptive filenames
