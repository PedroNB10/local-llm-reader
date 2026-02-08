import argparse
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from config import config

console = Console()


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        console.print("✨ [bold red]Clearing Database[/bold red]")
        clear_database()

    # Create (or update) the data store.
    with console.status("[bold green]Loading documents...", spinner="dots"):
        documents = load_documents()
    console.print(f"[green]✓[/green] Loaded {len(documents)} documents\n")
    
    with console.status("[bold green]Splitting documents into chunks...", spinner="dots"):
        chunks = split_documents(documents)
    console.print(f"[green]✓[/green] Created {len(chunks)} chunks\n")
    
    add_to_chroma(chunks)


def load_documents():
    # Load PDF files
    pdf_loader = PyPDFDirectoryLoader(config.DATA_PATH)
    pdf_documents = pdf_loader.load()
    
    # Load markdown files (excluding README and .gitkeep)
    md_documents = []
    data_path = Path(config.DATA_PATH)
    exclude_files = {'README.md', '.gitkeep'}
    
    for md_file in data_path.glob("**/*.md"):
        if md_file.name not in exclude_files:
            loader = TextLoader(str(md_file))
            md_documents.extend(loader.load())
    
    # Combine all documents
    return pdf_documents + md_documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    with console.status("[bold green]Loading vector database...", spinner="dots"):
        db = Chroma(
            persist_directory=config.CHROMA_PATH, embedding_function=get_embedding_function()
        )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    console.print(f"[blue]ℹ️  Number of existing documents in DB: {len(existing_ids)}[/blue]")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        console.print(f"[yellow]👉 Adding {len(new_chunks)} new documents...[/yellow]")
        with console.status("[bold green]Generating embeddings and storing...", spinner="dots"):
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
        console.print(f"[green]✅ Successfully added {len(new_chunks)} documents![/green]")
    else:
        console.print("[green]✅ No new documents to add[/green]")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(config.CHROMA_PATH):
        shutil.rmtree(config.CHROMA_PATH)


if __name__ == "__main__":
    main()