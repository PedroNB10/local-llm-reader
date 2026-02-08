#!/usr/bin/env python3
"""
Interactive RAG Chat Terminal Application
"""
import sys
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from get_embedding_function import get_embedding_function
from config import config

console = Console()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def initialize_system():
    """Initialize the RAG system"""
    console.print("\n[bold cyan]🚀 Initializing RAG System...[/bold cyan]")
    console.print(f"[dim]Embedding Model: {config.EMBEDDING_MODEL}[/dim]")
    console.print(f"[dim]LLM Model: {config.LLM_MODEL}[/dim]\n")
    
    with console.status("[bold green]Loading vector database...", spinner="dots"):
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=config.CHROMA_PATH, embedding_function=embedding_function)
    
    with console.status("[bold green]Loading language model...", spinner="dots"):
        model = OllamaLLM(model=config.LLM_MODEL)
    
    console.print("[green]✓[/green] System ready!\n")
    return db, model


def query_rag(db, model, query_text: str):
    """Query the RAG system"""
    # Search the DBconfig.TOP_K_RESULTS
    with console.status("[bold green]Searching documents...", spinner="dots"):
        results = db.similarity_search_with_score(query_text, k=5)

    if not results:
        console.print("[yellow]⚠️  No relevant documents found.[/yellow]\n")
        return None

    # Prepare context and prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    with console.status("[bold green]Generating response...", spinner="dots"):
        response_text = model.invoke(prompt)

    # Display results
    console.print(Panel(response_text, title="[bold blue]Response[/bold blue]", border_style="blue"))
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    console.print(f"[dim]📚 Sources: {', '.join(sources[:3])}...[/dim]\n")
    
    return response_text


def main():
    """Main interactive chat loop"""
    # Print welcome banner
    console.print(Panel.fit(
        "[bold cyan]Local LLM Reader[/bold cyan]\n"
        "[dim]Ask questions about your documents[/dim]\n\n"
        "Type [bold]'exit'[/bold] or [bold]'quit'[/bold] to leave",
        border_style="cyan"
    ))
    
    # Initialize system
    try:
        db, model = initialize_system()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] Failed to initialize system: {e}")
        sys.exit(1)
    
    # Main chat loop
    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                console.print("\n[cyan]👋 Goodbye![/cyan]\n")
                break
            
            # Skip empty queries
            if not query.strip():
                continue
            
            # Process query
            query_rag(db, model, query)
            
        except KeyboardInterrupt:
            console.print("\n\n[cyan]👋 Goodbye![/cyan]\n")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}\n")


if __name__ == "__main__":
    main()
