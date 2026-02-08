import argparse
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live

from get_embedding_function import get_embedding_function
from config import config

console = Console()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    
    console.print(f"\n[bold cyan]🔍 Query:[/bold cyan] {query_text}\n")
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    with console.status("[bold green]Loading database...", spinner="dots"):
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=config.CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    with console.status("[bold green]Searching for relevant documents...", spinner="dots"):
        results = db.similarity_search_with_score(query_text, k=config.TOP_K_RESULTS)

    if not results:
        console.print("[yellow]⚠️  No results found.[/yellow]")
        return None

    console.print(f"[green]✓[/green] Found {len(results)} relevant documents\n")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate response
    with console.status("[bold green]Generating response...", spinner="dots"):
        model = OllamaLLM(model=config.LLM_MODEL)
        response_text = model.invoke(prompt)

    # Display results
    console.print(Panel(response_text, title="[bold blue]Response[/bold blue]", border_style="blue"))
    
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    console.print(f"\n[dim]📚 Sources: {', '.join(sources)}[/dim]\n")
    
    return response_text


if __name__ == "__main__":
    main()