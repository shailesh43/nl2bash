import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Setup rich console
console = Console()

# Load model and data
model = SentenceTransformer('all-MiniLM-L6-v2')
with open('cmd.json', 'r') as f:
    data = json.load(f)

queries = [item['query'] for item in data]
commands = [item['command'] for item in data]
query_embeddings = model.encode(queries)

def get_command(user_input, top_k=1):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(user_embedding, query_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(commands[i], queries[i], similarities[i]) for i in top_indices]

def main():
    console.print(Panel.fit("🤖 [bold cyan]Welcome to NL2Bash Assistant![/bold cyan]", subtitle="Translate NL to terminal", padding=(1, 4)))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]> Describe your command[/bold green] (or type 'exit')").strip()
            if user_input.lower() == 'exit':
                console.print("\n[bold yellow]Goodbye! 👋[/bold yellow]")
                break

            results = get_command(user_input, top_k=3)
            if results:
                table = Table(title="Top Matching Commands", show_lines=False)
                table.add_column("Rank", style="bold blue")
                table.add_column("Command", style="bold magenta")
                table.add_column("Original Query", style="green")
                table.add_column("Similarity", justify="right", style="cyan")

                for idx, (cmd, query, score) in enumerate(results, start=1):
                    table.add_row(str(idx), cmd, query, f"{score:.2f}")
                console.print(table)

            else:
                console.print("[red]No command found.[/red]")

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Interrupted. Exiting...[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")

if __name__ == "__main__":
    main()
