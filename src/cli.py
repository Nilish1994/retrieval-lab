"""Parse local document commands and the in-memory benchmark command.

Run with python -m src.cli. Each invocation is a separate Python process.
"""
import argparse

from .benchmark import run_benchmark
from .bm25_search import BM25Index
from .store import add_document, load_documents


def positive_int(value: str) -> int:
    """Require a positive count for top-k and generated documents."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    """Dispatch commands while keeping storage and scoring in their modules."""
    parser = argparse.ArgumentParser(description="Local BM25 document search")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Add a text document")
    add.add_argument("--title", required=True)
    add.add_argument("--text", required=True)
    find = commands.add_parser("search", help="Rank documents using BM25")
    find.add_argument("query")
    find.add_argument("--top-k", type=positive_int, default=5)
    commands.add_parser("list", help="List stored documents")
    benchmark = commands.add_parser("benchmark", help="Time a reusable BM25 index")
    benchmark.add_argument("--documents", type=positive_int, default=10000)
    benchmark.add_argument("--top-k", type=positive_int, default=3)
    args = parser.parse_args()
    try:
        if args.command == "add":
            document = add_document(args.title, args.text)
            print(f"Added [{document['id']}] {document['title']}")
        elif args.command == "list":
            documents = load_documents()
            for document in documents:
                print(f"[{document['id']}] {document['title']}")
            if not documents:
                print("No documents yet.")
        elif args.command == "benchmark":
            run_benchmark(args.documents, args.top_k)
        else:
            documents = load_documents()
            index = BM25Index(documents)
            results = index.search(args.query, args.top_k)
            for rank, result in enumerate(results, start=1):
                print(f"{rank}. [{result['id']}] {result['title']} | score={result['score']:.4f}")
                print(f"   {' '.join(result['text'].split())[:160]}")
            if not results:
                print("No matching documents.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
