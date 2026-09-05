"""Run with python -m src.cli."""
import argparse

from .bm25_search import search
from .store import add_document, load_documents


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def main() -> None:
    parser = argparse.ArgumentParser(description="Local BM25 document search")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Add a text document")
    add.add_argument("--title", required=True)
    add.add_argument("--text", required=True)
    find = commands.add_parser("search", help="Rank documents using BM25")
    find.add_argument("query")
    find.add_argument("--top-k", type=positive_int, default=5)
    commands.add_parser("list", help="List stored documents")
    args = parser.parse_args()
    try:
        if args.command == "add":
            doc = add_document(args.title, args.text)
            print(f"Added [{doc['id']}] {doc['title']}")
        elif args.command == "list":
            documents = load_documents()
            for doc in documents:
                print(f"[{doc['id']}] {doc['title']}")
            if not documents:
                print("No documents yet.")
        else:
            results = search(load_documents(), args.query, args.top_k)
            for rank, result in enumerate(results, start=1):
                print(f"{rank}. [{result['id']}] {result['title']} | score={result['score']:.4f}")
                print(f"   {' '.join(result['text'].split())[:160]}")
            if not results:
                print("No matching documents.")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
