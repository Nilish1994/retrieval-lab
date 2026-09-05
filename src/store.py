"""JSON document storage. Private files are never read or indexed."""
import json
import os
import tempfile
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "documents.json"


def load_documents(path: Path = DEFAULT_PATH) -> list[dict]:
    if not path.exists():
        return []
    documents = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(documents, list):
        raise ValueError("Document store must contain a JSON list")
    ids = set()
    for doc in documents:
        if (not isinstance(doc, dict)
                or type(doc.get("id")) is not int or doc["id"] < 1
                or not isinstance(doc.get("title"), str)
                or not isinstance(doc.get("text"), str)):
            raise ValueError("Each document needs a positive integer id, title, and text")
        if doc["id"] in ids:
            raise ValueError("Document ids must be unique")
        ids.add(doc["id"])
    return documents


def add_document(title: str, text: str, path: Path = DEFAULT_PATH) -> dict:
    if not title.strip() or not text.strip():
        raise ValueError("Title and text must not be blank")
    documents = load_documents(path)
    doc = {"id": max((d["id"] for d in documents), default=0) + 1,
           "title": title.strip(), "text": text.strip()}
    documents.append(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(documents, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return doc
