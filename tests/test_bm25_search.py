import math

import pytest

from src.bm25_search import search
from src.store import add_document, load_documents
from src.tokenizer import tokenize


def test_retrieves_relevant_sample_first():
    results = search(load_documents(), "BM25 search ranking")
    assert results[0]["id"] == 2
    assert len(results) > 1
    assert all(r["score"] > 0 for r in results)
    assert [r["score"] for r in results] == sorted(
        (r["score"] for r in results), reverse=True
    )


def test_top_k_and_case_punctuation():
    docs = load_documents()
    results = search(docs, "PYTHON, programming!", top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == 1
    assert results == search(docs, "python programming", top_k=1)


def test_length_normalization_and_known_scores():
    docs = [{"id": 1, "title": "", "text": "apple"},
            {"id": 2, "title": "", "text": "apple pear pear"}]
    results = search(docs, "apple")
    assert [r["id"] for r in results] == [1, 2]
    assert results[0]["score"] == pytest.approx(math.log(1.2) * 2.5 / 1.9375)
    assert results[1]["score"] == pytest.approx(math.log(1.2) * 2.5 / 3.0625)


def test_empty_and_unmatched_queries():
    assert search([], "python") == []
    assert search(load_documents(), "!!!") == []
    assert search(load_documents(), "zzzzunmatched") == []
    assert search([{"id": 1, "title": "", "text": ""}], "python") == []
    with pytest.raises(ValueError):
        search([], "python", top_k=0)


def test_tokenizer():
    assert tokenize("Hello, WORLD! bread-water_123.") == [
        "hello", "world", "bread", "water", "123"
    ]


def test_add_persists_and_is_searchable(tmp_path):
    path = tmp_path / "data" / "documents.json"
    assert load_documents(path) == []
    added = add_document("Example", "Example text", path)
    second = add_document("Bread", "Flour and water", path)
    assert added["id"] != second["id"]
    assert search(load_documents(path), "example")[0]["id"] == added["id"]
    with pytest.raises(ValueError):
        add_document(" ", "text", path)
