"""Check optional Porter normalization through indexing, retrieval, and CLI use."""
import subprocess
import sys

import pytest
from nltk.stem import PorterStemmer

from src.bm25_search import BM25Index, search
from src.tokenizer import normalize_tokens, tokenize


@pytest.mark.parametrize("word, expected_stem", [
    ("server", "server"), ("servers", "server"),
    ("secure", "secur"), ("security", "secur"),
    ("automate", "autom"), ("automation", "autom"),
])
def test_actual_porter_normalization(word, expected_stem):
    """Record actual NLTK output, not a hand-written substitution table."""
    original_tokens = tokenize(word.upper() + "!")
    porter = PorterStemmer(mode=PorterStemmer.NLTK_EXTENSIONS)
    assert porter.stem(word) == expected_stem
    assert normalize_tokens(original_tokens, stemming=True) == [expected_stem]
    assert normalize_tokens(original_tokens) == [word]
    assert original_tokens == [word]


@pytest.mark.parametrize("first_form, second_form", [
    ("server", "servers"), ("secure", "security"), ("automate", "automation"),
])
def test_related_forms_match_only_when_normalized(first_form, second_form):
    documents = [
        {"id": 1, "title": "", "text": first_form},
        {"id": 2, "title": "", "text": second_form},
        {"id": 3, "title": "Bread", "text": "flour water"},
    ]
    exact_index = BM25Index(documents)
    stemmed_index = BM25Index(documents, stemming=True)
    assert [result["id"] for result in exact_index.search(first_form)] == [1]
    assert [result["id"] for result in exact_index.search(second_form)] == [2]
    first_results = stemmed_index.search(first_form)
    second_results = stemmed_index.search(second_form)
    assert first_results == second_results
    assert [result["id"] for result in first_results] == [1, 2]
    assert stemmed_index.last_candidate_count == 2


def test_plural_query_does_not_match_singular_in_exact_mode():
    documents = [{"id": 1, "title": "Server", "text": "SSH access"}]
    assert BM25Index(documents).search("servers") == []
    assert BM25Index(documents).search("server")[0]["id"] == 1
    assert BM25Index(documents, stemming=True).search("servers")[0]["id"] == 1
    assert search(documents, "servers", stemming=True)[0]["id"] == 1


def test_normalized_statistics_candidates_and_scores(monkeypatch):
    documents = [
        {"id": 12, "title": "Secure", "text": "security server servers"},
        {"id": 81, "title": "Server", "text": "bread bread bread bread bread"},
        {"id": 5, "title": "Cooking", "text": "flour"},
    ]
    index = BM25Index(documents, stemming=True)
    assert index.tokenized_documents[0] == ["secur", "secur", "server", "server"]
    assert index.term_frequencies[0] == {"secur": 2, "server": 2}
    assert index.document_frequencies["server"] == 2
    assert index.document_frequencies["secur"] == 1
    assert "servers" not in index.inverted_index
    assert "security" not in index.idf_values
    assert index.inverted_index["server"] == {12, 81}
    assert index.inverted_index["secur"] == {12}
    query_terms = normalize_tokens(tokenize("security servers"), stemming=True)
    assert index.candidate_document_ids(query_terms) == {12, 81}

    scored_ids = []
    original_score = index._score_document

    def record_score(document_position, normalized_query_terms):
        assert normalized_query_terms == ["secur", "server"]
        scored_ids.append(index.documents[document_position]["id"])
        return original_score(document_position, normalized_query_terms)

    monkeypatch.setattr(index, "_score_document", record_score)
    results = index.search("security servers")
    assert [result["id"] for result in results] == [12, 81]
    assert set(scored_ids) == {12, 81}
    assert index.last_candidate_count == 2
    # Original document content must remain visible in results.
    assert results[0]["title"] == "Secure"
    assert results[0]["text"] == "security server servers"

    # Feed independently stemmed text through exact BM25 as a scoring reference.
    porter = PorterStemmer()
    normalized_documents = []
    for document in documents:
        normalized_document = document.copy()
        for field in ["title", "text"]:
            normalized_words = []
            for token in tokenize(document[field]):
                normalized_words.append(porter.stem(token))
            normalized_document[field] = " ".join(normalized_words)
        normalized_documents.append(normalized_document)
    reference = BM25Index(normalized_documents).search("secur server")
    assert [result["id"] for result in results] == [result["id"] for result in reference]
    assert [result["score"] for result in results] == [result["score"] for result in reference]


def test_stemmed_reuse_empty_unknown_and_repeated_terms():
    documents = [
        {"id": 2, "title": "Servers", "text": "secure automation"},
        {"id": 1, "title": "Server", "text": "security automate"},
    ]
    index = BM25Index(documents, stemming=True)
    assert index.search("server servers SERVER!") == index.search("server")
    assert index.search("secure security") == index.search("secure")
    assert index.search("automate automation") == index.search("automation")
    assert [result["id"] for result in index.search("servers", top_k=1)] == [1]
    for query in ["", "  ", "!!!", "zzzzunknown"]:
        assert index.search(query) == []
        assert index.last_candidate_count == 0
    assert BM25Index([], stemming=True).search("servers") == []


def test_analyze_command(monkeypatch, capsys):
    from src.cli import main

    monkeypatch.setattr(sys, "argv", ["cli", "analyze", "SERVERS, security automation!"])
    main()
    assert capsys.readouterr().out == (
        "Original tokens: servers security automation\n"
        "Porter stems: server secur autom\n"
    )


def test_analyze_empty_command(monkeypatch, capsys):
    from src.cli import main

    monkeypatch.setattr(sys, "argv", ["cli", "analyze", "!!!"])
    main()
    assert capsys.readouterr().out == "Original tokens: (none)\nPorter stems: (none)\n"


def test_cli_search_flag(monkeypatch, capsys):
    from src import cli

    documents = [{"id": 1, "title": "Server", "text": "SSH access"}]
    monkeypatch.setattr(cli, "load_documents", lambda: documents)
    monkeypatch.setattr(sys, "argv", ["cli", "search", "servers"])
    cli.main()
    assert capsys.readouterr().out == "No matching documents.\n"
    monkeypatch.setattr(sys, "argv", ["cli", "search", "servers", "--stemming"])
    cli.main()
    assert "1. [1] Server | score=" in capsys.readouterr().out


def test_benchmark_stemming_option(monkeypatch, capsys):
    from src import benchmark

    modes = []
    original_index = benchmark.BM25Index

    def record_mode(documents, stemming=False):
        modes.append(stemming)
        return original_index(documents, stemming=stemming)

    monkeypatch.setattr(benchmark, "BM25Index", record_mode)
    benchmark.run_benchmark(100, top_k=1, stemming=True)
    output = capsys.readouterr().out
    assert modes == [True]
    assert "Mode: Porter stemming" in output
    assert output.count("Candidate documents:") == 4
    assert output.count("   Score:") == 4


def test_porter_needs_no_external_nltk_data():
    """Import and search in a fresh process with data lookup/download disabled."""
    code = '''
import nltk
import nltk.downloader

def forbidden(*args, **kwargs):
    raise AssertionError("No external NLTK data is needed")

nltk.data.path = []
nltk.data.find = forbidden
nltk.download = forbidden
nltk.downloader.Downloader.download = forbidden
from src.bm25_search import BM25Index
index = BM25Index([{"id": 1, "title": "Server", "text": "security"}], stemming=True)
assert index.search("servers secure")[0]["id"] == 1
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
