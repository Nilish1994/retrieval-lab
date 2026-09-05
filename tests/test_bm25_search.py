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


def test_index_stores_expected_statistics():
    from src.bm25_search import BM25Index

    documents = [
        {"id": 1, "title": "Apple", "text": "apple"},
        {"id": 2, "title": "", "text": "apple pear pear"},
    ]
    index = BM25Index(documents)
    assert index.tokenized_documents == [["apple", "apple"], ["apple", "pear", "pear"]]
    assert index.document_count == 2
    assert index.document_lengths == [2, 3]
    assert index.average_document_length == 2.5
    assert index.term_frequencies == [{"apple": 2}, {"apple": 1, "pear": 2}]
    assert index.document_frequencies == {"apple": 2, "pear": 1}
    assert index.idf_values["apple"] == pytest.approx(math.log(1.2))
    assert index.idf_values["pear"] == pytest.approx(math.log(2))


def test_multiple_searches_only_tokenize_queries(monkeypatch):
    import src.bm25_search as bm25

    index = bm25.BM25Index(load_documents())
    tokenized_inputs = []

    def record_tokenization(text):
        tokenized_inputs.append(text)
        return tokenize(text)

    def unexpected_recalculation(*args, **kwargs):
        pytest.fail("Search must reuse corpus statistics")

    monkeypatch.setattr(bm25, "tokenize", record_tokenization)
    monkeypatch.setattr(bm25, "Counter", unexpected_recalculation)
    monkeypatch.setattr(bm25, "log", unexpected_recalculation)
    assert index.search("python programming")[0]["id"] == 1
    assert index.search("bread water")[0]["id"] == 4
    assert index.search("BM25 search ranking")[0]["id"] == 2
    assert tokenized_inputs == ["python programming", "bread water", "BM25 search ranking"]


def test_unrelated_documents_preserve_obvious_ranking():
    from src.bm25_search import BM25Index

    documents = load_documents()
    original_document_count = len(documents)
    original_index = BM25Index(documents)
    for document_id in range(100, 120):
        documents.append({"id": document_id, "title": "Astronomy", "text": "stars planets"})
    expanded_index = BM25Index(documents)
    assert original_index.search("python programming")[0]["id"] == 1
    assert expanded_index.search("python programming")[0]["id"] == 1
    assert original_index.document_count == original_document_count


def test_index_top_k_empty_queries_and_ties():
    from src.bm25_search import BM25Index

    documents = [
        {"id": 2, "title": "Bread", "text": "water"},
        {"id": 1, "title": "Bread", "text": "water"},
    ]
    index = BM25Index(documents)
    assert [result["id"] for result in index.search("BREAD!", top_k=1)] == [1]
    assert [result["id"] for result in index.search("bread", top_k=10)] == [1, 2]
    assert index.search("bread bread") == index.search("bread")
    for query in ["", "  ", "!!!", "unknown"]:
        assert index.search(query) == []
    assert BM25Index([]).search("bread") == []
    assert BM25Index([{"id": 1, "title": "", "text": ""}]).search("bread") == []
    with pytest.raises(ValueError):
        index.search("bread", top_k=0)


def test_index_snapshot_is_independent_of_input_and_results():
    from src.bm25_search import BM25Index

    documents = [{"id": 1, "title": "Bread", "text": "water"}]
    index = BM25Index(documents)
    documents[0]["title"] = "Changed"
    first_results = index.search("bread")
    first_results[0]["title"] = "Changed again"
    assert index.search("bread")[0]["title"] == "Bread"


def test_generated_documents_are_varied_and_searchable():
    from src.benchmark import generate_documents
    from src.bm25_search import BM25Index

    documents = generate_documents(200)
    assert documents == generate_documents(200)
    assert len(documents) == 200
    assert len({document["id"] for document in documents}) == 200
    index = BM25Index(documents)
    assert len(set(index.document_lengths)) > 1
    assert len(set(counts["python"] for counts in index.term_frequencies)) > 2
    assert index.search("python programming")[0]["title"].startswith("Python programming")
    assert index.search("ssh security server")[0]["title"].startswith("Linux servers")
    assert index.search("bread water")[0]["title"].startswith("Cooking")
    with pytest.raises(ValueError):
        generate_documents(0)


def test_benchmark_constructs_one_index_for_all_queries(monkeypatch, capsys):
    import src.benchmark as benchmark

    original_index = benchmark.BM25Index
    constructed_indexes = []

    def record_construction(documents):
        index = original_index(documents)
        constructed_indexes.append(index)
        return index

    monkeypatch.setattr(benchmark, "BM25Index", record_construction)
    benchmark.run_benchmark(100, top_k=2)
    output = capsys.readouterr().out
    assert len(constructed_indexes) == 1
    assert "Documents: 100" in output
    assert "Generation time:" in output
    assert "Index build time:" in output
    assert output.count("Search time:") == len(benchmark.BENCHMARK_QUERIES)
    assert output.count("   Score:") == 2 * len(benchmark.BENCHMARK_QUERIES)


@pytest.fixture
def candidate_documents():
    """Use nonconsecutive IDs, partial matches, repetitions, and an unrelated row."""
    return [
        {"id": 20, "title": "Programming", "text": "design"},
        {"id": 1, "title": "Python", "text": "python programming"},
        {"id": 6, "title": "Python", "text": "scripts"},
        {"id": 10, "title": "Python", "text": "scripts"},
        {"id": 81, "title": "Bread", "text": "water"},
        {"id": 90, "title": "", "text": ""},
    ]


def test_inverted_index_maps_terms_to_actual_document_ids(candidate_documents):
    from src.bm25_search import BM25Index

    index = BM25Index(candidate_documents)
    assert index.inverted_index == {
        "programming": {1, 20},
        "design": {20},
        "python": {1, 6, 10},
        "scripts": {6, 10},
        "bread": {81},
        "water": {81},
    }
    # Repetition belongs in term frequency, not duplicate postings.
    assert index.term_frequencies[index.document_positions[1]]["python"] == 2
    for term, document_ids in index.inverted_index.items():
        assert len(document_ids) == index.document_frequencies[term]


def test_candidate_union_scores_each_match_once(candidate_documents, monkeypatch):
    from src.bm25_search import BM25Index

    index = BM25Index(candidate_documents)
    expected_candidates = {1, 6, 10, 20}
    assert index.candidate_document_ids(["python", "programming"]) == expected_candidates
    assert index.candidate_document_ids(["python", "python", "unknown"]) == {1, 6, 10}
    scored_document_ids = []
    original_score_document = index._score_document

    def record_scoring(document_position, query_terms):
        document_id = index.documents[document_position]["id"]
        scored_document_ids.append(document_id)
        return original_score_document(document_position, query_terms)

    monkeypatch.setattr(index, "_score_document", record_scoring)
    results = index.search("PYTHON python programming unknown", top_k=10)
    assert set(scored_document_ids) == expected_candidates
    assert len(scored_document_ids) == len(expected_candidates)
    assert {result["id"] for result in results} == expected_candidates
    assert results[0]["id"] == 1
    assert index.last_candidate_count == 4

    # Reusing the index must update instrumentation, even when no term matches.
    for query in ["bread", "unknown", "", "!!!"]:
        scored_document_ids.clear()
        index.search(query)
        expected_ids = {81} if query == "bread" else set()
        assert set(scored_document_ids) == expected_ids
        assert index.last_candidate_count == len(expected_ids)


def full_scan_reference(documents, query, top_k):
    """Independently reproduce milestone 2 BM25 by scanning the entire corpus.

    Keep this slower reference in tests so candidate filtering is checked
    against the full formula, without calling production scoring helpers.
    """
    from collections import Counter

    query_terms = sorted(set(tokenize(query)))
    document_count = len(documents)
    if not document_count or not query_terms:
        return []

    term_frequencies = []
    document_lengths = []
    document_frequencies = Counter()
    total_tokens = 0
    for document in documents:
        tokens = tokenize(document["title"] + " " + document["text"])
        term_counts = Counter(tokens)
        term_frequencies.append(term_counts)
        document_lengths.append(len(tokens))
        total_tokens += len(tokens)
        for term in term_counts:
            document_frequencies[term] += 1

    average_document_length = total_tokens / document_count
    if average_document_length == 0:
        return []

    matching_results = []
    k1, b = 1.5, 0.75
    for position, document in enumerate(documents):
        score = 0.0
        for term in query_terms:
            term_frequency = term_frequencies[position][term]
            if term_frequency == 0:
                continue
            document_frequency = document_frequencies[term]
            idf = math.log(1 + (document_count - document_frequency + 0.5)
                           / (document_frequency + 0.5))
            length_ratio = document_lengths[position] / average_document_length
            normalization = 1 - b + b * length_ratio
            numerator = term_frequency * (k1 + 1)
            denominator = term_frequency + k1 * normalization
            score += idf * numerator / denominator
        if score > 0:
            result = document.copy()
            result["score"] = score
            matching_results.append(result)

    def ranking_key(result):
        return (-result["score"], result["id"])

    return sorted(matching_results, key=ranking_key)[:top_k]


@pytest.mark.parametrize("query", [
    "python programming", "scripts", "bread water", "python bread",
    "PYTHON python! unknown", "unknown", "", "!!!",
])
@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_inverted_search_matches_full_scan(candidate_documents, query, top_k):
    from src.bm25_search import BM25Index

    expected = full_scan_reference(candidate_documents, query, top_k)
    actual = BM25Index(candidate_documents).search(query, top_k)
    assert [result["id"] for result in actual] == [result["id"] for result in expected]
    for actual_result, expected_result in zip(actual, expected):
        assert actual_result["score"] == expected_result["score"]


def test_benchmark_reports_actual_candidate_counts(monkeypatch, capsys):
    import src.benchmark as benchmark

    generated_documents = benchmark.generate_documents(100)
    expected_counts = []
    for query in benchmark.BENCHMARK_QUERIES:
        # A full scan independently establishes how many documents match.
        matches = full_scan_reference(generated_documents, query, top_k=100)
        expected_counts.append(len(matches))

    benchmark.run_benchmark(100, top_k=1)
    output_lines = capsys.readouterr().out.splitlines()
    reported_counts = []
    for line in output_lines:
        if line.startswith("Candidate documents:"):
            reported_counts.append(int(line.split(":")[1]))
    assert reported_counts == expected_counts


def test_duplicate_ids_are_rejected():
    from src.bm25_search import BM25Index

    documents = [
        {"id": 1, "title": "Python", "text": "code"},
        {"id": 1, "title": "Bread", "text": "water"},
    ]
    with pytest.raises(ValueError, match="Document ids must be unique"):
        BM25Index(documents)
