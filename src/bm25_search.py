"""Build BM25 corpus statistics once, then reuse them to score queries.

Index construction tokenizes documents and calculates frequencies and IDF.
An inverted index maps each term to the IDs of documents containing it.
Search unions query-term postings, scores only those candidates, and ranks them.
"""
from collections import Counter
from math import log

from .tokenizer import normalize_tokens, tokenize


class BM25Index:
    """An in-memory snapshot of documents and their reusable BM25 statistics.

    Treat the stored attributes as read-only. Build a new instance when the
    corpus or normalization mode changes so frequencies, IDF, and postings stay
    consistent. Exact tokens are the default; stemming applies to both sides.
    """

    def __init__(self, documents: list[dict], stemming: bool = False) -> None:
        """Copy and tokenize documents; build corpus statistics and term postings."""
        self.stemming = stemming
        self.documents = []
        self.tokenized_documents = []
        self.document_lengths = []
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.idf_values = {}
        self.inverted_index: dict[str, set[int]] = {}
        self.document_positions: dict[int, int] = {}
        # Expose the count from the actual search, without a second lookup pass.
        self.last_candidate_count = 0
        self.document_count = len(documents)
        self.average_document_length = 0.0

        # k1 controls term-frequency saturation; b controls length correction.
        self.k1 = 1.5
        self.b = 0.75
        total_tokens = 0

        for document_position, document in enumerate(documents):
            document_id = document["id"]
            if document_id in self.document_positions:
                raise ValueError("Document ids must be unique")
            # IDs need not be consecutive or match positions in the stored lists.
            self.document_positions[document_id] = document_position
            # A copy prevents later edits to the input from changing our snapshot.
            self.documents.append(document.copy())
            document_text = document["title"] + " " + document["text"]
            document_tokens = tokenize(document_text)
            normalized_tokens = normalize_tokens(document_tokens, self.stemming)
            document_length = len(normalized_tokens)
            term_counts = Counter(normalized_tokens)

            self.tokenized_documents.append(normalized_tokens)
            self.document_lengths.append(document_length)
            self.term_frequencies.append(term_counts)
            total_tokens += document_length

            # Count a term once per document, regardless of its repetitions.
            for term in term_counts:
                self.document_frequencies[term] += 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = set()
                self.inverted_index[term].add(document_id)

        if self.document_count > 0:
            self.average_document_length = total_tokens / self.document_count

        for term, document_frequency in self.document_frequencies.items():
            # Positive Robertson IDF: terms found in fewer documents weigh more.
            numerator = self.document_count - document_frequency + 0.5
            denominator = document_frequency + 0.5
            self.idf_values[term] = log(1 + numerator / denominator)

    def _score_document(self, document_position: int, query_terms: list[str]) -> float:
        """Sum query-term contributions using statistics already in the index."""
        term_counts = self.term_frequencies[document_position]
        document_length = self.document_lengths[document_position]
        length_ratio = document_length / self.average_document_length
        length_normalization = 1 - self.b + self.b * length_ratio
        score = 0.0

        for term in query_terms:
            term_frequency = term_counts[term]
            if term_frequency == 0:
                continue

            # Repetition helps with diminishing returns. At equal term frequency,
            # a longer document receives a smaller contribution.
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * length_normalization
            term_score = self.idf_values[term] * numerator / denominator
            score += term_score

        return score

    def candidate_document_ids(self, query_terms: list[str]) -> set[int]:
        """Union postings for already-normalized terms; any term may match."""
        candidate_document_ids = set()
        for query_term in query_terms:
            matching_document_ids = self.inverted_index.get(query_term, set())
            candidate_document_ids.update(matching_document_ids)
        return candidate_document_ids

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return positive matches by descending score, breaking ties by ID."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        query_tokens = tokenize(query)
        normalized_tokens = normalize_tokens(query_tokens, self.stemming)
        # Deduplicate after normalization: forms sharing a stem count only once.
        # Sorting also makes floating-point addition order repeatable.
        query_terms = sorted(set(normalized_tokens))
        candidate_document_ids = self.candidate_document_ids(query_terms)
        self.last_candidate_count = len(candidate_document_ids)
        if not candidate_document_ids:
            return []

        matching_results = []
        # Only postings-selected IDs are visited. Final sorting settles score
        # ties by document ID, independent of the set's iteration order.
        for document_id in candidate_document_ids:
            document_position = self.document_positions[document_id]
            document = self.documents[document_position]
            score = self._score_document(document_position, query_terms)
            if score > 0:
                result = document.copy()
                result["score"] = score
                matching_results.append(result)

        sorted_results = sorted(matching_results, key=_ranking_key)
        return sorted_results[:top_k]


def _ranking_key(result: dict) -> tuple[float, int]:
    """Negate the score for highest-first ordering; use ID to settle ties."""
    return (-result["score"], result["id"])


def search(
    documents: list[dict], query: str, top_k: int = 5, stemming: bool = False
) -> list[dict]:
    """Keep the milestone 1 API; use BM25Index directly to reuse indexing work."""
    index = BM25Index(documents, stemming=stemming)
    return index.search(query, top_k)
