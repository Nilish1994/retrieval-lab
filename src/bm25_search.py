"""BM25 with positive Robertson IDF and saturated term frequency."""
from collections import Counter
from math import log

from .tokenizer import tokenize


def search(documents: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """Rank title + text; omit zero scores and break ties by document id."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    terms = sorted(set(tokenize(query)))
    if not documents or not terms:
        return []
    counts = [Counter(tokenize(d["title"] + " " + d["text"])) for d in documents]
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(documents)
    if not average_length:
        return []
    frequencies = Counter(term for count in counts for term in count)
    k1, b = 1.5, 0.75
    results = []
    for doc, count, length in zip(documents, counts, lengths):
        score = 0.0
        for term in terms:
            tf = count[term]
            if not tf:
                continue
            df = frequencies[term]
            idf = log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            score += idf * tf * (k1 + 1) / (
                tf + k1 * (1 - b + b * length / average_length)
            )
        if score > 0:
            results.append({**doc, "score": score})
    return sorted(results, key=lambda r: (-r["score"], r["id"]))[:top_k]
