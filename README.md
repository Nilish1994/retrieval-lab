# Retrieval lab — milestone 1

Local keyword retrieval in Python using BM25. Six sample documents live in
`data/documents.json`. No AI services, LLMs, embeddings, vector databases,
PostgreSQL, or web UI are used. Search uses the Python standard library;
pytest is the only development dependency.

## Setup and commands

Run from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q

python -m src.cli add --title "Example" --text "Example text"
python -m src.cli list
python -m src.cli search "search terms"
python -m src.cli search "python programming" --top-k 3
python -m src.cli search "BM25 search ranking" --top-k 3
python -m src.cli search "bread water" --top-k 3
```

On Debian/Ubuntu, creating a fresh environment requires the `python3-venv`
package. The existing `.venv` is ready to use; on this machine its pip was
bootstrapped from the official PyPA installer because `ensurepip` was absent.

`add` persists a document with a unique integer ID. `list` shows IDs and titles.
`search` shows ranks, IDs, titles, scores to four decimal places, and previews.
`--top-k` defaults to 5 and must be positive. Only positive-scoring matches are
returned, so fewer than k results is normal. Ties use ascending ID. Blank or
unknown queries return no matches. Storage paths resolve relative to the source.

## BM25, TF, IDF, and length normalization

BM25 sums a relevance contribution for each distinct query term found in a
document. Title and text are combined with equal weight.

```text
score(D, Q) = sum over distinct query terms t:
    IDF(t) * TF(t,D) * (k1 + 1)
    / (TF(t,D) + k1 * (1 - b + b * length(D) / average_length))

IDF(t) = ln(1 + (N - DF(t) + 0.5) / (DF(t) + 0.5))
k1 = 1.5; b = 0.75
```

- **TF (term frequency)** counts occurrences of a term in a document. Repetition
  increases the score with diminishing returns; `k1` controls saturation.
- **DF (document frequency)** counts documents containing a term. **IDF (inverse
  document frequency)** gives rarer terms more weight. `N` is the document count.
  This IDF variant is positive for every indexed term.
- **Document length normalization** compares token count with the corpus average.
  At equal TF, longer documents receive a smaller contribution. `b` controls the
  correction: 0 disables it and 1 applies full length correction.

Scores are relative relevance values, not probabilities, and change when the
corpus changes. Unmatched documents score zero. An empty corpus or one with no
tokens returns no results.

## Query flow

1. Load and validate the JSON document list.
2. Lowercase title + text and extract letter/digit runs, splitting on punctuation
   (including underscores) and whitespace. Tokenize the query identically.
3. Compute term counts, document lengths, average length, and document frequency.
4. Sum BM25 contributions for each distinct query term in each document.
5. Remove zero scores, sort descending, and display the top k matches.

There is no stemming, stop-word removal, phrase matching, or query expansion.
Repeated query terms count once. Statistics are rebuilt on every search, so new
documents are immediately searchable. This implementation scans all documents
and is intended for small local corpora.

## Storage and scope

`data/private/` is ignored by Git and never automatically read or indexed.
Documents explicitly added through the CLI go into Git-visible
`data/documents.json`; keep private material out of this file if publishing it.

Writes replace the JSON file atomically to prevent partial files. Concurrent
writers are not supported; run one add command at a time. Malformed stores
produce an error instead of being overwritten. Tests use temporary storage for
add operations and cover retrieval, scores, normalization, top-k, tokenization,
and empty/unmatched inputs. Milestone 1 ends at this local CLI.
