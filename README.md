# Retrieval lab — milestones 1, 2, 3, and 4A

[![DOI](https://zenodo.org/badge/1357773047.svg)](https://doi.org/10.5281/zenodo.24214970)

Local keyword retrieval in Python using BM25. The local document collection lives in
`data/documents.json`; milestone 1 included six sample documents. No AI services, LLMs, embeddings, vector databases,
PostgreSQL, or web UI are used. BM25 uses the Python standard library; optional
Porter stemming uses NLTK. Pytest is the development dependency.

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
2. Build an index: lowercase title + text and extract letter/digit runs, splitting
   on punctuation (including underscores) and whitespace.
3. Optionally apply Porter stemming to document tokens, then store normalized
   tokens, term counts, lengths, average length, document frequency, and IDF
   once during construction; also build term-to-document-ID postings.
4. For each query, apply the same tokenization and optional stemming, then union
   the matching postings.
5. Score only those candidate IDs using the existing BM25 statistics.
6. Remove zero scores, sort descending, and display the top k matches.

Exact tokens are the default; `--stemming` enables Porter normalization. There
is no stop-word removal, phrase matching, or query expansion.
Repeated query terms count once. An index reuses statistics across searches.
A new CLI search process builds a fresh index, so newly added documents are
searchable on the next invocation. Search visits only candidate documents.

## Storage and scope

`data/private/` is ignored by Git and never automatically read or indexed.
Documents explicitly added through the CLI go into Git-visible
`data/documents.json`; keep private material out of this file if publishing it.

Writes replace the JSON file atomically to prevent partial files. Concurrent
writers are not supported; run one add command at a time. Malformed stores
produce an error instead of being overwritten. Tests use temporary storage for
add operations and cover retrieval, scores, normalization, top-k, tokenization,
and empty/unmatched inputs. The project remains a local CLI.


## Milestone 2: reusable in-memory BM25 index

An **index** is a prepared representation of a document collection that makes
retrieval work reusable. Here it consists of token lists and calculated BM25
statistics held in a Python object. Milestone 2 did not include an inverted
index; milestone 3 adds the postings described below.

Milestone 1 repeated document tokenization, counting, and IDF calculation on every
query. These values depend on the corpus, not the query, so repeating that work
for an unchanged corpus is wasteful. Constructing `BM25Index` does that work once:

```python
from src.bm25_search import BM25Index
from src.store import load_documents

index = BM25Index(load_documents())
python_results = index.search("python programming")
server_results = index.search("SSH security server")
bread_results = index.search("bread water", top_k=3)
```

The index stores copied documents, tokenized title + text, document lengths,
document count, average length, per-document term frequencies, corpus document
frequencies, and IDF values for every indexed term. All per-document lists use
the same positions, so a document's tokens, length, and counts line up.

For each search, only the query is tokenized. Its distinct terms are scored
against the stored term counts and IDF, then positive matches are sorted and
truncated to top-k. Avoiding corpus tokenization and statistic construction
removes repeated work. Milestone 2 still scanned every document. Milestone 3
replaces that scan with postings-based candidate selection.

The counts mean different things:

| Value | Meaning | Example |
| --- | --- | --- |
| Document count | Total documents in the corpus | 10,000 documents |
| Document frequency | Documents containing a particular term | `python` appears in 1,000 documents |
| Term frequency | Occurrences of a term in one document | `python` occurs 3 times in one document |
| IDF | Weight calculated from document count and document frequency | Rarer terms receive a larger weight |

`search(documents, query)` remains available for milestone 1 compatibility, but
that convenience function constructs a new index each time. Reuse a `BM25Index`
instance for repeated queries. Separate `python -m src.cli search` invocations
are separate processes and cannot share an in-memory object; the benchmark
explicitly reuses one instance in a single process.

### Performance experiment

```bash
source .venv/bin/activate
python -m src.cli benchmark --documents 10000 --top-k 3
```

The generator cycles through ten subject categories and uses a fixed random seed.
Keyword repetitions, detail counts, and occasional cross-topic sentences vary
term frequencies, document lengths, and matches. The dataset is generated only
in memory; nothing is saved to `data/documents.json` or committed to Git.

The command uses `time.perf_counter()` to measure separately:

1. Document generation, including text assembly.
2. Construction of one BM25 index, including document copies and all statistics.
3. Each of four queries against that same index, including query tokenization,
   candidate selection, scoring, sorting, and selection of results.

Printing is outside the timed regions. Output includes actual durations and
ranked scores. The experiment demonstrates the one-time indexing cost separately
from repeated search costs on 10,000 documents. It does not claim a measured
speedup over milestone 1, which is not timed alongside it. Timings vary with
hardware, system load, and corpus; tests have no fixed speed thresholds.

### Source-code walkthrough

- `src/tokenizer.py`: `tokenize` produces lowercase tokens; `normalize_tokens`
  optionally reduces them with NLTK Porter. Both index and query use this layer.
- `src/store.py`: validates, loads, and atomically saves JSON documents.
- `src/bm25_search.py`: `BM25Index.__init__` builds corpus statistics, postings,
  and an ID-to-position map; `candidate_document_ids` unions postings;
  `_score_document` explains the unchanged formula; `search` scores candidates
  and ranks results.
  The module-level `search` preserves the original interface.
- `src/cli.py`: parses storage, search, benchmark, and `analyze` commands;
  forwards `--stemming` to search and benchmark.
- `src/benchmark.py`: generates varied documents and measures one build and
  multiple searches without writing a dataset; reports mode and actual candidate
  counts. The generator and queries are identical in both modes.
- `tests/test_bm25_search.py`: preserves milestone 1 tests and checks stored
  statistics, reuse without recounting or IDF calculation, rankings, snapshot
  behavior, edge cases, and benchmark generation and reuse. Milestone 3 adds
  postings checks, scorer-call checks, and an independent full-scan reference.

### Current limitations

The index lasts only for the Python process. It uses memory for both tokens and
counts for inspection and learning. Treat its public statistics as read-only.
Adding or editing documents requires constructing a new index: an existing
instance is a snapshot and does not watch the JSON file. There is no incremental
update, disk-persisted index, or concurrency control for writers. Milestone 3
uses an inverted index, but still sorts all positive matches even for small
values of top-k. The synthetic corpus is a controlled learning experiment, not
a substitute for measuring retrieval quality on real documents.


## Milestone 3: inverted index and candidate scoring

An **inverted index** maps a term to the documents containing it. A **posting
list** is simply the collection of document references stored for one term:

```text
ssh -> [12, 81, 1272, 1822]
```

This means those four document IDs contain `ssh`. Our implementation uses a
Python set for each term rather than a list, so repeated occurrences do not add
duplicate IDs. Term frequency is stored separately for BM25. Postings include
both title and text tokens. IDs are actual document IDs, not positions: a separate
`document_positions` dictionary locates each ID in the existing statistics lists.
IDs must be unique; construction rejects duplicates instead of silently losing
a document from the postings.

Search engines use these mappings to avoid inspecting documents that cannot
match. Milestone 2 reused statistics but its scoring loop still visited all N
documents, so the scanning cost grew roughly linearly with corpus size for a
fixed query. Milestone 3 selects **candidates**: documents with at least one
query term, which are the only documents capable of a positive BM25 score.

```text
python      -> {1, 6, 10}
programming -> {1, 20}
union       -> {1, 6, 10, 20}
```

We use the **union**, not the intersection, because normal BM25 also ranks partial
matches. Requiring every term would discard relevant partial matches and change
the existing behavior. Unknown terms have empty postings. Empty queries and
queries containing only unknown terms have no candidates and score no documents.
Repeated query terms cannot duplicate candidates or increase their scores.

Candidate selection only answers which documents might be relevant. BM25 still
orders them using term frequency, IDF, saturation, and length normalization. The
formula, `k1 = 1.5`, `b = 0.75`, corpus-wide statistics, descending score order,
and ascending-ID tie breaking are unchanged. IDF and average length continue to
use the entire corpus, not just the candidate subset.

### Work at construction and at query time

Once at construction:

1. Copy documents and tokenize their title + text.
2. Store lengths and term frequencies; count each term once per document.
3. Add the document ID to each unique term's postings during that same pass.
4. Calculate average length and IDF across the complete corpus.

Once per query:

1. Tokenize the query and deduplicate its terms.
2. Look up postings and explicitly union their document IDs.
3. Record the candidate set's size in `last_candidate_count`.
4. Locate and BM25-score only those IDs.
5. Sort positive results by score and ID, then return top-k.

The benchmark prints `last_candidate_count` immediately after each search. It is
the size of the actual set used for scoring, before top-k truncation, with no
second candidate lookup. Each successful search replaces this count, including
empty and unknown queries. It describes the most recent search on that instance
and is intended for sequential inspection, not concurrent query reporting.

### Benchmark and trade-offs

Run the original seeded generator and original four queries at both sizes:

```bash
python -m src.cli benchmark --documents 10000 --top-k 3
python -m src.cli benchmark --documents 100000 --top-k 3
```

Corpus generation and query strings are unchanged from milestone 2. Generation,
index construction, and each complete search are timed separately with
`perf_counter()`. Query timing includes candidate selection, scoring, sorting,
and top-k; printing happens afterwards. Generated documents stay in memory.

Postings and the ID lookup require additional memory and construction work. In
return, selective queries avoid most document scoring. Common query terms may
still select most or all of the corpus, so an inverted index does not guarantee
sublinear work for every query. Postings union costs depend on the matching
posting sizes; scoring depends on candidate count and query length. All matches
are still sorted, even when only three results are requested.

The index remains an in-memory snapshot and must be rebuilt when documents
change. Tokens, term counts, and postings are deliberately kept explicit for
learning, with no compression or index persistence. Milestone 3 used exact
tokens only; milestone 4A adds optional stemming below. There is still no phrase
search, database, API, UI, vector search, or AI.

### Measured milestone 3 run

On this VPS, using the commands above (seconds, one run at each size):

| Documents | Generation | Index build |
| --- | ---: | ---: |
| 10,000 | 0.034197 | 0.200664 |
| 100,000 | 0.308857 | 2.645455 |

| Query | Candidates (10k) | Search (10k) | Candidates (100k) | Search (100k) |
| --- | ---: | ---: | ---: | ---: |
| python programming | 1,000 | 0.002943 | 10,000 | 0.047310 |
| SSH security server | 1,141 | 0.002690 | 11,521 | 0.044349 |
| bread water | 2,063 | 0.005351 | 20,648 | 0.066024 |
| astronomy telescope planets | 1,113 | 0.002680 | 11,430 | 0.043557 |

Top three results are listed in rank order below. Titles follow the generator's
`<topic> notes <ID>` format. All three results in each row have the displayed
score (rounded to four decimal places).

| Documents | Query | Topic | Ranked IDs | Score |
| --- | --- | --- | --- | ---: |
| 10,000 | python programming | Python programming | 1, 61, 241 | 8.5778 |
| 10,000 | SSH security server | Linux servers | 1272, 1822, 1952 | 11.9385 |
| 10,000 | bread water | Cooking | 706, 856, 866 | 7.0440 |
| 10,000 | astronomy telescope planets | Astronomy | 738, 1118, 1518 | 12.5612 |
| 100,000 | python programming | Python programming | 1, 61, 241 | 8.5756 |
| 100,000 | SSH security server | Linux servers | 1272, 1822, 1952 | 11.9119 |
| 100,000 | bread water | Cooking | 706, 856, 866 | 7.0344 |
| 100,000 | astronomy telescope planets | Astronomy | 78, 228, 288 | 12.5060 |

At 100,000 documents, measured query times of 0.043557–0.066024 seconds are below
the supplied milestone 2 baseline range of approximately 0.095–0.119 seconds.
This run shows an improvement against that historical range, not a controlled
per-query speedup ratio. Construction took 2.645455 seconds versus the supplied
historical build time of about 1.65 seconds. Memory use was not measured. These
are observations from a single run, not speed guarantees; system load and
hardware affect timings. Scores and even rankings can change between corpus
sizes because IDF and average document length change.


## Milestone 4A: exact tokens versus Porter stems

The original failures came from exact token lookup: `servers` was absent from
postings containing `server`; `secure` and `security`, or `automate` and
`automation`, were separate terms. Lowercasing and punctuation splitting do not
connect these forms.

**Stemming** mechanically reduces word forms using spelling rules. **Porter** is
a classical suffix-stripping algorithm. A **token** is a unit extracted from
text, such as `security`. A **stem** is the algorithm's reduced form, such as
`secur`; it need not be a dictionary word. We use `nltk.stem.PorterStemmer` with
its explicit `NLTK_EXTENSIONS` mode, which is NLTK's default Porter variant.
See the [official NLTK Porter documentation](https://www.nltk.org/api/nltk.stem.porter.html)
for its modes. `requirements.txt` pins the installed NLTK version to 3.10.3.
No WordNet, tokenizer datasets, or other NLTK corpora are downloaded or required.
Our existing regular-expression tokenizer still handles tokenization.

### Selecting and inspecting normalization

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.cli search "servers"
python -m src.cli search "servers" --stemming
python -m src.cli search "security" --stemming
python -m src.cli analyze "servers security automation"
```

The analyze command runs the same tokenizer and normalization function used in
search. Its actual output is:

```text
Original tokens: servers security automation
Porter stems: server secur autom
```

Actual verified pairs in this version:

| Original forms | Porter stem |
| --- | --- |
| server / servers | server |
| secure / security | secur |
| automate / automation | autom |

These are algorithm outputs, not manually coded substitutions. Original document
titles and text remain intact for result display and storage.

```python
from src.bm25_search import BM25Index
from src.store import load_documents

documents = load_documents()
exact_index = BM25Index(documents, stemming=False)
stemmed_index = BM25Index(documents, stemming=True)
exact_results = exact_index.search("servers")
stemmed_results = stemmed_index.search("servers")
print(stemmed_index.last_candidate_count)
```

The module-level `search(documents, query, top_k=5, stemming=False)` wrapper also
supports the option. Build a new index to change modes; treat `index.stemming`
and the other index attributes as read-only. Each index must use one consistent
normalization mode for both documents and queries.

### Document and query flow

```text
text
 ↓
tokenize and lowercase
 ↓
stem each token (optional)
 ↓
inverted index
 ↓
BM25
```

On the document side, `tokenize` extracts lowercase tokens and `normalize_tokens`
returns exact tokens or stems. Term counts, document frequency, IDF, and postings
are all built from that output. `tokenized_documents` holds the terms actually
indexed, so it contains stems when stemming is enabled. Occurrences are preserved:
`secure security` becomes `secur secur`, with term frequency 2. Document frequency
counts the merged term once per document, and postings contain each ID once.
Stemming changes spelling, not token count, so document lengths are preserved.

On the query side, the same functions run with the index's mode. Deduplication
happens **after** normalization: `server servers` contributes `server` once.
Already-normalized terms are passed to `candidate_document_ids`, which unions
postings as before. Normalizing only one side would cause missed matches.

BM25 itself needs no English rules and is unchanged. It uses the frequencies of
whatever terms it receives. Merging word forms can change document frequencies,
IDF, candidates, scores, and ranking. The same `k1`, `b`, IDF formula, length
normalization, union behavior, descending scores, and ascending-ID tie breaking
still apply. Exact mode preserves milestone 3 results.

### Limits: over-stemming and under-stemming

**Over-stemming** merges words too aggressively. In this implementation,
`universe` and `university` both become `univers`, even though a university and
the universe are different things. Extra matches can reduce precision.

**Under-stemming** leaves related forms separate. Here, `generous` becomes
`gener`, while `generosity` becomes `generos`. The spelling rules do not connect
this pair, so a relevant document can still be missed. These examples were
checked against the installed implementation.

Porter does not understand meaning, context, or intent. It is not semantic
search and is not linguistically perfect. It is most appropriate for English
word forms; the tokenizer accepting Unicode does not make the stemmer
multilingual. Lemmatization, synonyms, spelling correction, and semantic methods
remain outside milestone 4A so the effect of this one classical technique is
visible in isolation.

### Validation and performance experiment

`tests/test_stemming.py` covers the verified Porter forms, exact-mode isolation,
normalized statistics/postings, candidate-only scoring, scores against an
independently normalized corpus, query deduplication, empty and unknown queries,
CLI search/analyze, benchmark mode selection, and a fresh process with NLTK data
lookup and download functions disabled. The prior tests retain their assertions;
the benchmark constructor spy forwards the new optional mode argument.

```bash
python -m pytest -q
python -m src.cli benchmark --documents 100000 --top-k 3
python -m src.cli benchmark --documents 100000 --top-k 3 --stemming
```

Both benchmark modes use the same seeded documents and four queries. The only
normalization difference is Porter stemming. Timed index construction includes
normalizing every document token; timed queries include query normalization.
Generation and printing boundaries are unchanged. Stemming adds work and may
merge postings, so timing changes are not evidence of a speed optimization.
This milestone evaluates retrieval behavior, not performance improvements.


### Observed local comparison

These results use the existing nine local documents, with no dataset edits.
The query column is also the original token sequence; all examples are already
lowercase and punctuation-free. Exact mode indexes those tokens unchanged.
Results list `[document ID] score` in rank order, using default top-k 5.

Titles: [1] Python programming; [4] Sourdough bread; [6] Python text processing;
[7] VPS SSH Security; [8] Linux Python Automation; [9] Python Cooking Notes.

| Query / original tokens | Porter tokens | Candidates exact / Porter | Exact ranked results | Porter ranked results |
| --- | --- | --- | --- | --- |
| automate | autom | 1 / 2 | [8] 2.2160 | [8] 2.2073; [1] 1.4235 |
| automation | autom | 2 / 2 | [8] 1.6193; [1] 1.4235 | [8] 2.2073; [1] 1.4235 |
| server | server | 2 / 2 | [7] 1.8887; [8] 1.6193 | [7] 1.8887; [8] 1.6193 |
| servers | server | 0 / 2 | No matches | [7] 1.8887; [8] 1.6193 |
| secure | secur | 1 / 2 | [7] 1.7763 | [7] 2.2264; [8] 1.6193 |
| security | secur | 2 / 2 | [7] 1.8887; [8] 1.6193 | [7] 2.2264; [8] 1.6193 |
| python bread | python bread | 5 / 5 | [9] 2.6996; [4] 2.2548; [1] 1.3505; [8] 1.2714; [6] 1.1429 | [9] 2.6996; [4] 2.2548; [1] 1.3505; [8] 1.2714; [6] 1.1429 |

Exact mode was also compared directly with the committed milestone 3 index on
11 queries and produced identical results and scores. `python bread` produced
identical results in both normalization modes on this local corpus.

### Observed 100,000-document benchmark

Actual seconds from one run in each mode:

| Mode | Generation | Index build |
| --- | ---: | ---: |
| Exact | 0.305901 | 3.712621 |
| Porter | 0.331322 | 27.678905 |

| Query | Exact candidates | Exact search | Porter candidates | Porter search |
| --- | ---: | ---: | ---: | ---: |
| python programming | 10,000 | 0.038003 | 10,000 | 0.038397 |
| SSH security server | 11,521 | 0.042809 | 11,521 | 0.040521 |
| bread water | 20,648 | 0.065155 | 21,354 | 0.069290 |
| astronomy telescope planets | 11,430 | 0.038962 | 11,430 | 0.042642 |

Both modes returned these top-three IDs in order. Each row's three results
share the displayed score in that mode, rounded to four decimal places.

| Query | Result topic | Ranked IDs | Exact score | Porter score |
| --- | --- | --- | ---: | ---: |
| python programming | Python programming | 1, 61, 241 | 8.5756 | 8.5756 |
| SSH security server | Linux servers | 1272, 1822, 1952 | 11.9119 | 12.1425 |
| bread water | Cooking | 706, 856, 866 | 7.0344 | 6.9723 |
| astronomy telescope planets | Astronomy | 78, 228, 288 | 12.5060 | 12.5060 |

Stemming added substantial indexing work in this implementation. Query times
were similar in this run; no speed improvement is claimed. Merged terms can
increase candidate counts and change scores, as seen above. Timings depend on
system load and are observations, not guarantees or timing-test thresholds.
