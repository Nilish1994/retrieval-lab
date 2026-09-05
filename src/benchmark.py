"""Generate a varied local corpus and time one index build plus several queries.

The seeded generator keeps experiments reproducible without writing a dataset.
Timings use perf_counter and exclude printing from the measured operations.
"""
import random
from time import perf_counter

from .bm25_search import BM25Index

# Each topic has a focused vocabulary and several details. Random repetitions
# and detail counts vary term frequency and document length within each topic.
TOPICS = [
    ("Python programming", "python programming functions scripts", [
        "Classes organize reusable code and modules.",
        "Testing helps find bugs in automation scripts.",
        "Lists and dictionaries store application data.",
    ]),
    ("Linux servers", "linux server ssh security", [
        "SSH keys control remote access to the server.",
        "Security updates and firewall rules protect services.",
        "Administrators inspect logs and schedule backups.",
    ]),
    ("Databases", "database tables sql transactions", [
        "Queries select rows and join tables.",
        "Transactions preserve consistency during updates.",
        "Backups help recover stored records.",
    ]),
    ("Networking", "network routing packets tcp", [
        "Routers forward packets between networks.",
        "DNS resolves names for remote services.",
        "Latency and bandwidth affect connection performance.",
    ]),
    ("Gardening", "garden tomatoes soil water", [
        "Tomatoes need sunlight and regular watering.",
        "Compost improves soil for growing vegetables.",
        "Pruning and mulch help plants thrive.",
    ]),
    ("Cooking", "bread water flour cooking", [
        "Bread dough needs water and time to ferment.",
        "A hot oven creates a crisp crust.",
        "Salt and yeast affect flavor and texture.",
    ]),
    ("Cars", "cars engine tires maintenance", [
        "Oil changes protect engine components.",
        "Tire pressure affects handling and fuel use.",
        "Brake inspections support reliable driving.",
    ]),
    ("Astronomy", "astronomy stars planets telescope", [
        "A telescope reveals distant galaxies.",
        "Planets orbit stars under gravity.",
        "Dark skies improve observations of the moon.",
    ]),
    ("Finance", "finance savings interest budget", [
        "A budget tracks income and expenses.",
        "Compound interest changes savings over time.",
        "Diversification spreads investment risk.",
    ]),
    ("History", "history ancient empire trade", [
        "Ancient cities grew around trade routes.",
        "Written records help historians study an empire.",
        "Archaeology uncovers artifacts from past societies.",
    ]),
]

BENCHMARK_QUERIES = [
    "python programming",
    "SSH security server",
    "bread water",
    "astronomy telescope planets",
]


def generate_documents(document_count: int, seed: int = 42) -> list[dict]:
    """Create evenly distributed topics with repeatable vocabulary variation."""
    if document_count < 1:
        raise ValueError("document_count must be at least 1")

    random_generator = random.Random(seed)
    documents = []
    for document_position in range(document_count):
        topic_position = document_position % len(TOPICS)
        title, keywords, details = TOPICS[topic_position]
        text_parts = []

        repetition_count = random_generator.randint(1, 3)
        for _ in range(repetition_count):
            text_parts.append(keywords + ".")

        detail_count = random_generator.randint(1, len(details))
        selected_details = random_generator.sample(details, detail_count)
        text_parts.extend(selected_details)

        # Some cross-topic content gives broad queries partial matches too.
        if random_generator.random() < 0.25:
            other_topic = random_generator.choice(TOPICS)
            text_parts.append(random_generator.choice(other_topic[2]))

        document_id = document_position + 1
        document = {
            "id": document_id,
            "title": f"{title} notes {document_id}",
            "text": " ".join(text_parts),
        }
        documents.append(document)

    return documents


def run_benchmark(
    document_count: int = 10000, top_k: int = 3, stemming: bool = False
) -> None:
    """Time generation, one construction, and individual searches on that index."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    start_time = perf_counter()
    documents = generate_documents(document_count)
    generation_seconds = perf_counter() - start_time

    start_time = perf_counter()
    index = BM25Index(documents, stemming=stemming)
    index_seconds = perf_counter() - start_time

    mode = "Porter stemming" if stemming else "Exact tokens"
    print(f"Mode: {mode}")
    print(f"Documents: {index.document_count}")
    print(f"Generation time: {generation_seconds:.6f} seconds")
    print(f"Index build time: {index_seconds:.6f} seconds")

    for query in BENCHMARK_QUERIES:
        start_time = perf_counter()
        results = index.search(query, top_k)
        search_seconds = perf_counter() - start_time

        print(f"\nQuery: {query}")
        print(f"Candidate documents: {index.last_candidate_count}")
        print(f"Search time: {search_seconds:.6f} seconds")
        for rank, result in enumerate(results, start=1):
            print(f"{rank}. [{result['id']}] {result['title']}")
            print(f"   Score: {result['score']:.4f}")
        if not results:
            print("No matching documents.")
