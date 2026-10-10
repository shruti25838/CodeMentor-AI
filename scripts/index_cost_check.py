"""Check that batching order does not change a vector, and what an index really costs.

    python scripts/index_cost_check.py --check-order        # 2 calls of 50 documents
    python scripts/index_cost_check.py --index 600          # one real index, sorted
    python scripts/index_cost_check.py --index 600 --unsorted

Both modes **call the API for real and bypass the vector cache**, because the question in each
case is about what Cloudflare does, not about what is already on disk. Each prints its
estimate before sending anything.

`--check-order` is the safety check behind length-sorted batching. Sorting documents by length
before batching costs about half as much, because a batch is billed by its longest member and
sorted batches hold texts of similar length - but it is only worth anything if the vectors come
back identical, since a vector that depends on which other texts shared its call would make
every index quietly dependent on the order its documents happened to be read in. So the same
fifty documents are embedded twice, once in file order and once in length order, and the
smallest cosine between the two vectors of the same document is reported.

`--index N` builds one real index of N documents and prints the estimate beside what the ledger
actually recorded, which is the figure to compare against the account dashboard.

Nothing here prints a document, a URL or a credential.
"""

import argparse
import math
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import cloudflare_settings, get_config  # noqa: E402
from codeatlas.models.repository import Repository  # noqa: E402
from codeatlas.services.eval.retrieval_eval import checkout, load_question_set  # noqa: E402
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.retrieval import neuron_ledger  # noqa: E402
from codeatlas.services.retrieval.cloudflare_embedder import CloudflareEmbeddingService  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402
from codeatlas.utils.config import load_cloudflare_credentials  # noqa: E402

# A vector that moved by more than this between two batchings would mean the batching is part
# of what an index is, and length-sorting would not be safe to apply.
REQUIRED_COSINE = 0.9999
DEFAULT_SET = ROOT / "eval" / "questions" / "flask.json"
REPO_CACHE = ROOT / ".codeatlas" / "eval-repos"


def documents_for(count: int | None, clip: int, question_file: Path) -> tuple[list[str], object, object]:
    question_set = load_question_set(question_file)
    root = checkout(question_set, REPO_CACHE)
    repository = Repository(
        repo_id=f"cost-{question_set.name}",
        name=question_set.name,
        url=question_set.repo_url,
        root_path=str(root.resolve()),
        ingested_at=datetime.now(UTC),
        commit=question_set.commit,
    )
    parsed = TreeSitterAstParser().parse_repository(repository)
    indexer = CodeIndexService(
        embedder=_embedder(clip), retriever=FaissCodeRetriever(), max_chars=clip, prefix_metadata=True
    )
    documents, _ = indexer.documents(repository, parsed)
    return (documents if count is None else documents[:count]), repository, parsed


def _embedder(clip: int) -> CloudflareEmbeddingService:
    settings = replace(cloudflare_settings(get_config()), max_chars=clip)
    return CloudflareEmbeddingService(credentials=load_cloudflare_credentials(), settings=settings)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def check_order(clip: int, batch_size: int, sample: int, question_file: Path) -> int:
    documents, _, _ = documents_for(sample, clip, question_file)
    embedder = _embedder(clip)
    in_file_order = list(documents)
    in_length_order = sorted(range(len(documents)), key=lambda i: len(documents[i]))

    estimate = embedder.estimated_neurons(in_file_order) + embedder.estimated_neurons(
        [documents[i] for i in in_length_order]
    )
    print(f"documents: {len(documents)}, clip {clip}, batch size {batch_size}")
    print(f"estimate for both calls: {estimate:.2f} estimated neurons")
    print(f"ledger before: {neuron_ledger.spent_today().estimated_neurons:.2f}")

    unsorted_vectors = embedder.embed_texts(in_file_order)
    sorted_vectors = embedder.embed_texts([documents[i] for i in in_length_order])
    restored = [None] * len(documents)
    for position, index in enumerate(in_length_order):
        restored[index] = sorted_vectors[position]

    similarities = [cosine(unsorted_vectors[i], restored[i]) for i in range(len(documents))]
    worst = min(similarities)
    print(f"ledger after: {neuron_ledger.spent_today().estimated_neurons:.2f}")
    print(f"minimum cosine between the two vectors of the same document: {worst:.6f}")
    print(f"required: {REQUIRED_COSINE}")
    if worst >= REQUIRED_COSINE:
        print("PASS: batching order does not change a vector")
        return 0
    print("FAIL: a vector depends on what shared its call; do not sort")
    return 1


def build_index(count: int, clip: int, batch_size: int, sort: bool, question_file: Path) -> int:
    """Embed N documents for real, exactly as indexing would, and report estimate against ledger."""
    documents, _, _ = documents_for(count, clip, question_file)
    embedder = _embedder(clip)
    in_file_order = neuron_ledger.estimated_neurons(documents, batch_size=batch_size)
    in_length_order = neuron_ledger.estimated_neurons(sorted(documents, key=len), batch_size=batch_size)
    before = neuron_ledger.spent_today().estimated_neurons
    print(f"documents: {len(documents)}, clip {clip}, batch size {batch_size}, length-sorted: {sort}")
    print(f"estimate in file order:   {in_file_order:.2f} estimated neurons")
    print(f"estimate in length order: {in_length_order:.2f} estimated neurons")
    print(f"ledger before: {before:.2f}")

    indexer = CodeIndexService(
        embedder=embedder,
        retriever=FaissCodeRetriever(),
        max_chars=clip,
        prefix_metadata=True,
        batch_size=batch_size,
        sort_by_length=sort,
    )
    vectors = indexer.embed_documents(documents)
    after = neuron_ledger.spent_today()
    print(f"vectors: {len(vectors)}")
    print(f"ledger after:  {after.estimated_neurons:.2f} ({after.estimated_neurons - before:.2f} for this index)")
    print(f"calls today: {after.calls}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-order", action="store_true", help="embed the same documents two ways and compare")
    parser.add_argument("--index", type=int, metavar="N", help="build one real index of N documents")
    parser.add_argument("--sample", type=int, default=50, help="documents for --check-order (default 50)")
    parser.add_argument("--clip", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--unsorted", action="store_true", help="with --index, batch in file order")
    parser.add_argument("--set", dest="question_file", type=Path, default=DEFAULT_SET)
    args = parser.parse_args()

    if load_cloudflare_credentials() is None:
        print("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set (.env is read).")
        return 1
    if args.check_order:
        return check_order(args.clip, args.batch_size, args.sample, args.question_file)
    if args.index:
        return build_index(args.index, args.clip, args.batch_size, not args.unsorted, args.question_file)
    parser.error("choose --check-order or --index N")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
