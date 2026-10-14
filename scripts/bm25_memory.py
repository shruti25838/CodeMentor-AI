"""What the in-process BM25 cache actually costs, in the index itself and in the whole process.

    python scripts/bm25_memory.py
    python scripts/bm25_memory.py --json eval/results/experiments/bm25-memory.json

The proposal at d056989 ended on a cost with no number against it: with BM25 on, `AnswerService`
builds a lexical index on the first question about a repository and keeps it, and on a free
Render instance that is memory which grows with the number of repositories a visitor has
analysed. This measures it, two ways, because the two answer different questions.

**The index itself** is walked with `sys.getsizeof`, recursively and deduplicated by object id.
That is a deterministic number: it does not move with the allocator, the garbage collector or
what else the process happens to be holding. It is reported in three parts, because one of them
is not a cost at all - the record *objects* a `Bm25Index` points at are the retriever's own and
are shared with it, so only the list of pointers is new. The postings and the lengths are new.

**The process** is read from the operating system: the working set on Windows, `VmRSS` on
Linux. That is the number a host's memory limit is actually compared against, and it is the
only one that can answer "does this fit". It is also noisy - an allocator does not return freed
pages promptly and nothing here pretends otherwise, which is why the deterministic measurement
is reported beside it rather than instead of it. Reading it through `ctypes` with a `/proc`
fallback and no new dependency is how `scripts/answer_smoke.py` already reads memory.

Everything runs on the **hash embedder only**: this script must not reach a provider, and it
refuses to run under one that would. Which configuration is measured comes from the environment,
exactly as the server reads it, so an arm is selected by setting variables rather than by a flag
this script invents:

    python scripts/bm25_memory.py                                   # whatever is configured
    $env:CODEATLAS_HASH_RECORD_FORMAT = "structured"
    $env:CODEATLAS_HASH_BM25 = "true"
    python scripts/bm25_memory.py                                   # the other arm

Run the two arms in **separate processes**. A process that has already held one arm's indexes
does not give the next arm's number back.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.app.di import get_config, get_embedder, get_retrieval_settings, index_options  # noqa: E402
from codeatlas.models.repository import Repository  # noqa: E402
from codeatlas.services.eval.retrieval_eval import checkout, load_question_set  # noqa: E402
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser  # noqa: E402
from codeatlas.services.qa.answer_service import AnswerService  # noqa: E402
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever  # noqa: E402
from codeatlas.services.retrieval.indexing import CodeIndexService  # noqa: E402

REPO_CACHE = ROOT / ".codeatlas" / "eval-repos"
QUESTIONS = ROOT / "eval" / "questions"
SETS = [QUESTIONS / "itsdangerous.json", QUESTIONS / "flask.json"]

MB = 1024 * 1024


# ---------------------------------------------------------------- reading memory


def working_set_bytes() -> int:
    """This process's resident memory, or -1 when it cannot be read.

    The Windows working set and Linux's VmRSS are the same idea - pages this process has in
    physical memory - and both are what a host's limit is compared against.
    """
    try:
        if sys.platform == "win32":

            class _Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = _Counters()
            counters.cb = ctypes.sizeof(_Counters)
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            handle = kernel32.GetCurrentProcess()
            # Named K32GetProcessMemoryInfo in kernel32 since Windows 7 and still exported from
            # psapi; whichever this build has is the one that answers.
            read = getattr(kernel32, "K32GetProcessMemoryInfo", None) or ctypes.windll.psapi.GetProcessMemoryInfo
            read.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
            if not read(handle, ctypes.byref(counters), counters.cb):
                return -1
            return int(counters.WorkingSetSize)
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        return -1
    return -1


def deep_size(obj, seen: set[int] | None = None) -> int:
    """`sys.getsizeof` over a whole structure, counting each object once.

    Deduplicating by id is the point: a term string that appears in the postings dictionary and
    in a key is one string, and counting it twice would invent memory that is not there.
    """
    seen = set() if seen is None else seen
    marker = id(obj)
    if marker in seen:
        return 0
    seen.add(marker)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += deep_size(key, seen) + deep_size(value, seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            size += deep_size(item, seen)
    return size


def index_bytes(index) -> dict:
    """One BM25 index, in the three parts a reader needs kept apart.

    `records` is the list of pointers only. The `EmbeddingRecord` objects it points at belong to
    the retriever and are shared with it, so they are reported separately and are **not** part
    of what this cache costs: evicting the index does not free them.
    """
    postings = deep_size(index._postings)
    lengths = deep_size(index._lengths)
    pointers = sys.getsizeof(index._records)
    shared = deep_size(index._records) - pointers
    return {
        "records": len(index),
        "terms": len(index._postings),
        "postings_bytes": postings,
        "lengths_bytes": lengths,
        "record_pointers_bytes": pointers,
        "own_bytes": postings + lengths + pointers + sys.getsizeof(index),
        "shared_record_bytes": shared,
    }


# ---------------------------------------------------------------- the run


def repository_of(question_set, root: Path) -> Repository:
    return Repository(
        repo_id=f"eval-{question_set.name}",
        name=question_set.name,
        url=question_set.repo_url,
        root_path=str(root.resolve()),
        ingested_at=datetime.now(UTC),
        commit=question_set.commit,
    )


def measure(config, embedder, settings) -> dict:
    """Both repositories indexed and asked one question each, through the server's own path.

    The order matters and is the order a visitor produces: index, then ask. The working set is
    read at each boundary so that the indexes and the lexical caches can be told apart.
    """
    started = time.perf_counter()
    steps = [{"after": "imports and configuration", "working_set_bytes": working_set_bytes()}]

    answers: list[tuple[str, str, AnswerService]] = []
    repos = []
    for path in SETS:
        question_set = load_question_set(path)
        root = checkout(question_set, REPO_CACHE)
        repository = repository_of(question_set, root)
        parsed = TreeSitterAstParser().parse_repository(repository)
        retriever = FaissCodeRetriever()
        CodeIndexService(
            embedder=embedder, retriever=retriever, name=config.embedding_provider, **index_options(config)
        ).index_repository(repository, parsed)
        service = AnswerService(retriever=retriever, embedder=embedder, llm=None, settings=settings)
        answers.append((question_set.name, question_set.questions[0].question, service))
        repos.append(
            {
                "repo": question_set.name,
                "commit": question_set.commit[:12],
                "files": len(parsed.files),
                "records": len(retriever.records(repository.repo_id)),
                "repo_id": repository.repo_id,
            }
        )
    steps.append({"after": "both repositories indexed", "working_set_bytes": working_set_bytes()})

    indexes = []
    for (name, question, service), row in zip(answers, repos):
        service.retrieve(row["repo_id"], question, top_k=5)
        for (_, repo_id), (_, index) in service._bm25.items():
            indexes.append({"repo": name, "repo_id": repo_id, **index_bytes(index)})
    steps.append({"after": "one question against each", "working_set_bytes": working_set_bytes()})

    return {
        "repos": repos,
        "bm25_indexes": indexes,
        "steps": steps,
        "seconds": round(time.perf_counter() - started, 1),
    }


def _commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return "unknown"
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, help="write the full result here")
    args = parser.parse_args()

    config = get_config()
    if config.embedding_provider != "hash":
        print(
            f"This measurement runs on the hash embedder only; CODEATLAS_EMBEDDING_PROVIDER is "
            f"{config.embedding_provider!r}. It must not reach a provider."
        )
        return 1

    embedder, settings = get_embedder(), get_retrieval_settings()
    print(f"codementor commit: {_commit()}")
    print(f"platform: {sys.platform}, python {sys.version.split()[0]}")
    print(f"record format: {config.record_format_for('hash')}, bm25: {settings.bm25_enabled}")
    print(f"bm25 cache cap: {settings.bm25_max_indexes} index(es), least recently used evicted first")

    result = measure(config, embedder, settings)

    print("\nthe lexical index itself, walked with sys.getsizeof")
    print(
        f"  {'repo':<14}{'records':>9}{'terms':>9}{'postings KB':>14}{'lengths KB':>13}{'own MB':>10}{'shared MB':>12}"
    )
    for row in result["bm25_indexes"]:
        print(
            f"  {row['repo']:<14}{row['records']:>9}{row['terms']:>9}{row['postings_bytes'] / 1024:>14.1f}"
            f"{row['lengths_bytes'] / 1024:>13.1f}{row['own_bytes'] / MB:>10.2f}{row['shared_record_bytes'] / MB:>12.2f}"
        )
    total = sum(row["own_bytes"] for row in result["bm25_indexes"])
    print(f"  {'both':<14}{'':>9}{'':>9}{'':>14}{'':>13}{total / MB:>10.2f}")

    print("\nthe process, read from the operating system")
    print(f"  {'after':<34}{'working set MB':>16}{'step MB':>10}")
    previous = None
    for step in result["steps"]:
        megabytes = step["working_set_bytes"] / MB
        delta = "-" if previous is None else f"{megabytes - previous:+.1f}"
        print(f"  {step['after']:<34}{megabytes:>16.1f}{delta:>10}")
        previous = megabytes

    result = {
        "codementor_commit": _commit(),
        "platform": sys.platform,
        "record_format": config.record_format_for("hash"),
        "bm25_enabled": settings.bm25_enabled,
        "bm25_max_indexes": settings.bm25_max_indexes,
        "bm25_own_bytes_total": total,
        **result,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")

    print(f"\nseconds: {result['seconds']}")
    print(
        "\nThe working set is this machine's, not the host's. Nothing here has read the memory "
        "limit of any deployment, and no such limit is claimed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
