"""Retrieval evaluation against fixed question sets (eval/questions/*.json).

Each set pins a public repository at one commit. The repository is fetched at that commit,
parsed, embedded and indexed with the same classes the server uses, and every question goes
through ``AnswerService.retrieve``, the retrieval step of the chat. No language model is called.
"""

import json
import shutil
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from codeatlas.models.repository import Repository
from codeatlas.services.eval.basic_eval import RankingMetrics, hit_at_k, ranking_metrics, reciprocal_rank
from codeatlas.services.parsing.tree_sitter_parser import TreeSitterAstParser
from codeatlas.services.qa.answer_service import AnswerService
from codeatlas.services.retrieval.embedding import EmbeddingService
from codeatlas.services.retrieval.faiss_retriever import FaissCodeRetriever
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.indexing import CodeIndexService
from codeatlas.services.retrieval.sentence_transformer_embedder import SentenceTransformerEmbeddingService

SPLITS = ("dev", "test", "all")
# Results kept per question; MRR is computed over these.
RANKED_DEPTH = 10


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    question: str
    expected_files: frozenset[str]


@dataclass(frozen=True)
class QuestionSet:
    name: str
    repo_url: str
    commit: str
    questions: list[Question]

    def select(self, split: str) -> list[Question]:
        return [q for q in self.questions if split == "all" or q.split == split]


def load_question_set(path: Path) -> QuestionSet:
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = [
        Question(
            id=q["id"],
            split=q["split"],
            question=q["question"],
            expected_files=frozenset(q["expected_files"]),
        )
        for q in data["questions"]
    ]
    for q in questions:
        if q.split not in ("dev", "test") or not q.expected_files:
            raise ValueError(f"{path}: question {q.id} needs split dev/test and at least one expected file")
    return QuestionSet(name=data["name"], repo_url=data["repo_url"], commit=data["commit"], questions=questions)


def checkout(question_set: QuestionSet, cache_dir: Path) -> Path:
    """The repository at the pinned commit, fetched once into cache_dir and reused afterwards."""
    dest = cache_dir / f"{question_set.name}-{question_set.commit[:12]}"
    if (dest / ".git").is_dir() and _git(dest, "rev-parse", "HEAD") == question_set.commit:
        return dest
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    _git(dest, "init", "-q")
    _git(dest, "fetch", "-q", "--depth", "1", question_set.repo_url, question_set.commit)
    _git(dest, "checkout", "-q", "--detach", "FETCH_HEAD")
    return dest


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def describe_embedder(embedder: EmbeddingService) -> str:
    if isinstance(embedder, HashEmbeddingService):
        return f"hash (HashEmbeddingService, {embedder._dimension} dims, hashed bag of words, not semantic)"
    if isinstance(embedder, SentenceTransformerEmbeddingService):
        return f"sentence-transformers ({embedder._model_name})"
    return type(embedder).__name__


@dataclass(frozen=True)
class RepoRun:
    name: str
    commit: str
    files_indexed: int
    records_indexed: int
    metrics: RankingMetrics
    per_question: list[dict]


def evaluate_set(
    question_set: QuestionSet,
    root: Path,
    embedder: EmbeddingService,
    split: str,
    ks: Sequence[int] = (1, 3, 5),
) -> RepoRun:
    """Index the checkout at root like /analyze-repo does, then score each question's retrieval."""
    repo_id = f"eval-{question_set.name}"
    repository = Repository(
        repo_id=repo_id,
        name=question_set.name,
        url=question_set.repo_url,
        root_path=str(root.resolve()),
        ingested_at=datetime.now(UTC),
        commit=question_set.commit,
    )
    parsed = TreeSitterAstParser().parse_repository(repository)
    retriever = FaissCodeRetriever()  # in memory only
    CodeIndexService(embedder=embedder, retriever=retriever).index_repository(repository, parsed)
    answers = AnswerService(retriever=retriever, embedder=embedder, llm=None)

    pairs: list[tuple[list[str], set[str]]] = []
    per_question: list[dict] = []
    for q in question_set.select(split):
        records = answers.retrieve(repo_id, q.question, top_k=RANKED_DEPTH)
        ranked = [_relative(record.metadata.get("path", ""), root) for record in records]
        expected = set(q.expected_files)
        pairs.append((ranked, expected))
        per_question.append(
            {
                "id": q.id,
                "top5": ranked[:5],
                "hit@5": hit_at_k(ranked, expected, 5),
                "rr": round(reciprocal_rank(ranked, expected), 4),
            }
        )
    return RepoRun(
        name=question_set.name,
        commit=question_set.commit,
        files_indexed=len(parsed.files),
        records_indexed=len(parsed.files) + len(parsed.functions),
        metrics=ranking_metrics(pairs, ks),
        per_question=per_question,
    )


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def metrics_dict(m: RankingMetrics) -> dict:
    return {
        "questions": m.questions,
        "hit_rate": {f"@{k}": round(v, 4) for k, v in m.hit_rate.items()},
        "precision": {f"@{k}": round(v, 4) for k, v in m.precision.items()},
        f"mrr@{RANKED_DEPTH}": round(m.mrr, 4),
    }


def run(
    question_files: Sequence[Path],
    embedder: EmbeddingService,
    split: str,
    cache_dir: Path,
    ks: Sequence[int] = (1, 3, 5),
) -> dict:
    """Evaluate every question set; 'overall' averages over all questions (each question weighs the same)."""
    start = time.perf_counter()
    runs: list[RepoRun] = []
    for path in question_files:
        question_set = load_question_set(path)
        root = checkout(question_set, cache_dir)
        runs.append(evaluate_set(question_set, root, embedder, split, ks))

    total = sum(r.metrics.questions for r in runs)

    def weighted(get) -> float:
        return sum(get(r.metrics) * r.metrics.questions for r in runs) / total if total else 0.0

    overall = RankingMetrics(
        questions=total,
        hit_rate={k: weighted(lambda m, k=k: m.hit_rate[k]) for k in ks},
        precision={k: weighted(lambda m, k=k: m.precision[k]) for k in ks},
        mrr=weighted(lambda m: m.mrr),
    )
    return {
        "embedder": describe_embedder(embedder),
        "split": split,
        "repos": {
            r.name: {
                "commit": r.commit,
                "files_indexed": r.files_indexed,
                "records_indexed": r.records_indexed,
                **metrics_dict(r.metrics),
            }
            for r in runs
        },
        "overall": metrics_dict(overall),
        "per_question": {r.name: r.per_question for r in runs},
        "seconds": round(time.perf_counter() - start, 1),
    }
