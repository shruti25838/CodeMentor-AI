"""Ask the same questions under several server settings, then score the answers blind.

Two modes, and a person in between:

    python scripts/blind_compare.py run --repo itsdangerous --limit 5
    # ... read sheet.md, fill in sheet.csv ...
    python scripts/blind_compare.py score --run-id default

**Why blind.** Every automatic check in this repository is weak on purpose -- `answer_smoke`
asks whether the expected file is cited and whether one hand-written word appears, and says
so loudly. Nothing measures whether an explanation is correct, because nothing here can. A
person can, but a person who knows which arm wrote which answer is not measuring the answer.
So the arms' answers are shuffled per question into A, B and C, the key is written to a
different folder, and the sheet to be filled in contains no arm name anywhere.

**An arm is a set of environment overrides** -- the same variables a server would be started
with. Only two are accepted, `CODEATLAS_CHAT_MODE` and `CODEATLAS_REASONING_EFFORT`, because
an override that is not applied is worse than one that is rejected: the run would compare two
identical arms and report a dead heat. Each arm is served by a real local uvicorn server on
127.0.0.1, asked over a real socket, exactly as `chat_latency_probe` does, because an
in-process client hides the event loop and the executor.

**Spending is bounded the same way the probe bounds it.** The cap is checked inside
`budget_guard`, imported from `chat_latency_probe` rather than written again, so it carries
the reservation: the run stops while there is still room for the next question rather than
discovering the cost after paying it. The pacer keeps a trailing minute under the provider's
allowance. **Any 429 ends the run cleanly** -- the app's own rate limiter answering with a
status, or the provider's refusal arriving as the sentence the app shows a visitor.

**Everything is written under `.codeatlas/`, which is gitignored, and nowhere else.** A path
outside it is refused rather than written. The answers are a model's prose and have no
business in the repository. Nothing reads a credential: the key is checked for by name only
and never printed, and no value from the environment reaches a file or a log line here.

**It is resumable, and resuming is the default.** The run folder is keyed by `--run-id`, every
finished run is appended to `raw/runs.json` **after every single question**, and a second
invocation with the same id skips what is already answered. A different id starts fresh.

Layout of a run, under `.codeatlas/blind-compare/<run-id>/`:

    raw/runs.json     every answer, with its arm, its cited files and its timings
    raw/key.json      which arm each of A, B, C is, per question
    sheet/sheet.md    the answers to read, under their labels, no arm named
    sheet/sheet.csv   the grid to fill in: correctness, cited_support, note
    summary.json      written by `score`, after the sheet is filled in

This spends real money and needs a model key in the project's .env.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from codeatlas.observability import token_ledger  # noqa: E402
from codeatlas.services.llm.quota import MESSAGES  # noqa: E402
from codeatlas.services.qa.grounded_chat import GROUNDED, LEGACY  # noqa: E402
from scripts.answer_smoke import QUESTIONS, REPOS  # noqa: E402
from scripts.chat_latency_probe import ERROR_PREFIX, Pacer, budget_guard  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Everything this script writes lives under here. Nothing else may be written at all.
WRITES_UNDER = PROJECT / ".codeatlas"
RUNS_ROOT = WRITES_UNDER / "blind-compare"

# The settings an arm may override, and the config field each one sets. These are exactly the
# variables `load_config` reads, so an arm describes a server someone could really start.
# Anything else is refused: an override that silently did nothing would make two arms
# identical and the comparison would report a tie it had not measured.
ARM_SETTINGS = {
    "CODEATLAS_CHAT_MODE": "chat_mode",
    "CODEATLAS_REASONING_EFFORT": "llm_reasoning_effort",
}

DEFAULT_ARMS: dict[str, dict[str, str]] = {
    "legacy": {"CODEATLAS_CHAT_MODE": LEGACY},
    "grounded": {"CODEATLAS_CHAT_MODE": GROUNDED},
    "grounded-low": {"CODEATLAS_CHAT_MODE": GROUNDED, "CODEATLAS_REASONING_EFFORT": "low"},
}

# Labels the answers are masked behind, in order. Three arms gives A, B, C.
LABELS = "ABCDEFGH"

DEFAULT_SEED = 1315

# The columns the person fills in. `note` is free text; the other two are 0, 1 or 2.
SCORE_COLUMNS = ("correctness", "cited_support")
SHEET_COLUMNS = ("question_id", "question", "label", "cited_files", *SCORE_COLUMNS, "note")

SCORE_GUIDE = {
    "correctness": "0 wrong or evasive, 1 partly right, 2 right",
    "cited_support": "0 the cited files do not support it, 1 partly, 2 they do",
}

# What a question is assumed to cost before anything has been measured. Only the reserve and
# the pacer use it, and the pacer replaces it with the worst question seen so far.
FIRST_ESTIMATE = 5_000


# ---------- where things are written ----------


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def runs(self) -> Path:
        return self.root / "raw" / "runs.json"

    @property
    def key(self) -> Path:
        return self.root / "raw" / "key.json"

    @property
    def sheet_md(self) -> Path:
        return self.root / "sheet" / "sheet.md"

    @property
    def sheet_csv(self) -> Path:
        return self.root / "sheet" / "sheet.csv"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    def prepare(self) -> None:
        for path in (self.runs, self.key, self.sheet_md, self.sheet_csv, self.summary):
            path.parent.mkdir(parents=True, exist_ok=True)


def inside_writable_root(path: Path, root: Path = WRITES_UNDER) -> Path:
    """The resolved path, or a refusal. Nothing outside `.codeatlas/` is ever written.

    Checked on the resolved path, so `--run-id ../../somewhere` is caught rather than being
    taken literally: the answers are a model's prose and the run folder is scratch, and
    neither has any business anywhere else in the working tree.
    """
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    if resolved != root and root not in resolved.parents:
        raise SystemExit(f"blind_compare writes only under {root}; refusing {resolved}")
    return resolved


def paths_for(run_id: str) -> Paths:
    return Paths(inside_writable_root(RUNS_ROOT / run_id))


# ---------- arms ----------


def parse_arm(text: str) -> tuple[str, dict[str, str]]:
    """`name=VAR=value[,VAR=value]` into (name, overrides). Unknown variables are refused."""
    name, sep, rest = text.partition("=")
    if not sep or not name.strip():
        raise SystemExit(f"an arm looks like name=VAR=value[,VAR=value]; got {text!r}")
    overrides: dict[str, str] = {}
    for pair in rest.split(","):
        var, has, value = pair.partition("=")
        var = var.strip()
        if not has or not var:
            raise SystemExit(f"an arm looks like name=VAR=value[,VAR=value]; got {text!r}")
        if var not in ARM_SETTINGS:
            raise SystemExit(f"{var} is not an arm setting; choose from {', '.join(sorted(ARM_SETTINGS))}")
        overrides[var] = value.strip()
    if not overrides:
        raise SystemExit(f"arm {name!r} overrides nothing")
    return name.strip(), overrides


def config_for(base, overrides: dict[str, str]):
    """`base` with this arm's overrides applied, validated as `load_config` validates them."""
    from codeatlas.services.qa.grounded_chat import validate_chat_mode
    from codeatlas.utils.config import validate_reasoning_effort

    fields = {}
    for var, value in overrides.items():
        setting = ARM_SETTINGS[var]
        try:
            if setting == "chat_mode":
                fields[setting] = validate_chat_mode(value)
            else:
                # Empty means the arm leaves the setting unset, which is not the same as any
                # of its values: unset sends the parameter not at all.
                fields[setting] = validate_reasoning_effort(value) if value else None
        except ValueError as exc:
            raise SystemExit(str(exc))
    return replace(base, **fields)


def describe_arm(overrides: dict[str, str]) -> str:
    """An arm's settings as a server would be started with them. No value is a secret."""
    return " ".join(f"{var}={value}" for var, value in sorted(overrides.items()))


# ---------- the seeded mask ----------


def labels_for(question_id: str, arms: list[str], seed: int) -> dict[str, str]:
    """Which arm each of A, B, C is, for this question. Seeded, so two runs agree.

    The seed is mixed with the question id, so one question's order says nothing about the
    next one's -- a grader who worked out that A was the same arm every time would not be
    blind any more. The digest is taken with hashlib rather than by seeding `Random` with a
    string, so the order does not depend on how a Python version happens to hash one.
    """
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    shuffled = list(arms)
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(shuffled)
    return {LABELS[i]: arm for i, arm in enumerate(shuffled)}


def key_for(question_ids: list[str], arms: list[str], seed: int) -> dict:
    return {
        "seed": seed,
        "arms": list(arms),
        "labels": {question_id: labels_for(question_id, arms, seed) for question_id in question_ids},
    }


# ---------- asking ----------


@dataclass
class Asked:
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    agents_used: list[str] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)
    first_token_seconds: float | None = None
    total_seconds: float = 0.0
    status: int = 200
    refused: str = ""


def cited_files(citations: list[str]) -> list[str]:
    """Just the paths, the way `answer_smoke.judge` reads them out of a citation line."""
    files = {c.split(" (")[0].split(" | ")[0].strip().replace("\\", "/") for c in citations}
    return sorted(f for f in files if f)


def refusal_in(answer: str, status: int) -> str:
    """The refusal this question hit, as a short label, or "" when it was answered.

    A 429 reaches this script two ways and both end the run: the app's own rate limiter
    answers with the status, and the provider's refusal arrives as a 200 whose body is the
    sentence the app shows a visitor instead of an exception.
    """
    if status != 200:
        return f"http {status}"
    for kind, message in MESSAGES.items():
        if answer.startswith(message):
            return kind
    if answer.startswith(ERROR_PREFIX):
        return "model error"
    return ""


def ask(client, repo_id: str, question: str, session_id: str | None = None) -> Asked:
    """One streamed question over a real socket, timing the first token a visitor would see."""
    payload = {"question": question, "repo_id": repo_id, "session_id": session_id or uuid.uuid4().hex}
    started = time.perf_counter()
    parts: list[str] = []
    done: dict = {}
    first_token: float | None = None

    with client.stream("POST", "/ask/stream", json=payload) as response:
        if response.status_code != 200:
            response.read()
            return Asked(
                status=response.status_code,
                total_seconds=time.perf_counter() - started,
                refused=refusal_in("", response.status_code),
            )
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event.get("type")
            if kind == "token":
                if first_token is None:
                    first_token = time.perf_counter() - started
                parts.append(str(event.get("content", "")))
            elif kind in ("done", "error"):
                done = event

    answer = "".join(parts) if done.get("type") != "error" else ""
    asked = Asked(
        answer=answer,
        citations=list(done.get("citations") or []),
        agents_used=list(done.get("agents_used") or []),
        timings_ms=dict(done.get("timings_ms") or {}),
        first_token_seconds=first_token,
        total_seconds=time.perf_counter() - started,
    )
    asked.refused = "stream error" if done.get("type") == "error" else refusal_in(answer, 200)
    return asked


# ---------- the run ----------


def key_of(row: dict) -> tuple[str, str]:
    return (row["question_id"], row["arm"])


def load_runs(path: Path) -> list[dict]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return rows if isinstance(rows, list) else []


def finished(rows: list[dict]) -> set[tuple[str, str]]:
    """What need not be asked again. A refused run is not finished work: it is redone."""
    return {key_of(row) for row in rows if not row.get("refused")}


def write_runs(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def collect(
    clients: dict,
    repo_id: str,
    items: list[dict],
    arms: list[str],
    paths: Paths,
    token_cap: int,
    ledger: Path | None = None,
    prepare=lambda arm: None,
    tpm_limit: int = 8_000,
    seed: int = DEFAULT_SEED,
    report=print,
) -> tuple[list[dict], str]:
    """Ask every unanswered (question, arm), writing after each one. Returns (rows, why it stopped).

    Interleaved by question rather than run arm by arm, so a run the cap cuts short leaves a
    question either answered under every arm or under none. Half an arm compares nothing.
    """
    # A refused run is not finished work -- it measured the provider's mood -- so it is
    # dropped and done again rather than being carried forward and scored.
    rows = [row for row in load_runs(paths.runs) if not row.get("refused")]
    already = finished(rows)
    baseline = token_ledger.spent_today(path=ledger).total
    pacer = Pacer(limit=tpm_limit, first_estimate=FIRST_ESTIMATE)
    stopped = ""

    plan = [(item, arm) for item in items for arm in arms if (item["question_id"], arm) not in already]
    report(f"{len(plan)} run(s) to do of {len(items) * len(arms)}; {len(already)} already answered")

    for index, (item, arm) in enumerate(plan):
        # The worst question so far, which is the only estimate that cannot be optimistic
        # about what has already happened. The cap holds on the way out because of it.
        reserve = pacer.estimate()
        may_continue, spent, left = budget_guard(token_cap, baseline, ledger, reserve)
        if not may_continue:
            stopped = (
                f"token cap: this run has spent {spent} of {token_cap}, leaving {left}, and the "
                f"next question is expected to cost about {reserve}"
            )
            break
        # Nothing to pace against before the first question of this invocation; after it,
        # the window has to have room for one as expensive as the worst so far.
        if index:
            pacer.wait()

        prepare(arm)
        before = token_ledger.spent_today(path=ledger).total
        asked = ask(clients[arm], repo_id, item["question"])
        after = token_ledger.spent_today(path=ledger).total
        pacer.record(after - before)

        label = labels_for(item["question_id"], arms, seed)
        rows.append(
            {
                "question_id": item["question_id"],
                "repo": item["repo"],
                "question": item["question"],
                "arm": arm,
                "label": next(letter for letter, name in label.items() if name == arm),
                "answer": asked.answer,
                "cited_files": cited_files(asked.citations),
                "citations": asked.citations,
                "agents_used": asked.agents_used,
                "timings_ms": asked.timings_ms,
                "first_token_seconds": round(asked.first_token_seconds, 3)
                if asked.first_token_seconds is not None
                else None,
                "total_seconds": round(asked.total_seconds, 3),
                "tokens": after - before,
                "status": asked.status,
                "refused": asked.refused,
                "asked_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        # After every single run, so a run stopped by the cap, by a 429 or from outside keeps
        # everything it has already paid for.
        write_runs(paths.runs, rows)
        report(
            f"{item['question_id']:16s} {arm:16s} {asked.total_seconds:5.1f}s "
            f"{len(asked.answer):5d} chars {after - before:5d} tokens"
            + (f"  REFUSED: {asked.refused}" if asked.refused else "")
        )
        # Any refusal ends the run, which covers every way a 429 can arrive: the app's own
        # limiter answers with the status, and the provider's reaches here as the sentence the
        # app shows a visitor rather than as an exception. Carrying on would spend the cap on
        # questions the provider is going to refuse too.
        if asked.refused:
            stopped = f"refused on {item['question_id']} under {arm}: {asked.refused}"
            break

    return rows, stopped


# ---------- the sheet ----------


def answered(rows: list[dict]) -> list[dict]:
    return [row for row in rows if not row.get("refused")]


def complete_questions(rows: list[dict], arms: list[str]) -> list[str]:
    """Question ids every arm answered, in the order they were first asked.

    A question one arm answered and another did not would put an unanswered label on the
    sheet, and an arm scored on questions the others never saw.
    """
    seen: list[str] = []
    done: dict[str, set[str]] = {}
    for row in answered(rows):
        if row["question_id"] not in seen:
            seen.append(row["question_id"])
        done.setdefault(row["question_id"], set()).add(row["arm"])
    return [q for q in seen if done[q] >= set(arms)]


def sheet_rows(rows: list[dict], arms: list[str], seed: int) -> list[dict]:
    """One row per question and label, in label order, with no arm named anywhere."""
    by_key = {key_of(row): row for row in answered(rows)}
    out = []
    for question_id in complete_questions(rows, arms):
        labels = labels_for(question_id, arms, seed)
        for label in sorted(labels):
            run = by_key[(question_id, labels[label])]
            out.append(
                {
                    "question_id": question_id,
                    "question": run["question"],
                    "label": label,
                    "cited_files": "; ".join(run["cited_files"]),
                    **{column: "" for column in SCORE_COLUMNS},
                    "note": "",
                }
            )
    return out


def write_sheet_csv(path: Path, sheet: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SHEET_COLUMNS))
        writer.writeheader()
        writer.writerows(sheet)


def write_sheet_md(path: Path, rows: list[dict], arms: list[str], seed: int) -> None:
    """The answers to read while scoring. Labels only -- no arm name appears in this file."""
    by_key = {key_of(row): row for row in answered(rows)}
    parts = [
        "# Blind comparison\n\n",
        "Read each answer and fill in `sheet.csv`. Score two things per answer:\n\n",
        f"- **correctness** -- {SCORE_GUIDE['correctness']}\n",
        f"- **cited_support** -- {SCORE_GUIDE['cited_support']}\n\n",
        "The letters mean a different thing for each question, and nothing in this file or in\n",
        "`sheet.csv` says which setting wrote which answer.\n\n",
    ]
    for question_id in complete_questions(rows, arms):
        labels = labels_for(question_id, arms, seed)
        first = by_key[(question_id, labels["A"])]
        parts.append(f"## {question_id}\n\n**{first['question']}**\n\n")
        for label in sorted(labels):
            run = by_key[(question_id, labels[label])]
            files = "\n".join(f"- `{name}`" for name in run["cited_files"]) or "- (nothing cited)"
            parts.append(f"### {question_id} / {label}\n\nCited files:\n\n{files}\n\n")
            parts.append((run["answer"] or "(no answer)") + "\n\n")
        parts.append("---\n\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def write_key(path: Path, rows: list[dict], arms: list[str], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = key_for(complete_questions(rows, arms), arms, seed)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# ---------- scoring ----------


def read_sheet(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_score(value: str, where: str, column: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        score = int(text)
    except ValueError:
        raise SystemExit(f"{where}: {column} is {text!r}; it must be 0, 1 or 2")
    if score not in (0, 1, 2):
        raise SystemExit(f"{where}: {column} is {score}; it must be 0, 1 or 2")
    return score


def unmask(sheet: list[dict], key: dict) -> list[dict]:
    """The filled sheet with each row's arm put back, and its scores read as numbers."""
    labels = key["labels"]
    out = []
    for row in sheet:
        question_id, label = row["question_id"], row["label"]
        if question_id not in labels or label not in labels[question_id]:
            raise SystemExit(f"{question_id}/{label} is not in the key; the sheet and the key disagree")
        where = f"{question_id}/{label}"
        scores = {column: parse_score(row.get(column, ""), where, column) for column in SCORE_COLUMNS}
        out.append(
            {
                "question_id": question_id,
                "arm": labels[question_id][label],
                "label": label,
                "note": (row.get("note") or "").strip(),
                **scores,
            }
        )
    return out


def _mean(values: list[int]) -> float | None:
    return round(sum(values) / len(values), 2) if values else None


def summarise(scored: list[dict], arms: list[str]) -> dict:
    """Per-arm averages and wins, over the questions scored under every arm.

    A question is won by the arm with the highest total of the two scores; a draw at the top
    is a tie and is counted as one, not shared out. Only questions with every arm scored are
    counted at all, so no arm is credited for a question the others were not judged on.
    """
    complete = [
        question_id
        for question_id in dict.fromkeys(row["question_id"] for row in scored)
        if {row["arm"] for row in scored if row["question_id"] == question_id and _is_scored(row)} >= set(arms)
    ]
    usable = [row for row in scored if row["question_id"] in complete and _is_scored(row)]

    per_arm = {}
    for arm in arms:
        mine = [row for row in usable if row["arm"] == arm]
        per_arm[arm] = {
            "questions": len(mine),
            **{f"{column}_mean": _mean([row[column] for row in mine]) for column in SCORE_COLUMNS},
            "total_mean": _mean([sum(row[column] for column in SCORE_COLUMNS) for row in mine]),
            "wins": 0,
        }

    ties = 0
    for question_id in complete:
        totals = {
            row["arm"]: sum(row[column] for column in SCORE_COLUMNS)
            for row in usable
            if row["question_id"] == question_id
        }
        best = max(totals.values())
        winners = [arm for arm, total in totals.items() if total == best]
        if len(winners) == 1:
            per_arm[winners[0]]["wins"] += 1
        else:
            ties += 1

    return {
        "arms": list(arms),
        "questions_scored": len(complete),
        "ties": ties,
        "per_arm": per_arm,
        "skipped": sorted({row["question_id"] for row in scored} - set(complete)),
    }


def _is_scored(row: dict) -> bool:
    return all(row[column] is not None for column in SCORE_COLUMNS)


def sample_size_note(n: int) -> str:
    """Said plainly, every time, because the table looks more decisive than it is."""
    return (
        f"{n} question(s) scored by one person. A sample this size cannot separate small "
        "differences between arms: a gap of a few tenths of a point, or one or two wins, is "
        "what the same arms would produce twice in a row by chance. Treat anything short of a "
        "clear, repeated gap as no result, and read the notes rather than the averages."
    )


def print_summary(summary: dict, report=print) -> None:
    columns = " ".join(f"{column + '_mean':>18}" for column in SCORE_COLUMNS)
    report(f"\n{'arm':<18}{'n':>4}{columns}{'total_mean':>12}{'wins':>6}")
    for arm in summary["arms"]:
        stats = summary["per_arm"][arm]
        cells = " ".join(f"{_or_dash(stats[f'{column}_mean']):>18}" for column in SCORE_COLUMNS)
        report(f"{arm:<18}{stats['questions']:>4}{cells}{_or_dash(stats['total_mean']):>12}{stats['wins']:>6}")
    report(f"\nties (no single highest total): {summary['ties']}")
    if summary["skipped"]:
        report(f"not scored under every arm, so left out: {', '.join(summary['skipped'])}")
    report(f"\n{sample_size_note(summary['questions_scored'])}")


def _or_dash(value) -> str:
    return "-" if value is None else str(value)


# ---------- modes ----------


def questions_for(repo: str, limit: int) -> list[dict]:
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    mine = [item for item in items if item["repo"] == repo]
    if not mine:
        raise SystemExit(f"no smoke questions for {repo!r}; choose from {', '.join(sorted(REPOS))}")
    numbered = [{**item, "question_id": f"{repo}-{n + 1}"} for n, item in enumerate(mine)]
    return numbered[:limit] if limit else numbered


def run(args) -> int:
    arms = dict(DEFAULT_ARMS) if not args.arm else dict(parse_arm(text) for text in args.arm)
    if len(arms) > len(LABELS):
        raise SystemExit(f"at most {len(LABELS)} arms; got {len(arms)}")
    names = list(arms)
    items = questions_for(args.repo, args.limit)
    paths = paths_for(args.run_id)
    paths.prepare()

    # Set before the config is first read, because `get_config` is cached for the process.
    # Both directories are under `.codeatlas/`, inside the run's own folder, so a resumed run
    # finds a warm index and nothing outside the run is touched.
    os.environ.update(
        CODEATLAS_INDEX_DIR=str(paths.root / "indexes"),
        CODEATLAS_STATE_DIR=str(paths.root / "state"),
        CODEATLAS_TOKEN_LEDGER=str(WRITES_UNDER / "token-spend.json"),
        CODEATLAS_CLONE_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_CLONE_GLOBAL_PER_MINUTE="1000",
        CODEATLAS_LLM_PER_CLIENT_PER_MINUTE="1000",
        CODEATLAS_LLM_GLOBAL_PER_MINUTE="1000",
    )
    # Whatever is in the environment or in .env must not decide what an arm sends; each arm
    # states its own settings and a value left over from a shell would make it a different arm.
    for var in ARM_SETTINGS:
        os.environ.pop(var, None)

    from dotenv import load_dotenv

    load_dotenv(PROJECT / ".env")
    # By name only. No value is read into a variable, printed or written anywhere.
    if not (os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise SystemExit("no model key found in .env; this comparison needs a real model")

    import httpx

    from codeatlas.app import di
    from codeatlas.app.main import create_app
    from codeatlas.services.llm.provider import LlmProvider
    from scripts.mode_compare import serve
    from scripts.reasoning_effort_compare import swap_model

    base_config = di.get_config()
    configs = {name: config_for(base_config, overrides) for name, overrides in arms.items()}

    # One server per distinct chat mode: the mode is read off `app.state.config`, so two arms
    # that differ only in reasoning effort share a server and differ only in the model.
    servers: dict[str, object] = {}
    clients: dict[str, object] = {}
    for name, config in configs.items():
        if config.chat_mode not in servers:
            base_url, _ = serve(create_app(config))
            servers[config.chat_mode] = base_url
        clients[name] = httpx.Client(base_url=str(servers[config.chat_mode]), timeout=args.timeout)

    # The reasoning effort is set when the client is built, and the DI singletons are shared
    # between the servers, so it is swapped per question rather than per server.
    models = {name: LlmProvider(config).get_chat_model() for name, config in configs.items()}
    orchestrator = di.get_agent_orchestrator()
    answer_service = di.get_answer_service()
    provider = di.get_llm_provider()

    print(f"model: {base_config.llm_provider} / {base_config.llm_model}")
    print(f"run folder: {paths.root}")
    for name, overrides in arms.items():
        print(f"arm {name:16s} {describe_arm(overrides)}")
    print(f"token cap for this invocation: {args.token_cap:,}\n")

    body = clients[names[0]].post("/analyze-repo", json={"repo_url": REPOS[args.repo]}, timeout=args.timeout).json()
    repo_id = body["repository_id"]
    print(f"indexed {args.repo}: {body['file_count']} files\n")

    rows, stopped = collect(
        clients,
        repo_id,
        items,
        names,
        paths,
        token_cap=args.token_cap,
        prepare=lambda arm: swap_model(orchestrator, answer_service, provider, models[arm]),
        tpm_limit=args.tpm_limit,
        seed=args.seed,
    )

    write_key(paths.key, rows, names, args.seed)
    write_sheet_md(paths.sheet_md, rows, names, args.seed)
    write_sheet_csv(paths.sheet_csv, sheet_rows(rows, names, args.seed))

    complete = complete_questions(rows, names)
    print(f"\n{len(complete)} question(s) answered under every arm")
    if stopped:
        print(f"STOPPED: {stopped}")
    print(f"wrote {paths.sheet_md}")
    print(f"wrote {paths.sheet_csv}  <- fill in {', '.join(SCORE_COLUMNS)} and note")
    print(f"wrote {paths.key}  <- do not open this until the sheet is filled in")
    print(f"then: python scripts/blind_compare.py score --run-id {args.run_id}")
    return 0


def score(args) -> int:
    paths = paths_for(args.run_id)
    if not paths.sheet_csv.exists() or not paths.key.exists():
        raise SystemExit(f"no filled sheet and key under {paths.root}; run the comparison first")
    key = json.loads(paths.key.read_text(encoding="utf-8"))
    scored = unmask(read_sheet(paths.sheet_csv), key)
    summary = summarise(scored, key["arms"])
    summary["run_id"] = args.run_id
    summary["seed"] = key["seed"]
    summary["sample_size_note"] = sample_size_note(summary["questions_scored"])
    summary["recorded"] = datetime.now(UTC).strftime("%Y-%m-%d")

    print_summary(summary)
    notes = [row for row in scored if row["note"]]
    if notes:
        print("\nnotes:")
        for row in sorted(notes, key=lambda r: (r["question_id"], r["arm"])):
            print(f"  {row['question_id']:16s} {row['arm']:16s} {row['note']}")

    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.write_text(json.dumps({**summary, "rows": scored}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {paths.summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    asking = sub.add_parser("run", help="ask every arm every question, then write the sheet")
    asking.add_argument("--repo", default="itsdangerous", choices=sorted(REPOS))
    asking.add_argument("--limit", type=int, default=0, help="first N of this repo's smoke questions")
    asking.add_argument("--run-id", default="default", help="the folder under .codeatlas/blind-compare; resumes it")
    asking.add_argument("--arm", action="append", help="name=VAR=value[,VAR=value]; repeatable")
    asking.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the masking seed; two runs of it agree")
    asking.add_argument("--token-cap", type=int, default=40_000, help="hard cap on this invocation's tokens")
    asking.add_argument("--tpm-limit", type=int, default=8_000, help="the provider's per-minute token allowance")
    asking.add_argument("--timeout", type=float, default=600.0)
    asking.set_defaults(func=run)

    scoring = sub.add_parser("score", help="read the filled sheet and the key, and unmask it")
    scoring.add_argument("--run-id", default="default")
    scoring.set_defaults(func=score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
