"""One search, one model call, and the model's own tokens streamed as they arrive.

What the legacy path does, and why this exists. In repo mode today a question is answered by
the agent graph: retrieval summarises the snippets through the model, the mentor re-runs
retrieval and summarises again, then the mentor answers - three model calls - and only when
all three have finished does the server start sending the answer, four words every eight
milliseconds. `docs/RESULTS.md` records the consequence: "the time to the first token the user
sees is the sum of three full model calls", and what is streamed is a finished string being
typed out rather than a model thinking.

Grounded mode is the other shape:

* **one** retrieval, through the same `AnswerService` the evaluation scores, so the recorded
  retrieval numbers describe what this mode searches with;
* **one** model call, whose prompt holds the capped snippets with their paths and line ranges,
  the session history, and the question;
* the model's tokens forwarded to the browser as they arrive.

**Citations are the snippets in the prompt.** Not a re-ranked list, not what the model said it
read - the same objects, built once, rendered into the prompt and into the citation lines by
`grounded_context`. A model cannot cite a file it was not shown, and the browser cannot show a
citation the model did not have.

Two provider behaviours are handled here rather than being left to surprise a reader. A
reasoning model emits tokens that carry no visible content first, so the first thing the
visitor sees can be seconds after the call started: empty chunks are skipped, nothing is
mistaken for the first token, and the caller is told when the real first token arrives. And a
model can finish having said nothing at all, which must read as a plain sentence rather than an
empty bubble.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from codeatlas.services.llm.quota import TRUNCATED_NOTE, classify_quota, quota_answer, was_truncated
from codeatlas.services.memory.conversation import Turn
from codeatlas.services.qa.answer_service import AnswerService, citation_text
from codeatlas.services.retrieval.snippets import Snippet, render_snippets

logger = logging.getLogger(__name__)

LEGACY = "legacy"
GROUNDED = "grounded"
CHAT_MODES = (LEGACY, GROUNDED)

SYSTEM_PROMPT = (
    "You are a senior engineer explaining a codebase to a colleague. Answer only from the code "
    "snippets below. Each snippet is introduced by its file path and line range; cite those when "
    "you refer to code. If the snippets do not contain the answer, say so plainly rather than "
    "guessing."
)

# What the visitor reads when the model returned nothing at all. An empty bubble looks like a
# bug in the page; this says what happened.
EMPTY_REPLY = "The model returned an empty answer. The files below are what the search found."


def unknown_chat_mode_message(value: str) -> str:
    return (
        f"Unknown chat mode {value!r}. Set CODEATLAS_CHAT_MODE to one of: {', '.join(CHAT_MODES)}. "
        f"'{LEGACY}' is the default and is the agent pipeline every recorded measurement describes."
    )


def validate_chat_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in CHAT_MODES:
        raise ValueError(unknown_chat_mode_message(value))
    return normalized


@dataclass(frozen=True)
class GroundedContext:
    """What one search produced: the snippets, and the citations that describe exactly them."""

    snippets: list[Snippet]
    citations: list[str]
    note: str = ""
    reasoning_steps: list[str] = field(default_factory=list)


def grounded_context(
    answers: AnswerService,
    repo_id: str,
    question: str,
    top_k: int,
    total_max_chars: int,
) -> GroundedContext:
    """One retrieval, capped to the context budget, with the citations built from the survivors.

    The snippets are cut to the budget *before* the citations are made, so a snippet dropped for
    size cannot be cited: the two lists are the same list seen twice.
    """
    records, note = answers.retrieve_with_note(repo_id, question, top_k)
    snippets = answers.snippets_for(records, total_max_chars)
    steps = [
        f"Searched the index once and kept {len(snippets)} snippet(s) within the context budget.",
        "Answered with one model call over those snippets; the citations are exactly them.",
    ]
    return GroundedContext(
        snippets=snippets,
        citations=[citation_text(snippet) for snippet in snippets],
        note=note,
        reasoning_steps=steps,
    )


def build_messages(question: str, context: GroundedContext, history: list[Turn]) -> list:
    """The one prompt: the rules, the earlier turns, the snippets, and the question.

    History goes in as real turns rather than as text inside the human message, so a provider
    that prices or truncates by message sees the shape it expects.
    """
    messages: list = [SystemMessage(SYSTEM_PROMPT)]
    for turn in history:
        messages.append(HumanMessage(turn.question))
        messages.append(AIMessage(turn.answer))
    rendered = render_snippets(context.snippets) if context.snippets else "(no code snippets were found)"
    messages.append(HumanMessage(f"{question}\n\nCode snippets:\n{rendered}"))
    return messages


@dataclass
class StreamOutcome:
    """What a streamed call produced, once it has finished one way or another."""

    answer: str = ""
    truncated: bool = False
    quota: str | None = None
    failed: bool = False


def stream_tokens(
    llm: BaseChatModel,
    messages: list,
    context: GroundedContext,
    on_first_token=None,
) -> Iterator[tuple[str, StreamOutcome | None]]:
    """Yield (token, None) as the model speaks, then one final ("", outcome).

    A chunk with no visible content is a reasoning model thinking; it is skipped and does not
    count as the first token. `on_first_token` fires once, when something a visitor can read
    actually arrives, so a caller can time the wait that matters rather than the first packet.

    A provider refusal is not raised: it is classified and turned into the same plain sentence
    the non-streaming path uses, with the files underneath, because by this point the browser
    has a 200 and an open stream and an exception would reach it as a raw error string.
    """
    outcome = StreamOutcome()
    last_chunk = None
    try:
        for chunk in llm.stream(messages):
            last_chunk = chunk
            token = getattr(chunk, "content", None)
            if not token:
                continue
            if not outcome.answer and on_first_token is not None:
                on_first_token()
            outcome.answer += token
            yield token, None
    except Exception as exc:  # noqa: BLE001 - every failure has to reach the visitor as words
        kind = classify_quota(exc)
        if kind:
            logger.warning("Grounded answer refused on the %s quota", kind)
            outcome.quota = kind
            message = quota_answer(kind, context.snippets)
        else:
            logger.warning("Grounded answer failed: %s", type(exc).__name__)
            outcome.failed = True
            message = _failure_message(context)
        # Nothing readable has been sent yet in the usual case; if some had been, the message
        # follows it rather than replacing it, so the visitor is not shown a half answer alone.
        outcome.answer = f"{outcome.answer}\n\n{message}".strip() if outcome.answer else message
        yield message, None
        yield "", outcome
        return

    if was_truncated(last_chunk):
        outcome.truncated = True
        outcome.answer += TRUNCATED_NOTE
        yield TRUNCATED_NOTE, None
    elif not outcome.answer:
        outcome.answer = _empty_message(context)
        yield outcome.answer, None
    yield "", outcome


def _empty_message(context: GroundedContext) -> str:
    parts = [EMPTY_REPLY]
    if context.snippets:
        parts.append("\n".join(f"- {snippet.location}" for snippet in context.snippets))
    return "\n\n".join(parts)


def _failure_message(context: GroundedContext) -> str:
    parts = ["The model could not answer that question just now. The files below are what the search found."]
    if context.snippets:
        parts.append("\n".join(f"- {snippet.location}" for snippet in context.snippets))
    return "\n\n".join(parts)
