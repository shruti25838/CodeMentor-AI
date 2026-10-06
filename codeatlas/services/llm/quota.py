"""What the visitor is told when the provider refuses a call.

The demo runs on a free provider tier with a per-minute token allowance, a per-day one, and
a limit on how much can be sent at once. When one of those is hit the visitor used to see
the raw exception text — `Error generating answer: Error code: 429 - {'error': ...}` — or,
worse, a bare list of file paths under the heading "Top relevant locations:", which reads as
though it were the answer to the question.

Both are replaced here by a short plain sentence, followed by the files that did match,
under a heading that says what they are. The wording differs by cause, because what the
visitor should do differs: a per-minute limit clears by itself, a daily one does not.
"""

from __future__ import annotations

from codeatlas.services.retrieval.snippets import Snippet

# Causes, in the order they are checked. A daily limit must be recognised before the general
# rate-limit case, because the provider reports it as a 429 too.
DAILY = "daily"
PER_MINUTE = "per_minute"
TOO_LARGE = "too_large"

MESSAGES = {
    DAILY: "The demo's AI quota for today is used up. Please try again later.",
    PER_MINUTE: "The demo's AI is busy right now — its per-minute quota is used up. Wait a minute and ask again.",
    TOO_LARGE: "That question needed more context than the demo's AI accepts at once, so it could not answer.",
}

# Said before the file list so it is never mistaken for an answer.
FILES_HEADING = "The search did find these files, which may still help:"
NO_FILES = "No matching files were found either."

# Appended when the provider says the answer ran out of output budget.
TRUNCATED_NOTE = "\n\n_This answer was cut off because it reached the model's output limit._"


def classify_quota(error: BaseException | str) -> str | None:
    """Which quota the provider refused on, or None when the failure is something else."""
    text = str(error)
    lowered = text.lower()
    status = getattr(error, "status_code", None)

    if "tokens per day" in lowered or "(tpd)" in lowered or "per day" in lowered:
        return DAILY
    if status == 413 or "413" in text or "request too large" in lowered or "too large" in lowered:
        return TOO_LARGE
    if status == 429 or "429" in text or "rate limit" in lowered or "rate_limit" in lowered:
        return PER_MINUTE
    return None


def quota_answer(kind: str, snippets: list[Snippet]) -> str:
    """The message a visitor sees, with the retrieved files beneath it."""
    parts = [MESSAGES.get(kind, MESSAGES[PER_MINUTE])]
    if snippets:
        parts.append(FILES_HEADING)
        parts.append("\n".join(f"- {snippet.location}" for snippet in snippets))
    else:
        parts.append(NO_FILES)
    return "\n\n".join(parts)


def was_truncated(response: object) -> bool:
    """Whether the provider said the reply stopped because it ran out of output budget."""
    metadata = getattr(response, "response_metadata", None) or {}
    if metadata.get("finish_reason") == "length":
        return True
    # Some providers report it per generation rather than on the message.
    info = metadata.get("generation_info") or {}
    return info.get("finish_reason") == "length"
