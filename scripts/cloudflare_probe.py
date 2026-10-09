"""Ask the Cloudflare Workers AI embeddings endpoint the questions its documentation does not answer.

    python scripts/cloudflare_probe.py --dry-run   # the plan and the cost estimate, no calls
    python scripts/cloudflare_probe.py             # makes the calls

Four things are not in Cloudflare's docs and decide how the provider must be written:

1. **How many texts fit in one `text` array.** The widely repeated "100" traces back to a
   LlamaIndex page dated 2024-03-31 and a 2023 docs issue, not to any current schema. Arrays
   of 50, 100 and 101 are sent to find the real edge.
2. **What happens to a text past the model's 512-token window** - truncation or an error. A
   single 8,000-character text is sent whose first 2,400 characters are about one subject and
   whose remaining 5,600 are about an unrelated one. Each subject is also embedded alone. If
   the long text's vector sits on top of the first subject's and far from the second's, the
   tail was thrown away; if it sits between them, it was not.
3. **Whether `pooling: "cls"` is accepted over REST** for this model, and whether it really is
   a different vector space (the docs say cls and mean embeddings are not comparable).
4. **What a rejected call looks like**, which is what the fallback-to-hash path has to react
   to. One call is made with a deliberately invalid token to see the status and the error body.

**What this prints, and what it never prints.** Per call: a label, the HTTP status, the
`success` flag, `result.shape`, the pooling the response echoes, and any error codes. Never
the URL (it contains the account id), never a header, never a token, and never any input
text - only its length. The inputs are invented strings in this file; no repository content
is involved.

The call budget is capped at MAX_CALLS and every input is tiny: the whole run is a few
neurons of the 10,000-per-day free allowance, and the estimate is printed before anything
is sent.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from codeatlas.services.http.json_client import JsonClientError, JsonResponse, post_json  # noqa: E402
from codeatlas.utils.config import load_cloudflare_credentials  # noqa: E402

MODEL = "@cf/baai/bge-small-en-v1.5"
ENDPOINT = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
# A hard ceiling, not a target: the probe must never turn into a load test against a free plan.
MAX_CALLS = 10
DEFAULT_TIMEOUT_SECONDS = 60.0

# Cloudflare's published price for this model, and the model's input window.
NEURONS_PER_MILLION_INPUT_TOKENS = 1841
MAX_TOKENS_PER_DOCUMENT = 512
# The free allowance this run has to stay a rounding error against.
DAILY_FREE_NEURONS = 10_000
# Deliberately pessimistic: 3 characters per token over-counts English prose (nearer 4) and is
# about right for code, so an estimate built on it is a ceiling rather than a hope.
CHARS_PER_TOKEN = 3

_SIGNING = (
    "The signer serializes the payload, derives a key from the secret and appends an HMAC "
    "signature so that a tampered cookie fails verification before it is deserialized. "
)
_BREAD = (
    "Fold the dough gently every half hour while the levain ferments, then shape the loaf and "
    "let it proof in a floured basket overnight in the refrigerator before baking. "
)

HEAD_CHARS = 2400
LONG_CHARS = 8000


def long_text() -> str:
    """8,000 characters: 2,400 on one subject followed by 5,600 on an unrelated one."""
    return (_repeat(_SIGNING, HEAD_CHARS) + _repeat(_BREAD, LONG_CHARS - HEAD_CHARS))[:LONG_CHARS]


def head_text() -> str:
    """The first 2,400 characters of long_text(): roughly 600 tokens, comfortably past the 512 cap."""
    return _repeat(_SIGNING, HEAD_CHARS)


def tail_sample() -> str:
    """The same length again from the second subject, as the contrast for the truncation test."""
    return _repeat(_BREAD, HEAD_CHARS)


def _repeat(sentence: str, chars: int) -> str:
    return (sentence * (chars // len(sentence) + 1))[:chars]


def short_texts(count: int) -> list[str]:
    return [f"probe item {i} describing a small function in a repository" for i in range(count)]


@dataclass(frozen=True)
class Call:
    label: str
    question: str
    texts: list[str]
    pooling: str | None = None
    # The one call that is meant to fail, so the fallback path knows what a refusal looks like.
    invalid_token: bool = False


def plan() -> list[Call]:
    return [
        Call("baseline", "does a single short text work at all", short_texts(1)),
        Call("array-50", "is an array of 50 accepted", short_texts(50)),
        Call("array-100", "is an array of 100 accepted", short_texts(100)),
        Call("array-101", "is 101 over the edge", short_texts(101)),
        Call("long-8000", "is an 8,000-character text truncated or refused", [long_text()]),
        Call("head-2400", "the first subject alone, to compare against long-8000", [head_text()]),
        Call("tail-2400", "the second subject alone, as the contrast", [tail_sample()]),
        Call("pooling-cls", "is pooling=cls accepted, and is it a different space", short_texts(1), pooling="cls"),
        Call("invalid-token", "what a refused call looks like", short_texts(1), invalid_token=True),
    ]


@dataclass
class Outcome:
    label: str
    texts: int
    chars: int
    status: int | None = None
    success: bool | None = None
    shape: list | None = None
    pooling: str | None = None
    errors: list = field(default_factory=list)
    transport_error: str | None = None
    vectors: list = field(default_factory=list)

    def result(self) -> str:
        if self.transport_error:
            return f"no response ({self.transport_error})"
        parts = [f"HTTP {self.status}", f"success={self.success}"]
        if self.shape is not None:
            parts.append(f"shape={self.shape}")
        if self.pooling:
            parts.append(f"pooling={self.pooling}")
        if self.errors:
            parts.append(f"errors={json.dumps(self.errors)}")
        return ", ".join(parts)

    def line(self) -> str:
        return f"{self.label:<14} {self.texts:>4} texts {self.chars:>6} chars  {self.result()}"


def estimated_tokens(texts: list[str]) -> int:
    """The estimator the provider will use: ceil(chars / 3) per text, capped at the model's window."""
    return sum(min(math.ceil(len(text) / CHARS_PER_TOKEN), MAX_TOKENS_PER_DOCUMENT) for text in texts)


def estimated_neurons(tokens: int) -> float:
    return tokens * NEURONS_PER_MILLION_INPUT_TOKENS / 1_000_000


def run_call(call: Call, account_id: str, api_token: str, timeout: float) -> Outcome:
    outcome = Outcome(label=call.label, texts=len(call.texts), chars=sum(len(t) for t in call.texts))
    body: dict = {"text": call.texts}
    if call.pooling:
        body["pooling"] = call.pooling
    token = "invalid-token-on-purpose" if call.invalid_token else api_token
    url = ENDPOINT.format(account_id=account_id, model=MODEL)
    try:
        response = post_json(url, body, timeout=timeout, headers={"Authorization": f"Bearer {token}"})
    except JsonClientError as exc:
        outcome.transport_error = str(exc)
        return outcome
    _read(outcome, response)
    return outcome


def _read(outcome: Outcome, response: JsonResponse) -> None:
    outcome.status = response.status
    outcome.success = response.body.get("success")
    result = response.body.get("result")
    if isinstance(result, dict):
        outcome.shape = result.get("shape")
        outcome.pooling = result.get("pooling")
        data = result.get("data")
        if isinstance(data, list) and data:
            outcome.vectors = data
    errors = response.body.get("errors")
    if isinstance(errors, list):
        # Codes and messages only, and the error bodies do not echo the request. This shape is
        # what the fallback path will have to recognise.
        outcome.errors = [{"code": e.get("code"), "message": e.get("message")} for e in errors if isinstance(e, dict)]


def cosine(a: list, b: list) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else None


def report(outcomes: dict[str, Outcome]) -> list[str]:
    """What each of the four questions turned out to be, in words."""
    lines: list[str] = []
    sizes = ("array-50", "array-100", "array-101")
    accepted = [label for label in sizes if _ok(outcomes.get(label))]
    refused = [label for label in sizes if _refused(outcomes.get(label))]
    lines.append(f"batch size: accepted {accepted or 'none'}; refused {refused or 'none'}")

    long_out, head_out, tail_out = outcomes.get("long-8000"), outcomes.get("head-2400"), outcomes.get("tail-2400")
    if _ok(long_out) and _ok(head_out) and _ok(tail_out):
        to_head = cosine(long_out.vectors[0], head_out.vectors[0]) or 0.0
        to_tail = cosine(long_out.vectors[0], tail_out.vectors[0]) or 0.0
        verdict = "the tail was dropped" if to_head > to_tail else "the whole text was read"
        lines.append(
            f"over-length text: accepted; cos(long, head)={to_head:.4f} cos(long, tail)={to_tail:.4f} -> {verdict}"
        )
    elif _refused(long_out):
        lines.append(f"over-length text: refused, {long_out.result()}")

    cls_out, base_out = outcomes.get("pooling-cls"), outcomes.get("baseline")
    if _ok(cls_out) and _ok(base_out):
        same = cosine(cls_out.vectors[0], base_out.vectors[0]) or 0.0
        lines.append(f"pooling=cls: accepted, echoed {cls_out.pooling!r}; cos(cls, mean) same text = {same:.4f}")
    elif _refused(cls_out):
        lines.append(f"pooling=cls: refused, {cls_out.result()}")

    bad = outcomes.get("invalid-token")
    if bad is not None:
        lines.append(f"invalid token: {bad.result()}")
    return lines


def _ok(outcome: Outcome | None) -> bool:
    return bool(outcome and outcome.success and outcome.vectors)


def _refused(outcome: Outcome | None) -> bool:
    return bool(outcome and outcome.status is not None and not outcome.success)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and the estimate, call nothing")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    calls = plan()
    if len(calls) > MAX_CALLS:
        print(f"refusing to run: {len(calls)} calls is over the cap of {MAX_CALLS}")
        return 1

    tokens = sum(estimated_tokens(call.texts) for call in calls if not call.invalid_token)
    print(f"model: {MODEL}")
    print(f"calls: {len(calls)} of at most {MAX_CALLS}")
    print(f"estimated cost: {tokens} input tokens, {estimated_neurons(tokens):.2f} estimated neurons")
    for call in calls:
        chars = sum(len(t) for t in call.texts)
        print(f"  {call.label:<14} {len(call.texts):>4} texts {chars:>6} chars  {call.question}")

    if args.dry_run:
        print("\ndry run: nothing was sent")
        return 0

    credentials = load_cloudflare_credentials()
    if credentials is None:
        print("\nCLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must both be set (.env is read).")
        return 1

    print()
    outcomes: dict[str, Outcome] = {}
    for call in calls:
        outcome = run_call(call, credentials.account_id, credentials.api_token, args.timeout)
        outcomes[call.label] = outcome
        print(outcome.line())

    print()
    for line in report(outcomes):
        print(line)
    print(f"\nestimated spend for this run: {estimated_neurons(tokens):.2f} neurons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
