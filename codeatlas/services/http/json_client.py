"""One JSON POST over the standard library, and nothing else.

Why not httpx or requests: both are importable here, but only because something else pulled
them in (openai and langsmith need httpx, langchain and tiktoken need requests). Neither is
in requirements.txt, so an app path built on one would depend on a pin nobody declares, and a
dependency bump could remove it. urllib.request is already used for real HTTP in
scripts/benchmark.py and costs nothing.

Two rules this module exists to keep:

* **An HTTP error status is an answer, not a failure.** 4xx and 5xx come back as a
  JsonResponse with the provider's own body, because the caller has to tell "out of budget"
  from "bad token" and only the body says which.
* **Nothing that fails carries the request with it.** The URL holds an account id and the
  headers hold a token, so a transport failure is re-raised as JsonClientError carrying the
  exception's type name alone, with the original cause dropped so a traceback cannot print
  it either.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class JsonResponse:
    status: int
    # The parsed JSON body, or {} when the body was empty or not JSON. Headers are deliberately
    # not kept: one of them is the Authorization header that was sent.
    body: dict

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class JsonClientError(Exception):
    """The request never produced an HTTP response: DNS, connection, TLS or timeout.

    Its message is the underlying exception's type name and nothing else.
    """


def post_json(
    url: str,
    body: dict,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    headers: dict[str, str] | None = None,
) -> JsonResponse:
    """POST `body` as JSON and return the status and parsed body.

    Raises JsonClientError if there was no response at all. Any other exception is a bug in
    the caller (an unserialisable body, say) and is left alone.
    """
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return JsonResponse(status=response.status, body=_parse(response.read()))
    except urllib.error.HTTPError as error:
        return JsonResponse(status=error.code, body=_parse(error.read()))
    except OSError as exc:
        # URLError, TimeoutError and the ssl errors are all OSError.
        transport_failure = type(exc).__name__
    # Raised out here rather than inside the except block, and that is the whole point: raised
    # inside it, the original would hang off this exception as __context__, and its message
    # holds the URL and therefore the account id. `raise ... from None` only sets a flag that
    # asks tracebacks not to print it; leaving the block clears it outright.
    raise JsonClientError(transport_failure)


def _parse(raw: bytes) -> dict:
    """The body as a dict; {} for anything that is not a JSON object (an HTML error page, say)."""
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
