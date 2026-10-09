"""The stdlib JSON client: what it returns, what it raises, and what it must never carry.

No test here touches the network; every one of them replaces urlopen.
"""

import io
import json
import urllib.error

import pytest

from codeatlas.services.http.json_client import JsonClientError, JsonResponse, post_json

URL = "https://api.example.invalid/accounts/acct-123/ai/run/model"
SECRET = "token-that-must-not-leak"


class _FakeResponse:
    def __init__(self, status: int, raw: bytes) -> None:
        self.status = status
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _capture(monkeypatch, status: int = 200, raw: bytes = b"{}") -> list:
    """Record the request urlopen was given, and answer with a canned response."""
    seen: list = []

    def fake_urlopen(request, timeout=None):
        seen.append((request, timeout))
        return _FakeResponse(status, raw)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def _raise(monkeypatch, exc: Exception) -> None:
    def fake_urlopen(request, timeout=None):
        raise exc

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)


def test_a_success_returns_the_status_and_the_parsed_body(monkeypatch) -> None:
    _capture(monkeypatch, 200, json.dumps({"success": True, "result": {"shape": [1, 384]}}).encode())
    response = post_json(URL, {"text": ["x"]}, timeout=5)
    assert response == JsonResponse(status=200, body={"success": True, "result": {"shape": [1, 384]}})
    assert response.ok


def test_the_body_is_sent_as_json_with_the_headers_the_caller_asked_for(monkeypatch) -> None:
    seen = _capture(monkeypatch)
    post_json(URL, {"text": ["x"], "pooling": "cls"}, timeout=7, headers={"Authorization": f"Bearer {SECRET}"})
    request, timeout = seen[0]
    assert timeout == 7
    assert request.method == "POST"
    assert json.loads(request.data.decode()) == {"text": ["x"], "pooling": "cls"}
    # urllib title-cases header names.
    assert request.headers["Content-type"] == "application/json"
    assert request.headers["Authorization"] == f"Bearer {SECRET}"


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_an_error_status_is_an_answer_not_a_failure(monkeypatch, status) -> None:
    """The caller has to tell 'out of budget' from 'bad token', and only the body says which."""
    raw = json.dumps({"success": False, "errors": [{"code": 10000, "message": "Authentication error"}]}).encode()
    error = urllib.error.HTTPError(URL, status, "nope", {}, io.BytesIO(raw))
    _raise(monkeypatch, error)

    response = post_json(URL, {"text": ["x"]}, timeout=5)
    assert response.status == status
    assert not response.ok
    assert response.body["errors"] == [{"code": 10000, "message": "Authentication error"}]


def test_a_body_that_is_not_json_becomes_an_empty_dict(monkeypatch) -> None:
    """A proxy's HTML error page must not crash the caller on the way to the fallback."""
    _capture(monkeypatch, 502, b"<html>bad gateway</html>")
    response = post_json(URL, {"text": ["x"]}, timeout=5)
    assert response == JsonResponse(status=502, body={})


def test_a_json_body_that_is_not_an_object_becomes_an_empty_dict(monkeypatch) -> None:
    _capture(monkeypatch, 200, b"[1, 2, 3]")
    assert post_json(URL, {"text": ["x"]}, timeout=5).body == {}


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError(f"failed to reach {URL}"),
        TimeoutError(f"timed out talking to {URL}"),
        OSError(f"connection reset by {URL}"),
    ],
)
def test_no_response_raises_an_error_that_names_only_the_exception_type(monkeypatch, exc) -> None:
    _raise(monkeypatch, exc)
    with pytest.raises(JsonClientError) as raised:
        post_json(URL, {"text": ["x"]}, timeout=5, headers={"Authorization": f"Bearer {SECRET}"})
    assert str(raised.value) == type(exc).__name__


def test_a_transport_failure_carries_neither_the_url_nor_the_token(monkeypatch) -> None:
    """The URL holds an account id and the header holds a token; a logged traceback must show neither.

    The cause chain is checked, not just the message: an exception raised inside the except block
    would keep the original as __context__, where the URL still is.
    """
    _raise(monkeypatch, urllib.error.URLError(f"no route to host for {URL}"))
    with pytest.raises(JsonClientError) as raised:
        post_json(URL, {"text": ["secret document text"]}, timeout=5, headers={"Authorization": f"Bearer {SECRET}"})

    error = raised.value
    assert error.__cause__ is None and error.__context__ is None
    for forbidden in (URL, SECRET, "acct-123", "secret document text"):
        assert forbidden not in str(error)
        assert forbidden not in repr(error)


def test_an_unserialisable_body_is_the_callers_bug_and_is_not_swallowed(monkeypatch) -> None:
    _capture(monkeypatch)
    with pytest.raises(TypeError):
        post_json(URL, {"text": object()}, timeout=5)
