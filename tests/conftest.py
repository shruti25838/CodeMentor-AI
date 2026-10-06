"""Shared test setup."""

import pytest

from codeatlas.observability import token_ledger


@pytest.fixture(autouse=True)
def _ledger_in_a_temp_file(tmp_path, monkeypatch):
    """Keep the token ledger out of the real one.

    Several tests drive a fake model through the usage callback, which records spend. Left
    alone they would write those invented tokens into `.codeatlas/token-spend.json`, the
    file a long run checks before deciding it has budget left, and a developer's count of
    what they had really spent today would be wrong.
    """
    monkeypatch.setenv(token_ledger.ENV_PATH, str(tmp_path / "token-spend.json"))
