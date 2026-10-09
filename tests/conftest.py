"""Shared test setup."""

import pytest

from codeatlas.observability import token_ledger
from codeatlas.services.retrieval import neuron_ledger


@pytest.fixture(autouse=True)
def _ledger_in_a_temp_file(tmp_path, monkeypatch):
    """Keep the token ledger out of the real one.

    Several tests drive a fake model through the usage callback, which records spend. Left
    alone they would write those invented tokens into `.codeatlas/token-spend.json`, the
    file a long run checks before deciding it has budget left, and a developer's count of
    what they had really spent today would be wrong.
    """
    monkeypatch.setenv(token_ledger.ENV_PATH, str(tmp_path / "token-spend.json"))


@pytest.fixture(autouse=True)
def _neurons_in_a_temp_file(tmp_path, monkeypatch):
    """Keep the estimated-neuron ledger out of the real one, for the same reason as above.

    This one guards a budget rather than a count: a test that wrote invented spend into
    `.codeatlas/cloudflare-neurons.json` could leave the provider believing the day's free
    allowance was gone and silently falling back to hash.
    """
    monkeypatch.setenv(neuron_ledger.ENV_PATH, str(tmp_path / "cloudflare-neurons.json"))
