"""Shared test setup."""

import os
import sys

import pytest

import codeatlas.services.llm.provider as llm_provider
import codeatlas.services.retrieval.cloudflare_embedder as cloudflare
import codeatlas.utils.config as app_config
from codeatlas.observability import token_ledger
from codeatlas.services.http.json_client import post_json
from codeatlas.services.retrieval import neuron_ledger

# Everything that could let a test reach a real provider, or quietly depend on a developer
# having one configured. Cleared for every test.
CREDENTIAL_VARS = ("OPENAI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY")
CREDENTIAL_PREFIXES = ("CLOUDFLARE_",)

BUILDS_A_CLIENT = "builds_model_client"


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


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        f"{BUILDS_A_CLIENT}: this test deliberately constructs a provider client and supplies its own fake key",
    )


@pytest.fixture(autouse=True)
def _no_real_providers(request, monkeypatch):
    """No test may reach a real provider, or depend on one being configured.

    This exists because of a failure that only CI could see. A test built the answer service
    through the real factory, which built a ChatOpenAI, which needs a key at construction
    time. On a developer's machine a key was always there - `load_dotenv()` runs when the app
    module is imported, so the whole .env had been read into the environment by whichever test
    happened to run first - and in CI there was none. The test was never isolated; it was
    sharing state with its neighbours, and the only thing that noticed was a machine without
    a .env file.

    So two things happen for every test. The credentials are removed from the environment and
    .env loading is disabled, which makes a local run behave like CI. And constructing a
    provider client raises, unless the test is marked `builds_model_client` - which three
    tests in test_llm_defaults.py are, because checking that settings reach the client is
    exactly what they are for, and they supply their own plainly fake key.

    For a chat model it is construction that is blocked, because construction is what needs
    the key - by the time the object exists the mistake has been made. For Cloudflare it is
    the call, because several tests build a provider only to read its signature and never
    embed anything; those get a client that refuses if it is ever used.
    """
    # Order matters, and getting it wrong is self-defeating. Both places the app reads .env
    # from are silenced *first*, because patching a module by name imports it, and importing
    # codeatlas.app.main runs load_dotenv() at module level - which would put the whole file
    # back into the environment immediately after it had been cleared.
    monkeypatch.setattr(app_config, "load_dotenv", lambda *a, **k: False)
    if "codeatlas.app.main" in sys.modules:
        monkeypatch.setattr("codeatlas.app.main.load_dotenv", lambda *a, **k: False, raising=False)

    for name in CREDENTIAL_VARS:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        if name.startswith(CREDENTIAL_PREFIXES):
            monkeypatch.delenv(name, raising=False)

    if request.node.get_closest_marker(BUILDS_A_CLIENT):
        return

    def refuse(name: str):
        def constructor(*args, **kwargs):
            raise AssertionError(
                f"This test built a real {name}. A test must not construct a provider client: "
                f"it needs a credential, so it passes only where one happens to be configured. "
                f"Use a fake, or CODEATLAS_LLM_PROVIDER=none for the app's own offline model. "
                f"If constructing one is the point of the test, mark it @pytest.mark.{BUILDS_A_CLIENT}."
            )

        return constructor

    monkeypatch.setattr(llm_provider, "ChatOpenAI", refuse("ChatOpenAI"))
    for module_name in ("langchain_groq",):
        try:  # pragma: no cover - only if the package is ever added
            __import__(module_name)
        except ImportError:
            continue
        monkeypatch.setattr(f"{module_name}.ChatGroq", refuse("ChatGroq"), raising=False)

    # For Cloudflare the danger is the call, not the object: several tests build a provider
    # only to read its signature, and never embed anything. So a provider built without a fake
    # client gets one that refuses, which turns a real request into a named failure instead of
    # a request. The default is captured when the function is defined, so this has to be done
    # by wrapping __init__ rather than by patching the module's name for it.
    original = cloudflare.CloudflareEmbeddingService.__init__

    def guarded(self, *args, **kwargs):
        if kwargs.get("post", post_json) is post_json:
            kwargs["post"] = refuse("call to the Cloudflare API")
        original(self, *args, **kwargs)

    monkeypatch.setattr(cloudflare.CloudflareEmbeddingService, "__init__", guarded)
