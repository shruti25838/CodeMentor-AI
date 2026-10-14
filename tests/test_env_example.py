"""Every environment variable the config reads is documented in .env.example.

A setting that exists only in `codeatlas/utils/config.py` is a setting nobody deploying this
app can find. The file is the only list of them there is, so it drifting behind the loader is
the failure worth catching: the variables added here in batch 13 - the per-index record format
and BM25 pair, the BM25 cache cap, the window size and overlap, and the chat mode - had all
been reachable for some time with no line in `.env.example` saying so.

What is compared is not a regex over the source but the names `load_config()` and
`load_cloudflare_credentials()` actually ask the environment for, recorded by standing in for
`os.getenv` while they run. A variable read through a helper, or renamed, is still caught;
a string that merely looks like a variable name is not mistaken for one.
"""

from pathlib import Path

import pytest

import codeatlas.utils.config as config_module
from codeatlas.utils.config import load_cloudflare_credentials, load_config

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def _names_read_by_the_config(monkeypatch) -> set[str]:
    """Every name the two loaders pass to os.getenv, with the defaults left in place."""
    seen: set[str] = set()
    real = config_module.os.getenv

    def recording(name, default=None):
        seen.add(name)
        return real(name, default)

    monkeypatch.setattr(config_module.os, "getenv", recording)
    load_config()
    load_cloudflare_credentials()
    return seen


def _names_documented() -> set[str]:
    """Every NAME= in .env.example, whether the line is commented out or not.

    A commented line still documents the variable - several here are examples of what to set
    rather than defaults to keep - so `# CODEATLAS_MAX_DOWNLOAD_MB=200` counts.
    """
    documented: set[str] = set()
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.lstrip("#").strip()
        name, sep, _ = stripped.partition("=")
        if sep and name and name.replace("_", "").isalnum() and name.isupper():
            documented.add(name)
    return documented


def _assert_all_documented(read: set[str], documented: set[str]) -> None:
    missing = sorted(read - documented)

    assert not missing, (
        f"{len(missing)} setting(s) are read by codeatlas/utils/config.py but appear nowhere in "
        f".env.example, so nobody deploying this app can find them: {', '.join(missing)}. "
        f"Add each with its default and one line saying what it does."
    )


def test_every_variable_the_config_reads_is_in_env_example(monkeypatch) -> None:
    _assert_all_documented(_names_read_by_the_config(monkeypatch), _names_documented())


def test_the_recording_getenv_sees_settings_that_are_really_read(monkeypatch) -> None:
    """The guard above is only worth anything if it observes the reads. Prove that it does."""
    seen = _names_read_by_the_config(monkeypatch)

    assert "CODEATLAS_EMBEDDING_PROVIDER" in seen
    assert "CODEATLAS_HASH_BM25" in seen
    assert "CLOUDFLARE_API_TOKEN" in seen


def test_a_variable_absent_from_env_example_fails_the_check() -> None:
    """And that a missing one is reported rather than passed over."""
    with pytest.raises(AssertionError, match="CODEATLAS_A_SETTING_NOBODY_DOCUMENTED"):
        _assert_all_documented({"CODEATLAS_A_SETTING_NOBODY_DOCUMENTED"}, {"CODEATLAS_INDEX_DIR"})
