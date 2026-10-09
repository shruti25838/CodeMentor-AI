"""The seven retrieval settings read from the environment.

Two things are under test and they pull in opposite directions: the defaults must be exactly
what docs/RESULTS.md measured, and a value that is not a real setting must stop the app rather
than quietly becoming one.
"""

import pytest

from codeatlas.app.di import get_retrieval_settings, index_options
from codeatlas.utils.config import load_config

SETTINGS = {
    "CODEATLAS_RETRIEVAL_CANDIDATES": ("retrieval_candidates", 20),
    "CODEATLAS_RERANK_WEIGHT": ("rerank_weight", 1.0),
    "CODEATLAS_RERANK_SUBTOKENS": ("rerank_subtokens", False),
    "CODEATLAS_DROP_STOPWORDS": ("drop_stopwords", True),
    "CODEATLAS_SKIP_TESTS": ("skip_tests", False),
    "CODEATLAS_EMBED_MAX_CHARS": ("embed_max_chars", None),
    "CODEATLAS_EMBED_PREFIX_METADATA": ("embed_prefix_metadata", True),
}


@pytest.fixture(autouse=True)
def _nothing_set(monkeypatch):
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_every_default_is_the_one_that_was_measured() -> None:
    config = load_config()
    for name, (field, default) in SETTINGS.items():
        assert getattr(config, field) == default, name


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "40", 40),
        ("CODEATLAS_RERANK_WEIGHT", "0", 0.0),
        ("CODEATLAS_RERANK_WEIGHT", "0.35", 0.35),
        ("CODEATLAS_RERANK_WEIGHT", "1", 1.0),
        ("CODEATLAS_RERANK_SUBTOKENS", "true", True),
        ("CODEATLAS_RERANK_SUBTOKENS", "ON", True),
        ("CODEATLAS_DROP_STOPWORDS", "false", False),
        ("CODEATLAS_DROP_STOPWORDS", "0", False),
        ("CODEATLAS_SKIP_TESTS", "yes", True),
        ("CODEATLAS_EMBED_MAX_CHARS", "3000", 3000),
        ("CODEATLAS_EMBED_PREFIX_METADATA", "off", False),
    ],
)
def test_a_valid_value_is_taken(monkeypatch, name, value, expected) -> None:
    monkeypatch.setenv(name, value)
    assert getattr(load_config(), SETTINGS[name][0]) == expected


def test_zero_characters_means_no_limit(monkeypatch) -> None:
    """0 is how an environment variable says "unlimited"; inside the app that is None."""
    monkeypatch.setenv("CODEATLAS_EMBED_MAX_CHARS", "0")
    assert load_config().embed_max_chars is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "0"),
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "-5"),
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "twenty"),
        ("CODEATLAS_RETRIEVAL_CANDIDATES", "20.5"),
        ("CODEATLAS_RERANK_WEIGHT", "1.5"),
        ("CODEATLAS_RERANK_WEIGHT", "-0.1"),
        ("CODEATLAS_RERANK_WEIGHT", "heavy"),
        ("CODEATLAS_RERANK_SUBTOKENS", "truthy"),
        ("CODEATLAS_RERANK_SUBTOKENS", "maybe"),
        ("CODEATLAS_DROP_STOPWORDS", "ture"),
        ("CODEATLAS_SKIP_TESTS", "2"),
        ("CODEATLAS_EMBED_MAX_CHARS", "-1"),
        ("CODEATLAS_EMBED_MAX_CHARS", "all"),
        ("CODEATLAS_EMBED_PREFIX_METADATA", ""),
    ],
)
def test_a_value_that_is_not_a_setting_stops_the_app(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    if value == "":
        # The one exception, stated outright: an empty variable is how a deployment unsets
        # something, so it means "use the default" rather than "fail".
        assert load_config().embed_prefix_metadata is True
        return
    with pytest.raises(ValueError) as raised:
        load_config()
    assert name in str(raised.value)


def test_a_typo_is_never_quietly_read_as_off(monkeypatch) -> None:
    """The failure this guards against: the app runs, the answers change, nobody is told."""
    monkeypatch.setenv("CODEATLAS_DROP_STOPWORDS", "flase")
    with pytest.raises(ValueError):
        load_config()


def test_the_settings_reach_the_retrieval_pipeline(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RETRIEVAL_CANDIDATES", "33")
    monkeypatch.setenv("CODEATLAS_RERANK_WEIGHT", "0.25")
    monkeypatch.setenv("CODEATLAS_RERANK_SUBTOKENS", "true")
    monkeypatch.setenv("CODEATLAS_DROP_STOPWORDS", "false")
    monkeypatch.setenv("CODEATLAS_SKIP_TESTS", "true")
    from codeatlas.app.di import get_config

    get_config.cache_clear()
    get_retrieval_settings.cache_clear()
    try:
        settings = get_retrieval_settings()
        assert settings.candidates == 33
        assert settings.rerank_weight == 0.25
        assert settings.rerank_subtokens is True
        assert settings.drop_stopwords is False
        assert settings.skip_tests is True
    finally:
        get_config.cache_clear()
        get_retrieval_settings.cache_clear()


def test_the_indexing_settings_reach_indexing(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_EMBED_MAX_CHARS", "1234")
    monkeypatch.setenv("CODEATLAS_EMBED_PREFIX_METADATA", "false")
    options = index_options(load_config())
    assert options["max_chars"] == 1234
    assert options["prefix_metadata"] is False


def test_a_provider_may_still_have_its_own_defaults(monkeypatch) -> None:
    """embed_max_chars is the shared setting; cloudflare clips at its own figure regardless."""
    from dataclasses import replace

    monkeypatch.setenv("CODEATLAS_EMBED_MAX_CHARS", "9999")
    config = replace(load_config(), embedding_provider="cloudflare")
    assert index_options(config)["max_chars"] == config.cloudflare_max_chars == 2000
