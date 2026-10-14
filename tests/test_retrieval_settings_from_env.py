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


# ---------------------------------------------------------------- BM25, the second retriever


BM25_SETTINGS = {
    "CODEATLAS_BM25": ("bm25_enabled", True),
    "CODEATLAS_BM25_K1": ("bm25_k1", 1.2),
    "CODEATLAS_BM25_B": ("bm25_b", 0.75),
    "CODEATLAS_BM25_RRF_K": ("bm25_rrf_k", 60),
    "CODEATLAS_BM25_DEPTH": ("bm25_depth", 0),
}


@pytest.fixture(autouse=True)
def _no_bm25_variables(monkeypatch):
    for name in BM25_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_bm25_is_on_by_default_and_off_for_cloudflare() -> None:
    """The default change: fusion is worth +0.105 of MRR@10 to hash and cost cloudflare 0.031.

    The parameters beside it did not move, so every recorded number that describes *them* still
    does. What the recorded numbers under BM25 off describe is kept in eval/results/legacy/.
    """
    config = load_config()
    for name, (field, default) in BM25_SETTINGS.items():
        assert getattr(config, field) == default, name
    assert config.bm25_enabled_for("hash") is True
    assert config.bm25_enabled_for("cloudflare") is False


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("CODEATLAS_BM25", "on", True),
        ("CODEATLAS_BM25", "true", True),
        ("CODEATLAS_BM25", "off", False),
        ("CODEATLAS_BM25_K1", "1.5", 1.5),
        ("CODEATLAS_BM25_K1", "2", 2.0),
        ("CODEATLAS_BM25_B", "0", 0.0),
        ("CODEATLAS_BM25_B", "0.4", 0.4),
        ("CODEATLAS_BM25_B", "1", 1.0),
        ("CODEATLAS_BM25_RRF_K", "10", 10),
        ("CODEATLAS_BM25_DEPTH", "0", 0),
        ("CODEATLAS_BM25_DEPTH", "50", 50),
    ],
)
def test_a_valid_bm25_value_is_taken(monkeypatch, name, value, expected) -> None:
    monkeypatch.setenv(name, value)
    assert getattr(load_config(), BM25_SETTINGS[name][0]) == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_BM25", "maybe"),
        ("CODEATLAS_BM25", "1.0"),
        ("CODEATLAS_BM25_K1", "0"),
        ("CODEATLAS_BM25_K1", "-1"),
        ("CODEATLAS_BM25_K1", "high"),
        # b is a fraction of the length normalisation, so outside 0..1 it means nothing.
        ("CODEATLAS_BM25_B", "1.5"),
        ("CODEATLAS_BM25_B", "-0.1"),
        ("CODEATLAS_BM25_B", "most"),
        ("CODEATLAS_BM25_RRF_K", "0"),
        ("CODEATLAS_BM25_RRF_K", "-60"),
        ("CODEATLAS_BM25_RRF_K", "sixty"),
        ("CODEATLAS_BM25_DEPTH", "-1"),
        ("CODEATLAS_BM25_DEPTH", "deep"),
    ],
)
def test_an_invalid_bm25_value_stops_the_app(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as raised:
        load_config()
    # The variable's name, never the value a deployment set: this message reaches a log.
    assert name in str(raised.value)


def test_the_bm25_settings_reach_the_retrieval_pipeline(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_BM25", "on")
    monkeypatch.setenv("CODEATLAS_BM25_K1", "1.8")
    monkeypatch.setenv("CODEATLAS_BM25_B", "0.3")
    monkeypatch.setenv("CODEATLAS_BM25_RRF_K", "12")
    monkeypatch.setenv("CODEATLAS_BM25_DEPTH", "40")
    from codeatlas.app.di import get_config

    get_config.cache_clear()
    get_retrieval_settings.cache_clear()
    try:
        settings = get_retrieval_settings()
        assert settings.bm25_enabled is True
        assert (settings.bm25_k1, settings.bm25_b) == (1.8, 0.3)
        assert (settings.bm25_rrf_k, settings.bm25_depth) == (12, 40)
    finally:
        get_config.cache_clear()
        get_retrieval_settings.cache_clear()


# ---------------------------------------------------------------- what a record is


RECORD_SETTINGS = {
    "CODEATLAS_RECORD_FORMAT": ("record_format", "structured"),
    "CODEATLAS_RECORD_WINDOW_CHARS": ("record_window_chars", 2000),
    "CODEATLAS_RECORD_WINDOW_OVERLAP": ("record_window_overlap", 200),
}


@pytest.fixture(autouse=True)
def _no_record_variables(monkeypatch):
    for name in RECORD_SETTINGS:
        monkeypatch.delenv(name, raising=False)


def test_the_record_format_defaults_to_the_one_that_was_measured() -> None:
    """Structured for hash, whole for cloudflare. Both were measured, once each, on holdout3."""
    config = load_config()
    assert config.record_format_for("hash") == "structured"
    assert config.record_format_for("cloudflare") == "whole"
    for name, (field, default) in RECORD_SETTINGS.items():
        assert getattr(config, field) == default, name


@pytest.mark.parametrize(
    ("name", "value", "expected"),
    [
        ("CODEATLAS_RECORD_FORMAT", "structured", "structured"),
        ("CODEATLAS_RECORD_FORMAT", "STRUCTURED", "structured"),
        ("CODEATLAS_RECORD_FORMAT", " whole ", "whole"),
        ("CODEATLAS_RECORD_WINDOW_CHARS", "1500", 1500),
        ("CODEATLAS_RECORD_WINDOW_OVERLAP", "0", 0),
    ],
)
def test_a_valid_record_value_is_taken(monkeypatch, name, value, expected) -> None:
    monkeypatch.setenv(name, value)
    assert getattr(load_config(), RECORD_SETTINGS[name][0]) == expected


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CODEATLAS_RECORD_FORMAT", "chunks"),
        ("CODEATLAS_RECORD_FORMAT", "windows"),
        ("CODEATLAS_RECORD_WINDOW_CHARS", "0"),
        ("CODEATLAS_RECORD_WINDOW_CHARS", "-100"),
        ("CODEATLAS_RECORD_WINDOW_CHARS", "big"),
        ("CODEATLAS_RECORD_WINDOW_OVERLAP", "-1"),
        ("CODEATLAS_RECORD_WINDOW_OVERLAP", "some"),
    ],
)
def test_an_invalid_record_value_stops_the_app(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError) as raised:
        load_config()
    message = str(raised.value)
    assert name in message or "record format" in message.lower()


def test_the_format_is_in_the_signature_so_a_stale_index_is_refused(monkeypatch) -> None:
    """The default moved, so the signature moved with it, which is the guard working.

    Every index built under the old default is refused rather than searched: `has_index` says
    no, the index cache misses, and the repository is reindexed on its next analysis. `whole`
    is still the unwrapped signature, so a server set back to it reads those indexes again.
    """
    from codeatlas.app.di import build_embedder

    assert build_embedder(load_config()).signature() == "hash:384:lowercase=0:subtokens=1|records=structured"
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", "whole")
    assert build_embedder(load_config()).signature() == "hash:384:lowercase=0:subtokens=1"


# ---------------------------------------------------------------- how a repository question is answered


def test_the_chat_mode_defaults_to_legacy(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_CHAT_MODE", raising=False)
    assert load_config().chat_mode == "legacy"


@pytest.mark.parametrize("value,expected", [("grounded", "grounded"), ("LEGACY", "legacy"), (" grounded ", "grounded")])
def test_a_valid_chat_mode_is_taken(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("CODEATLAS_CHAT_MODE", value)
    assert load_config().chat_mode == expected


@pytest.mark.parametrize("value", ["agents", "fast", "deep", "streaming"])
def test_an_unknown_chat_mode_stops_the_app(monkeypatch, value) -> None:
    monkeypatch.setenv("CODEATLAS_CHAT_MODE", value)
    with pytest.raises(ValueError) as raised:
        load_config()
    assert value in str(raised.value)
