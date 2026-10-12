"""The settings `scripts/eval_retrieval.py` builds, and the guard that stops it lying.

The trap these tests close is recorded in docs/RESULTS.md at ce05005. The script took three of
its index settings from the server and let the record format fall back to `whole`, while the
embedder it scored with was the server's own and signed itself `|records=structured`. Set
`CODEATLAS_RECORD_FORMAT=structured` and it printed `"record_format":"whole"` beside a
structured signature, and a held-out MRR@10 for a pair the server refuses to put together.

So two things are under test: every index setting comes from `di.index_options`, and a
signature that disagrees with those settings stops the run instead of scoring it.
"""

import argparse

import pytest

from codeatlas.app import di
from codeatlas.services.eval.retrieval_eval import EvalSettings, IndexSettings
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.services.retrieval.record_format import STRUCTURED, WHOLE
from codeatlas.utils.config import load_config
from scripts import eval_retrieval as script

RECORD_VARS = (
    "CODEATLAS_RECORD_FORMAT",
    "CODEATLAS_RECORD_WINDOW_CHARS",
    "CODEATLAS_RECORD_WINDOW_OVERLAP",
    "CODEATLAS_EMBEDDING_PROVIDER",
)


def _clear() -> None:
    """di caches the config, the embedder and the retrieval settings; a test changes the environment."""
    for cached in (di.get_config, di.get_embedder, di.get_retrieval_settings):
        cached.cache_clear()


@pytest.fixture(autouse=True)
def _fresh_di(monkeypatch):
    for name in RECORD_VARS:
        monkeypatch.delenv(name, raising=False)
    _clear()
    yield
    _clear()


def _args(**overrides) -> argparse.Namespace:
    """The parsed arguments of a plain run: every tuning flag unset."""
    defaults = {
        "candidates": None,
        "rerank_weight": None,
        "rerank_subtokens": None,
        "drop_stopwords": None,
        "skip_tests": None,
        "embed_max_chars": None,
        "prefix_metadata": None,
        "hash_lowercase": None,
        "hash_subtokens": None,
    }
    return argparse.Namespace(**{**defaults, **overrides})


# ---------------------------------------------------------------- the server's index settings


def test_a_plain_run_still_describes_the_default() -> None:
    """No environment setting: whole records, the window defaults, and an unwrapped signature."""
    settings, embedder = script._settings(_args())

    assert settings.index.record_format == WHOLE
    assert settings.index.window_chars == 2000
    assert settings.index.window_overlap == 200
    assert script.signed_record_format(embedder) == WHOLE
    assert script.mismatch(embedder, settings) == ""


@pytest.mark.parametrize("record_format", [WHOLE, STRUCTURED])
def test_every_index_setting_is_the_servers(monkeypatch, record_format) -> None:
    """Field by field against di.index_options, which is what the server builds its index with."""
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", record_format)
    _clear()
    options = di.index_options(load_config())

    settings, _ = script._settings(_args())

    assert settings.index.record_format == options["record_format"]
    assert settings.index.window_chars == options["window_chars"]
    assert settings.index.window_overlap == options["window_overlap"]
    assert settings.index.max_chars == options["max_chars"]
    assert settings.index.prefix_metadata == options["prefix_metadata"]


def test_the_record_format_reaches_the_index_settings(monkeypatch) -> None:
    """The failure itself: the format was read for the embedder and dropped for the index."""
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", STRUCTURED)
    _clear()

    settings, embedder = script._settings(_args())

    assert settings.index.record_format == STRUCTURED
    assert embedder.signature().endswith("|records=structured")
    assert script.mismatch(embedder, settings) == ""


def test_the_window_settings_are_read_too(monkeypatch) -> None:
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", STRUCTURED)
    monkeypatch.setenv("CODEATLAS_RECORD_WINDOW_CHARS", "1200")
    monkeypatch.setenv("CODEATLAS_RECORD_WINDOW_OVERLAP", "50")
    _clear()

    settings, _ = script._settings(_args())

    assert (settings.index.window_chars, settings.index.window_overlap) == (1200, 50)


def test_a_hash_flag_rebuilds_the_embedder_and_keeps_the_format(monkeypatch) -> None:
    """--hash-subtokens builds a fresh HashEmbeddingService; it must come back wrapped."""
    monkeypatch.setenv("CODEATLAS_RECORD_FORMAT", STRUCTURED)
    _clear()

    settings, embedder = script._settings(_args(hash_subtokens="off"))

    assert "subtokens=0" in embedder.signature()
    assert script.signed_record_format(embedder) == STRUCTURED
    assert script.mismatch(embedder, settings) == ""


def test_the_tuning_flags_still_reach_the_retrieval_settings() -> None:
    settings, _ = script._settings(_args(candidates=40, drop_stopwords="off"))

    assert settings.retrieval.candidates == 40
    assert settings.retrieval.drop_stopwords is False


# ---------------------------------------------------------------- the guard


def test_an_unwrapped_signature_reads_as_whole() -> None:
    assert script.signed_record_format(HashEmbeddingService()) == WHOLE


def test_a_disagreement_is_reported_with_both_sides() -> None:
    message = script.mismatch(
        HashEmbeddingService(),
        EvalSettings(index=IndexSettings(record_format=STRUCTURED)),
    )

    assert message.startswith("FAIL:")
    assert "records=whole" in message
    assert "record_format=structured" in message
    assert "CODEATLAS_RECORD_FORMAT" in message, "the message has to say which setting to look at"


def test_the_other_direction_is_a_disagreement_too() -> None:
    from codeatlas.services.retrieval.record_format import with_record_format

    message = script.mismatch(
        with_record_format(HashEmbeddingService(), STRUCTURED),
        EvalSettings(index=IndexSettings(record_format=WHOLE)),
    )

    assert "records=structured" in message and "record_format=whole" in message


def test_main_exits_1_and_scores_nothing_on_a_disagreement(monkeypatch, capsys) -> None:
    """The point of the guard: a mismatched pair must not produce a number at all."""
    monkeypatch.setattr(
        script,
        "_settings",
        lambda args: (EvalSettings(index=IndexSettings(record_format=STRUCTURED)), HashEmbeddingService()),
    )

    def refuse(*args, **kwargs):
        raise AssertionError("a mismatched pair was scored")

    monkeypatch.setattr(script, "run", refuse)
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--split", "test"])

    assert script.main() == 1
    printed = capsys.readouterr().out
    assert "FAIL:" in printed


def test_main_prints_the_effective_record_format(monkeypatch, capsys) -> None:
    """A reader must be able to see what a record was, without reading this file."""
    monkeypatch.setattr(
        script,
        "_settings",
        lambda args: (EvalSettings(index=IndexSettings(record_format=STRUCTURED)), HashEmbeddingService()),
    )
    monkeypatch.setattr("sys.argv", ["eval_retrieval.py", "--split", "test"])

    script.main()

    assert "record format: structured (window 2000, overlap 200)" in capsys.readouterr().out
