"""The legacy hash baseline: whole records, BM25 off, hit@5 0.750 and MRR@10 0.645 held out.

Every number in `docs/RESULTS.md` up to the holdout3 entries was measured under that pair, and
hundreds of lines of that file end with *"--split test -> overall hit@5 0.750, MRR@10 0.645,
unchanged"*. When the default moves, `eval/results/{dev,test}.json` move with it and describe
whatever the default is now, which is correct and which would leave every one of those lines
describing a configuration nothing points at any more.

So the baseline is kept as its own committed pair of result files, and this pins them. What is
asserted is deliberately two different kinds of thing:

* the **numbers**, so a result file that drifted would be caught rather than quietly believed;
* that the **configuration those files name is reachable** - that setting the two variables in
  `eval/results/legacy/README.md` really does produce whole records with BM25 off. A baseline
  nobody can get back to is a souvenir, not a baseline.

Nothing here runs the evaluation. It reads committed files and the configuration loader, so it
needs no checkout, no network and no provider.
"""

import json
from pathlib import Path

import pytest

from codeatlas.app.di import build_embedder, index_options, retrieval_settings_for
from codeatlas.services.retrieval.record_format import WHOLE
from codeatlas.utils.config import load_config

LEGACY = Path(__file__).resolve().parent.parent / "eval" / "results" / "legacy"

# The two variables the README tells a reader to set, and what they have to produce.
LEGACY_ENV = {"CODEATLAS_HASH_RECORD_FORMAT": "whole", "CODEATLAS_HASH_BM25": "false"}

# The held-out pair that docs/RESULTS.md quotes throughout.
HELD_OUT_HIT_AT_5 = 0.750
HELD_OUT_MRR_AT_10 = 0.645


def _read(name: str) -> dict:
    return json.loads((LEGACY / name).read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _nothing_set(monkeypatch):
    for name in ("CODEATLAS_RECORD_FORMAT", "CODEATLAS_BM25", *LEGACY_ENV):
        monkeypatch.delenv(name, raising=False)


def test_the_held_out_baseline_is_the_pair_this_file_has_always_quoted() -> None:
    overall = _read("test.json")["overall"]

    assert overall["questions"] == 20
    assert overall["hit_rate"]["@5"] == HELD_OUT_HIT_AT_5
    assert overall["mrr@10"] == HELD_OUT_MRR_AT_10


def test_the_dev_baseline_is_recorded_too() -> None:
    overall = _read("dev.json")["overall"]

    assert overall["questions"] == 20
    assert (overall["hit_rate"]["@5"], overall["mrr@10"]) == (0.750, 0.6625)


@pytest.mark.parametrize("name", ["dev.json", "test.json"])
def test_each_legacy_file_names_the_configuration_it_was_measured_under(name) -> None:
    """A number without the settings beside it is not reproducible, however exactly it is pinned."""
    settings = _read(name)["settings"]

    assert settings["index"]["record_format"] == WHOLE
    assert settings["retrieval"]["bm25_enabled"] is False


@pytest.mark.parametrize("name", ["dev.json", "test.json"])
def test_each_legacy_file_was_measured_by_the_hash_embedder_with_no_format_in_its_signature(name) -> None:
    """`whole` is the unwrapped signature, which is why no index built before all this is invalidated."""
    embedder = _read(name)["embedder"]

    assert embedder.startswith("hash (HashEmbeddingService")
    assert "records " not in embedder


def test_the_variables_in_the_readme_really_produce_that_configuration(monkeypatch) -> None:
    """The half that makes it a baseline: the configuration can be got back to, not just read about."""
    for name, value in LEGACY_ENV.items():
        monkeypatch.setenv(name, value)
    config = load_config()

    assert config.record_format_for("hash") == WHOLE
    assert config.bm25_enabled_for("hash") is False
    assert index_options(config)["record_format"] == WHOLE
    assert retrieval_settings_for(config, "hash").bm25_enabled is False
    # And the embedder signs itself as it always did, so an index built under the legacy
    # configuration is one a legacy-configured server will still search.
    assert build_embedder(config).signature() == "hash:384:lowercase=0:subtokens=1"


def test_the_readme_tells_a_reader_to_set_exactly_those_variables() -> None:
    readme = (LEGACY / "README.md").read_text(encoding="utf-8")

    for name, value in LEGACY_ENV.items():
        assert f'$env:{name} = "{value}"' in readme
    assert "0.750" in readme and "0.645" in readme


def test_the_legacy_files_are_not_the_ones_the_default_regenerates() -> None:
    """They are a separate pair on purpose: eval/results/{dev,test}.json follow the default."""
    current = Path(__file__).resolve().parent.parent / "eval" / "results"

    assert (LEGACY / "test.json").resolve() != (current / "test.json").resolve()
    assert (LEGACY / "test.json").exists() and (LEGACY / "dev.json").exists()
