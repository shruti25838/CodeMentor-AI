"""The fastembed provider is opt-in: selecting it must change indexing, and nothing else should.

Most of these tests never build a model. The two that do are skipped when fastembed is not
installed, which is the default for this project.
"""

from dataclasses import replace

import pytest

from codeatlas.app.di import build_embedder, index_options
from codeatlas.services.retrieval.fastembed_embedder import (
    DEFAULT_FASTEMBED_BATCH_SIZE,
    DEFAULT_FASTEMBED_MODEL,
    NOT_INSTALLED,
    FastEmbedEmbeddingService,
)
from codeatlas.services.retrieval.hash_embedder import HashEmbeddingService
from codeatlas.utils.config import load_config


def _config(provider: str):
    return replace(load_config(), embedding_provider=provider)


def test_hash_is_still_the_default(monkeypatch) -> None:
    monkeypatch.delenv("CODEATLAS_EMBEDDING_PROVIDER", raising=False)
    config = load_config()
    assert config.embedding_provider == "hash"
    assert isinstance(build_embedder(config), HashEmbeddingService)


def test_fastembed_is_selected_by_name_without_loading_a_model() -> None:
    embedder = build_embedder(_config("fastembed"))
    assert isinstance(embedder, FastEmbedEmbeddingService)
    assert embedder._model_name == DEFAULT_FASTEMBED_MODEL


def test_fastembed_embeds_one_document_at_a_time() -> None:
    assert DEFAULT_FASTEMBED_BATCH_SIZE == 1
    assert build_embedder(_config("fastembed"))._batch_size == 1
    assert index_options(_config("fastembed"))["batch_size"] == 1


def test_fastembed_clips_each_document_to_3000_characters() -> None:
    assert index_options(_config("fastembed"))["max_chars"] == 3000


def test_the_longer_index_timeout_applies_only_to_fastembed() -> None:
    config = _config("fastembed")
    assert index_options(config)["timeout_seconds"] == config.fastembed_index_timeout_seconds
    assert config.fastembed_index_timeout_seconds > config.index_timeout_seconds


@pytest.mark.parametrize("provider", ["hash", "sentence"])
def test_other_providers_keep_the_settings_tuned_for_hash(provider) -> None:
    config = _config(provider)
    options = index_options(config)
    assert options["timeout_seconds"] == config.index_timeout_seconds
    assert options["max_chars"] == config.embed_max_chars
    # No batch size of its own, so CodeIndexService uses its default.
    assert "batch_size" not in options


def test_the_fastembed_only_settings_are_ignored_by_other_providers() -> None:
    """Changing a fastembed knob must not move a hash index, or the index cache would miss."""
    base = _config("hash")
    loud = replace(base, fastembed_batch_size=32, fastembed_max_chars=10, fastembed_index_timeout_seconds=1)
    assert index_options(base) == index_options(loud)


def test_fastembed_and_hash_do_not_share_an_index_signature() -> None:
    assert FastEmbedEmbeddingService().signature() != HashEmbeddingService().signature()
    assert FastEmbedEmbeddingService().signature().startswith("fastembed:")


def test_the_signature_names_the_model_so_two_models_are_not_mixed() -> None:
    assert (
        FastEmbedEmbeddingService(model_name="a").signature() != FastEmbedEmbeddingService(model_name="b").signature()
    )


def test_batch_size_is_not_part_of_the_signature() -> None:
    """It changes how long indexing takes, not what the vectors are."""
    assert FastEmbedEmbeddingService(batch_size=1).signature() == FastEmbedEmbeddingService(batch_size=8).signature()


def test_a_missing_install_says_what_to_do(monkeypatch) -> None:
    monkeypatch.setattr("codeatlas.services.retrieval.fastembed_embedder.TextEmbedding", None)
    with pytest.raises(RuntimeError) as exc:
        FastEmbedEmbeddingService().embed_texts(["x"])
    assert "requirements-fastembed.txt" in str(exc.value)
    assert "CODEATLAS_EMBEDDING_PROVIDER=hash" in str(exc.value)
    assert str(exc.value) == NOT_INSTALLED


def test_no_texts_needs_no_model(monkeypatch) -> None:
    monkeypatch.setattr("codeatlas.services.retrieval.fastembed_embedder.TextEmbedding", None)
    assert FastEmbedEmbeddingService().embed_texts([]) == []


def test_real_vectors_are_unit_length_and_the_right_width() -> None:
    pytest.importorskip("fastembed", reason="optional provider; pip install -r requirements-fastembed.txt")
    embedder = FastEmbedEmbeddingService()
    vectors = embedder.embed_texts(["def sign(value): ...", "how are cookies signed"])
    assert len(vectors) == 2
    assert {len(v) for v in vectors} == {384}
    assert all(abs(sum(x * x for x in v) - 1.0) < 1e-3 for v in vectors)


def test_a_question_is_closer_to_the_code_it_is_about_than_to_unrelated_code() -> None:
    """The point of the provider: matching on meaning, which the hash embedder cannot do."""
    pytest.importorskip("fastembed", reason="optional provider; pip install -r requirements-fastembed.txt")
    embedder = FastEmbedEmbeddingService()
    about, unrelated = embedder.embed_texts(
        [
            "def verify_signature(self, value, sig):\n    return hmac.compare_digest(sig, self.get_signature(value))",
            "def render_template(name, **context):\n    return current_app.jinja_env.get_template(name).render(context)",
        ]
    )
    query = embedder.embed_query("how is a signature checked?")
    assert sum(a * b for a, b in zip(query, about)) > sum(a * b for a, b in zip(query, unrelated))
