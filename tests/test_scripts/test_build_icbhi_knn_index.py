"""Guards the CLAP embedding contract.

`get_audio_features` returns a model output object, not a tensor. Taking `[0]` of it yields
`last_hidden_state` - a per-frame sequence - which stacks into a 3-D array and only fails later,
inside the similarity matmul, with an opaque shape error. These tests pin the contract at the point
where it is actually readable.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from assertpy import assert_that
from transformers.modeling_outputs import BaseModelOutputWithPooling

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_icbhi_knn_index.py"

NUM_CLIPS = 4
EMBED_DIM = 512
SEQ_LEN = 32


@pytest.fixture(scope="module")
def knn():
    spec = importlib.util.spec_from_file_location("build_icbhi_knn_index", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_output(pooled: torch.Tensor) -> BaseModelOutputWithPooling:
    """What transformers >= 5 hands back: a sequence first, the pooled embedding second."""
    return BaseModelOutputWithPooling(last_hidden_state=torch.randn(1, SEQ_LEN, EMBED_DIM), pooler_output=pooled)


@pytest.mark.parametrize(
    "output_factory",
    [
        pytest.param(model_output, id="transformers_5_model_output"),
        pytest.param(lambda pooled: pooled, id="legacy_bare_tensor"),
    ],
)
def test_pooled_embedding_returns_a_unit_norm_vector(knn, output_factory):
    pooled = torch.arange(EMBED_DIM, dtype=torch.float32).unsqueeze(0)

    embedding = knn.pooled_embedding(output_factory(pooled))

    assert_that(embedding.shape).is_equal_to((EMBED_DIM,))
    assert_that(float(np.linalg.norm(embedding))).is_close_to(1.0, 1e-5)


def test_pooled_embedding_rejects_a_sequence(knn):
    """This is the failure that used to surface as an unreadable matmul error."""
    assert_that(knn.pooled_embedding).raises(ValueError).when_called_with(torch.randn(1, SEQ_LEN, EMBED_DIM))


class FakeProcessor:
    def __call__(self, audio, sampling_rate, return_tensors):
        return {"input_features": torch.randn(1, 1, 8, 8)}


class FakeModel:
    def get_audio_features(self, **_):
        return model_output(torch.randn(1, EMBED_DIM))


def test_embed_produces_a_two_dimensional_matrix(knn):
    clips = [(f"audio{i}", "COPD", np.zeros(16, dtype=np.float32)) for i in range(NUM_CLIPS)]

    embeddings = knn.embed(clips, FakeModel(), FakeProcessor(), "cpu")

    assert_that(embeddings.shape).is_equal_to((NUM_CLIPS, EMBED_DIM))
    assert_that(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0)).is_true()


def test_similarity_of_unit_rows_is_cosine(knn):
    """A dot product of unit-norm rows is exactly cosine similarity; the diagonal proves it."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((NUM_CLIPS, EMBED_DIM))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    item_ids = [f"audio{i}" for i in range(NUM_CLIPS)]

    neighbors = knn.nearest_neighbors(embeddings, item_ids, {}, top_k=2)

    similarity = embeddings @ embeddings.T
    assert_that(np.allclose(np.diag(similarity), 1.0)).is_true()
    assert_that(float(similarity.max())).is_less_than_or_equal_to(1.0 + 1e-9)
    for item_id, indices in neighbors.items():
        assert_that(indices).is_length(2)
        assert_that(item_ids.index(item_id)).is_not_in(*indices)


def test_nearest_neighbors_excludes_near_duplicates(knn):
    embeddings = np.eye(NUM_CLIPS, EMBED_DIM)
    embeddings[1] = embeddings[0]  # audio1 is a copy of audio0
    item_ids = [f"audio{i}" for i in range(NUM_CLIPS)]

    neighbors = knn.nearest_neighbors(embeddings, item_ids, {"audio0": ["audio1"]}, top_k=1)

    assert_that(item_ids[neighbors["audio0"][0]]).is_not_equal_to("audio1")


def test_nearest_neighbors_rejects_a_non_matrix(knn):
    assert_that(knn.nearest_neighbors).raises(ValueError).when_called_with(
        np.zeros((NUM_CLIPS, 1, EMBED_DIM)), ["a", "b", "c", "d"], {}, 2
    )
