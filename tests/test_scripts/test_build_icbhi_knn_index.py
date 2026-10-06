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


def test_similarity_of_unit_rows_is_cosine():
    """A dot product of unit-norm rows is exactly cosine similarity; the diagonal proves it."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((NUM_CLIPS, EMBED_DIM))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity = embeddings @ embeddings.T

    assert_that(np.allclose(np.diag(similarity), 1.0)).is_true()
    assert_that(float(similarity.max())).is_less_than_or_equal_to(1.0 + 1e-9)


ITEM_IDS = ["eval0", "eval1", "demo0", "demo1"]
CANDIDATES = {"demo0", "demo1"}


def test_neighbours_come_only_from_the_held_out_candidates(knn):
    """Retrieving from the evaluation set would hand the model other scored items' labels."""
    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((len(ITEM_IDS), EMBED_DIM))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    neighbors = knn.nearest_neighbors(embeddings, ITEM_IDS, CANDIDATES, {}, top_k=5)

    assert_that(set(neighbors)).is_equal_to({"eval0", "eval1"})  # candidates are never queries
    for indices in neighbors.values():
        assert_that({ITEM_IDS[i] for i in indices}).is_subset_of(CANDIDATES)
        assert_that(indices).is_length(len(CANDIDATES))  # top_k capped at the pool size


def test_nearest_neighbors_excludes_near_duplicates(knn):
    embeddings = np.eye(len(ITEM_IDS), EMBED_DIM)
    embeddings[2] = embeddings[0]  # demo0 is a copy of eval0

    neighbors = knn.nearest_neighbors(embeddings, ITEM_IDS, CANDIDATES, {"eval0": ["demo0"]}, top_k=1)

    assert_that(ITEM_IDS[neighbors["eval0"][0]]).is_equal_to("demo1")


def test_nearest_neighbors_rejects_a_non_matrix(knn):
    assert_that(knn.nearest_neighbors).raises(ValueError).when_called_with(
        np.zeros((len(ITEM_IDS), 1, EMBED_DIM)), ITEM_IDS, CANDIDATES, {}, 2
    )


def test_missing_split_refuses_to_run(knn, tmp_path):
    assert_that(knn.load_candidate_ids).raises(SystemExit).when_called_with(str(tmp_path / "nope.json"))


def test_knn_control_is_scored_on_evaluation_items_only(knn):
    labels = ["COPD", "URTI", "COPD", "URTI"]
    neighbors = {"eval0": [2], "eval1": [2]}  # both retrieve demo0 (COPD)

    assert_that(knn.knn_vote_accuracy(neighbors, labels, ITEM_IDS)).is_close_to(50.0, 0.01)
