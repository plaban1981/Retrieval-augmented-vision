import numpy as np
import pytest
from src.index import build_index, load_index, search


@pytest.fixture
def embeddings_5x8():
    np.random.seed(42)
    vecs = np.random.rand(5, 8).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms  # L2-normalised, mirrors embed.py output


@pytest.fixture
def metadata_5():
    return [{"path": f"tile_{i}.png", "page": i + 1} for i in range(5)]


def test_build_index_ntotal(tmp_path, embeddings_5x8, metadata_5):
    index, _ = build_index(embeddings_5x8, metadata_5, tmp_path)
    assert index.ntotal == 5


def test_build_index_saves_files(tmp_path, embeddings_5x8, metadata_5):
    build_index(embeddings_5x8, metadata_5, tmp_path)
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "metadata.json").exists()


def test_load_index_roundtrip(tmp_path, embeddings_5x8, metadata_5):
    build_index(embeddings_5x8, metadata_5, tmp_path)
    index, meta = load_index(tmp_path)
    assert index.ntotal == 5
    assert len(meta) == 5
    assert meta[0]["page"] == 1


def test_load_index_returns_none_when_missing(tmp_path):
    index, meta = load_index(tmp_path)
    assert index is None
    assert meta is None


def test_search_returns_k_results(tmp_path, embeddings_5x8, metadata_5):
    index, meta = build_index(embeddings_5x8, metadata_5, tmp_path)
    results = search(embeddings_5x8[0], index, meta, k=3)
    assert len(results) == 3


def test_search_exact_match_is_first(tmp_path, embeddings_5x8, metadata_5):
    index, meta = build_index(embeddings_5x8, metadata_5, tmp_path)
    results = search(embeddings_5x8[0], index, meta, k=3)
    assert results[0]["page"] == 1  # exact match for tile_0 → page 1


def test_search_score_in_unit_range(tmp_path, embeddings_5x8, metadata_5):
    index, meta = build_index(embeddings_5x8, metadata_5, tmp_path)
    query = np.random.rand(8).astype(np.float32)
    for r in search(query, index, meta, k=2):
        assert 0.0 <= r["score"] <= 1.0
