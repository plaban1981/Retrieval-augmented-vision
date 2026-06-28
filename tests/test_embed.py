import json
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from src.embed import get_device, _encode, embed_query, embed_tiles


@pytest.fixture
def mock_model():
    output = MagicMock()
    output.pooler_output = None
    output.last_hidden_state = torch.rand(1, 10, 16)
    return MagicMock(return_value=output)


@pytest.fixture
def mock_processor():
    proc = MagicMock()
    mock_batch = MagicMock()
    mock_batch.to.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}
    proc.return_value = mock_batch
    return proc


def test_get_device_returns_valid_string():
    assert get_device() in ("cuda", "cpu")


def test_encode_text_returns_normalised_float32(mock_model, mock_processor):
    vec = _encode(mock_model, mock_processor, "cpu", text="hello world")
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_encode_image_returns_normalised_float32(mock_model, mock_processor, sample_png):
    from PIL import Image
    vec = _encode(mock_model, mock_processor, "cpu", image=Image.open(sample_png))
    assert vec.dtype == np.float32
    assert vec.ndim == 1
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_query_returns_ndarray(mock_model, mock_processor):
    vec = embed_query("test question", mock_model, mock_processor, "cpu")
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1


def test_embed_tiles_saves_cache(tmp_path, mock_model, mock_processor, sample_png):
    tiles = [{"path": str(sample_png), "page": 1}]
    embed_tiles(tiles, mock_model, mock_processor, "cpu", tmp_path)
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "metadata.json").exists()


def test_embed_tiles_uses_cache_on_second_call(tmp_path, mock_model, mock_processor, sample_png):
    tiles = [{"path": str(sample_png), "page": 1}]
    embs1, _ = embed_tiles(tiles, mock_model, mock_processor, "cpu", tmp_path)
    calls_after_first = mock_model.call_count

    embs2, _ = embed_tiles(tiles, mock_model, mock_processor, "cpu", tmp_path)

    assert mock_model.call_count == calls_after_first  # no extra model calls
    np.testing.assert_array_equal(embs1, embs2)


def test_embed_tiles_rebuilds_on_corrupt_cache(tmp_path, mock_model, mock_processor, sample_png):
    tiles = [{"path": str(sample_png), "page": 1}]
    # Write corrupt cache
    (tmp_path / "embeddings.npy").write_bytes(b"not a numpy file")
    (tmp_path / "metadata.json").write_text("not json{{{")

    # Should not raise — should re-embed
    embs, meta = embed_tiles(tiles, mock_model, mock_processor, "cpu", tmp_path)
    assert len(embs) == 1
