import base64
import pytest
from unittest.mock import MagicMock, patch
from src.reader import answer, _encode_image


def test_encode_image_returns_valid_base64(sample_png):
    result = _encode_image(sample_png)
    assert isinstance(result, str)
    base64.b64decode(result)  # must not raise


def test_answer_raises_without_api_key(sample_png, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    tile_results = [{"path": str(sample_png), "page": 1, "score": 0.9}]
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        answer("What is this?", tile_results, [], api_key=None)


def test_answer_returns_concatenated_stream(sample_png):
    tile_results = [{"path": str(sample_png), "page": 1, "score": 0.9}]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["Hello ", "world."])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    with patch("src.reader.anthropic.Anthropic", return_value=mock_client):
        ans_text, tile_paths = answer(
            "What is on this page?", tile_results, [], api_key="test-key"
        )

    assert ans_text == "Hello world."
    assert tile_paths == [str(sample_png)]


def test_answer_caps_history_at_20_messages(sample_png):
    tile_results = [{"path": str(sample_png), "page": 1, "score": 0.9}]
    # 24 messages = 12 turns — should be capped to last 20
    long_history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
        for i in range(24)
    ]

    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(["ok"])

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = mock_stream

    with patch("src.reader.anthropic.Anthropic", return_value=mock_client):
        answer("q", tile_results, long_history, api_key="test-key")

    call_kwargs = mock_client.messages.stream.call_args[1]
    # 20 history messages + 1 current user message = 21 total
    assert len(call_kwargs["messages"]) == 21
