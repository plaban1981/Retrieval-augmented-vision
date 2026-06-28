import base64
import logging
import os
from pathlib import Path

import anthropic

MODEL = "claude-sonnet-4-6"
logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a precise document Q&A assistant. "
    "The user has retrieved the most visually relevant page images from a document. "
    "Answer the question based solely on what you can see in these page images. "
    "Be concise and cite specific page numbers when relevant."
)


def _encode_image(image_path):
    """Base64-encode a PNG file for the Claude API."""
    return base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")


def answer(question, tile_results, chat_history, api_key=None):
    """
    Send question + retrieved tile images to Claude Sonnet 4.6 Vision.

    Args:
        question: User question string.
        tile_results: List of {"path": str, "page": int, "score": float}.
        chat_history: List of {"role": str, "content": str} (text only, no images).
        api_key: Anthropic key; falls back to ANTHROPIC_API_KEY env var.

    Returns:
        (answer_text: str, tile_paths_used: list[str])

    Raises:
        ValueError: If no API key is available.
    """
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    # Cap history at last 20 messages (10 turns) to bound token use
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in chat_history[-20:]
    ]

    # Current user message: tile images + page labels + question
    content = []
    for tile in tile_results:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": _encode_image(tile["path"]),
            },
        })
        content.append({
            "type": "text",
            "text": f"[Page {tile['page']}, similarity score {tile['score']:.2f}]",
        })
    content.append({"type": "text", "text": question})

    messages = history + [{"role": "user", "content": content}]

    full_text = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM,
        messages=messages,
    ) as stream:
        for chunk in stream.text_stream:
            full_text.append(chunk)

    answer_text = "".join(full_text)
    tile_paths_used = [t["path"] for t in tile_results]
    logger.info(f"Answer: {len(answer_text)} chars from {len(tile_results)} tiles")
    return answer_text, tile_paths_used
