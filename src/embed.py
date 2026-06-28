import json
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
LORA_ID = "Chrisyichuan/wiki-screenshot-embedding-lora"

logger = logging.getLogger(__name__)


def get_device():
    """Return 'cuda' if a usable GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            logger.warning("CUDA available but unusable — falling back to CPU")
    return "cpu"


def load_model(device=None):
    """
    Load Qwen3-VL-Embedding-2B + LoRA adapters.
    Falls back to base model if LoRA download fails.

    Returns:
        (model, processor, device: str, lora_loaded: bool)
    """
    from peft import PeftModel
    from transformers import AutoModel, AutoProcessor

    if device is None:
        device = get_device()

    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Loading {MODEL_ID} on {device} ({dtype})")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=dtype)

    lora_loaded = False
    try:
        model = PeftModel.from_pretrained(base, LORA_ID)
        lora_loaded = True
        logger.info("LoRA adapters loaded")
    except Exception as exc:
        logger.warning(f"LoRA load failed ({exc}) — using base model")
        model = base

    model = model.to(device).eval()
    return model, processor, device, lora_loaded


def _encode(model, processor, device, *, image=None, text=None):
    """
    Embed one PIL image or one text string.
    Exactly one of image/text must be provided.
    Returns an L2-normalised float32 numpy vector.
    """
    if image is not None:
        inputs = processor(images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(
            text=text, return_tensors="pt",
            padding=True, truncation=True, max_length=512,
        ).to(device)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        vec = out.pooler_output.squeeze()
    else:
        vec = out.last_hidden_state.mean(dim=1).squeeze()

    vec = vec.float().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_tiles(tiles, model, processor, device, cache_dir):
    """
    Embed all tiles. Loads from cache if available; rebuilds on corrupt cache.

    Args:
        tiles: List of {"path": str, "page": int} dicts from render_pdf().
        model, processor, device: From load_model().
        cache_dir: Directory to store embeddings.npy + metadata.json.

    Returns:
        (embeddings: np.ndarray shape (N, D), metadata: list)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "embeddings.npy"
    meta_path = cache_dir / "metadata.json"

    if emb_path.exists() and meta_path.exists():
        try:
            embeddings = np.load(str(emb_path))
            metadata = json.loads(meta_path.read_text())
            logger.info(f"Embedding cache hit: {embeddings.shape}")
            return embeddings, metadata
        except Exception:
            logger.warning("Embedding cache corrupt — re-embedding")
            emb_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    vecs, metadata = [], []
    for tile in tqdm(tiles, desc="Embedding tiles"):
        img = Image.open(tile["path"]).convert("RGB")
        vec = _encode(model, processor, device, image=img)
        vecs.append(vec)
        metadata.append(tile)

    embeddings = np.array(vecs, dtype=np.float32)
    np.save(str(emb_path), embeddings)
    meta_path.write_text(json.dumps(metadata))
    logger.info(f"Embeddings saved: {embeddings.shape}")
    return embeddings, metadata


def embed_query(text, model, processor, device):
    """Embed a text query. Returns an L2-normalised float32 numpy vector."""
    return _encode(model, processor, device, text=text)
