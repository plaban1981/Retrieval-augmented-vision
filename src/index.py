import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def build_index(embeddings, metadata, index_dir):
    """
    Build FAISS flat L2 index and save to disk.
    Returns (index, metadata).
    """
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "metadata.json").write_text(json.dumps(metadata))
    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index, metadata


def load_index(index_dir):
    """
    Load FAISS index + metadata from disk.
    Returns (index, metadata) or (None, None) if files not present.
    """
    index_dir = Path(index_dir)
    index_path = index_dir / "index.faiss"
    meta_path = index_dir / "metadata.json"

    if not index_path.exists() or not meta_path.exists():
        return None, None

    try:
        index = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text())
        logger.info(f"Index loaded: {index.ntotal} vectors")
        return index, metadata
    except Exception as e:
        logger.warning(f"Index load failed: {e}")
        return None, None


def search(query_vec, index, metadata, k=3):
    """
    Return top-k tiles closest to query_vec.
    Each result is the metadata dict plus a "score" key in [0, 1].
    Higher score = more similar.
    """
    query = np.array([query_vec], dtype=np.float32)
    distances, indices = index.search(query, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        tile = metadata[idx].copy()
        tile["score"] = float(1.0 / (1.0 + dist))
        results.append(tile)
    return results
