import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def render_pdf(pdf_path, tiles_dir, dpi=200):
    """
    Render each PDF page to a PNG tile at ~1700px wide (200 DPI).
    Returns list of {"path": str, "page": int} dicts.
    Skips re-rendering if tiles exist and PDF mtime is unchanged.
    """
    pdf_path = Path(pdf_path)
    tiles_dir = Path(tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    meta_path = tiles_dir / "meta.json"
    current_mtime = pdf_path.stat().st_mtime

    if meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text())
            if cached.get("pdf_mtime") == current_mtime:
                logger.info(f"Tile cache hit — {len(cached['tiles'])} tiles")
                return cached["tiles"]
        except Exception:
            pass  # corrupt cache — fall through to re-render

    logger.info(f"Rendering {pdf_path.name} at {dpi} DPI…")
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    doc = fitz.open(str(pdf_path))
    tiles = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        tile_path = tiles_dir / f"page_{page_num + 1:03d}.png"
        pix.save(str(tile_path))
        tiles.append({"path": str(tile_path), "page": page_num + 1})

    doc.close()
    meta_path.write_text(json.dumps({"pdf_mtime": current_mtime, "tiles": tiles}))
    logger.info(f"Rendered {len(tiles)} tiles → {tiles_dir}")
    return tiles
