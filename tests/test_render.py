import time
from pathlib import Path
import pytest
from src.render import render_pdf


def test_render_pdf_creates_png_tiles(sample_pdf, tmp_path):
    tiles = render_pdf(sample_pdf, tmp_path / "tiles")
    assert len(tiles) == 3
    for tile in tiles:
        assert Path(tile["path"]).exists()
        assert tile["path"].endswith(".png")


def test_render_pdf_page_numbers_are_correct(sample_pdf, tmp_path):
    tiles = render_pdf(sample_pdf, tmp_path / "tiles")
    assert [t["page"] for t in tiles] == [1, 2, 3]


def test_render_pdf_cache_skips_rerender(sample_pdf, tmp_path):
    tiles_dir = tmp_path / "tiles"
    tiles1 = render_pdf(sample_pdf, tiles_dir)
    first_tile = Path(tiles1[0]["path"])
    mtime_before = first_tile.stat().st_mtime

    tiles2 = render_pdf(sample_pdf, tiles_dir)

    assert first_tile.stat().st_mtime == mtime_before  # NOT re-written
    assert [t["path"] for t in tiles1] == [t["path"] for t in tiles2]


def test_render_pdf_cache_rebuilds_when_pdf_changes(sample_pdf, tmp_path):
    import fitz
    tiles_dir = tmp_path / "tiles"
    tiles1 = render_pdf(sample_pdf, tiles_dir)

    time.sleep(0.05)  # ensure mtime changes on fast filesystems
    doc = fitz.open(str(sample_pdf))
    doc.new_page()
    # Save to temp file then replace original to update mtime
    temp_path = tmp_path / "temp.pdf"
    doc.save(str(temp_path), garbage=4, deflate=True)
    doc.close()
    # Replace original with modified version
    import shutil
    shutil.move(str(temp_path), str(sample_pdf))

    tiles2 = render_pdf(sample_pdf, tiles_dir)
    assert len(tiles2) == 4  # 3 original + 1 new
