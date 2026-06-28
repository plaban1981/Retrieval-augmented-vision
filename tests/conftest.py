import pytest


@pytest.fixture
def sample_pdf(tmp_path):
    import fitz
    doc = fitz.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text((50, 100), f"Test page {i + 1} with some content")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def sample_png(tmp_path):
    from PIL import Image
    img = Image.new("RGB", (64, 64), color=(200, 200, 200))
    path = tmp_path / "tile.png"
    img.save(path, format="PNG")
    return path
