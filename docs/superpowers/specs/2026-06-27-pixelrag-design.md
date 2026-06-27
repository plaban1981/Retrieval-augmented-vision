# PixelRAG Application Design

**Date:** 2026-06-27  
**Status:** Approved  
**Document:** `pixelrag-paper.pdf`

---

## 1. Overview

A working PixelRAG implementation that treats documents as pixel-native image tiles rather than parsed text. The system ingests a complex PDF (the PixelRAG paper itself — dense with figures, architecture diagrams, and tables) and enables conversational Q&A over it using visual retrieval.

**Deliverables:**
- `app.py` — Streamlit chat app with visual tile citations
- `notebook.ipynb` — Step-by-step Jupyter walkthrough of the full pipeline
- `src/` — Four focused pipeline modules (render, embed, index, reader)

---

## 2. Architecture

Five-stage pipeline split into one-time ingestion and per-query retrieval:

### Ingestion (one-time, cached)
1. **Render** — PyMuPDF converts each PDF page to a 1600px-wide PNG tile saved to `data/tiles/`
2. **Embed** — Qwen3-VL-Embedding-2B + LoRA adapters encode each tile to a 1536-dim float32 vector; cached to `data/embeddings/embeddings.npy` + `metadata.json`
3. **Index** — FAISS flat L2 index built from vectors; saved to `data/index/index.faiss`

### Query (per question)
4. **Retrieve** — User query embedded with the same model; FAISS returns top-3 most similar tiles
5. **Read** — Retrieved tile images + question + chat history sent to Claude Sonnet 4.6 vision API; answer streamed back with tile citations

### Device selection
Auto-detected at startup:
- CUDA GPU available → Qwen3-VL-Embedding-2B runs on GPU (fast, ~seconds per tile)
- CPU only → same model on CPU (slow, ~minutes per tile, warned in UI)

---

## 3. File Layout

```
pixel-rag/
├── src/
│   ├── render.py        # PDF → PNG tiles (PyMuPDF)
│   ├── embed.py         # Tile + query embedding (Qwen3-VL + LoRA, auto GPU/CPU)
│   ├── index.py         # FAISS build / load / search
│   └── reader.py        # Claude Vision answer generation
├── app.py               # Streamlit chat app
├── notebook.ipynb       # Jupyter walkthrough (7 cells)
├── data/
│   ├── tiles/           # Rendered PNG files (page_001.png …)
│   ├── embeddings/      # embeddings.npy + metadata.json
│   └── index/           # index.faiss
├── pixelrag-paper.pdf
└── requirements.txt
```

---

## 4. Components

### `src/render.py`
- `render_pdf(pdf_path, tiles_dir, dpi=200)` — renders all pages to PNGs (~1700px wide); returns list of tile paths + metadata
- Skips rendering if tiles already exist and PDF mtime hasn't changed

### `src/embed.py`
- `load_model(device)` — loads Qwen3-VL-Embedding-2B + LoRA adapters via PEFT; falls back to base model if adapter download fails
- `embed_tiles(tile_paths, model, device, cache_dir)` — batch embeds tiles; loads from cache if present
- `embed_query(text, model, device)` — embeds a single text query at query time

### `src/index.py`
- `build_index(embeddings, index_dir)` — builds FAISS flat L2 index and saves to disk
- `load_index(index_dir)` — loads saved index + metadata
- `search(query_vec, index, metadata, k=3)` — returns top-k tile paths + similarity scores

### `src/reader.py`
- `answer(question, tile_paths, chat_history, api_key)` — encodes tiles as base64, sends to Claude Sonnet 4.6 with last 10 chat turns (to keep token use bounded); streams response
- Returns `(answer_text, tile_paths_used)`

---

## 5. Streamlit App (`app.py`)

**Layout:** Two-column — narrow sidebar left, main chat panel right.

**Sidebar:**
- Document name + tile count
- Device badge (CUDA / CPU with warning if CPU)
- Index status (built/building/not built)
- Top-K slider (default 3)
- Re-index button, Clear Chat button

**Main panel:**
- Multi-turn chat history (user bubbles blue, assistant bubbles white)
- Each assistant message has an expandable `📎 N source tiles` section showing tile thumbnails + similarity scores
- Text input + Send button at bottom
- Answer streams token-by-token

**Startup sequence:**
1. Detect device
2. Load model + LoRA (progress spinner)
3. Check cache → load index or trigger build
4. App ready

---

## 6. Jupyter Notebook (`notebook.ipynb`)

7 cells, each producing visible output:

| Cell | Title | Inline Output |
|------|-------|---------------|
| 1 | Setup & Imports | Device info printout |
| 2 | Render PDF → Tiles | Thumbnail grid of all pages |
| 3 | Load Embedding Model + LoRA | Model summary + device placement |
| 4 | Embed Tiles → Vectors | Shape `(N, 1536)` + t-SNE scatter coloured by page |
| 5 | Build FAISS Index | Index stats printout |
| 6 | Query Demo | 3 example queries: retrieved tiles displayed inline, then Claude's answer |
| 7 | Interactive Query Cell | `query = "..."` — re-run to explore freely |

---

## 7. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| No GPU / CUDA OOM | Auto-fallback to CPU; sidebar shows `⚠️ CPU mode — embedding slow` |
| Missing `ANTHROPIC_API_KEY` | Indexing works; answering shows `❌ Set ANTHROPIC_API_KEY env var` inline |
| LoRA adapter download fails | Falls back to base Qwen3-VL-Embedding-2B; warning logged |
| Corrupt cache files | Exception caught → stale files deleted → rebuild triggered automatically |

---

## 8. Caching Strategy

Three-layer cache; each layer checked before re-doing work:

1. **Tiles** — skip rendering if `data/tiles/` populated and PDF mtime unchanged
2. **Embeddings** — skip embedding if `data/embeddings/embeddings.npy` exists
3. **FAISS index** — skip build if `data/index/index.faiss` exists

**Cold start (first run):** 3–10 min depending on device.  
**Warm start (all cached):** < 30 seconds.

---

## 9. Dependencies (`requirements.txt`)

```
pymupdf           # PDF rendering
torch             # GPU/CPU backend
transformers      # Qwen3-VL model
peft              # LoRA adapter loading
faiss-cpu         # FAISS index; install faiss-gpu instead if CUDA is available for faster search
anthropic         # Claude API
streamlit         # Web app
pillow            # Image handling
numpy             # Embedding arrays
scikit-learn      # t-SNE for notebook visualisation
tqdm              # Progress bars
```

---

## 10. Success Criteria

- Cold start renders all PDF pages, embeds, indexes without error on both GPU and CPU
- Warm start loads in under 30 seconds
- A factual question about the paper (e.g. "What is the FAISS index size?") returns the correct answer with the right page tile cited
- Streamlit chat preserves multi-turn history across questions
- Notebook runs top-to-bottom without manual intervention and all cells produce visible output
