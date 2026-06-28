import logging
import os
import shutil
from pathlib import Path

import streamlit as st

from src.embed import embed_query, embed_tiles, get_device, load_model
from src.index import build_index, load_index, search
from src.reader import answer
from src.render import render_pdf

logging.basicConfig(level=logging.INFO)

PDF_PATH = "pixelrag-paper.pdf"
TILES_DIR = "data/tiles"
EMB_DIR = "data/embeddings"
INDEX_DIR = "data/index"

st.set_page_config(page_title="PixelRAG", page_icon="🔬", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history", []),
    ("model", None),
    ("processor", None),
    ("device", None),
    ("lora_loaded", False),
    ("faiss_index", None),
    ("index_metadata", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 PixelRAG")
    st.divider()
    st.markdown("**Document**")
    st.caption(PDF_PATH)

    device_badge = st.empty()
    lora_badge = st.empty()
    index_status = st.empty()

    st.markdown("**Anthropic API Key**")
    api_key_input = st.text_input(
        "Anthropic API Key",
        value=st.session_state.get("api_key", os.environ.get("ANTHROPIC_API_KEY", "")),
        type="password",
        placeholder="sk-ant-...",
        label_visibility="collapsed",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input
    st.divider()

    top_k = st.slider("Top-K tiles", min_value=1, max_value=5, value=3)
    st.divider()

    col1, col2 = st.columns(2)
    do_reindex = col1.button("🔄 Re-index")
    if col2.button("🗑️ Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

# ── Load model (once per session) ─────────────────────────────────────────────
if st.session_state.model is None:
    with st.spinner("Loading Qwen3-VL-Embedding-2B…"):
        model, processor, device, lora_loaded = load_model()
        st.session_state.model = model
        st.session_state.processor = processor
        st.session_state.device = device
        st.session_state.lora_loaded = lora_loaded

device = st.session_state.device
if device == "cuda":
    device_badge.success("✅ CUDA GPU")
else:
    device_badge.warning("⚠️ CPU mode — embedding will be slow")

if st.session_state.lora_loaded:
    lora_badge.success("✅ LoRA adapters loaded")
else:
    lora_badge.warning("⚠️ Base model only (no LoRA)")

# ── Build or load FAISS index ─────────────────────────────────────────────────
def _build_index(force=False):
    if not force:
        idx, meta = load_index(INDEX_DIR)
        if idx is not None:
            return idx, meta, 0

    tiles = render_pdf(PDF_PATH, TILES_DIR)
    embeddings, meta = embed_tiles(
        tiles,
        st.session_state.model,
        st.session_state.processor,
        st.session_state.device,
        EMB_DIR,
    )
    idx, meta = build_index(embeddings, meta, INDEX_DIR)
    return idx, meta, len(tiles)


if st.session_state.faiss_index is None or do_reindex:
    if do_reindex:
        for d in [TILES_DIR, EMB_DIR, INDEX_DIR]:
            if Path(d).exists():
                shutil.rmtree(d)

    with st.spinner("Building index… (may take several minutes on first run)"):
        idx, meta, tile_count = _build_index(force=do_reindex)
        st.session_state.faiss_index = idx
        st.session_state.index_metadata = meta

n_tiles = len(st.session_state.index_metadata) if st.session_state.index_metadata else 0
index_status.success(f"✅ Ready — {n_tiles} tiles")

# ── Main chat panel ───────────────────────────────────────────────────────────
st.title("PixelRAG Chat")
st.caption(
    f"Pixel-native Q&A over **{PDF_PATH}** · {n_tiles} page tiles "
    f"· top-{top_k} retrieval · Claude Sonnet 4.6"
)

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tiles"):
            with st.expander(f"📎 {len(msg['tiles'])} source tiles"):
                cols = st.columns(min(len(msg["tiles"]), 5))
                for col, tile in zip(cols, msg["tiles"]):
                    col.image(
                        tile["path"],
                        caption=f"Page {tile['page']} · {tile['score']:.2f}",
                        use_container_width=True,
                    )

question = st.chat_input("Ask a question about the paper…")

if question:
    api_key = st.session_state.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("❌ Enter your Anthropic API key in the sidebar.")
        st.stop()

    with st.chat_message("user"):
        st.markdown(question)

    query_vec = embed_query(
        question,
        st.session_state.model,
        st.session_state.processor,
        st.session_state.device,
    )
    results = search(
        query_vec,
        st.session_state.faiss_index,
        st.session_state.index_metadata,
        k=top_k,
    )

    api_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history
    ]

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            ans_text, _ = answer(question, results, api_history, api_key)
        st.markdown(ans_text)
        with st.expander(f"📎 {len(results)} source tiles"):
            cols = st.columns(min(len(results), 5))
            for col, tile in zip(cols, results):
                col.image(
                    tile["path"],
                    caption=f"Page {tile['page']} · {tile['score']:.2f}",
                    use_container_width=True,
                )

    st.session_state.chat_history.append({"role": "user", "content": question})
    st.session_state.chat_history.append(
        {"role": "assistant", "content": ans_text, "tiles": results}
    )
