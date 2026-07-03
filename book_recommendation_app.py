"""
Book Recommendation System — v2
Fixed HTML rendering + complete redesign.

Root causes of raw HTML showing in v1:
  1. All cards joined into one giant st.markdown() blob — any special char
     in a book title/author broke Streamlit's HTML parser for everything after it.
  2. The SVG placeholder embedded in every onerror="" attr inflated the string
     to tens of thousands of chars, making the parser bail out.

Fixes applied:
  • Each card is rendered in its own st.columns() slot as small, complete HTML.
  • html.escape() on all dynamic text (titles, authors).
  • Simple CSS-layer image fallback — no inline SVG, no data URIs.

Run with:  streamlit run app.py
"""

import html as html_lib
import os
import pickle

import numpy as np
import streamlit as st

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Shelfwise · Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

POPULAR_PATH    = "popular.pkl"
PT_PATH         = "pt.pkl"
BOOKS_PATH      = "books.pkl"
SIMILARITY_PATH = "similarity_scores.pkl"

COLS = 5          # cards per row

# --------------------------------------------------------------------------- #
# Global CSS
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── App background ── */
.stApp {
    background: #0C1929;
    color: #EDE8DF;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0A1520;
    border-right: 1px solid rgba(232,180,75,0.15);
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p {
    color: #C8C0AE !important;
    font-size: 0.9rem;
}
section[data-testid="stSidebar"] .stRadio label { color: #EDE8DF !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #112033 0%, #0C1929 100%);
    border: 1px solid rgba(232,180,75,0.18);
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-bottom: 1.6rem;
}
.hero .tag {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #E8B44B;
    margin-bottom: 0.5rem;
    display: block;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F4EDE4;
    margin: 0 0 0.35rem;
}
.hero p { font-size: 0.92rem; color: #9BA8B5; margin: 0; }

/* ── Section title ── */
.sec-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    color: #F4EDE4;
    margin-bottom: 0.15rem;
}
.sec-sub { font-size: 0.83rem; color: #7A8899; margin-bottom: 1.1rem; }

/* ── Book card ── */
.bk-card {
    border: 1px solid rgba(232,180,75,0.13);
    border-radius: 12px;
    padding: 0.7rem;
    background: rgba(255,255,255,0.03);
    height: 100%;
    transition: border-color 0.2s;
}
.bk-card:hover { border-color: rgba(232,180,75,0.4); }

/* Cover image wrapper — book emoji sits behind the <img> via stacking */
.cover-wrap {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    background: #112033;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
/* Fallback emoji — always present, hidden by successful <img> */
.cover-wrap::before {
    content: '📖';
    font-size: 2.8rem;
    opacity: 0.35;
    position: absolute;
}
.cover-wrap img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.bk-rank {
    position: absolute;
    top: 6px; left: 6px;
    background: rgba(12,25,41,0.85);
    color: #E8B44B;
    font-size: 0.72rem;
    font-weight: 700;
    border: 1px solid rgba(232,180,75,0.45);
    border-radius: 6px;
    padding: 1px 7px;
    z-index: 1;
    font-family: 'Inter', sans-serif;
}
.bk-match {
    position: absolute;
    top: 6px; right: 6px;
    background: rgba(139,35,35,0.88);
    color: #F6E4D8;
    font-size: 0.68rem;
    font-weight: 600;
    border-radius: 999px;
    padding: 2px 8px;
    z-index: 1;
    font-family: 'Inter', sans-serif;
}
.bk-title {
    font-family: 'Playfair Display', serif;
    font-size: 0.87rem;
    font-weight: 700;
    color: #F0EBE0;
    line-height: 1.3;
    margin-bottom: 0.2rem;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.bk-author {
    font-size: 0.74rem;
    color: #7A8899;
    margin-bottom: 0.3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bk-meta { font-size: 0.73rem; color: #E8B44B; font-weight: 600; }

/* ── Queried book spotlight ── */
.spotlight {
    background: linear-gradient(120deg, #112033, #0C1929);
    border: 1px solid rgba(232,180,75,0.25);
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.spotlight .sp-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; color: #E8B44B; }
.spotlight .sp-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    color: #F4EDE4;
    margin: 0.1rem 0 0;
}

/* ── CTA button ── */
.stButton > button {
    background: #E8B44B;
    color: #0C1929;
    font-weight: 700;
    border: none;
    border-radius: 10px;
    width: 100%;
    font-family: 'Inter', sans-serif;
}
.stButton > button:hover { background: #F0C76A; }

/* ── Column gutter ── */
[data-testid="column"] { padding: 0 0.3rem !important; }

.footnote { font-size: 0.76rem; color: #4A5568; margin-top: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- #
# Load artifacts
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_artifacts():
    paths = [POPULAR_PATH, PT_PATH, BOOKS_PATH, SIMILARITY_PATH]
    if not all(os.path.exists(p) for p in paths):
        return None
    with open(POPULAR_PATH, "rb") as f:   popular_df       = pickle.load(f)
    with open(PT_PATH, "rb") as f:        pt               = pickle.load(f)
    with open(BOOKS_PATH, "rb") as f:     books_df         = pickle.load(f)
    with open(SIMILARITY_PATH, "rb") as f: similarity_scores = pickle.load(f)
    return popular_df, pt, books_df, similarity_scores


artifacts = load_artifacts()

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="hero">
    <span class="tag">Popularity ranking · Item-based collaborative filtering</span>
    <h1>📚 Shelfwise</h1>
    <p>Browse the most acclaimed titles, or pick a book you love and discover what readers
    with similar taste rated highly too.</p>
</div>
""", unsafe_allow_html=True)

if artifacts is None:
    st.warning(
        "⚠️ Artifacts not found. Run `data_preprocessing_and_model_training.ipynb` first "
        "to generate `popular.pkl`, `pt.pkl`, `books.pkl`, and `similarity_scores.pkl`."
    )
    st.stop()

popular_df, pt, books_df, similarity_scores = artifacts

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🗂 Browse Mode")
    mode = st.radio(
        "mode", ["🏆 Top Rated", "🔍 Discover Similar"],
        label_visibility="collapsed"
    )
    st.markdown("---")

    if mode == "🏆 Top Rated":
        top_n = st.slider("Books to show", 5, min(50, len(popular_df)), min(15, len(popular_df)), 5)
        selected_book = None
        find_clicked  = False
    else:
        book_titles   = sorted(pt.index.tolist())
        selected_book = st.selectbox("Pick a book you enjoyed", book_titles)
        top_n         = st.slider("Recommendations", 2, 10, 4)
        find_clicked  = st.button("✨ Find Similar Books")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_img(url) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    return url.strip()


def card_html(
    title: str, author: str, img_url: str,
    meta: str, badge: str = "", badge_class: str = "bk-rank"
) -> str:
    """
    Return a self-contained, safely-escaped card HTML string.
    Each card is rendered individually — never joined into a blob.
    """
    t  = html_lib.escape(title)
    a  = html_lib.escape(author)
    img_tag = (
        f'<img src="{html_lib.escape(img_url)}" '
        f'onerror="this.style.display=\'none\'">'
        if img_url else ""
    )
    badge_html = f'<span class="{badge_class}">{html_lib.escape(badge)}</span>' if badge else ""
    return f"""
<div class="bk-card">
    <div class="cover-wrap">
        {badge_html}
        {img_tag}
    </div>
    <div class="bk-title">{t}</div>
    <div class="bk-author">{a}</div>
    <div class="bk-meta">{meta}</div>
</div>"""


def render_grid(items):
    """
    Render a list of dicts (keys: title, author, img, meta, badge, badge_class)
    in a COLS-wide column grid. Each card is a separate st.markdown() call —
    no joining, no giant HTML blob.
    """
    for row_start in range(0, len(items), COLS):
        batch = items[row_start: row_start + COLS]
        cols  = st.columns(COLS)
        for col, item in zip(cols, batch):
            with col:
                st.markdown(
                    card_html(
                        title      = item["title"],
                        author     = item["author"],
                        img_url    = item["img"],
                        meta       = item["meta"],
                        badge      = item.get("badge", ""),
                        badge_class= item.get("badge_class", "bk-rank"),
                    ),
                    unsafe_allow_html=True,
                )


def get_recommendations(book_name: str, n: int):
    matches = np.where(pt.index == book_name)[0]
    if len(matches) == 0:
        return []
    idx     = matches[0]
    similar = sorted(
        enumerate(similarity_scores[idx]), key=lambda x: x[1], reverse=True
    )[1: n + 1]
    results = []
    for i, score in similar:
        title   = pt.index[i]
        tmp     = books_df[books_df["Book-Title"] == title].drop_duplicates("Book-Title")
        if tmp.empty:
            continue
        row = tmp.iloc[0]
        results.append({
            "title":       str(row.get("Book-Title", title)),
            "author":      str(row.get("Book-Author", "Unknown")),
            "img":         safe_img(row.get("Image-URL-M", "")),
            "meta":        f"{score * 100:.0f}% match",
            "badge":       f"{score * 100:.0f}% match",
            "badge_class": "bk-match",
        })
    return results


# --------------------------------------------------------------------------- #
# Top Rated
# --------------------------------------------------------------------------- #
if mode == "🏆 Top Rated":
    st.markdown('<div class="sec-title">Most Acclaimed Books</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-sub">Ranked by average rating · requires ≥ 250 reader ratings</div>',
        unsafe_allow_html=True,
    )
    subset = popular_df.head(top_n).reset_index(drop=True)
    items  = [
        {
            "title":  str(row.get("Book-Title", "")),
            "author": str(row.get("Book-Author", "")),
            "img":    safe_img(row.get("Image-URL-M", "")),
            "meta":   f"⭐ {row['avg_rating']:.2f} · {int(row['num_ratings']):,} ratings",
            "badge":  f"#{rank + 1}",
            "badge_class": "bk-rank",
        }
        for rank, (_, row) in enumerate(subset.iterrows())
    ]
    render_grid(items)

# --------------------------------------------------------------------------- #
# Discover Similar
# --------------------------------------------------------------------------- #
else:
    if selected_book:
        st.markdown(f"""
        <div class="spotlight">
            <div>
                <div class="sp-label">Finding books similar to</div>
                <div class="sp-title">{html_lib.escape(selected_book)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if find_clicked and selected_book:
        st.session_state["recs"]     = get_recommendations(selected_book, top_n)
        st.session_state["recs_for"] = selected_book

    recs = st.session_state.get("recs")
    if recs and st.session_state.get("recs_for") == selected_book:
        st.markdown('<div class="sec-title">Readers Also Loved</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sec-sub">Books most similar to <i>{html_lib.escape(selected_book)}</i>, '
            'based on shared reader ratings</div>',
            unsafe_allow_html=True,
        )
        render_grid(recs)
    else:
        st.info("👈 Pick a book in the sidebar and click **Find Similar Books**.")

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="footnote">Recommendations are generated from historical reader ratings '
    "(Book-Crossing dataset). Popularity ranking requires ≥ 250 ratings per title; "
    "similarity matching is computed only among frequently-rated books and active readers, "
    "so very niche titles may not appear.</div>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ How it works"):
    st.markdown("""
- **Top Rated** — groups all ratings by title, filters to ≥ 250 ratings, sorts by average.
- **Discover Similar** — builds a Book × User rating matrix (active readers only), computes
  **cosine similarity** between every book pair. The match % is that score.
    """)