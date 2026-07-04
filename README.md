<div align="center">

# 📚 Book Recommendation System

### Discover your next read with popularity-based and item-based collaborative filtering

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)

[Overview](#-overview) •
[Dataset](#-dataset) •
[Installation](#-installation) •
[Usage](#-usage) •
[Pipeline](#-recommendation-pipeline) •
[App](#-streamlit-app) •
[Future Work](#-future-improvements)

</div>

---

## 📌 Overview

With millions of books in print, the hardest part of reading isn't finishing a book — it's choosing the next one. This project builds two complementary recommendation engines on top of real reader rating data:

| Technique | What it answers |
|---|---|
| 🏆 **Popularity-based** | "What are the most beloved books, by readers who actually rated a lot of them?" |
| 🔍 **Item-based collaborative filtering** | "Readers who liked *this* book also tended to like…?" (via cosine similarity over a user–book rating matrix) |

The repo contains three things:

| Component | What it does |
|---|---|
| 🧪 **Training notebook/script** | Loads the ratings data, builds the popularity table and the similarity matrix, and exports both as reusable artifacts |
| 🌐 **Streamlit app** | A browsable bookshelf UI — top-rated books, plus a "find similar books" search |
| 📦 **Reusable artifacts** | Pickled DataFrames and a similarity matrix so the app never has to recompute from raw data |

---

## 📊 Dataset

The project uses the classic **Book-Crossing dataset**, split across three files:

| File | Key columns | Description |
|---|---|---|
| `books.csv` | `ISBN`, `Book-Title`, `Book-Author`, `Year-Of-Publication`, `Publisher`, `Image-URL-S/M/L` | Metadata for ~270K books |
| `users.csv` | `User-ID`, `Location`, `Age` | ~280K registered readers |
| `ratings.csv` | `User-ID`, `ISBN`, `Book-Rating` | ~1.1M explicit (1–10) and implicit (0) ratings |

`ratings.csv` is joined to `books.csv` on `ISBN` to attach titles, authors, and cover art to every rating.

---

## 🗂 Project Structure

```
book-recommendation-system/
│
├── books.csv                           # Book metadata (not included — see Dataset section)
├── users.csv                           # User metadata
├── ratings.csv                         # User–book ratings
├── data_preprocessing_and_model_training.ipynb   # EDA, popularity table, similarity matrix
│
├── app.py                              # Streamlit web app
├── requirements.txt                    # Python dependencies
│
├── popular.pkl                         # Top 50 books by avg rating (≥250 ratings) — generated
├── pt.pkl                              # User–book pivot table — generated
├── books.pkl                           # Deduplicated book metadata — generated
├── similarity_scores.pkl               # Cosine similarity matrix over pt — generated
│
└── README.md                           # You are here
```

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/zakir-maswani/Book-Recommendation-System.git
cd book-recommendation-system
```

**2. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add the dataset**

Place `books.csv`, `users.csv`, and `ratings.csv` in the project root. The Book-Crossing dataset is available from sources such as [Kaggle](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

---

## 🚀 Usage

### Build the recommendation artifacts

Run the notebook (or the equivalent `.py` script) top to bottom:

```bash
jupyter notebook data_preprocessing_and_model_training.ipynb
```

This will:
1. Load `books.csv`, `users.csv`, and `ratings.csv`, and check for nulls
2. Merge ratings with book metadata on `ISBN`
3. Build the **popularity table** — books with ≥250 ratings, ranked by average rating, top 50
4. Filter to "well-read" users (>200 ratings each) and "famous" books (≥50 ratings among those users) to keep the similarity matrix dense and meaningful
5. Pivot into a Book-Title × User-ID rating matrix and compute **cosine similarity** between books
6. Export four artifacts to the project root: `popular.pkl`, `pt.pkl`, `books.pkl`, `similarity_scores.pkl`

### Launch the app

Once the artifacts above exist:

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`).

---

## 🔬 Recommendation Pipeline

```
ratings.csv ──merge(ISBN)──> books.csv
        │
        ▼
┌─────────────────────────────┐      ┌──────────────────────────────────┐
│   Popularity-Based Engine   │      │  Item-Based Collaborative Engine  │
│                             │      │                                   │
│  group by Book-Title        │      │  keep users with > 200 ratings    │
│  ├─ count  → num_ratings    │      │  keep books with ≥ 50 ratings     │
│  └─ mean    → avg_rating    │      │       among those users           │
│                             │      │              │                    │
│  keep books with            │      │              ▼                    │
│  num_ratings ≥ 250          │      │  pivot: Book-Title × User-ID      │
│       │                     │      │  (missing ratings → 0)            │
│       ▼                     │      │              │                    │
│  sort by avg_rating, top 50 │      │              ▼                    │
│       │                     │      │  cosine_similarity(pivot table)   │
│       ▼                     │      │              │                    │
│   popular.pkl               │      │              ▼                    │
└─────────────────────────────┘      │  pt.pkl + similarity_scores.pkl   │
                                      └──────────────────────────────────┘
```

**Why these choices?**
- **The ≥250 / >200 / ≥50 thresholds** filter out the long tail of one-off ratings, which would otherwise make both the popularity ranking and the similarity matrix noisy and unreliable.
- **Cosine similarity** on a Book-Title × User-ID matrix measures how similarly two books are rated *by the same readers* — it's a standard, lightweight choice for item-based collaborative filtering and needs no training/hyperparameters.
- **Filling missing ratings with 0** is a simplification: it treats "didn't rate" the same as "rated zero." This is a known limitation worth revisiting (see [Future Improvements](#-future-improvements)).

---

## 🌐 Streamlit App

The app (`app.py`) provides a two-mode bookshelf interface:

- **🏆 Top Rated** — browse the most acclaimed books (by readers who rated heavily), shown as a cover-art grid with average rating and review count, with a slider to control how many books are displayed.
- **🔍 Discover Similar** — pick any book from the catalog and instantly see the books most similar to it (by cosine similarity over shared readers), each tagged with a match-percentage badge.
- **Custom-styled book cards** — a library-inspired dark green & brass color palette with serif display type for titles, rendered with custom HTML/CSS rather than plain Streamlit widgets.
- **Graceful fallbacks** — if the `.pkl` artifacts aren't found yet, or a book has no cover image, the app degrades cleanly instead of crashing.

---

## 🛠 Tech Stack

- **Language:** Python 3.9+
- **Data handling:** pandas, NumPy
- **Similarity computation:** scikit-learn (`cosine_similarity`)
- **Serialization:** pickle
- **Web app:** Streamlit + custom HTML/CSS

---

## 🗺 Future Improvements

- Replace zero-filled missing ratings with a more principled approach (e.g., mean-centering, or matrix factorization that natively handles sparsity)
- Add matrix factorization (SVD / ALS) as a third recommendation engine and compare against cosine similarity
- Build a hybrid recommender that blends collaborative filtering with content-based signals (genre, author, description)
- Add offline evaluation (precision@k, recall@k) using a held-out split of `ratings.csv`
- Cache cover images locally to avoid relying on external image URLs at runtime
- Containerize the app with Docker for one-command deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request for bug fixes, new features, or documentation improvements.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

## 🙏 Acknowledgments

- Dataset: the **Book-Crossing dataset**, originally collected by Cai-Nicolas Ziegler, commonly redistributed via Kaggle.
- Built with [scikit-learn](https://scikit-learn.org/) and [Streamlit](https://streamlit.io/).

<div align="center">

Made with ☕ and `cosine_similarity`

</div>
