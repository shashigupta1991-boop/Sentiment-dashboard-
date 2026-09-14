# Social Media Sentiment Dashboard

An interactive dashboard that analyzes consumer feedback, product
reviews, or social media posts and classifies each one as
**Positive**, **Negative**, or **Neutral** using NLTK's **VADER**
sentiment analyzer — built specifically for short, informal text
like tweets and reviews.

## Project structure

| File | Purpose |
|---|---|
| `preprocess.py` | Cleans raw text (strips URLs, @mentions, hashtag symbols, junk characters) |
| `sentiment_analyzer.py` | Core NLP: runs VADER, scores text, classifies sentiment |
| `dashboard.py` | Streamlit web dashboard (charts, word clouds, live testing, CSV export) |
| `sample_reviews.csv` | Sample dataset of 20 product reviews/social posts to try immediately |
| `requirements.txt` | All dependencies |

## How the sentiment engine works

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
lexicon + rule-based model — no training data or GPU required. It
scores text on 4 dimensions (negative, neutral, positive, and an
overall **compound** score from -1 to +1) and is aware of things
like:
- Capitalization and punctuation emphasis (`"GREAT!!!"` scores
  higher than `"great"`)
- Negation (`"not good"` correctly flips sentiment)
- Emoticons and common slang

Classification uses the standard thresholds:
- `compound >= 0.05` → **Positive**
- `compound <= -0.05` → **Negative**
- otherwise → **Neutral**

## Setup (local)

```bash
pip install -r requirements.txt
```

The first run automatically downloads the small VADER lexicon file
via NLTK (no manual step needed).

## Run locally

```bash
streamlit run dashboard.py
```

This opens the dashboard in your browser at `http://localhost:8501`.

## Run it online (no local install needed)

This project is a great fit for **Streamlit Community Cloud**
(free), since it has no webcam/hardware dependency:

1. Push this folder to a public GitHub repository.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app," select your repo, and set the main file to
   `dashboard.py`.
4. Click Deploy. You'll get a public URL you can share with anyone.

## Using the dashboard

1. Upload your own CSV (any column of text works — tweets, reviews,
   comments) or use the bundled sample data.
2. Pick which column holds the text, and optionally a date column
   for the trend chart.
3. View:
   - Overall sentiment breakdown (metrics + pie chart)
   - Sentiment trend over time (if dates are provided)
   - Word clouds for positive vs. negative posts
   - A filterable, searchable results table
   - A "try it yourself" box to test any sentence live
4. Download the full analyzed results as CSV.

## Extending this project

- **Live social feeds**: to pull real-time data instead of a CSV,
  add the `tweepy` library and X/Twitter API v2 credentials (or
  Reddit's `praw`), fetch recent posts for a keyword/hashtag, and
  feed them into `analyze_dataframe()` — the rest of the dashboard
  works unchanged.
- **More granular emotions**: swap/add a transformer-based model
  (e.g. `cardiffnlp/twitter-roberta-base-sentiment` via Hugging Face
  `transformers`) for higher accuracy on sarcasm/nuance, at the cost
  of needing more compute.
- **Database storage**: log each analysis batch into SQLite/Postgres
  to track sentiment trends across multiple uploads over time.
