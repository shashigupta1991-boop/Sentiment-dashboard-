"""
sentiment_analyzer.py
----------------------
Core NLP logic for the Social Media Sentiment Dashboard.

Uses NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner),
a lexicon- and rule-based sentiment analyzer purpose-built for short,
informal text like tweets, reviews, and comments (handles slang,
emoticons, punctuation emphasis, and negation well without needing
any training data).

Usage:
    from sentiment_analyzer import analyze_text, analyze_dataframe

    result = analyze_text("I absolutely love this product!!")
    # {'text': ..., 'compound': 0.83, 'sentiment': 'Positive', ...}
"""

import nltk
import pandas as pd
from preprocess import clean_text, clean_series


def ensure_vader_lexicon():
    """Download the VADER lexicon once if it isn't already available."""
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")


ensure_vader_lexicon()

from nltk.sentiment.vader import SentimentIntensityAnalyzer  # noqa: E402

_analyzer = SentimentIntensityAnalyzer()

# Standard VADER thresholds for classifying the compound score
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def classify_compound(compound: float) -> str:
    """Map a VADER compound score (-1 to +1) to a sentiment label."""
    if compound >= POSITIVE_THRESHOLD:
        return "Positive"
    elif compound <= NEGATIVE_THRESHOLD:
        return "Negative"
    return "Neutral"


def analyze_text(raw_text: str) -> dict:
    """
    Run sentiment analysis on a single piece of text.
    Returns the cleaned text, individual VADER scores, the compound
    score, and the final Positive/Negative/Neutral label.
    """
    cleaned = clean_text(raw_text)
    scores = _analyzer.polarity_scores(cleaned)

    return {
        "text": raw_text,
        "cleaned_text": cleaned,
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "positive": scores["pos"],
        "compound": scores["compound"],
        "sentiment": classify_compound(scores["compound"]),
    }


def analyze_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Run sentiment analysis on every row of a DataFrame's text column.
    Returns a new DataFrame with the original columns plus:
    cleaned_text, negative, neutral, positive, compound, sentiment.
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")

    results = df[text_column].apply(lambda t: analyze_text(t))
    results_df = pd.DataFrame(list(results))

    # Combine with the original dataframe (minus the duplicate raw text col)
    combined = pd.concat(
        [df.reset_index(drop=True), results_df.drop(columns=["text"])],
        axis=1,
    )
    return combined


def sentiment_summary(df: pd.DataFrame) -> pd.Series:
    """Return counts of each sentiment label, e.g. for a pie chart."""
    return df["sentiment"].value_counts()


if __name__ == "__main__":
    # Quick manual test
    samples = [
        "I absolutely love this product, best purchase ever!!",
        "This is the worst customer service I've ever experienced.",
        "The package arrived on Tuesday.",
        "Not bad, but not great either. It's okay I guess.",
        "Terrible battery life, very disappointed :(",
    ]
    for s in samples:
        print(analyze_text(s))
