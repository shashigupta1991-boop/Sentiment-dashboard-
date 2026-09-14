"""
preprocess.py
-------------
Basic text-cleaning utilities for social media posts / product reviews
before sentiment scoring. Keeps things lightweight (no heavy tokenizer
downloads beyond what NLTK's VADER needs).
"""

import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
SPECIAL_CHARS_PATTERN = re.compile(r"[^A-Za-z0-9\s.,!?'\"]+")
MULTI_SPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Clean a single piece of text (tweet / review / comment) for sentiment
    analysis:
      - lowercases nothing (VADER is case-aware: CAPS/exclamation marks
        affect intensity, so we deliberately keep original casing)
      - strips URLs
      - strips @mentions
      - keeps hashtag *words* but removes the '#' symbol
        (e.g. "#GreatProduct" -> "GreatProduct")
      - removes other special characters/emojis that aren't useful text
      - collapses extra whitespace
    """
    if not isinstance(text, str):
        return ""

    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_SYMBOL_PATTERN.sub("", text)
    text = SPECIAL_CHARS_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def clean_series(series):
    """Apply clean_text to a pandas Series of raw text."""
    return series.astype(str).apply(clean_text)
