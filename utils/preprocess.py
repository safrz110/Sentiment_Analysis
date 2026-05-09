# ============================================================
#  utils/preprocess.py  –  NLTK text-preprocessing pipeline
# ============================================================

import re
import string
import logging

import nltk
from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── Download required NLTK assets (one-time) ─────────────────
def download_nltk_assets() -> None:
    assets = ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]
    for asset in assets:
        try:
            nltk.download(asset, quiet=True)
        except Exception as e:
            logger.warning(f"Could not download NLTK asset '{asset}': {e}")

download_nltk_assets()

# ── Preprocessing helpers ─────────────────────────────────────
_lemmatizer  = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))

# Negation words we intentionally KEEP (removing them hurts sentiment)
_NEGATION_WORDS = {
    "no", "not", "nor", "neither", "never", "nothing",
    "nobody", "nowhere", "hardly", "barely", "scarcely",
}
_KEEP_WORDS = _NEGATION_WORDS  # extend here if needed


def clean_text(text: str) -> str:
    """
    Light-touch cleaning that preserves sentiment signal:
      1. Lower-case
      2. Expand common contractions
      3. Strip URLs, HTML tags, extra whitespace
      4. Remove non-ASCII characters
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()

    # Expand contractions
    contractions = {
        r"won't":   "will not",
        r"can't":   "cannot",
        r"n't":     " not",
        r"'re":     " are",
        r"'s":      " is",
        r"'d":      " would",
        r"'ll":     " will",
        r"'ve":     " have",
        r"'m":      " am",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)

    text = re.sub(r"http\S+|www\.\S+",              "",  text)   # URLs
    text = re.sub(r"<[^>]+>",                        "",  text)   # HTML tags
    text = re.sub(r"[^a-z0-9\s]",                   " ", text)   # punctuation
    text = re.sub(r"\s+",                            " ", text).strip()

    return text


def tokenize_and_filter(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize → optionally remove stop-words (keeping negations)."""
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [
            t for t in tokens
            if t not in _stop_words or t in _KEEP_WORDS
        ]
    return tokens


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens."""
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, remove_stopwords: bool = True) -> str:
    """
    Full pipeline:
      clean_text → tokenize_and_filter → lemmatize → rejoin
    Returns a clean string ready for the BERT tokenizer.
    """
    cleaned  = clean_text(text)
    tokens   = tokenize_and_filter(cleaned, remove_stopwords=remove_stopwords)
    lemmas   = lemmatize(tokens)
    return " ".join(lemmas)


def batch_preprocess(texts: list[str], remove_stopwords: bool = True) -> list[str]:
    """Apply `preprocess` to a list of texts."""
    return [preprocess(t, remove_stopwords) for t in texts]


# ── Quick self-test ───────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "The product quality is absolutely AMAZING! I can't believe how good it is.",
        "This item was okay, nothing special. It does what it's supposed to do.",
        "Terrible experience. The product broke after two days. Do NOT buy this!!",
        "Check out our website at https://example.com for more deals!",
    ]
    print("=" * 60)
    print("  Preprocessing Pipeline – Quick Test")
    print("=" * 60)
    for s in samples:
        print(f"\nORIGINAL : {s}")
        print(f"PROCESSED: {preprocess(s)}")
