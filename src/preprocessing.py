"""
preprocessing.py
================
NLP preprocessing pipeline for storage type classification.
Handles text cleaning, tokenization, stopword removal, lemmatization,
and feature extraction for both traditional ML and deep learning models.

Author: NLP Engineering Team
"""

import re
import string
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── NLTK Resources ───────────────────────────────────────────────────────────
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOP_WORDS   = set(stopwords.words("english"))
LEMMATIZER   = WordNetLemmatizer()

# Storage-domain keywords (preserve these even if stopwords)
DOMAIN_KEYWORDS = {
    "freeze", "frozen", "freezer", "fridge", "refrigerate", "refrigerator",
    "cold", "chill", "chilled", "cool", "cooled", "ice", "store", "storage",
    "room", "temperature", "dry", "shelf", "cabinet", "pantry", "cellar",
}

# ─── Text Cleaner ─────────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    End-to-end NLP preprocessing pipeline.

    Steps:
    1. Lowercase
    2. Expand contractions
    3. Remove URLs / emails
    4. Remove punctuation
    5. Tokenize
    6. Remove stopwords (preserving domain keywords)
    7. Lemmatize
    8. Rejoin tokens
    """

    CONTRACTIONS = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will",
        "'ve": " have", "'m": " am", "it's": "it is", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "wasn't": "was not",
        "weren't": "were not", "haven't": "have not", "hasn't": "has not",
        "hadn't": "had not", "shouldn't": "should not", "wouldn't": "would not",
        "couldn't": "could not", "mustn't": "must not",
    }

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize        = lemmatize

    def expand_contractions(self, text: str) -> str:
        for contraction, expansion in self.CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    def clean(self, text: str) -> str:
        """Full preprocessing pipeline for one text string."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = self.expand_contractions(text)
        text = re.sub(r"http\S+|www\S+", " ", text)          # remove URLs
        text = re.sub(r"\S+@\S+", " ", text)                  # remove emails
        text = re.sub(r"\d+", " ", text)                       # remove digits
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)

        if self.remove_stopwords:
            tokens = [
                t for t in tokens
                if t not in STOP_WORDS or t in DOMAIN_KEYWORDS
            ]

        if self.lemmatize:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def transform(self, texts: List[str]) -> List[str]:
        """Apply clean() to a list of texts."""
        return [self.clean(t) for t in texts]


# ─── TF-IDF Feature Extractor ─────────────────────────────────────────────────

class TFIDFExtractor:
    """Wraps sklearn TfidfVectorizer with fit/transform interface."""

    def __init__(self, max_features: int = 10_000, ngram_range: Tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts)

    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()


# ─── Label Encoder Wrapper ────────────────────────────────────────────────────

class StorageLabelEncoder:
    """Encode/decode storage labels with a fixed class order."""

    CLASSES = ["freezer", "fridge", "normal"]

    def __init__(self):
        self.encoder = LabelEncoder()
        self.encoder.fit(self.CLASSES)

    def encode(self, labels) -> np.ndarray:
        return self.encoder.transform(labels)

    def decode(self, indices) -> np.ndarray:
        return self.encoder.inverse_transform(indices)

    def num_classes(self) -> int:
        return len(self.CLASSES)


# ─── Dataset Loader ───────────────────────────────────────────────────────────

class DataLoader:
    """Loads CSVs and applies preprocessing for model consumption."""

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.label_enc    = StorageLabelEncoder()

    def load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        df["cleaned"] = self.preprocessor.transform(df["sentence"].tolist())
        df["label_id"] = self.label_enc.encode(df["storage_label"])
        return df

    def get_texts_labels(self, df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        return df["cleaned"].tolist(), df["label_id"].values


# ─── Sequence Padding (for LSTM) ──────────────────────────────────────────────

def build_vocab(texts: List[str], max_vocab: int = 15_000) -> dict:
    """Build a word → index vocabulary from a list of cleaned texts."""
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

def texts_to_sequences(texts: List[str], vocab: dict, max_len: int = 64) -> np.ndarray:
    """Convert texts to padded integer sequences."""
    seqs = []
    for text in texts:
        ids = [vocab.get(w, 1) for w in text.split()][:max_len]
        ids = ids + [0] * (max_len - len(ids))
        seqs.append(ids)
    return np.array(seqs, dtype=np.int32)


if __name__ == "__main__":
    proc = TextPreprocessor()
    samples = [
        "Please refrigerate the insulin after opening.",
        "Store the laptop in a dry place.",
        "The frozen chicken nuggets should go in the freezer.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {proc.clean(s)}\n")