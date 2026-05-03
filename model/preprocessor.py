"""
Text Preprocessing Module for Fake News Detection.

Handles text cleaning, tokenization, feature extraction,
and keyword highlighting for the ML pipeline.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK data (only once)
def ensure_nltk_data():
    """Download required NLTK datasets if not already present."""
    resources = {
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
    }
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_data()

# Patterns typical of sensational / fake news writing
SENSATIONAL_PATTERNS = [
    r'\bBREAKING\b', r'\bSHOCKING\b', r'\bEXPOSED\b', r'\bURGENT\b',
    r'\bBOMBSHELL\b', r'\bMUST READ\b', r'\bUNBELIEVABLE\b',
    r'\bYOU WON\'T BELIEVE\b', r'\bSECRET\b', r'\bCONSPIRACY\b',
    r'\bCOVER.?UP\b', r'\bWHISTLEBLOWER\b', r'\bHIDDEN TRUTH\b',
    r'\bWAKE UP\b', r'\bTHEY DON\'T WANT YOU\b', r'\bBIG PHARMA\b',
    r'\bDEEP STATE\b', r'\bFALSE FLAG\b', r'\bHOAX\b', r'\bPROPAGANDA\b',
    r'\bALERT\b', r'\bMUST SEE\b', r'\bSHOCKING REVEAL\b', r'\bBREAKING NEWS\b',
]

CREDIBLE_PATTERNS = [
    r'\baccording to\b', r'\bstudies show\b', r'\bresearchers\b',
    r'\bpublished in\b', r'\bpeer.reviewed\b', r'\bofficial\b',
    r'\bgovernment\b', r'\buniversity\b', r'\bscientists\b',
    r'\bstatistics\b', r'\bdata shows\b', r'\breport(ed|s)?\b',
    r'\banalysis\b', r'\bexpert(s)?\b', r'\bconfirmed\b',
    r'\bjournal\b', r'\binstitution\b', r'\bprofessor\b', r'\bclinical trial\b',
]


class TextPreprocessor:
    """Preprocesses text for fake news classification."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep negations — they matter for sentiment
        self.stop_words -= {'no', 'not', 'nor', 'neither', 'never', 'none'}
        # Also keep comparative/superlative — important for claim strength
        self.stop_words -= {'more', 'most', 'less', 'least', 'very', 'too'}

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text.

        Steps:
        1. Lowercase
        2. Remove URLs, HTML, emails, mentions
        3. Remove special characters and numbers
        4. Remove extra whitespace
        5. Remove stopwords
        6. Lemmatize tokens
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[@#]\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(tokens)

    def extract_features(self, text: str) -> dict:
        """
        Extract rich linguistic and stylistic features from raw text.

        Returns a dict of features used both for display and the
        extra-features classifier layer.
        """
        if not isinstance(text, str) or not text.strip():
            return self._empty_features()

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)

        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        avg_sentence_length = word_count / sentence_count
        exclamation_count = text.count('!')
        question_count = text.count('?')
        upper_chars = sum(1 for c in text if c.isupper())
        capital_ratio = upper_chars / max(char_count, 1)
        url_count = len(re.findall(r'https?://\S+|www\.\S+', text))

        text_upper = text.upper()
        sensational_hits = sum(
            1 for p in SENSATIONAL_PATTERNS
            if re.search(p, text_upper)
        )
        credible_hits = sum(
            1 for p in CREDIBLE_PATTERNS
            if re.search(p, text, re.IGNORECASE)
        )

        # Unique word ratio (lexical diversity)
        unique_ratio = len(set(w.lower() for w in words)) / max(word_count, 1)

        return {
            'char_count': char_count,
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_ratio': round(capital_ratio, 4),
            'url_count': url_count,
            'sensational_score': sensational_hits,
            'credibility_score': credible_hits,
            'lexical_diversity': round(unique_ratio, 4),
        }

    def get_top_keywords(self, text: str, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Return the top N significant keywords from the text,
        scored by inverse-stopword-frequency heuristic.
        Used for explainability display.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        freq: dict[str, int] = {}
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1

        total = max(sum(freq.values()), 1)
        scored = [(word, count / total) for word, count in freq.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def _empty_features(self) -> dict:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'capital_ratio': 0,
            'url_count': 0,
            'sensational_score': 0,
            'credibility_score': 0,
            'lexical_diversity': 0,
        }


# ── Utility ────────────────────────────────────────────────────────────

def count_sensational_hits(text: str) -> int:
    """Return the number of sensational pattern matches in text (case-insensitive)."""
    text_upper = text.upper()
    return sum(1 for p in SENSATIONAL_PATTERNS if __import__('re').search(p, text_upper))
