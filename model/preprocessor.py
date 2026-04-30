"""
Text Preprocessing Module for Fake News Detection.

Handles text cleaning, tokenization, and feature extraction
for the ML pipeline.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (only once)
def ensure_nltk_data():
    """Download required NLTK datasets if not already present."""
    for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource != 'punkt' and resource != 'punkt_tab' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

ensure_nltk_data()


class TextPreprocessor:
    """Preprocesses text for fake news classification."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some negation words as they can be important for sentiment
        self.stop_words -= {'no', 'not', 'nor', 'neither', 'never', 'none'}

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text.

        Steps:
        1. Lowercase
        2. Remove URLs
        3. Remove HTML tags
        4. Remove special characters and numbers
        5. Remove extra whitespace
        6. Remove stopwords
        7. Lemmatize tokens
        """
        if not isinstance(text, str):
            return ""

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove mentions and hashtags
        text = re.sub(r'[@#]\w+', '', text)

        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Tokenize, remove stopwords, and lemmatize
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 2
        ]

        return ' '.join(tokens)

    def extract_features(self, text: str) -> dict:
        """
        Extract additional features from raw text for analysis display.

        Returns a dict of features useful for the UI.
        """
        if not isinstance(text, str):
            return {
                'char_count': 0,
                'word_count': 0,
                'avg_word_length': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'capital_ratio': 0,
                'url_count': 0,
            }

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        upper_chars = sum(1 for c in text if c.isupper())
        capital_ratio = upper_chars / max(char_count, 1)
        url_count = len(re.findall(r'https?://\S+|www\.\S+', text))

        return {
            'char_count': char_count,
            'word_count': word_count,
            'avg_word_length': round(avg_word_length, 2),
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'capital_ratio': round(capital_ratio, 4),
            'url_count': url_count,
        }
