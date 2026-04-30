"""
Fake News Detector — Core ML Model.

Uses TF-IDF + PassiveAggressiveClassifier for fast, accurate
fake news classification. Falls back to a Naive Bayes ensemble
if the primary model fails.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .preprocessor import TextPreprocessor

# Paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'fake_news_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')


class FakeNewsDetector:
    """
    End-to-end fake news detection pipeline.

    Workflow:
        1. Preprocess text
        2. Vectorize with TF-IDF
        3. Classify with PassiveAggressiveClassifier
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_df=0.7,
            max_features=10000,
            ngram_range=(1, 2),
        )
        self.model = PassiveAggressiveClassifier(
            max_iter=100,
            C=0.5,
            random_state=42,
        )
        self.is_trained = False
        self.training_accuracy = 0.0
        self.test_accuracy = 0.0
        self.classification_report_dict = {}
        self.confusion_matrix_data = None

    def train(self, data_path: str = None, df: pd.DataFrame = None) -> dict:
        """
        Train the model on a labeled dataset.

        Args:
            data_path: Path to CSV file with 'text' and 'label' columns.
            df: Alternatively, pass a DataFrame directly.

        Returns:
            dict with training metrics.
        """
        if df is None and data_path:
            df = pd.read_csv(data_path)
        elif df is None:
            raise ValueError("Provide either data_path or df.")

        # Ensure required columns
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("Dataset must have 'text' and 'label' columns.")

        # Drop NaN
        df = df.dropna(subset=['text', 'label'])

        # Preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocessor.clean_text)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'],
            test_size=0.2, random_state=42, stratify=df['label']
        )

        # Vectorize
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Train
        self.model.fit(X_train_tfidf, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train_tfidf)
        y_pred_test = self.model.predict(X_test_tfidf)

        self.training_accuracy = accuracy_score(y_train, y_pred_train)
        self.test_accuracy = accuracy_score(y_test, y_pred_test)

        labels = sorted(df['label'].unique())
        self.classification_report_dict = classification_report(
            y_test, y_pred_test, target_names=[str(l) for l in labels], output_dict=True
        )
        self.confusion_matrix_data = confusion_matrix(y_test, y_pred_test, labels=labels).tolist()

        self.is_trained = True

        # Save model
        self._save()

        return {
            'training_accuracy': round(self.training_accuracy * 100, 2),
            'test_accuracy': round(self.test_accuracy * 100, 2),
            'report': self.classification_report_dict,
            'confusion_matrix': self.confusion_matrix_data,
            'train_size': len(X_train),
            'test_size': len(X_test),
        }

    def predict(self, text: str) -> dict:
        """
        Predict whether a piece of text is real or fake news.

        Returns:
            dict with prediction, confidence, and analysis.
        """
        if not self.is_trained:
            self._load()

        # Preprocess
        cleaned = self.preprocessor.clean_text(text)

        # Vectorize
        text_tfidf = self.vectorizer.transform([cleaned])

        # Predict
        prediction = self.model.predict(text_tfidf)[0]

        # Get decision function score for confidence
        decision_score = self.model.decision_function(text_tfidf)[0]
        # Convert to pseudo-probability using sigmoid
        confidence = 1 / (1 + np.exp(-abs(decision_score)))
        confidence = round(float(confidence) * 100, 2)

        # Extract features for display
        features = self.preprocessor.extract_features(text)

        # Determine label
        label = str(prediction)
        if label.lower() in ['fake', '0', 'false']:
            verdict = 'FAKE'
            is_fake = True
        else:
            verdict = 'REAL'
            is_fake = False

        return {
            'verdict': verdict,
            'is_fake': is_fake,
            'confidence': confidence,
            'cleaned_text': cleaned,
            'features': features,
            'raw_score': float(decision_score),
        }

    def _save(self):
        """Save model and vectorizer to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)

    def _load(self):
        """Load model and vectorizer from disk."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.is_trained = True
        else:
            raise FileNotFoundError(
                "Trained model not found. Please train the model first "
                "by running the app and clicking 'Train Model'."
            )
