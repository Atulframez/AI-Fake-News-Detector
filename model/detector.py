"""
Fake News Detector — Real-World ML Ensemble Model.

Architecture:
  - TF-IDF (unigrams + bigrams, 50k features) as primary representation
  - Voting Ensemble: LogisticRegression + SGDClassifier + PassiveAggressive
  - Soft-voting with calibrated probabilities for confidence scores
  - LIME-style keyword explainability via TF-IDF weight extraction
  - Cross-validation during training for robust metrics
  - Supports real datasets (WELFake, ISOT, custom CSV)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .preprocessor import TextPreprocessor

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'saved_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
META_PATH = os.path.join(MODEL_DIR, 'model_meta.pkl')


class FakeNewsDetector:
    """
    Production-grade fake news detection ensemble.

    Pipeline:
        raw text → clean → TF-IDF → Voting(LR + SGD + PAC) → calibrated prob
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.label_encoder = LabelEncoder()

        # TF-IDF: large vocab, bigrams for phrase capture
        self.vectorizer = TfidfVectorizer(
            max_df=0.85,
            min_df=2,
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,       # log-scaled TF for more balanced weighting
            strip_accents='unicode',
            analyzer='word',         # word-level tokens (not char n-grams)
        )

        # Individual calibrated classifiers
        lr = CalibratedClassifierCV(
            LogisticRegression(
                C=1.0, max_iter=1000, solver='lbfgs',
                class_weight='balanced', random_state=42
            ),
            cv=3, method='isotonic'
        )
        sgd = CalibratedClassifierCV(
            SGDClassifier(
                loss='modified_huber', alpha=1e-4,
                max_iter=200, random_state=42, class_weight='balanced'
            ),
            cv=3, method='isotonic'
        )
        pac = CalibratedClassifierCV(
            PassiveAggressiveClassifier(
                C=0.5, max_iter=200, random_state=42, class_weight='balanced'
            ),
            cv=3, method='isotonic'
        )

        # Soft-voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[('lr', lr), ('sgd', sgd), ('pac', pac)],
            voting='soft',
        )

        self.is_trained = False
        self.meta = {}
        self.feature_names = []
        self._prediction_count = 0  # track total predictions made this session

    # ── Training ───────────────────────────────────────────────────────

    def train(self, data_path: str = None, df: pd.DataFrame = None) -> dict:
        """
        Train the ensemble on a labeled dataset.

        Accepts:
            data_path: path to CSV with 'text' and 'label' columns.
            df:        DataFrame with 'text' and 'label' columns.

        Supports WELFake, ISOT, and generic two-column CSVs.

        Returns dict with comprehensive training metrics.
        """
        df = self._load_data(data_path, df)
        df = self._normalize_labels(df)

        print(f"📊 Dataset: {len(df)} samples | "
              f"FAKE: {(df['label']=='FAKE').sum()} | "
              f"REAL: {(df['label']=='REAL').sum()}")

        # Preprocess text
        print("🔤 Preprocessing text...")
        df['cleaned'] = df['text'].apply(self.preprocessor.clean_text)
        df = df[df['cleaned'].str.strip().astype(bool)]  # drop empty

        X = df['cleaned'].values
        y = df['label'].values

        # Encode labels
        self.label_encoder.fit(y)
        y_enc = self.label_encoder.transform(y)

        # Train / test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        # Fit TF-IDF
        print("🔢 Vectorizing...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Train ensemble
        print("🤖 Training ensemble (LR + SGD + PAC)...")
        self.ensemble.fit(X_train_tfidf, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.ensemble.predict(X_test_tfidf)
        y_prob = self.ensemble.predict_proba(X_test_tfidf)

        train_acc = accuracy_score(y_train, self.ensemble.predict(X_train_tfidf))
        test_acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        except Exception:
            auc = None

        labels_decoded = self.label_encoder.classes_.tolist()
        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(
            y_test, y_pred,
            target_names=labels_decoded,
            output_dict=True
        )

        # Cross-validation on full data (quick 3-fold)
        print("📐 Running cross-validation...")
        X_all_tfidf = self.vectorizer.transform(X)
        cv_scores = cross_val_score(
            self.ensemble, X_all_tfidf, y_enc, cv=3, scoring='accuracy', n_jobs=-1
        )

        self.meta = {
            'training_accuracy': round(train_acc * 100, 2),
            'test_accuracy': round(test_acc * 100, 2),
            'f1_score': round(f1 * 100, 2),
            'precision': round(precision * 100, 2),
            'recall': round(recall * 100, 2),
            'roc_auc': round(auc * 100, 2) if auc else None,
            'cv_mean': round(cv_scores.mean() * 100, 2),
            'cv_std': round(cv_scores.std() * 100, 2),
            'train_size': len(X_train),
            'test_size': len(X_test),
            'total_samples': len(df),
            'confusion_matrix': cm,
            'report': report,
            'labels': labels_decoded,
        }

        from datetime import datetime, timezone
        self.meta['trained_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        self._save()
        print(f"✅ Training complete! Test accuracy: {test_acc*100:.2f}%")
        return self.meta

    # ── Prediction ─────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict whether text is FAKE or REAL news.

        Returns:
            dict with verdict, confidence, top contributing keywords,
            stylistic features, and raw probability breakdown.
        """
        if not self.is_trained:
            self._load()

        cleaned = self.preprocessor.clean_text(text)
        if not cleaned.strip():
            return self._empty_result(text)

        vec = self.vectorizer.transform([cleaned])
        y_enc = self.ensemble.predict(vec)[0]
        probs = self.ensemble.predict_proba(vec)[0]

        label = self.label_encoder.inverse_transform([y_enc])[0]
        is_fake = label == 'FAKE'
        self._prediction_count += 1

        # Confidence: use the winning class probability
        confidence = round(float(probs[y_enc]) * 100, 2)

        # Keyword explainability: top TF-IDF features for this text
        keywords = self._explain_prediction(vec, y_enc, top_n=12)

        # Stylistic features
        features = self.preprocessor.extract_features(text)

        # Individual model agreement
        model_votes = self._get_model_votes(vec)

        return {
            'verdict': label,
            'is_fake': is_fake,
            'confidence': confidence,
            'confidence_label': self.confidence_label(confidence),
            'fake_prob': round(float(probs[self._fake_idx()]) * 100, 2),
            'real_prob': round(float(probs[self._real_idx()]) * 100, 2),
            'keywords': keywords,
            'features': features,
            'model_votes': model_votes,
            'cleaned_text': cleaned,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """
        Predict a list of texts.

        Args:
            texts: list of raw news article strings.

        Returns:
            List of result dicts in the same order as input.
            Empty or whitespace-only strings return an UNKNOWN verdict.
        """
        return [self.predict(t) for t in texts]

    # ── Explainability ─────────────────────────────────────────────────

    def _explain_prediction(self, vec, predicted_class: int, top_n: int = 12) -> list[dict]:
        """
        Return top-N TF-IDF feature weights for the predicted class,
        used as keyword explanation (LIME-lite).
        """
        try:
            # Get the LR estimator's coefficients (most interpretable)
            lr_cal = self.ensemble.estimators_[0]  # CalibratedClassifierCV(LR)
            base_lr = lr_cal.calibrated_classifiers_[0].estimator
            coef = base_lr.coef_

            # For binary or multi-class
            if coef.shape[0] == 1:
                class_coef = coef[0] if predicted_class == 1 else -coef[0]
            else:
                class_coef = coef[predicted_class]

            # Get non-zero TF-IDF indices for this text
            nz = vec.nonzero()[1]
            scored = [
                (self.feature_names[i], float(class_coef[i]) * float(vec[0, i]))
                for i in nz
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = scored[:top_n]

            return [
                {'word': word, 'score': round(score, 4), 'positive': score > 0}
                for word, score in top
            ]
        except Exception:
            return []

    def _get_model_votes(self, vec) -> dict:
        """Get individual predictions from each sub-model."""
        votes = {}
        names = ['Logistic Regression', 'SGD Classifier', 'Passive Aggressive']
        for name, est in zip(names, self.ensemble.estimators_):
            try:
                pred = est.predict(vec)[0]
                label = self.label_encoder.inverse_transform([pred])[0]
                votes[name] = label
            except Exception:
                votes[name] = 'N/A'
        return votes

    def _fake_idx(self) -> int:
        """Index of the FAKE class in the label encoder."""
        classes = list(self.label_encoder.classes_)
        return classes.index('FAKE') if 'FAKE' in classes else 0

    def _real_idx(self) -> int:
        """Index of the REAL class in the label encoder."""
        classes = list(self.label_encoder.classes_)
        return classes.index('REAL') if 'REAL' in classes else 1

    def label_classes(self) -> list[str]:
        """Return the list of recognized class labels."""
        return list(self.label_encoder.classes_)

    def top_fake_keywords(self, text: str, top_n: int = 5) -> list:
        """
        Return up to top_n keywords from text that match known
        sensational patterns — useful for quick explainability display.
        """
        import re
        from .preprocessor import SENSATIONAL_PATTERNS
        text_upper = text.upper()
        matched = []
        for pattern in SENSATIONAL_PATTERNS:
            m = re.search(pattern, text_upper)
            if m:
                matched.append(m.group(0).title())
            if len(matched) >= top_n:
                break
        return matched

    def verdict_emoji(self, verdict: str) -> str:
        """Return an emoji representing the verdict for display purposes."""
        return {'FAKE': '🔴', 'REAL': '🟢'}.get(verdict.upper(), '⚪')

    def confidence_label(self, confidence: float) -> str:
        """
        Convert a numeric confidence percentage to a human-readable label.

        Ranges:
            >= 90  → 'Very High'
            >= 75  → 'High'
            >= 60  → 'Moderate'
            >= 45  → 'Low'
            <  45  → 'Very Low'
        """
        if confidence >= 90:
            return 'Very High'
        elif confidence >= 75:
            return 'High'
        elif confidence >= 60:
            return 'Moderate'
        elif confidence >= 45:
            return 'Low'
        return 'Very Low'

    def is_model_loaded(self) -> bool:
        """Return True if the model is trained or loaded and ready for inference."""
        return self.is_trained

    def model_summary(self) -> dict:
        """
        Return a brief summary of the current model state.

        Useful for displaying model info in a UI or API response.
        """
        return {
            'is_loaded': self.is_trained,
            'classes': self.label_classes() if self.is_trained else [],
            'vocab_size': len(self.feature_names),
            'total_predictions': self._prediction_count,
            'test_accuracy': self.meta.get('test_accuracy'),
            'f1_score': self.meta.get('f1_score'),
        }

    # ── Data Loading ───────────────────────────────────────────────────

    def _load_data(self, data_path, df) -> pd.DataFrame:
        if df is None and data_path:
            df = pd.read_csv(data_path)
        elif df is None:
            raise ValueError("Provide either data_path or df.")

        # Auto-detect WELFake format (has 'title' + 'text')
        if 'title' in df.columns and 'text' in df.columns:
            df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        elif 'title' in df.columns and 'text' not in df.columns:
            df = df.rename(columns={'title': 'text'})

        # Auto-detect ISOT format (separate True/Fake files merged)
        if 'text' not in df.columns:
            candidates = [c for c in df.columns if 'text' in c.lower() or 'content' in c.lower() or 'article' in c.lower()]
            if candidates:
                df = df.rename(columns={candidates[0]: 'text'})

        if 'label' not in df.columns:
            label_candidates = [c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'fake' in c.lower()]
            if label_candidates:
                df = df.rename(columns={label_candidates[0]: 'label'})

        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns.")

        df = df.dropna(subset=['text', 'label'])
        df = df[df['text'].str.strip().astype(bool)]
        return df[['text', 'label']].copy()

    def _normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize various label formats to FAKE / REAL.

        Handles: 0/1, true/false, real/fake, reliable/unreliable,
        pants-fire, barely-true, half-true, mostly-true from LIAR dataset.
        """
        mapping = {
            '0': 'FAKE', 0: 'FAKE', 'fake': 'FAKE', 'FALSE': 'FAKE',
            'false': 'FAKE', 'unreliable': 'FAKE', 'pants-fire': 'FAKE',
            'barely-true': 'FAKE', 'half-true': 'REAL',
            '1': 'REAL', 1: 'REAL', 'real': 'REAL', 'TRUE': 'REAL',
            'true': 'REAL', 'reliable': 'REAL', 'mostly-true': 'REAL',
        }
        df['label'] = df['label'].apply(
            lambda x: mapping.get(str(x).strip(), str(x).strip().upper())
        )
        # Keep only FAKE / REAL
        df = df[df['label'].isin(['FAKE', 'REAL'])].copy()
        return df

    # ── Persistence ────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.ensemble, MODEL_PATH)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        joblib.dump({
            'meta': self.meta,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
        }, META_PATH)
        print(f"💾 Model saved to {MODEL_DIR}")

    def _load(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
            raise FileNotFoundError(
                "No trained model found. Please train the model first."
            )
        self.ensemble = joblib.load(MODEL_PATH)
        self.vectorizer = joblib.load(VECTORIZER_PATH)
        if os.path.exists(META_PATH):
            saved = joblib.load(META_PATH)
            self.meta = saved.get('meta', {})
            self.label_encoder = saved.get('label_encoder', self.label_encoder)
            self.feature_names = saved.get('feature_names', [])
        self.is_trained = True

    def get_prediction_count(self) -> int:
        """Return the number of predictions made this session."""
        return self._prediction_count

    def reset_prediction_count(self) -> None:
        """Reset the session prediction counter back to zero."""
        self._prediction_count = 0

    # ── Helpers ────────────────────────────────────────────────────────

    def _empty_result(self, text: str) -> dict:
        return {
            'verdict': 'UNKNOWN',
            'is_fake': None,
            'confidence': 0,
            'fake_prob': 0,
            'real_prob': 0,
            'keywords': [],
            'features': self.preprocessor.extract_features(text),
            'model_votes': {},
            'cleaned_text': '',
        }
