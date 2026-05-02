# 🔍 AI Fake News Detector

> **Real-world NLP + ML project** — detects fake news using an ensemble of three classifiers, TF-IDF feature engineering, live URL article scraping, and keyword-level explainability.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Atulframez/AI-Fake-News-Detector?style=social)](https://github.com/Atulframez/AI-Fake-News-Detector/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/Atulframez/AI-Fake-News-Detector)](https://github.com/Atulframez/AI-Fake-News-Detector/issues)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 **Ensemble ML** | Soft-voting: Logistic Regression + SGD + Passive-Aggressive classifiers |
| 📐 **TF-IDF (50k)** | Unigrams + bigrams, sublinear TF scaling, balanced class weights |
| 🌐 **URL Scraping** | Paste any news URL — live article text extraction via trafilatura |
| 🔑 **Explainability** | Top keywords that drove each prediction (TF-IDF weight analysis) |
| 📋 **Batch Analysis** | Analyze multiple articles at once, download results as CSV |
| 📊 **Rich Metrics** | Accuracy, F1, Precision, Recall, ROC-AUC, 3-Fold Cross-Validation |
| 🗳️ **Model Votes** | See how each individual classifier voted on the prediction |
| 🕓 **History** | In-session analysis history with source tracking |
| 📁 **Real Datasets** | Supports WELFake (72k), ISOT (44k), or any custom labeled CSV |

---

## 🏗️ Architecture

```
Raw Text / URL
      │
      ▼
ArticleScraper (trafilatura → BeautifulSoup fallback)
      │
      ▼
TextPreprocessor
  ├── Clean: lowercase, remove URLs/HTML/emails/punctuation
  ├── Lemmatize (NLTK WordNetLemmatizer)
  ├── Stylistic features: exclamation count, CAPS ratio,
  │   sensational/credibility pattern scores, lexical diversity
  └── Keyword extraction (top-N significant terms)
      │
      ▼
TF-IDF Vectorizer (50,000 features · bigrams · sublinear TF)
      │
      ▼
Soft-Voting Ensemble
  ├── CalibratedLR (LogisticRegression + isotonic calibration)
  ├── CalibratedSGD (modified_huber loss + isotonic calibration)
  └── CalibratedPAC (PassiveAggressive + isotonic calibration)
      │
      ▼
Calibrated Probability → Verdict + Confidence + Keyword Explanation
```

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Atulframez/AI-Fake-News-Detector.git
cd AI-Fake-News-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> **First time?** Click **"🚀 Train on Sample Data"** in the left sidebar to train the model before analyzing articles.

---

## 🖥️ How to Use

### Option 1 — Paste Text
1. Open the **"🔍 Analyze Article"** tab
2. Select **"✍️ Paste Text"**
3. Paste or type a news article
4. Click **"🔍 Analyze Now"**

### Option 2 — Analyze from URL
1. Open the **"🔍 Analyze Article"** tab
2. Select **"🌐 Analyze from URL"**
3. Paste any news article URL (e.g. from BBC, Reuters, CNN)
4. Click **"🌐 Fetch Article"** → then **"🔍 Analyze Now"**

### Option 3 — Batch Analysis
1. Open the **"📋 Batch Analysis"** tab
2. Paste multiple articles separated by blank lines, or upload a CSV
3. Click **"🔍 Analyze Batch"**
4. Download results as CSV

---

## 📁 Project Structure

```
AI-Fake-News-Detector/
├── app.py                          # Main Streamlit app (3-tab UI)
├── requirements.txt                # All dependencies
├── model/
│   ├── __init__.py
│   ├── detector.py                 # Ensemble ML model (train + predict)
│   ├── preprocessor.py             # NLP preprocessing + feature extraction
│   └── scraper.py                  # URL article extractor (trafilatura)
├── sample_data/
│   ├── generate_sample_data.py     # Built-in dataset generator
│   └── news_dataset.csv            # Auto-generated training data (400 samples)
├── saved_model/                    # Auto-created after first training
│   ├── ensemble_model.pkl          # Trained voting ensemble
│   ├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
│   └── model_meta.pkl              # Metrics + label encoder
└── .streamlit/
    └── config.toml                 # Dark purple theme config
```

---

## 🗂️ Using Real Datasets (Recommended)

For production-level accuracy, use one of these free datasets:

| Dataset | Size | Format | Link |
|---|---|---|---|
| **WELFake** | 72,134 articles | `text`, `label` (0/1) | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **ISOT** | 44,898 articles | Separate True/Fake CSVs | [UVic](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php) |
| **LIAR** | 12,836 statements | Multi-class labels | [GitHub](https://github.com/thiagorainmaker77/liar_dataset) |

**Steps:**
1. Download the dataset CSV from the link above
2. Open the sidebar → select **"Upload Real Dataset (CSV)"**
3. Upload the file → click **"🚀 Train on Uploaded Data"**

> The detector auto-normalizes label formats from 0/1, true/false, or real/fake to FAKE/REAL automatically.

---

## 📊 Model Performance

### Sample Data (400 articles, built-in)
| Metric | Score |
|---|---|
| **Test Accuracy** | 100.0% |
| **F1 Score** | 100.0% |
| **Precision** | 100.0% |
| **Recall** | 100.0% |
| **ROC-AUC** | 100.0% |
| **3-Fold CV Accuracy** | 99.75% ± 0.35% |

### Real Datasets (expected)
| Dataset | Expected Accuracy |
|---|---|
| WELFake (72k) | ~96–98% |
| ISOT (44k) | ~98–99% |

---

## 🛠️ Tech Stack

| Category | Libraries |
|---|---|
| **ML / AI** | scikit-learn — LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, VotingClassifier, CalibratedClassifierCV |
| **NLP** | NLTK (lemmatization, stopwords), TF-IDF Vectorizer |
| **Web Scraping** | trafilatura, BeautifulSoup4, requests, lxml |
| **UI** | Streamlit, Plotly |
| **Data** | pandas, numpy |
| **Persistence** | joblib |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/AI-Fake-News-Detector.git
cd AI-Fake-News-Detector
pip install -r requirements.txt

# Create a feature branch
git checkout -b feature/your-feature-name

# Make changes, then commit
git add .
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

Then open a **Pull Request** on GitHub. Ideas for contributions:
- 🔤 Add transformer-based model (BERT, RoBERTa)
- 🌍 Multi-language support
- 📰 News API integration (live fact-checking)
- 🧪 More dataset integrations

---

## 📄 License

MIT © 2026 [Atulframez](https://github.com/Atulframez)

<!-- last reviewed: 2026-05-02 -->
