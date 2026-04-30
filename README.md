# 🔍 AI Fake News Detector

> **Real-world NLP + ML project** — detects fake news using an ensemble of three classifiers, TF-IDF feature engineering, URL article scraping, and keyword-level explainability.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

| Feature | Description |
|---|---|
| 🤖 **Ensemble ML** | Soft-voting: Logistic Regression + SGD + Passive-Aggressive |
| 📐 **TF-IDF (50k)** | Unigrams + bigrams, sublinear TF, balanced class weights |
| 🌐 **URL Scraping** | Paste any news URL — extracts article text via trafilatura |
| 🔑 **Explainability** | Top keywords that drove the prediction (TF-IDF weight analysis) |
| 📋 **Batch Analysis** | Analyze multiple articles at once, download results as CSV |
| 📊 **Rich Metrics** | Accuracy, F1, Precision, Recall, ROC-AUC, Cross-Validation |
| 🕓 **History** | In-session analysis history with source tracking |
| 📁 **Real Datasets** | Supports WELFake (72k), ISOT (44k), or any labeled CSV |

---

## 🏗️ Architecture

```
Raw Text / URL
      │
      ▼
ArticleScraper (trafilatura + BeautifulSoup)
      │
      ▼
TextPreprocessor (clean → lemmatize → feature extraction)
      │
      ▼
TF-IDF Vectorizer (50,000 features, bigrams, sublinear TF)
      │
      ▼
Voting Ensemble
  ├── CalibratedLR (isotonic)
  ├── CalibratedSGD (modified_huber)
  └── CalibratedPAC
      │
      ▼
Soft-voted probability → Verdict + Confidence + Keyword Explanation
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/AI-Fake-News-Detector.git
cd AI-Fake-News-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

Then open **http://localhost:8501** and click **"Train on Sample Data"** in the sidebar.

---

## 📁 Project Structure

```
AI-Fake-News-Detector/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── model/
│   ├── detector.py                 # Ensemble ML model
│   ├── preprocessor.py             # NLP preprocessing + features
│   └── scraper.py                  # URL article extractor
├── sample_data/
│   └── generate_sample_data.py     # Built-in training data generator
├── saved_model/                    # Auto-created after training
│   ├── ensemble_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_meta.pkl
└── .streamlit/
    └── config.toml                 # Dark theme config
```

---

## 🗂️ Using Real Datasets

For production-level accuracy, use one of these free datasets:

| Dataset | Size | Link |
|---|---|---|
| **WELFake** | 72,134 articles | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| **ISOT** | 44,898 articles | [UVic](https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php) |
| **LIAR** | 12,836 statements | [GitHub](https://github.com/thiagorainmaker77/liar_dataset) |

Download → upload via **"Upload Real Dataset (CSV)"** in the sidebar.

---

## 🧪 Model Performance (Sample Data)

| Metric | Score |
|---|---|
| Test Accuracy | ~85–92% |
| F1 Score | ~85–92% |
| Cross-Validation | ±2–3% std |

*Performance scales significantly with real datasets (WELFake: ~97%+ accuracy)*

---

## 🛠️ Tech Stack

- **ML**: scikit-learn (LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, VotingClassifier)
- **NLP**: NLTK (lemmatization, stopwords), TF-IDF
- **Scraping**: trafilatura, BeautifulSoup4, requests
- **UI**: Streamlit, Plotly
- **Persistence**: joblib

---

## 📄 License

MIT © 2025
