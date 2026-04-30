# 🔍 AI Fake News Detector

An AI-powered tool that uses **Natural Language Processing (NLP)** and **Machine Learning** to detect fake news articles in real-time.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45-red?logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikit-learn)


## ✨ Features

- **Real-time Detection** — Paste any news text and get instant REAL/FAKE classification
- **Confidence Score** — See how confident the model is with an interactive gauge
- **Text Analysis** — View word count, caps ratio, exclamation usage, and more
- **Interactive Dashboard** — Beautiful dark-themed UI with charts and metrics
- **Train Your Own Model** — Use built-in sample data or upload your own CSV
- **Model Performance Visualization** — Confusion matrix and accuracy charts

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Atulframez/AI-Fake-News-Detector.git
cd AI-Fake-News-Detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

## 🧠 How It Works

1. **Text Preprocessing** — Cleans text (removes URLs, HTML, stopwords) and lemmatizes
2. **TF-IDF Vectorization** — Extracts important word features using TF-IDF (up to bigrams)
3. **ML Classification** — PassiveAggressiveClassifier predicts REAL vs FAKE
4. **Confidence Scoring** — Sigmoid-based confidence from decision function scores

## 📁 Project Structure

```
AI-Fake-News-Detector/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── model/
│   ├── __init__.py
│   ├── preprocessor.py             # Text cleaning & feature extraction
│   └── detector.py                 # ML model (train/predict/save/load)
├── sample_data/
│   └── generate_sample_data.py     # Generates training dataset
├── .streamlit/
│   └── config.toml                 # UI theme configuration
└── README.md
```

## 📊 Model Details

| Component | Details |
|-----------|---------|
| Algorithm | PassiveAggressiveClassifier |
| Features | TF-IDF (max 10k features, bigrams) |
| Preprocessing | Lemmatization, stopword removal |
| Training Data | 400 balanced samples (expandable) |

## 🤝 Contributing

Pull requests are welcome! Feel free to open issues for bugs or feature requests.


---

⭐ **Star this repo** if you find it useful!

