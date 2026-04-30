"""
AI Fake News Detector — Real-World Streamlit App
Ensemble ML + URL Scraping + Explainability + Batch Analysis
"""

import streamlit as st
import os, sys, re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.detector import FakeNewsDetector, MODEL_PATH, VECTORIZER_PATH
from model.scraper import ArticleScraper
from sample_data.generate_sample_data import generate_dataset

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; }

.main-header { text-align:center; padding:2rem 0 0.5rem; }
.main-header h1 {
    background: linear-gradient(135deg,#6C63FF,#FF6584,#FFBE0B);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:3rem; font-weight:800; margin-bottom:0.3rem;
}
.main-header p { color:#8B8D97; font-size:1.05rem; }

.badge {
    display:inline-block; padding:0.25rem 0.75rem; border-radius:20px;
    font-size:0.75rem; font-weight:600; margin:0.2rem;
}
.badge-purple { background:rgba(108,99,255,0.15); color:#8B85FF; border:1px solid rgba(108,99,255,0.3); }
.badge-green  { background:rgba(0,200,117,0.12); color:#00C875; border:1px solid rgba(0,200,117,0.3); }
.badge-red    { background:rgba(255,71,87,0.12); color:#FF4757; border:1px solid rgba(255,71,87,0.3); }

.verdict-card {
    border-radius:16px; padding:1.8rem; text-align:center;
    margin:1rem 0; backdrop-filter:blur(10px);
}
.verdict-real { background:linear-gradient(135deg,rgba(0,200,117,0.15),rgba(0,200,117,0.04)); border:2px solid rgba(0,200,117,0.4); }
.verdict-fake { background:linear-gradient(135deg,rgba(255,71,87,0.15),rgba(255,71,87,0.04)); border:2px solid rgba(255,71,87,0.4); }
.verdict-label { font-size:2.2rem; font-weight:800; }
.real-text { color:#00C875; } .fake-text { color:#FF4757; }
.conf-text { color:#8B8D97; font-size:1rem; margin-top:0.3rem; }

.metric-card {
    background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.2);
    border-radius:12px; padding:1.1rem; text-align:center;
}
.metric-value { font-size:1.6rem; font-weight:700; color:#6C63FF; }
.metric-label { font-size:0.8rem; color:#8B8D97; margin-top:0.2rem; }

.keyword-positive {
    display:inline-block; padding:0.3rem 0.7rem; margin:0.2rem;
    border-radius:8px; font-size:0.8rem; font-weight:500;
    background:rgba(255,71,87,0.12); color:#FF4757; border:1px solid rgba(255,71,87,0.25);
}
.keyword-neutral {
    display:inline-block; padding:0.3rem 0.7rem; margin:0.2rem;
    border-radius:8px; font-size:0.8rem; font-weight:500;
    background:rgba(0,200,117,0.1); color:#00C875; border:1px solid rgba(0,200,117,0.25);
}

.model-vote {
    display:inline-block; padding:0.4rem 1rem; border-radius:8px;
    margin:0.25rem; font-size:0.82rem; font-weight:600;
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1);
}

.history-item {
    background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:0.8rem 1rem; margin:0.4rem 0;
}
.sidebar-info {
    background:rgba(108,99,255,0.08); border-radius:10px;
    padding:0.9rem; margin:0.5rem 0;
}
.stTextArea textarea {
    border-radius:12px !important; border:2px solid rgba(108,99,255,0.3) !important;
}
.stTextArea textarea:focus {
    border-color:#6C63FF !important; box-shadow:0 0 0 3px rgba(108,99,255,0.15) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────
for key, default in [
    ('detector', None), ('trained', False), ('metrics', None),
    ('history', []), ('scraper', ArticleScraper()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

if st.session_state.detector is None:
    st.session_state.detector = FakeNewsDetector()

# ── Auto-load saved model ──────────────────────────────────────────────
if not st.session_state.trained and os.path.exists(MODEL_PATH):
    try:
        st.session_state.detector._load()
        st.session_state.metrics = st.session_state.detector.meta
        st.session_state.trained = True
    except Exception:
        pass


# ── Sidebar ───────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        if st.session_state.trained:
            m = st.session_state.metrics or {}
            st.success("✅ Model Ready")
            st.markdown(f"""
            <div class="sidebar-info">
            <b>Test Accuracy:</b> {m.get('test_accuracy','—')}%<br>
            <b>F1 Score:</b> {m.get('f1_score','—')}%<br>
            <b>CV Accuracy:</b> {m.get('cv_mean','—')} ± {m.get('cv_std','—')}%<br>
            <b>ROC-AUC:</b> {m.get('roc_auc','—')}%<br>
            <b>Samples:</b> {m.get('total_samples','—'):,}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Model not trained yet")

        st.markdown("---")
        st.markdown("### 🎯 Train Model")

        data_src = st.radio("Data Source", [
            "Built-in Sample Data",
            "Upload Real Dataset (CSV)",
            "WELFake / ISOT Instructions",
        ])

        if data_src == "Built-in Sample Data":
            if st.button("🚀 Train on Sample Data", use_container_width=True):
                with st.spinner("Generating & training..."):
                    try:
                        path = generate_dataset()
                        metrics = st.session_state.detector.train(data_path=path)
                        st.session_state.trained = True
                        st.session_state.metrics = metrics
                        st.success(f"✅ Accuracy: {metrics['test_accuracy']}%")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training failed: {e}")

        elif data_src == "Upload Real Dataset (CSV)":
            st.markdown("""
            <div class="sidebar-info">
            <small>Upload a CSV with <b>text</b> and <b>label</b> columns.<br>
            Compatible with WELFake, ISOT, and custom datasets.</small>
            </div>
            """, unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            if uploaded:
                if st.button("🚀 Train on Uploaded Data", use_container_width=True):
                    with st.spinner("Training ensemble model..."):
                        try:
                            df = pd.read_csv(uploaded)
                            metrics = st.session_state.detector.train(df=df)
                            st.session_state.trained = True
                            st.session_state.metrics = metrics
                            st.success(f"✅ Accuracy: {metrics['test_accuracy']}%")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Training failed: {e}")

        else:
            st.markdown("""
            <div class="sidebar-info">
            <b>🔗 Real Datasets:</b><br><small>
            • <b>WELFake</b> — 72k articles<br>
              kaggle.com/datasets/saurabhshahane/fake-news-classification<br><br>
            • <b>ISOT</b> — 44k articles<br>
              uvic.ca/engineering/ece/isot/datasets<br><br>
            Download CSV → Upload above
            </small></div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
        <b>🧠 Architecture</b><br><small>
        • TF-IDF (50k features, bigrams)<br>
        • Ensemble: LR + SGD + PassiveAgg<br>
        • Soft-voting with calibrated probs<br>
        • Keyword explainability (TF-IDF weights)<br>
        • URL article scraping (trafilatura)
        </small></div>
        """, unsafe_allow_html=True)

        if st.session_state.history:
            st.markdown("---")
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()


# ── Verdict Card ──────────────────────────────────────────────────────
def render_verdict(result: dict, article_title: str = ""):
    verdict = result['verdict']
    is_fake = result['is_fake']
    conf = result['confidence']

    css = "verdict-fake" if is_fake else "verdict-real"
    tc  = "fake-text"   if is_fake else "real-text"
    emoji = "🚨" if is_fake else "✅"

    title_html = f"<div style='color:#8B8D97;font-size:0.85rem;margin-bottom:0.5rem;'>{article_title[:120]}</div>" if article_title else ""

    st.markdown(f"""
    <div class="verdict-card {css}">
        {title_html}
        <div class="verdict-label {tc}">{emoji} {verdict} NEWS</div>
        <div class="conf-text">Confidence: <b>{conf}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    # Probability donut
    fig = go.Figure(go.Pie(
        values=[result['fake_prob'], result['real_prob']],
        labels=['FAKE', 'REAL'],
        hole=0.65,
        marker_colors=['#FF4757', '#00C875'],
        textinfo='label+percent',
        textfont=dict(color='#FAFAFA', size=13),
    ))
    fig.update_layout(
        height=240, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        annotations=[dict(
            text=f"<b>{conf}%</b>",
            x=0.5, y=0.5, font=dict(size=20, color='#FAFAFA'), showarrow=False
        )]
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Text Analysis Metrics ─────────────────────────────────────────────
def render_features(features: dict):
    st.markdown("#### 📊 Text Analysis")
    items = [
        ("📝 Words",          features['word_count']),
        ("📏 Avg Word Len",   features['avg_word_length']),
        ("❗ Exclamations",   features['exclamation_count']),
        ("🔠 CAPS Ratio",     f"{features['capital_ratio']:.1%}"),
        ("🎭 Sensational",    features['sensational_score']),
        ("✅ Credibility",    features['credibility_score']),
        ("📖 Lexical Div.",   f"{features['lexical_diversity']:.2f}"),
        ("🔗 URLs",           features['url_count']),
    ]
    cols = st.columns(4)
    for i, (label, value) in enumerate(items):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div><br>
            """, unsafe_allow_html=True)


# ── Keyword Explainability ────────────────────────────────────────────
def render_keywords(keywords: list, is_fake: bool):
    if not keywords:
        return
    st.markdown("#### 🔑 Key Contributing Words")
    st.caption("Words that most influenced this prediction (based on TF-IDF model weights)")
    html = ""
    for kw in keywords:
        css = "keyword-positive" if kw['positive'] else "keyword-neutral"
        html += f'<span class="{css}">{kw["word"]} ({kw["score"]:.3f})</span>'
    st.markdown(html, unsafe_allow_html=True)


# ── Model Votes ───────────────────────────────────────────────────────
def render_model_votes(votes: dict):
    if not votes:
        return
    st.markdown("#### 🗳️ Model Agreement")
    html = ""
    for model, vote in votes.items():
        color = "#FF4757" if vote == "FAKE" else "#00C875"
        html += f'<span class="model-vote" style="color:{color};">{model}: <b>{vote}</b></span>'
    st.markdown(html, unsafe_allow_html=True)


# ── Performance Charts ────────────────────────────────────────────────
def render_performance():
    m = st.session_state.metrics
    if not m:
        return
    st.markdown("---")
    st.markdown("### 📈 Model Performance")

    col1, col2, col3, col4 = st.columns(4)
    for col, (label, val, suffix) in zip(
        [col1, col2, col3, col4],
        [
            ("Test Accuracy", m.get('test_accuracy', 0), "%"),
            ("F1 Score",      m.get('f1_score', 0), "%"),
            ("Precision",     m.get('precision', 0), "%"),
            ("Recall",        m.get('recall', 0), "%"),
        ]
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}{suffix}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        cm = m.get('confusion_matrix', [[0,0],[0,0]])
        labels = m.get('labels', ['FAKE', 'REAL'])
        fig = px.imshow(
            cm, x=labels, y=labels,
            labels=dict(x="Predicted", y="Actual"),
            color_continuous_scale=[[0,'#1A1D23'],[1,'#6C63FF']],
            text_auto=True,
        )
        fig.update_layout(
            title="Confusion Matrix", height=320,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#FAFAFA'},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        metrics_bar = {
            'Metric': ['Train Acc', 'Test Acc', 'F1 Score', 'Precision', 'Recall'],
            'Value': [
                m.get('training_accuracy', 0), m.get('test_accuracy', 0),
                m.get('f1_score', 0), m.get('precision', 0), m.get('recall', 0),
            ]
        }
        fig = go.Figure(go.Bar(
            x=metrics_bar['Metric'], y=metrics_bar['Value'],
            marker_color=['#6C63FF','#FF6584','#FFBE0B','#00C875','#FF9F43'],
            text=[f"{v}%" for v in metrics_bar['Value']],
            textposition='outside', textfont={'color':'#FAFAFA'},
        ))
        fig.update_layout(
            title="All Metrics", yaxis_range=[0, 108], height=320,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font={'color':'#FAFAFA'}, showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # CV score callout
    cv_m = m.get('cv_mean')
    cv_s = m.get('cv_std')
    if cv_m:
        st.info(f"📐 **3-Fold Cross-Validation Accuracy:** {cv_m}% ± {cv_s}%")


# ── History Panel ─────────────────────────────────────────────────────
def render_history():
    if not st.session_state.history:
        return
    st.markdown("---")
    st.markdown("### 🕓 Analysis History")
    for i, entry in enumerate(reversed(st.session_state.history[-10:])):
        verdict = entry['verdict']
        conf    = entry['confidence']
        snippet = entry['text'][:100] + "..." if len(entry['text']) > 100 else entry['text']
        color   = "#FF4757" if entry['is_fake'] else "#00C875"
        emoji   = "🚨" if entry['is_fake'] else "✅"
        source  = entry.get('source', 'text')
        st.markdown(f"""
        <div class="history-item">
            <span style="color:{color};font-weight:700;">{emoji} {verdict}</span>
            <span style="color:#8B8D97;font-size:0.8rem;"> · {conf}% confidence · via {source}</span><br>
            <span style="font-size:0.85rem;color:#CCC;">{snippet}</span>
        </div>
        """, unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Fake News Detector</h1>
        <p>Ensemble ML · URL Scraping · Keyword Explainability · Real-World Ready</p>
        <div>
            <span class="badge badge-purple">TF-IDF + LR + SGD + PAC</span>
            <span class="badge badge-green">Soft-Voting Ensemble</span>
            <span class="badge badge-red">Live URL Scraping</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Article", "📋 Batch Analysis", "📈 Model Performance"])

    # ── Tab 1: Single Article ──────────────────────────────────────────
    with tab1:
        input_mode = st.radio(
            "Input Method",
            ["✍️ Paste Text", "🌐 Analyze from URL"],
            horizontal=True,
        )

        article_text = ""
        article_title = ""
        source_label = "text"

        if input_mode == "✍️ Paste Text":
            samples = {
                "Select a sample...": "",
                "🟢 Real: Fed Rate Hike": (
                    "The Federal Reserve announced a 0.25 percentage point increase in "
                    "interest rates, citing continued inflation concerns and strong labor "
                    "market data. Chair Jerome Powell emphasized the decision was data-driven "
                    "and consistent with the central bank's dual mandate of price stability."
                ),
                "🔴 Fake: 5G Conspiracy": (
                    "BREAKING!! Scientists confirm that 5G towers are SPREADING a new virus "
                    "affecting brain cells! Thousands of people living near towers report "
                    "severe memory loss. The government is COVERING IT UP! Share before deleted!"
                ),
                "🟢 Real: NASA Moon Mission": (
                    "NASA's Artemis program successfully completed its lunar orbit mission, "
                    "with astronauts conducting a range of scientific experiments aboard the "
                    "Orion spacecraft. Mission scientists published findings in Nature."
                ),
            }
            selected = st.selectbox("Try a sample:", list(samples.keys()))
            article_text = st.text_area(
                "Paste or type news article text:",
                value=samples.get(selected, ""),
                height=200,
                placeholder="Paste a full news article or headline here...",
            )
            source_label = "text"

        else:
            url_input = st.text_input(
                "Enter article URL:",
                placeholder="https://www.bbc.com/news/science-environment-...",
            )
            if url_input and st.button("🌐 Fetch Article", use_container_width=False):
                with st.spinner("Scraping article..."):
                    scraped = st.session_state.scraper.scrape(url_input)
                if scraped['success']:
                    st.success(f"✅ Extracted {scraped['word_count']:,} words via {scraped['method']} from **{scraped['domain']}**")
                    article_text  = scraped['text']
                    article_title = scraped['title']
                    source_label  = scraped['domain']
                    st.text_area("Extracted text (preview):", value=article_text[:600] + "...", height=130, disabled=True)
                else:
                    st.error(f"❌ Scraping failed: {scraped['error']}")

        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            analyze_btn = st.button("🔍 Analyze Now", use_container_width=True, type="primary")

        if analyze_btn:
            if not article_text.strip():
                st.warning("Please provide some text or fetch a URL first.")
            elif not st.session_state.trained:
                st.error("⚠️ Please train the model first using the sidebar.")
            else:
                with st.spinner("Analyzing with ensemble model..."):
                    result = st.session_state.detector.predict(article_text)

                # Store in history
                st.session_state.history.append({
                    **result,
                    'text': article_text,
                    'source': source_label,
                    'title': article_title,
                })

                col_v, col_k = st.columns([1, 1])
                with col_v:
                    render_verdict(result, article_title)
                    render_model_votes(result.get('model_votes', {}))
                with col_k:
                    render_keywords(result.get('keywords', []), result['is_fake'])
                    st.markdown("<br>", unsafe_allow_html=True)
                    render_features(result['features'])

        render_history()

    # ── Tab 2: Batch Analysis ──────────────────────────────────────────
    with tab2:
        st.markdown("### 📋 Batch Analysis")
        st.caption("Paste multiple articles (one per line, separated by blank lines) or upload a CSV.")

        batch_mode = st.radio("Batch Input", ["📝 Text (one per line)", "📂 Upload CSV"], horizontal=True)

        if batch_mode == "📝 Text (one per line)":
            raw_batch = st.text_area(
                "Enter articles (separate with blank lines):",
                height=250,
                placeholder="Article 1 text...\n\nArticle 2 text...\n\nArticle 3 text...",
            )
            if st.button("🔍 Analyze Batch", type="primary"):
                if not st.session_state.trained:
                    st.error("Train the model first.")
                elif not raw_batch.strip():
                    st.warning("Enter at least one article.")
                else:
                    articles = [a.strip() for a in re.split(r'\n\s*\n', raw_batch) if a.strip()]
                    with st.spinner(f"Analyzing {len(articles)} articles..."):
                        results = st.session_state.detector.predict_batch(articles)
                    _render_batch_results(articles, results)

        else:
            batch_file = st.file_uploader("Upload CSV (must have 'text' column)", type=['csv'])
            if batch_file and st.button("🔍 Analyze CSV", type="primary"):
                if not st.session_state.trained:
                    st.error("Train the model first.")
                else:
                    df_batch = pd.read_csv(batch_file)
                    if 'text' not in df_batch.columns:
                        st.error("CSV must have a 'text' column.")
                    else:
                        texts = df_batch['text'].dropna().tolist()
                        with st.spinner(f"Analyzing {len(texts)} articles..."):
                            results = st.session_state.detector.predict_batch(texts)
                        _render_batch_results(texts, results)
                        # Download results
                        out_df = pd.DataFrame([{
                            'text': t[:200],
                            'verdict': r['verdict'],
                            'confidence': r['confidence'],
                            'fake_prob': r['fake_prob'],
                            'real_prob': r['real_prob'],
                        } for t, r in zip(texts, results)])
                        st.download_button(
                            "⬇️ Download Results CSV",
                            out_df.to_csv(index=False).encode(),
                            "fake_news_results.csv", "text/csv",
                        )

    # ── Tab 3: Performance ─────────────────────────────────────────────
    with tab3:
        if st.session_state.trained:
            render_performance()
        else:
            st.info("Train the model first to see performance metrics.")


def _render_batch_results(articles, results):
    fake_count = sum(1 for r in results if r['is_fake'])
    real_count = len(results) - fake_count

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Articles", len(results))
    c2.metric("🚨 Fake", fake_count)
    c3.metric("✅ Real", real_count)

    # Pie
    fig = go.Figure(go.Pie(
        values=[fake_count, real_count], labels=['FAKE', 'REAL'],
        marker_colors=['#FF4757', '#00C875'], hole=0.5,
    ))
    fig.update_layout(
        height=250, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10,b=10,l=10,r=10), font={'color':'#FAFAFA'},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    rows = []
    for i, (article, result) in enumerate(zip(articles, results), 1):
        rows.append({
            '#': i,
            'Snippet': article[:80] + '...' if len(article) > 80 else article,
            'Verdict': result['verdict'],
            'Confidence': f"{result['confidence']}%",
            'FAKE Prob': f"{result['fake_prob']}%",
            'REAL Prob': f"{result['real_prob']}%",
        })
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
