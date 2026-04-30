"""
AI Fake News Detector — Streamlit Web App
"""

import streamlit as st
import os
import sys
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.detector import FakeNewsDetector
from sample_data.generate_sample_data import generate_dataset

# ── Page Config ──
st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif; }

.main-header {
    text-align: center;
    padding: 2rem 0 1rem;
}
.main-header h1 {
    background: linear-gradient(135deg, #6C63FF, #FF6584);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.main-header p {
    color: #8B8D97;
    font-size: 1.1rem;
}

.verdict-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
}
.verdict-real {
    background: linear-gradient(135deg, rgba(0,200,117,0.15), rgba(0,200,117,0.05));
    border: 2px solid rgba(0,200,117,0.4);
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,71,87,0.15), rgba(255,71,87,0.05));
    border: 2px solid rgba(255,71,87,0.4);
}
.verdict-label {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}
.real-text { color: #00C875; }
.fake-text { color: #FF4757; }

.confidence-text {
    font-size: 1.2rem;
    color: #8B8D97;
}

.metric-card {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #6C63FF;
}
.metric-label {
    font-size: 0.85rem;
    color: #8B8D97;
    margin-top: 0.3rem;
}

.feature-pill {
    display: inline-block;
    background: rgba(108,99,255,0.1);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 20px;
    padding: 0.4rem 1rem;
    margin: 0.3rem;
    font-size: 0.85rem;
}

.stTextArea textarea {
    border-radius: 12px !important;
    border: 2px solid rgba(108,99,255,0.3) !important;
    font-size: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.15) !important;
}

.sidebar-info {
    background: rgba(108,99,255,0.08);
    border-radius: 12px;
    padding: 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ──
if 'detector' not in st.session_state:
    st.session_state.detector = FakeNewsDetector()
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 AI Fake News Detector</h1>
        <p>Powered by NLP & Machine Learning — Detect misinformation instantly</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        st.markdown("---")

        # Model Status
        if st.session_state.trained:
            st.success("✅ Model is trained and ready!")
        else:
            st.warning("⚠️ Model needs training")

        st.markdown("---")
        st.markdown("### 🎯 Train Model")

        data_source = st.radio(
            "Data Source",
            ["Built-in Sample Data", "Upload CSV"],
            help="Choose training data source"
        )

        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload CSV (text, label columns)",
                type=['csv']
            )
            if uploaded and st.button("🚀 Train on Uploaded Data", use_container_width=True):
                with st.spinner("Training model..."):
                    df = pd.read_csv(uploaded)
                    metrics = st.session_state.detector.train(df=df)
                    st.session_state.trained = True
                    st.session_state.metrics = metrics
                    st.success(f"Accuracy: {metrics['test_accuracy']}%")
                    st.rerun()
        else:
            if st.button("🚀 Train on Sample Data", use_container_width=True):
                with st.spinner("Generating dataset & training..."):
                    data_path = generate_dataset()
                    metrics = st.session_state.detector.train(data_path=data_path)
                    st.session_state.trained = True
                    st.session_state.metrics = metrics
                    st.success(f"Accuracy: {metrics['test_accuracy']}%")
                    st.rerun()

        # Show metrics if trained
        if st.session_state.metrics:
            st.markdown("---")
            st.markdown("### 📊 Model Metrics")
            m = st.session_state.metrics
            st.markdown(f"""
            <div class="sidebar-info">
                <b>Train Accuracy:</b> {m['training_accuracy']}%<br>
                <b>Test Accuracy:</b> {m['test_accuracy']}%<br>
                <b>Train Size:</b> {m['train_size']}<br>
                <b>Test Size:</b> {m['test_size']}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div class="sidebar-info">
            <b>🧠 How it works</b><br>
            <small>
            1. Text is cleaned & preprocessed<br>
            2. TF-IDF extracts key features<br>
            3. ML classifier makes prediction<br>
            4. Confidence score is calculated
            </small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#555;font-size:0.8rem;'>"
            "Built with ❤️ using Streamlit & scikit-learn</p>",
            unsafe_allow_html=True
        )


def render_prediction(result):
    verdict = result['verdict']
    confidence = result['confidence']
    features = result['features']
    is_fake = result['is_fake']

    css_class = "verdict-fake" if is_fake else "verdict-real"
    text_class = "fake-text" if is_fake else "real-text"
    emoji = "🚨" if is_fake else "✅"

    st.markdown(f"""
    <div class="verdict-card {css_class}">
        <div class="verdict-label {text_class}">{emoji} {verdict} NEWS</div>
        <div class="confidence-text">Confidence: {confidence}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence Score", 'font': {'size': 16, 'color': '#FAFAFA'}},
        number={'suffix': '%', 'font': {'color': '#FAFAFA'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#555'},
            'bar': {'color': '#FF4757' if is_fake else '#00C875'},
            'bgcolor': '#1A1D23',
            'bordercolor': '#333',
            'steps': [
                {'range': [0, 50], 'color': 'rgba(255,255,255,0.03)'},
                {'range': [50, 75], 'color': 'rgba(255,255,255,0.05)'},
                {'range': [75, 100], 'color': 'rgba(255,255,255,0.08)'},
            ],
        }
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=40, b=0, l=30, r=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#FAFAFA'},
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature Analysis
    st.markdown("### 📋 Text Analysis")
    cols = st.columns(4)
    feature_items = [
        ("📝 Words", features['word_count']),
        ("📏 Avg Length", features['avg_word_length']),
        ("❗ Exclamations", features['exclamation_count']),
        ("🔠 CAPS Ratio", f"{features['capital_ratio']:.1%}"),
    ]
    for col, (label, value) in zip(cols, feature_items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_model_performance():
    if not st.session_state.metrics:
        return

    st.markdown("---")
    st.markdown("### 📈 Model Performance")
    m = st.session_state.metrics

    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        cm = m['confusion_matrix']
        labels = ['FAKE', 'REAL']
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            color_continuous_scale=[[0, '#1A1D23'], [1, '#6C63FF']],
            text_auto=True,
        )
        fig.update_layout(
            title="Confusion Matrix",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FAFAFA'},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Accuracy comparison bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Training', 'Testing'],
            y=[m['training_accuracy'], m['test_accuracy']],
            marker_color=['#6C63FF', '#FF6584'],
            text=[f"{m['training_accuracy']}%", f"{m['test_accuracy']}%"],
            textposition='outside',
            textfont={'color': '#FAFAFA'},
        ))
        fig.update_layout(
            title="Accuracy Comparison",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 105],
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#FAFAFA'},
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    render_header()
    render_sidebar()

    # Check if model is already saved
    from model.detector import MODEL_PATH, VECTORIZER_PATH
    if not st.session_state.trained and os.path.exists(MODEL_PATH):
        try:
            st.session_state.detector._load()
            st.session_state.trained = True
        except Exception:
            pass

    # Main input area
    st.markdown("### 📰 Enter News Text to Analyze")

    sample_texts = {
        "Select a sample...": "",
        "🟢 Real News Example": (
            "The Federal Reserve announced a 0.25 percentage point increase "
            "in interest rates, citing continued inflation concerns. Chair "
            "Powell emphasized the decision was data-driven."
        ),
        "🔴 Fake News Example": (
            "BREAKING: Scientists confirm that 5G towers spread a new virus "
            "affecting brain cells! Thousands report memory loss near towers. "
            "The government is covering it up!"
        ),
    }

    selected = st.selectbox("Try a sample:", list(sample_texts.keys()))
    default_text = sample_texts.get(selected, "")

    news_text = st.text_area(
        "Paste or type news article text below:",
        value=default_text,
        height=180,
        placeholder="Enter news text here to check if it's real or fake...",
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_btn = st.button(
            "🔍 Analyze News",
            use_container_width=True,
            type="primary",
        )

    if analyze_btn:
        if not news_text.strip():
            st.error("Please enter some text to analyze.")
        elif not st.session_state.trained:
            st.error("Please train the model first using the sidebar.")
        else:
            with st.spinner("Analyzing..."):
                result = st.session_state.detector.predict(news_text)
            render_prediction(result)

    render_model_performance()


if __name__ == "__main__":
    main()
