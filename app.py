from pathlib import Path
import pickle

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --bg-1: #060816;
        --bg-2: #0f172a;
        --accent-1: #7c3aed;
        --accent-2: #22d3ee;
        --accent-3: #34d399;
        --text: #ecf3ff;
        --muted: #97a6c6;
        --card: rgba(15, 23, 42, 0.62);
        --line: rgba(255,255,255,0.10);
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 20%, rgba(124, 58, 237, 0.28), transparent 22%),
            radial-gradient(circle at 80% 12%, rgba(34, 211, 238, 0.22), transparent 20%),
            radial-gradient(circle at 50% 78%, rgba(52, 211, 153, 0.12), transparent 24%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 1.2rem;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--line);
        background: linear-gradient(135deg, rgba(10,16,33,0.8), rgba(12,20,44,0.55));
        box-shadow: 0 25px 90px rgba(0,0,0,0.35);
        border-radius: 28px;
        padding: 34px 34px 26px 34px;
        backdrop-filter: blur(16px);
    }

    .hero-shell::before,
    .hero-shell::after {
        content: "";
        position: absolute;
        inset: auto;
        border-radius: 999px;
        filter: blur(10px);
        opacity: 0.9;
        animation: floatOrb 7s ease-in-out infinite;
    }

    .hero-shell::before {
        width: 220px;
        height: 220px;
        top: -90px;
        right: -50px;
        background: radial-gradient(circle at 30% 30%, rgba(34,211,238,0.9), rgba(34,211,238,0.06) 62%);
    }

    .hero-shell::after {
        width: 180px;
        height: 180px;
        bottom: -70px;
        left: -50px;
        background: radial-gradient(circle at 30% 30%, rgba(124,58,237,0.95), rgba(124,58,237,0.06) 62%);
        animation-delay: -2.5s;
    }

    @keyframes floatOrb {
        0%, 100% { transform: translateY(0px) translateX(0px) scale(1); }
        50% { transform: translateY(14px) translateX(8px) scale(1.05); }
    }

    .eyebrow {
        display: inline-block;
        color: #7dd3fc;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }

    .hero-title {
        font-size: clamp(2.5rem, 4vw, 4.4rem);
        line-height: 1.02;
        font-weight: 800;
        margin: 0;
        color: white;
        max-width: 10ch;
    }

    .hero-title .gradient {
        background: linear-gradient(135deg, #7dd3fc, #c084fc 52%, #86efac);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }

    .hero-subtitle {
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 64ch;
        margin-top: 1rem;
        margin-bottom: 0;
    }

    .glass-card {
        border: 1px solid var(--line);
        background: var(--card);
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
        border-radius: 24px;
        padding: 22px;
        backdrop-filter: blur(18px);
        height: 100%;
    }

    .mini-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 1.3rem;
    }

    .metric {
        padding: 14px;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .metric strong {
        display: block;
        font-size: 1rem;
        color: white;
        margin-bottom: 0.35rem;
    }

    .metric span {
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .panel-title {
        color: white;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .panel-copy {
        color: var(--muted);
        line-height: 1.65;
        margin-bottom: 1rem;
    }

    .pulse-dot {
        width: 12px;
        height: 12px;
        display: inline-block;
        border-radius: 999px;
        background: #34d399;
        box-shadow: 0 0 0 rgba(52, 211, 153, 0.6);
        animation: pulse 1.8s infinite;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }
        70% { box-shadow: 0 0 0 12px rgba(52, 211, 153, 0); }
        100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.06);
        color: white;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.12);
        height: 3.2rem;
        font-size: 1rem;
    }

    .stButton > button {
        width: 100%;
        border: 0;
        min-height: 3.25rem;
        font-size: 1rem;
        font-weight: 700;
        border-radius: 14px;
        color: #07111f;
        background: linear-gradient(135deg, #67e8f9, #c084fc);
        box-shadow: 0 16px 45px rgba(103, 232, 249, 0.22);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 50px rgba(103, 232, 249, 0.28);
    }

    .result-card {
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(10,16,33,0.92), rgba(16,24,48,0.8));
        border: 1px solid rgba(125, 211, 252, 0.18);
        padding: 26px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(0,0,0,0.24);
        animation: riseIn 0.45s ease;
    }

    @keyframes riseIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-label {
        color: #7dd3fc;
        text-transform: uppercase;
        letter-spacing: 0.16em;
        font-size: 0.76rem;
        font-weight: 700;
    }

    .result-word {
        font-size: clamp(1.7rem, 3vw, 2.8rem);
        font-weight: 800;
        margin: 0.65rem 0 0.25rem;
        color: #86efac;
    }

    .sentence-preview {
        color: var(--muted);
        line-height: 1.7;
        font-size: 1rem;
    }

    .tip-card {
        margin-top: 1rem;
        padding: 16px 18px;
        border-radius: 18px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--muted);
        line-height: 1.65;
    }

    .footer-note {
        text-align: center;
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_assets():
    missing = [
        path.name for path in (MODEL_PATH, TOKENIZER_PATH, MAX_LEN_PATH) if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required model assets: " + ", ".join(missing)
        )

    model = load_model(MODEL_PATH)
    with TOKENIZER_PATH.open("rb") as f:
        tokenizer = pickle.load(f)
    with MAX_LEN_PATH.open("rb") as f:
        max_len = pickle.load(f)

    if not isinstance(max_len, int):
        max_len = int(max_len)

    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    return model, tokenizer, max_len, index_to_word


def predict_next_word(text: str, model, tokenizer, max_len: int, index_to_word: dict) -> str:
    normalized_text = " ".join(text.strip().split())
    if not normalized_text:
        raise ValueError("Please enter some text.")

    sequence = tokenizer.texts_to_sequences([normalized_text])[0]
    if not sequence:
        raise ValueError(
            "The input does not contain words known to the model. Try simpler training-style text."
        )

    padded = pad_sequences([sequence], maxlen=max_len - 1, padding="pre")
    prediction = model.predict(padded, verbose=0)
    predicted_index = int(np.argmax(prediction, axis=-1)[0])

    predicted_word = index_to_word.get(predicted_index) or index_to_word.get(predicted_index + 1)
    if not predicted_word:
        raise ValueError("Prediction succeeded, but the predicted token was not found in the tokenizer vocabulary.")

    return predicted_word


left_col, right_col = st.columns([1.5, 1], gap="large")

with left_col:
    st.markdown(
        """
        <div class="hero-shell">
            <div class="eyebrow">Deep Learning • LSTM • NLP Demo</div>
            <h1 class="hero-title">Predict the <span class="gradient">next word</span> with a polished neural language interface.</h1>
            <p class="hero-subtitle">
                This app uses a trained LSTM language model to continue a text prompt.
                Enter a short sequence and get a predicted next token through a cleaner,
                more deployment-ready interface.
            </p>
            <div class="mini-metrics">
                <div class="metric">
                    <strong>Model</strong>
                    <span>LSTM-based neural sequence predictor</span>
                </div>
                <div class="metric">
                    <strong>Experience</strong>
                    <span>Animated UI with a modern glassmorphism layout</span>
                </div>
                <div class="metric">
                    <strong>Deployment</strong>
                    <span>Stable paths and dependency manifest included</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        """
        <div class="glass-card">
            <div class="panel-title"><span class="pulse-dot"></span>Ready to infer</div>
            <div class="panel-copy">
                Best results usually come from short phrases similar to the model's training text.
                Try simple prompts and iterate.
            </div>
            <div class="tip-card">
                <strong style="color:white;">Example prompts</strong><br>
                i love<br>
                deep learning is<br>
                artificial intelligence will
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

try:
    model, tokenizer, max_len, index_to_word = load_assets()
except Exception as exc:
    st.error(f"Unable to load model assets: {exc}")
    st.stop()

input_col, info_col = st.columns([1.3, 0.7], gap="large")

with input_col:
    st.subheader("Enter your prompt")
    text = st.text_input(
        "Type a phrase",
        placeholder="Example: I love",
        label_visibility="collapsed",
    )
    clicked = st.button("Predict Next Word")

with info_col:
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="panel-title">Model assets</div>
            <div class="panel-copy">
                Sequence length: <strong style='color:white;'>{max_len}</strong><br>
                Vocabulary size: <strong style='color:white;'>{len(index_to_word):,}</strong><br>
                Runtime: <strong style='color:white;'>Streamlit + TensorFlow</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if clicked:
    try:
        with st.spinner("Running prediction..."):
            word = predict_next_word(text, model, tokenizer, max_len, index_to_word)

        safe_text = " ".join(text.strip().split())
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted next word</div>
                <div class="result-word">{word}</div>
                <div class="sentence-preview">Prompt: <strong style='color:white;'>{safe_text}</strong><br>
                Full continuation: <strong style='color:#7dd3fc;'>{safe_text} {word}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.markdown("<div class='footer-note'>Built with Streamlit, TensorFlow, and a trained LSTM language model.</div>", unsafe_allow_html=True)
