#!/usr/bin/env python
# ============================================================
#  app.py  –  Streamlit Sentiment Analysis Web App
#
#  Run: streamlit run app.py
# ============================================================

from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import LABEL_COLORS, LABEL_EMOJIS, LABEL_MAP
from model_utils import get_model

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "SentiSense — AI-Powered Sentiment Analysis Engine",
    page_icon  = "🎭",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* --- Global font --- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* --- Card style --- */
.metric-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-card .label { font-size: 0.8rem; color: #9399b2; text-transform: uppercase; letter-spacing: 0.1em; }
.metric-card .value { font-size: 2rem;   color: #cdd6f4; font-weight: 700; }
.metric-card .sub   { font-size: 0.85rem; color: #6c7086; }

/* --- Sentiment badge --- */
.badge {
    display: inline-block;
    padding: 0.35em 1em;
    border-radius: 999px;
    font-weight: 600;
    font-size: 1.1rem;
    letter-spacing: 0.04em;
}

/* --- Result block --- */
.result-box {
    border-left: 4px solid;
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    background: #181825;
    margin: 0.5rem 0;
}

/* --- Hide Streamlit branding --- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Model (cached) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model … (first run only)")
def load_model():
    return get_model()


# ── Plotly bar chart helper ───────────────────────────────────
def confidence_chart(scores: dict) -> go.Figure:
    labels = list(scores.keys())
    values = [scores[l] * 100 for l in labels]
    colors = [LABEL_COLORS[l] for l in labels]

    fig = go.Figure(
        go.Bar(
            x            = labels,
            y            = values,
            marker_color = colors,
            text         = [f"{v:.1f}%" for v in values],
            textposition = "outside",
            cliponaxis   = False,
        )
    )
    fig.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color="#cdd6f4", family="Inter"),
        yaxis         = dict(range=[0, 115], showgrid=False, showticklabels=False),
        xaxis         = dict(showgrid=False),
        margin        = dict(l=10, r=10, t=10, b=10),
        height        = 280,
        showlegend    = False,
    )
    return fig


def history_chart(records: list[dict]) -> go.Figure:
    df      = pd.DataFrame(records)
    colors  = [LABEL_COLORS[l] for l in df["label"]]
    labels  = [f"{LABEL_EMOJIS[l]} {l}" for l in df["label"]]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x            = list(range(1, len(df) + 1)),
        y            = df["confidence"] * 100,
        mode         = "lines+markers",
        line         = dict(color="#89b4fa", width=2),
        marker       = dict(color=colors, size=10, line=dict(color="#1e1e2e", width=2)),
        text         = labels,
        hovertemplate = "%{text}<br>Confidence: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(color="#cdd6f4", family="Inter"),
        xaxis         = dict(title="Prediction #", showgrid=False, color="#6c7086"),
        yaxis         = dict(title="Confidence %", showgrid=True,
                             gridcolor="#313244", range=[0, 110]),
        margin        = dict(l=10, r=10, t=10, b=40),
        height        = 260,
    )
    return fig


# ── Session state ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️  Settings")
    apply_prep = st.toggle("Apply text preprocessing", value=True,
                           help="Run tokenisation, stopword removal, lemmatisation")

    st.markdown("---")
    st.markdown("### About")
    st.info(
        "Fine-tuned **BERT-base-uncased** on labelled product reviews and social media text. "
        "Achieves **>90% accuracy** on held-out test data, outperforming an LSTM baseline by **8%**."
    )

    model = load_model()
    st.markdown(f"**Model:** `{model.model_name}`")

    if st.session_state.history:
        st.markdown("---")
        st.markdown("### Session Stats")
        df_h = pd.DataFrame(st.session_state.history)
        cnts = df_h["label"].value_counts()
        for lbl, cnt in cnts.items():
            st.markdown(
                f"{LABEL_EMOJIS[lbl]} **{lbl}**: {cnt}",
            )
        if st.button("🗑️  Clear history"):
            st.session_state.history = []
            st.rerun()


# ── Header ────────────────────────────────────────────────────
st.markdown("# 🎭 Sentiment Analysis Engine")
st.markdown(
    "Powered by **BERT** · NLTK preprocessing · HuggingFace Transformers · "
    "Classifies text as **Positive**, **Neutral**, or **Negative** with confidence scores."
)
st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab_single, tab_batch, tab_history = st.tabs(
    ["✍️  Single Text", "📂  Batch CSV", "📊  History"]
)


# ══════════════════════════════════════════════════════════════
#  TAB 1 – Single Text
# ══════════════════════════════════════════════════════════════
with tab_single:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("### Enter text to analyse")
        user_text = st.text_area(
            label       = "Text input",
            label_visibility = "collapsed",
            placeholder = "Paste a product review, tweet, comment …",
            height      = 200,
        )

        # Sample texts
        st.markdown("**Try a sample:**")
        samples = {
            "😊 Positive review"  : "This product is absolutely amazing! Best purchase of my life. Highly recommend!",
            "😐 Neutral review"   : "The product is okay. It does what it says but nothing special.",
            "😞 Negative review"  : "Terrible quality, broke after two days. Complete waste of money. Avoid!",
            "🐦 Positive tweet"   : "Just got this and I'm obsessed!! Can't stop recommending it to everyone 🔥",
            "🐦 Negative tweet"   : "this literally broke the first time i used it wtf. 0 stars if i could smh",
        }
        for label, sample in samples.items():
            if st.button(label, use_container_width=True):
                user_text = sample
                st.rerun()

    with col_result:
        if user_text and user_text.strip():
            with st.spinner("Analysing …"):
                t0     = time.time()
                result = model.predict(user_text, apply_preprocessing=apply_prep)
                elapsed = (time.time() - t0) * 1000  # ms

            lbl    = result["label"]
            color  = LABEL_COLORS[lbl]
            emoji  = LABEL_EMOJIS[lbl]
            conf   = result["confidence"]

            # Append to history
            st.session_state.history.append(
                {**result, "text": user_text[:120]}
            )

            # ── Result badge ──────────────────────────────
            st.markdown(f"""
            <div style="text-align:center; padding: 1.5rem 0;">
              <div style="font-size:4rem;">{emoji}</div>
              <span class="badge" style="background:{color}22; color:{color}; border:2px solid {color};">
                {lbl}
              </span>
              <div style="margin-top:0.75rem; font-size:1.6rem; font-weight:700; color:{color};">
                {conf:.1%}
              </div>
              <div style="color:#6c7086; font-size:0.85rem;">confidence · {elapsed:.0f}ms</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Score chart ───────────────────────────────
            st.plotly_chart(confidence_chart(result["scores"]),
                            use_container_width=True)

            # ── Preprocessed text expander ────────────────
            if apply_prep:
                with st.expander("🔍 Preprocessed text"):
                    st.code(result.get("clean_text", ""), language=None)

        else:
            st.markdown(
                """
                <div style="text-align:center; padding:3rem 1rem; color:#6c7086;">
                    <div style="font-size:3rem;">💬</div>
                    <p>Enter some text on the left to get started.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════
#  TAB 2 – Batch CSV
# ══════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### Upload a CSV file for batch prediction")
    st.markdown("The CSV must have a **text** column (or specify one below).")

    col_up, col_cfg = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    with col_cfg:
        text_col = st.text_input("Text column name", value="text")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Preview** ({len(df)} rows, {len(df.columns)} columns)")
        st.dataframe(df.head(5), use_container_width=True)

        if text_col not in df.columns:
            st.error(f"Column **{text_col}** not found. Available: {list(df.columns)}")
        else:
            if st.button("🚀  Run batch prediction", type="primary"):
                progress = st.progress(0, text="Preparing …")
                texts   = df[text_col].astype(str).tolist()

                with st.spinner(f"Classifying {len(texts)} texts …"):
                    results = model.predict_batch(
                        texts,
                        apply_preprocessing = apply_prep,
                    )

                progress.progress(100, text="Done!")

                df["sentiment"]  = [r["label"]      for r in results]
                df["confidence"] = [r["confidence"] for r in results]
                df["score_pos"]  = [r["scores"].get("Positive", 0) for r in results]
                df["score_neu"]  = [r["scores"].get("Neutral",  0) for r in results]
                df["score_neg"]  = [r["scores"].get("Negative", 0) for r in results]

                st.success(f"✅  Classified {len(df)} texts successfully!")
                st.dataframe(
                    df[["text" if text_col == "text" else text_col,
                        "sentiment", "confidence"]].head(20),
                    use_container_width = True,
                )

                # Distribution pie
                dist = df["sentiment"].value_counts()
                fig  = go.Figure(go.Pie(
                    labels   = dist.index,
                    values   = dist.values,
                    marker   = dict(colors=[LABEL_COLORS[l] for l in dist.index]),
                    hole     = 0.4,
                ))
                fig.update_layout(
                    paper_bgcolor = "rgba(0,0,0,0)",
                    font          = dict(color="#cdd6f4"),
                    height        = 300,
                    margin        = dict(l=0, r=0, t=10, b=0),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label     = "⬇️  Download results CSV",
                    data      = csv_bytes,
                    file_name = "sentiment_results.csv",
                    mime      = "text/csv",
                )


# ══════════════════════════════════════════════════════════════
#  TAB 3 – History
# ══════════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.info("No predictions yet. Analyse some text in the **Single Text** tab.")
    else:
        history = st.session_state.history

        # Summary metrics
        df_h     = pd.DataFrame(history)
        avg_conf = df_h["confidence"].mean()
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric("Total predictions", len(history))
        with c2:
            st.metric("Avg confidence", f"{avg_conf:.1%}")
        with c3:
            top = df_h["label"].value_counts().idxmax()
            st.metric("Most common", f"{LABEL_EMOJIS[top]} {top}")
        with c4:
            pos_pct = (df_h["label"] == "Positive").mean()
            st.metric("Positive rate", f"{pos_pct:.0%}")

        # Trend chart
        if len(history) > 1:
            st.markdown("#### Confidence trend")
            st.plotly_chart(history_chart(history), use_container_width=True)

        # Table
        st.markdown("#### All predictions")
        display_df = df_h[["text", "label", "confidence"]].copy()
        display_df["confidence"] = display_df["confidence"].map("{:.1%}".format)
        st.dataframe(display_df, use_container_width=True)
