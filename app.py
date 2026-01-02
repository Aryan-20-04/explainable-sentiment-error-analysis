import json
import streamlit as st
import os

st.set_page_config(
    page_title="Explainable Sentiment Error Analysis",
    layout="wide"
)

st.title("🔍 Explainable Sentiment Error Analysis")
st.write("Explore **why** the sentiment model made incorrect predictions.")

# ---------- Load metadata ----------
meta_path = "analysis/misclassified_meta.json"

if not os.path.exists(meta_path):
    st.error("Run sentiment_imdb.py first to generate analysis data.")
    st.stop()

with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---------- Filters ----------
error_types = sorted(set(item["error_type"] for item in data))
selected_error = st.selectbox("Filter by error type", ["ALL"] + error_types)

filtered = data if selected_error == "ALL" else [
    d for d in data if d["error_type"] == selected_error
]

st.write(f"Showing **{len(filtered)}** misclassified samples")

# ---------- Select sample ----------
sample_ids = [d["id"] for d in filtered]
selected_id = st.selectbox("Select sample ID", sample_ids)

sample = next(d for d in filtered if d["id"] == selected_id)

# ---------- Display ----------
st.subheader("Review Text")
st.write(sample["text"])

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("True Label", sample["true_label"])

with col2:
    st.metric("Predicted Label", sample["pred_label"])

with col3:
    st.metric("Error Type", sample["error_type"])

st.subheader("Model Confidence")
st.json(sample["confidence"])

if os.path.exists(sample["pdf_path"]):
    st.success(f"PDF report available: {sample['pdf_path']}")
else:
    st.warning("PDF report not found (run model again if needed)")
