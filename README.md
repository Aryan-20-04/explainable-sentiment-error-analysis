# Explainable Sentiment Error Analysis

This project focuses on **understanding why transformer-based sentiment models make incorrect predictions**, rather than improving raw accuracy.

Using a pretrained **DistilBERT** sentiment classifier and **SHAP (SHapley Additive exPlanations)**, the system analyzes misclassified movie reviews and identifies **linguistic failure modes** such as negation, contrast, sarcasm, and lexical polarity traps.

The goal is to make model failures **transparent, interpretable, and auditable**.

---

## 🔍 Motivation

Modern NLP models often achieve high accuracy but still make **confidently wrong predictions**.  
Accuracy alone does not explain *why* these errors occur.

This project aims to:
- Diagnose transformer model failures
- Explain decisions at the token level
- Categorize common linguistic error patterns
- Promote trustworthy and explainable AI evaluation

---

## 🧠 What This Project Does

- Runs sentiment inference using a pretrained DistilBERT model
- Detects misclassified reviews
- Explains misclassifications using SHAP
- Automatically categorizes errors into:
  - Negation errors
  - Contrast clause dominance
  - Sarcasm misinterpretation
  - Lexical polarity traps
  - Long-review dilution
- Generates **standalone PDF audit reports** for misclassified samples
- Produces quantitative error analysis (CSV + bar graph)
- Provides a **Streamlit interface** to explore model failures interactively

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT (SST-2 fine-tuned)
- SHAP
- Scikit-learn
- ReportLab (PDF generation)
- Streamlit

---

## 📁 Project Structure

```bash
explainable-sentiment-error-analysis/
│
├── model/
│ ├── sentiment_imdb.py # Main pipeline
│ ├── error_analysis.py # Error distribution (CSV + graph)
│ ├── pdf_reporter.py # PDF report generator
│
├── app.py # Streamlit interface
├── README.md
├── requirements.txt
├── .gitignore
```

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

2. Prepare dataset
Organize cleaned IMDB-style reviews as:

data/cleaned_reviews/
 ├── pos/
 └── neg/

3. Run analysis pipeline

python model/sentiment_imdb.py
This will:
```

```text

Run inference

Generate PDF reports for misclassified samples

Save error distribution CSV and graph

Save metadata for the Streamlit app

4. Launch Streamlit UI

streamlit run app.py
```

## 📊 Outputs
```text
PDF reports explaining individual misclassifications

CSV file summarizing error category counts

Bar graph showing dominant error types

Interactive UI for exploring errors by category
```

## 📈 Key Insight
```bash

Transformer sentiment models often fail when strong polarity words appear inside negated or contrastive contexts, even when predictions are made with high confidence.

This highlights the gap between lexical cues and compositional semantic understanding.
```

## ⚠️ Limitations
```bash
Error categorization is heuristic-based

SHAP provides approximations, not ground truth explanations

Sarcasm detection is incomplete

Findings are dataset-dependent (IMDB reviews)
```

## 🎯 Project Scope
```bash
This project does not aim to maximize accuracy.
Its objective is to analyze, explain, and document model failures using explainable AI techniques.
```