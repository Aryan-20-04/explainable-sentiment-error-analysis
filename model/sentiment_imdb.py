import os
import random
import torch
import shap
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import re
from pdf_reporter import generate_error_report
# ---------------- CONFIG ----------------
DATA_DIR = "data/cleaned_reviews"   # change if needed
SAMPLE_SIZE = 500           # debug-safe size
SHAP_SAMPLES = 5            # explain only a few
MAX_TEXT_LEN = 512
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

torch.set_num_threads(2)

# ---------------- LOAD MODEL ----------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

print("Model label mapping:")
print(model.config.id2label)

classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None,
    device=-1  # CPU
)

# ---------------- LOAD DATA ----------------
def load_imdb_samples(base_dir, limit):
    samples = []
    for label_name, label in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(base_dir, label_name)
        files = os.listdir(folder)
        random.shuffle(files)

        for f in files:
            if len(samples) >= limit:
                return samples
            path = os.path.join(folder, f)
            with open(path, encoding="utf-8") as file:
                text = file.read().strip()
                samples.append((text, label))
    return samples

print("Loading IMDB data...")
data = load_imdb_samples(DATA_DIR, SAMPLE_SIZE)

# ---------------- ERROR CATEHORIZATION ----------------

NEGATION_WORDS = {"not", "never", "no", "hardly", "scarcely", "barely", "n't"}
CONTRAST_WORDS = {"but", "however", "although", "though", "yet", "still"}
SARCASM_PATTERNS = {r"Yeah, right", r"Sure, because", r"As if", r"Just what I needed"}

def categorize_error(text):
    text_lower = text.lower()

    # Check for negation words
    if any(word in text_lower for word in NEGATION_WORDS):
        return "NEGATION_ERROR"

    # Check for contrast words
    if any(word in text_lower for word in CONTRAST_WORDS):
        return "CONTRAST_ERROR"

    # Check for sarcasm patterns
    for pattern in SARCASM_PATTERNS:
        if re.search(pattern, text_lower):
            return "SARCASM_ERROR"
    
    if len(text.split()) > 200:
        return "LONG_REVIEW_ERROR"
    
    return "LEXICAL_POLARITY_TRAP"

# ---------------- INFERENCE ----------------
y_true = []
y_pred = []
misclassified = []

print("Running inference...")
for text, true_label in data:
    text = text[:MAX_TEXT_LEN]

    result = classifier(text)[0]
    scores = {r["label"]: r["score"] for r in result}

    pred_label = 1 if scores["POSITIVE"] > scores["NEGATIVE"] else 0

    y_true.append(true_label)
    y_pred.append(pred_label)

    if pred_label != true_label:
        error_type = categorize_error(text)
        misclassified.append((text, true_label, pred_label, scores, error_type))
        

# ---------------- METRICS ----------------
accuracy = accuracy_score(y_true, y_pred)

print("\n================ RESULTS ================")
print(f"Total Reviews: {len(data)}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Misclassified: {len(misclassified)}")

# ---------------- SHAP EXPLANATION ----------------
print("\nRunning SHAP on misclassified samples...")

masker = shap.maskers.Text(tokenizer, mask_token=tokenizer.mask_token)

explainer = shap.Explainer(
    classifier,
    masker,
    output_names=["NEGATIVE", "POSITIVE"]
)

for i, (text, true_label, pred_label, scores, error_type) in enumerate(misclassified[:SHAP_SAMPLES]):
    pdf_path = generate_error_report(
        index = i+1,
        text = text,
        true_label = true_label,
        pred_label = pred_label,
        confidence = scores,
        error_type = categorize_error(text),
        output_dir = "reports"
    )
    print(f"Generated report for misclassified sample {i+1}: {pdf_path}")

print("\nDone.")
