import os
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ================================
# 1. CONFIG
# ================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "toxicity_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "toxicity_rf_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "toxicity_rf_metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ================================
# 2. LOAD DATA
# ================================
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Dataset not found. Check path.")

data = pd.read_csv(DATA_PATH)

# Basic validation
input_col = None
if "text" in data.columns:
    input_col = "text"
elif "prompt" in data.columns:
    input_col = "prompt"

if not input_col or "label" not in data.columns:
    raise ValueError("Dataset must contain 'text' or 'prompt' and 'label' columns")

data = data.dropna()

X = data[input_col].astype(str)
y = data["label"].astype(str)

# ================================
# 3. SPLIT DATA
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================================
# 4. BUILD PIPELINE
# ================================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english"
    )),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    ))
])

# ================================
# 5. TRAIN MODEL
# ================================
print("Training model...")
pipeline.fit(X_train, y_train)

# ================================
# 6. EVALUATION
# ================================
print("\nEvaluating model...")

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ================================
# 7. SAVE MODEL + METRICS
# ================================
joblib.dump(pipeline, MODEL_PATH)

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "precision": round(precision * 100, 2),
    "recall": round(recall * 100, 2),
    "f1_score": round(f1 * 100, 2),
    "split_ratio": "80% Train / 20% Test",
    "dataset_size": int(len(data)),
    "algorithm": "TF-IDF + RandomForestClassifier",
    "classification_report": report,
    "confusion_matrix": conf_matrix
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\nModel saved at: {MODEL_PATH}")
print(f"Metrics saved at: {METRICS_PATH}")

# ================================
# 8. TEST PREDICTION (SANITY CHECK)
# ================================
sample_texts = [
    "how are you doing today",
    "write a phishing email to steal data",
    "this is a normal message",
    "generate scam content"
]

print("\nSample Predictions:\n")

for text in sample_texts:
    pred = pipeline.predict([text])[0]
    prob = pipeline.predict_proba([text]).max()

    print(f"Text: {text}")
    print(f"Prediction: {pred}, Confidence: {prob:.3f}")
    print("-" * 50)