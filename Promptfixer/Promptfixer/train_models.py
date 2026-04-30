import os
import pandas as pd
import json
import joblib
import warnings
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Ignore warnings for clean console output
warnings.filterwarnings('ignore')


def resolve_dataset_path(base_dir, csv_filename, output_prefix):
    data_dir = os.path.join(base_dir, "data")
    requested = os.path.join(data_dir, csv_filename)
    if os.path.exists(requested):
        return requested

    # Handle common naming typos in provided datasets (intent_dataset..csv).
    if output_prefix == "intent_detection":
        for candidate in ["intent_dataset.csv", "intent_dataset..csv"]:
            candidate_path = os.path.join(data_dir, candidate)
            if os.path.exists(candidate_path):
                return candidate_path

    return requested


def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    return " ".join(text.split())

def train_model(csv_filename, model_name, output_prefix):
    # Setup Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = resolve_dataset_path(BASE_DIR, csv_filename, output_prefix)
    model_dir = os.path.join(BASE_DIR, "models")
    
    # Create models folder if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print(f"\n⏳ Loading Dataset for {model_name}...")
    try:
        # Dynamic separator: Handles both Comma (CSV) and Tab (TSV) separated files automatically
        df = pd.read_csv(data_path, sep=None, engine='python')
    except FileNotFoundError:
        print(f"❌ Error: Dataset not found at {data_path}.")
        return

    # 🛑 1. Data Preprocessing (Updated for New Dataset Format)
    # Automatically detect if the input column is named 'text' (new) or 'prompt' (old)
    input_col = None
    if 'text' in df.columns:
        input_col = 'text'
    elif 'prompt' in df.columns:
        input_col = 'prompt'

    if not input_col or 'label' not in df.columns:
        print(f"❌ Error: Required columns missing in {csv_filename}. Found columns: {list(df.columns)}")
        print("💡 Hint: Ensure your file has a header with 'text' and 'label' (or 'prompt' and 'label').")
        return

    # Drop empty rows and convert to string
    df = df.dropna(subset=[input_col, 'label'])
    X = df[input_col].astype(str).map(normalize_text)
    y = df['label'].astype(str).str.strip()
    
    dataset_size = len(df)
    class_count = y.nunique()
    print(f"📊 {model_name} Dataset Size: {dataset_size} valid rows (Using '{input_col}' column)")

    # 🛑 2. Train/Test Split (adaptive for tiny multiclass datasets)
    # Need at least one sample per class in the test split when stratified.
    min_test_size = class_count / dataset_size if dataset_size else 0.2
    test_size = max(0.2, min_test_size)
    test_size = min(test_size, 0.4)

    stratify_labels = y if dataset_size >= (class_count * 2) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify_labels
    )

    print(f"⚙️ Training {model_name} Model (TF-IDF + Logistic Regression)...")
    
    # 🛑 3. Build ML Pipeline
    # Hybrid features (word + character n-grams) improve intent detection on typos and short prompts.
    model = make_pipeline(
        FeatureUnion([
            (
                "word_tfidf",
                TfidfVectorizer(
                    max_features=12000,
                    ngram_range=(1, 3),
                    lowercase=True,
                    strip_accents='unicode',
                    sublinear_tf=True,
                    min_df=1
                )
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    max_features=8000,
                    lowercase=True,
                    sublinear_tf=True,
                    min_df=1
                )
            )
        ]),
        LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            C=2.0,
            solver='lbfgs'
        )
    )
    
    # Train the model
    model.fit(X_train, y_train)

    # 🛑 4. Evaluate the Model
    predictions = model.predict(X_test)
    
    # Calculate Metrics
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, average='weighted', zero_division=0)
    rec = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

    metrics = {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "split_ratio": f"{round((1 - test_size) * 100, 1)}% Train / {round(test_size * 100, 1)}% Test",
        "dataset_size": dataset_size,
        "num_classes": int(class_count),
        "algorithm": "Hybrid TF-IDF (word+char) + Logistic Regression"
    }

    print(f"✅ {model_name} Trained Successfully!")
    print(f"   Accuracy:  {metrics['accuracy']}%")
    print(f"   F1 Score:  {metrics['f1_score']}%")

    # 🛑 5. Save the Model and Metrics
    model_path = os.path.join(model_dir, f"{output_prefix}_pipeline.pkl")
    metrics_path = os.path.join(model_dir, f"{output_prefix}_metrics.json")
    
    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"📁 Saved to {model_dir}")

if __name__ == "__main__":
    print("🚀 Initiating Machine Learning Training Sequence...")
    
    # 1. Train Intent Model
    train_model(
        csv_filename="intent_dataset..csv",
        model_name="Intent Detection", 
        output_prefix="intent_detection"
    )
    
    print("-" * 50)
    
    # 2. Train Toxicity Model
    train_model(
        csv_filename="toxicity_dataset.csv", 
        model_name="Toxicity Detection", 
        output_prefix="toxicity_detection"
    )
    
    print("\n🎉 All ML Models are trained and saved! You can now run 'app.py'.")