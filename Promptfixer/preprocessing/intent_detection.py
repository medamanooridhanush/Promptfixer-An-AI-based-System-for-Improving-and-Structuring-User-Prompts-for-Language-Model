import os
import joblib
import json

class IntentDetector:
    def __init__(self, base_dir):
        self.model_path = os.path.join(base_dir, "models", "intent_detection_pipeline.pkl")
        self.metrics_path = os.path.join(base_dir, "models", "intent_detection_metrics.json")
        try:
            self.model = joblib.load(self.model_path)
            with open(self.metrics_path, "r") as f:
                self.metrics = json.load(f)
        except Exception:
            self.model = None
            self.metrics = {"error": "Intent model not trained."}

    def detect(self, text):
        if not self.model:
            return "general_query"

        clean_text = " ".join(str(text).strip().lower().split())
        if not clean_text:
            return "general_query"

        return self.model.predict([clean_text])[0]