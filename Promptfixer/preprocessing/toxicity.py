import os
import joblib
import json

class ToxicityDetector:
    def __init__(self, base_dir):
        self.model_path = os.path.join(base_dir, "models", "toxicity_detection_pipeline.pkl")
        self.metrics_path = os.path.join(base_dir, "models", "toxicity_detection_metrics.json")
        try:
            self.model = joblib.load(self.model_path)
            with open(self.metrics_path, "r") as f:
                self.metrics = json.load(f)
        except Exception:
            self.model = None
            self.metrics = {"error": "Toxicity model not trained."}

    def is_toxic(self, text):
        if not self.model:
            return False
        pred = self.model.predict([text])[0]
        return str(pred).strip().lower() == 'toxic'