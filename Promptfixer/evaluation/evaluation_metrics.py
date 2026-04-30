import os
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_func
import logging

# Mute warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class EvaluationEngine:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

        print("⏳ Loading Semantic Evaluation Models... (Warming up memory)")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            # Warm up BERT score
            _ = bert_score_func(["warmup"], ["warmup"], lang="en", model_type="distilbert-base-uncased", verbose=False)
        except Exception:
            pass

    def calculate_context_density(self, text):
        if not text or len(text.strip()) == 0: 
            return "Low"
        words = text.split()
        unique_words = set(words)
        density = len(unique_words) / len(words)
        if density > 0.65: return "High"
        elif density > 0.45: return "Medium"
        else: return "Low"

    def evaluate(self, original, optimized, start_time):
        latency = round((time.time() - start_time), 2) 
        semantic_sim, bleu, bert_f1 = 0.0, 0.0, 0.0

        try:
            emb1 = self.sim_model.encode(original, convert_to_tensor=True)
            emb2 = self.sim_model.encode(optimized, convert_to_tensor=True)
            semantic_sim = round(util.pytorch_cos_sim(emb1, emb2).item() * 100, 2)
        except Exception: pass

        try:
            orig_tokens = nltk.word_tokenize(original.lower())
            opt_tokens = nltk.word_tokenize(optimized.lower())
            chencherry = SmoothingFunction()
            bleu = round(sentence_bleu([orig_tokens], opt_tokens, smoothing_function=chencherry.method1) * 100, 2)
        except Exception: pass

        try:
            P, R, F1 = bert_score_func([optimized], [original], lang="en", model_type="distilbert-base-uncased", verbose=False)
            bert_f1 = round(F1.item() * 100, 2)
        except Exception: pass

        return {
            "semantic_similarity": f"{semantic_sim}%",
            "bleu_score": f"{bleu}%",
            "bert_score": f"{bert_f1}%",
            "latency_ms": latency * 1000 
        }