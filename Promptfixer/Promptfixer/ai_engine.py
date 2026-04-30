import os
import warnings

# 🔥 MUST BE AT TOP: Kills TensorFlow/Torch Warnings completely
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# 🛑 FORCE 100% OFFLINE MODE
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

warnings.filterwarnings("ignore")

import joblib
import time
import ollama
import re
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score_func
import logging

# Mute transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

class PromptFixerEngine:
    def __init__(self):
        print("🚀 Booting Enterprise AI Engine (Meta/Google Level)...")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        def load_model(name):
            try:
                model = joblib.load(os.path.join(self.base_dir, "models", f"{name}_pipeline.pkl"))
                with open(os.path.join(self.base_dir, "models", f"{name}_metrics.json"), "r") as f:
                    metrics = json.load(f)
                return model, metrics
            except Exception as e:
                print(f"⚠️ Warning: {name} model/metrics not found. Run train_models.py first.")
                return None, {"error": "Model not trained."}

        self.intent_model, self.intent_metrics = load_model("intent_detection")
        self.toxicity_model, self.toxicity_metrics = load_model("toxicity_detection")
        
        print("⏳ Loading Semantic Evaluation Models... (Warming up memory to reduce latency)")
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            _ = bert_score_func(["warmup"], ["warmup"], lang="en", model_type="distilbert-base-uncased", verbose=False)
        except Exception:
            pass

        print("✅ Production AI Engine Ready! (100% Bulletproof Local Pipeline Active)")

        # Conservative fallback to avoid over-confident wrong intent labels.
        self.intent_confidence_threshold = 0.38

    def get_advanced_metrics(self, original, optimized, start_time):
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

    def _predict_intent(self, clean_prompt):
        if not self.intent_model or not clean_prompt:
            return "general_query"

        lowered = clean_prompt.lower()

        # Strong heuristic path for code requests.
        has_code_action = bool(re.search(r"\b(write|generate|create|build|make|implement|develop)\b", lowered))
        has_code_object = bool(re.search(r"\b(code|script|function|api|backend|frontend|app|program|algorithm|query|sql|endpoint|class)\b", lowered))
        has_tech_keyword = bool(re.search(r"\b(python|java|javascript|typescript|react|node|flask|django|fastapi|pandas|numpy|html|css|c\+\+|c#|go|rust)\b", lowered))
        if (has_code_action and (has_code_object or has_tech_keyword)) or has_tech_keyword:
            return "code_generation"

        # Strong heuristic path for image prompt requests.
        has_image_noun = bool(re.search(r"\b(image|photo|picture|illustration|artwork|poster|wallpaper|render|visual)\b", lowered))
        has_image_action = bool(re.search(r"\b(generate|create|make|design|craft|write|produce)\b", lowered))
        has_image_model = bool(re.search(r"\b(midjourney|stable diffusion|dalle|dall e|sdxl|flux)\b", lowered))
        if has_image_model or (has_image_noun and has_image_action):
            return "image_generation"

        # Rule hints improve stability on short, noisy inputs.
        intent_hints = {
            "image_generation": [
                "generate image", "image prompt", "create image", "text to image", "text-to-image",
                "midjourney", "stable diffusion", "dalle", "dall e", "photorealistic", "cinematic shot",
                "concept art", "render", "illustration", "wallpaper", "8k image"
            ],
            "translation": ["translate", "convert to", "into hindi", "into tamil", "into french", "into spanish"],
            "summarization": ["summarize", "summary", "short version", "brief"],
            "code_generation": ["write code", "generate code", "python code", "build app", "program","code"],
            "debugging": ["debug", "fix", "error", "not working", "traceback", "bug"],
            "comparison": ["difference", "compare", "vs", "which is better"],
            "planning": ["roadmap", "plan", "step by step", "learning path"],
            "analysis": ["analyze", "insights", "patterns", "evaluate performance"],
            "recommendation": ["recommend", "best", "suggest"],
            "content_generation": ["write blog", "write article", "create content"],
            "project_creation": ["create a project"],
            "info_request": ["what is", "explain", "how does"]
        }
        for intent, phrases in intent_hints.items():
            if any(phrase in lowered for phrase in phrases):
                return intent

        try:
            predicted = self.intent_model.predict([clean_prompt])[0]
        except Exception:
            return "general_query"

        try:
            if hasattr(self.intent_model, "predict_proba"):
                proba = self.intent_model.predict_proba([clean_prompt])[0]
                best_conf = float(max(proba))
                if best_conf < self.intent_confidence_threshold:
                    return "general_query"
        except Exception:
            pass

        return predicted

    def _build_system_prompt(self, intent):
        intent_guidance = {
            "image_generation": "Output a rich text-to-image prompt with subject, composition, style, lighting, camera/lens, color palette, scene details, and explicit negative constraints.",
            "code_generation": "Produce a production-grade coding prompt that explicitly captures language/framework, required features, inputs/outputs, error handling, edge cases, performance constraints, and expected deliverables (code + brief explanation + tests when relevant).",
            "debugging": "Ask for error context, environment, repro steps, logs, and desired fix behavior.",
            "summarization": "Preserve key claims, compress verbosity, and specify preferred summary length.",
            "translation": "Preserve meaning, tone, and proper nouns; specify source and target language clearly.",
            "analysis": "Request data assumptions, method, outputs, and interpretation boundaries.",
            "planning": "Define milestones, timeline, prerequisites, and measurable outcomes.",
            "comparison": "Define decision criteria and force side-by-side structure.",
            "recommendation": "Capture constraints, budget, scale, and rationale format.",
            "content_generation": "Specify audience, tone, structure, length, and must-cover points.",
            "info_request": "Request concise explanation depth and include examples if useful.",
            "general_query": "Clarify objective, context, constraints, and expected answer format."
        }
        guidance = intent_guidance.get(intent, intent_guidance["general_query"])

        return (
            "You are an elite prompt optimizer. Rewrite the user's raw prompt into a high-performance prompt "
            "for LLMs while preserving exact intent.\n\n"
            "Rules:\n"
            "1) Keep original meaning unchanged.\n"
            "2) Remove ambiguity and vague wording.\n"
            "3) Add missing context and constraints that improve answer quality.\n"
            "4) Make the optimized prompt detailed and substantial. Target 150 to 260 words unless the user asks for ultra-short output.\n"
            "5) Use plain text only (no markdown, no code fences).\n"
            "6) Do not add meta commentary. Return only the final optimized prompt.\n"
            f"7) Intent-specific guidance: {guidance}\n"
        )

    def _clean_generated_prompt(self, text):
        cleaned = text.strip()
        cleaned = cleaned.replace("**", "")
        cleaned = re.sub(r'#+\s*', '', cleaned)
        cleaned = re.sub(r'```[\s\S]*?```', '', cleaned)
        cleaned = re.sub(r'\r\n?', '\n', cleaned)

        # Remove chatty wrappers from LLM output while keeping the actual prompt body.
        lines = [line.strip() for line in cleaned.split('\n')]
        while lines and re.match(r'^(sure|great|okay|alright|here)\b', lines[0], flags=re.IGNORECASE):
            lines.pop(0)
        while lines and re.match(r'^(optimized\s*prompt|final\s*prompt|prompt)\s*:?\s*$', lines[0], flags=re.IGNORECASE):
            lines.pop(0)
        cleaned = '\n'.join(lines).strip()

        cleaned = re.sub(r'^\s*(optimized\s*prompt|final\s*prompt|prompt)\s*:\s*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
        return cleaned

    def _infer_code_stack(self, text):
        lowered = text.lower()
        if "react" in lowered:
            return "React (JavaScript/TypeScript)"
        if "flask" in lowered:
            return "Python with Flask"
        if "fastapi" in lowered:
            return "Python with FastAPI"
        if "django" in lowered:
            return "Python with Django"
        if any(token in lowered for token in ["python", "pandas", "numpy"]):
            return "Python"
        if any(token in lowered for token in ["csv", "plot", "matplotlib", "seaborn", "dataframe"]):
            return "Python with pandas and matplotlib"
        if any(token in lowered for token in ["node", "express", "javascript"]):
            return "JavaScript (Node.js)"
        if "java" in lowered:
            return "Java"
        return "the most appropriate language and framework"

    def _build_code_generation_prompt(self, user_prompt):
        stack = self._infer_code_stack(user_prompt)
        return (
            f"Generate production-ready {stack} code for this task: {user_prompt}. "
            "Do not provide pseudo code, placeholders, or TODO stubs. Build a complete, runnable solution with clear structure and maintainable naming. "
            "Explicitly include dependencies, setup/install commands, and execution steps. Implement robust input validation, error handling, and edge-case coverage. "
            "Use secure defaults and avoid unsafe patterns. If the task involves APIs or forms, include validation and proper failure responses. If it involves data processing, handle missing/invalid records and produce clean output. "
            "Output format must be: 1) final code, 2) short architecture explanation, 3) run instructions, 4) example usage with expected output, and 5) compact test cases that verify core behavior. "
            "Keep the solution concise but complete, and aligned with modern best practices."
        )

    def _code_generation_quality_fallback(self, user_prompt, candidate_prompt):
        stack = self._infer_code_stack(user_prompt)
        fallback = (
            f"Generate production-ready {stack} code for the following request: {user_prompt}. "
            "Return complete runnable code and avoid placeholders. Clearly define project structure, dependencies, and setup commands. "
            "Implement core features end-to-end with validation, error handling, and readable naming. Include edge-case handling and input sanitization where relevant. "
            "Use clean modular organization and include comments only for non-obvious logic. "
            "After the code, provide: 1) brief run instructions, 2) one realistic example input/output or usage flow, and 3) quick test cases to verify correctness. "
            "Keep the solution practical, maintainable, and aligned with current best practices."
        )

        candidate_words = len(candidate_prompt.split())
        fallback_words = len(fallback.split())
        if candidate_words < 140 or re.search(r'\b(sure|optimized prompt|final prompt)\b', candidate_prompt, flags=re.IGNORECASE):
            return fallback if fallback_words >= candidate_words else candidate_prompt
        return candidate_prompt

    def _expand_prompt_if_needed(self, user_prompt, optimized_prompt, intent):
        target_min_words = 140 if intent == "code_generation" else 120
        expanded = optimized_prompt

        for _ in range(2):
            word_count = len(expanded.split())
            if word_count >= target_min_words:
                break

            expansion_system = self._build_system_prompt(intent)
            lower_bound = 160 if intent == "code_generation" else 130
            upper_bound = 260 if intent == "code_generation" else 220
            expansion_request = (
                "Expand and enrich the prompt below without changing intent. "
                f"Return only plain text between {lower_bound} and {upper_bound} words with concrete constraints, context, and quality expectations.\n\n"
                f"Original user prompt: {user_prompt}\n\n"
                f"Current optimized prompt: {expanded}"
            )

            expanded_response = ollama.chat(
                model='gemma:2b',
                messages=[
                    {'role': 'system', 'content': expansion_system},
                    {'role': 'user', 'content': expansion_request}
                ],
                options={
                    'temperature': 0.15,
                    'top_p': 0.8
                }
            )
            candidate = self._clean_generated_prompt(expanded_response['message']['content'])
            if len(candidate.split()) > len(expanded.split()):
                expanded = candidate

        return expanded

    def _optimize_prompt_with_llm(self, user_prompt, intent):
        system_prompt = self._build_system_prompt(intent)

        # First pass: produce optimized prompt.
        response = ollama.chat(
            model='gemma:2b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Raw prompt: {user_prompt}\n\nReturn only optimized prompt."}
            ],
            options={
                'temperature': 0.15,
                'top_p': 0.8
            }
        )
        first_pass = response['message']['content'].strip()

        # Second pass: refine for precision and consistency.
        refine_prompt = (
            "Refine the optimized prompt below to maximize clarity, completeness, and instruction quality. "
            "Keep intent unchanged. Remove markdown and fluff. Return only the final prompt.\n\n"
            f"Original user prompt: {user_prompt}\n\n"
            f"Candidate optimized prompt: {first_pass}"
        )
        refine_response = ollama.chat(
            model='gemma:2b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': refine_prompt}
            ],
            options={
                'temperature': 0.1,
                'top_p': 0.75
            }
        )

        optimized_prompt = self._clean_generated_prompt(refine_response['message']['content'])

        # Safety fallback: avoid unusable tiny or empty outputs.
        if len(optimized_prompt) < 20:
            optimized_prompt = first_pass.strip()

        optimized_prompt = self._expand_prompt_if_needed(
            user_prompt=user_prompt,
            optimized_prompt=optimized_prompt,
            intent=intent
        )

        if intent == "code_generation":
            optimized_prompt = self._code_generation_quality_fallback(user_prompt, optimized_prompt)

        return optimized_prompt

    def get_research_specs(self):
        return {
            "intent_model_metrics": self.intent_metrics,
            "toxicity_model_metrics": self.toxicity_metrics,
            "generative_optimization": {
                "model_family": "Gemma (Google DeepMind)",
                "parameters": "2 Billion (2B)",
                "architecture": "Local Llama.cpp backend (Ollama)"
            },
            "evaluation_engine": {
                "semantic_similarity": "SentenceTransformers (all-MiniLM-L6-v2)",
                "bert_score": "distilbert-base-uncased",
                "bleu_score": "NLTK Sentence BLEU (Smoothing Function 1)"
            }
        }

    def analyze_and_fix(self, user_prompt):
        start_time = time.time()

        if self.toxicity_model:
            tox_pred = self.toxicity_model.predict([user_prompt])[0]
            if str(tox_pred).strip().lower() == 'toxic':
                return {
                    "status": "rejected",
                    "message": "⚠️ Harmful or toxic content detected. Blocked by Custom ML Filter.",
                    "intent": "Blocked",
                    "optimized_prompt": "",
                    "metrics": None
                }

        clean_prompt = " ".join(str(user_prompt).strip().lower().split())
        intent = self._predict_intent(clean_prompt)

        try:
            if intent == "code_generation":
                optimized_prompt = self._build_code_generation_prompt(user_prompt)
            else:
                optimized_prompt = self._optimize_prompt_with_llm(user_prompt=user_prompt, intent=intent)

        except Exception as e:
            print(f"❌ Ollama Error: {e}")
            optimized_prompt = (
                "Role: Error\n\n"
                "Context: System Failure\n\n"
                "Task: Ollama failed.\n\n"
                "Format: Text"
            )

        metrics = self.get_advanced_metrics(user_prompt, optimized_prompt, start_time)

        return {
            "status": "success",
            "message": "Prompt successfully optimized.",
            "intent": intent,
            "original_prompt": user_prompt,
            "optimized_prompt": optimized_prompt,
            "metrics": metrics
        }