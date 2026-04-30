from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pymongo import MongoClient
from ai_engine import PromptFixerEngine
from datetime import datetime, timezone
import os
import json

app = Flask(__name__)
CORS(app) # Enterprise standard to allow frontend requests
ai_engine = PromptFixerEngine()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_LOGS_PATH = os.path.join(BASE_DIR, "data", "local_logs.json")


def ensure_local_logs_file():
    os.makedirs(os.path.dirname(LOCAL_LOGS_PATH), exist_ok=True)
    if not os.path.exists(LOCAL_LOGS_PATH):
        with open(LOCAL_LOGS_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)


def read_local_logs():
    ensure_local_logs_file()
    try:
        with open(LOCAL_LOGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception:
        pass
    return []


def write_local_logs(logs):
    ensure_local_logs_file()
    with open(LOCAL_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def save_log_entry(entry):
    if db_connected:
        collection.insert_one(entry)
        return

    local_logs = read_local_logs()
    local_entry = dict(entry)
    local_entry["timestamp"] = entry["timestamp"].isoformat()
    local_logs.append(local_entry)
    write_local_logs(local_logs)


def get_all_logs():
    if db_connected:
        return list(collection.find({}, {"_id": 0}))
    return read_local_logs()


def get_success_history(limit=20):
    if db_connected:
        return list(collection.find(
            {"status": "success"},
            {"_id": 0, "timestamp": 1, "intent": 1, "original_prompt": 1, "optimized_prompt": 1}
        ).sort("timestamp", -1).limit(limit))

    local_logs = [log for log in read_local_logs() if log.get("status") == "success"]
    local_logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return local_logs[:limit]


def normalize_timestamp(value):
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return value
    return str(value)

# MongoDB Graceful Connection
db_connected = False
try:
    client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=3000)
    client.server_info() # Test connection
    db = client["PromptFixerProd"]
    collection = db["logs"]
    db_connected = True
    print("✅ MongoDB Connected Successfully")
except Exception:
    print("❌ MongoDB Connection Failed! App will run, but logs will NOT be saved.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
def process_prompt():
    try:
        data = request.json
        user_prompt = data.get("prompt", "")

        if not user_prompt.strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        # Pass through the core AI Engine
        result = ai_engine.analyze_and_fix(user_prompt)

        # Persist logs in MongoDB when available, else local JSON fallback.
        log_entry = {
            "timestamp": datetime.now(timezone.utc),
            "status": result.get("status"),
            "intent": result.get("intent", "None"),
            "original_prompt": result.get("original_prompt", ""),
            "optimized_prompt": result.get("optimized_prompt", ""),
            "latency_ms": result.get("metrics", {}).get("latency_ms", 0) if result.get("metrics") else 0
        }
        save_log_entry(log_entry)

        return jsonify(result), 200

    except Exception as e:
        print(f"🔥 Critical API Error: {e}")
        return jsonify({"status": "error", "message": "Internal Server Error during processing."}), 500

@app.route('/api/dashboard-metrics', methods=['GET'])
def get_dashboard_metrics():
    try:
        logs = get_all_logs()
        total_requests = len(logs)
        blocked_requests = sum(1 for log in logs if log.get("status") == "rejected")

        success_logs = [log for log in logs if log.get("status") == "success"]

        intent_map = {}
        for log in success_logs:
            key = log.get("intent", "unknown")
            intent_map[key] = intent_map.get(key, 0) + 1

        intents = list(intent_map.keys())
        intent_counts = list(intent_map.values())

        latencies = [float(log.get("latency_ms", 0) or 0) for log in success_logs]
        avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

        return jsonify({
            "total": total_requests,
            "blocked": blocked_requests,
            "intents": intents,
            "intent_counts": intent_counts,
            "avg_latency": avg_latency
        }), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch metrics: {e}"}), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        logs = get_success_history(limit=20)

        for log in logs:
            log["timestamp"] = normalize_timestamp(log.get("timestamp", ""))
            log["original_prompt"] = log.get("original_prompt", "⚠️ No raw input recorded (Old log entry).")
            log["optimized_prompt"] = log.get("optimized_prompt", "⚠️ No optimized output recorded (Old log entry).")

        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": f"Failed to fetch history: {e}"}), 500

# 🔬 Route to fetch Model Specs for Research Paper Dashboard
@app.route('/api/research-specs', methods=['GET'])
def get_research_specs():
    specs = ai_engine.get_research_specs()
    return jsonify(specs), 200

# 🔬 Route to Export raw database logs for Research graphs (JSON export)
@app.route('/api/export-logs', methods=['GET'])
def export_logs():
    try:
        logs = get_all_logs()
        for log in logs:
            if "timestamp" in log:
                ts = log["timestamp"]
                if hasattr(ts, "isoformat"):
                    log["timestamp"] = ts.isoformat()
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": f"Failed to export logs: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)