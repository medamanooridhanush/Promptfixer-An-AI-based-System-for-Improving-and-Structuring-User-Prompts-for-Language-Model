# 🚀 PromptFixer Pro | Enterprise AI Engine

A high-performance Prompt Engineering system powered by Google Gemma (via Ollama) and Scikit-Learn.
Designed for automated prompt structuring, toxicity filtering, and real-time research metrics.

---

## 📌 Prerequisites

Make sure the following are installed:

* Python 3.10+
* MongoDB (running on localhost:27017)
* Ollama → https://ollama.com/download

---

## 🛠️ Step-by-Step Installation

### 1️⃣ Setup Ollama (Core Engine)

Open your terminal and run:

ollama run gemma:2b

* Wait for the model to download
* Once ready, type:

/bye

⚠️ Keep Ollama running in the background

---

### 2️⃣ Project Setup

* Clone the repository or extract the project folder
* Open terminal inside the project directory

Install dependencies:

pip install -r requirements.txt

---

### 3️⃣ Initialize Model

Train the intent model (required once):

python train_intent.py

---

### 4️⃣ Run the Application

Make sure MongoDB is running, then start the app:

python app.py

---

### 5️⃣ Access the System

Open your browser and go to:

http://localhost:5000

---

## 📊 Features (Research-Oriented)

* Optimized Console
  Real-time prompt structuring using Gemma 2B

* Analytics Dashboard
  Visualizes:

  * Intent distribution
  * Latency metrics
    (Powered by Chart.js)

* Research Tab

  * Export performance logs (JSON)
  * View system hyperparameters
  * Useful for academic documentation

---

## ⚠️ Troubleshooting

MongoDB Error?
→ Ensure MongoDB service is running:
mongod

Ollama Error?
→ Check if Ollama is active in:

* System tray
* Task Manager

Format Error?
→ Ensure:

* models/ folder exists
* intent_pipeline.pkl is present

---

## 💡 Expert Delivery Tips

### 📁 data Folder

Ensure:
data/intent_detection_dataset.csv
is included in the project

---

### 🧠 models Folder

Include:
intent_pipeline.pkl

⚠️ If dataset changes, re-train:
python train_intent.py

---

### 📦 Before Zipping

Clean your project:

* Delete **pycache**/
* Delete all_files_content.txt

Result → Smaller, cleaner project zip

---

## 🎯 Final Outcome

This project includes:

* Clean UI (Markdown + Charts)
* Robust Backend (Error handling & resilience)
* Powerful AI Engine

  * Gemma 2B
  * Intent Model
  * Toxicity Filter
* Academic Proofs

  * System specs
  * Log exports

---

## 🏁 Conclusion

This project is ready for:

* University Submission
* Hackathons
* Client Portfolio

---

You built something powerful. Good luck with your submission! 🚀
