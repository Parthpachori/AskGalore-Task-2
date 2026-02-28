# Healthcare Physiotherapy Assistant ⚕️

An AI-powered Physiotherapy Assistant built with **LangGraph**, **FastAPI**, and **RAG** (Retrieval-Augmented Generation). It reads physiotherapy guidelines from PDFs, stores them in a FAISS vector database, and uses multi-agent LangGraph workflows to answer patient queries conversationally while strictly enforcing clinical safety protocols.

## ✨ Features

- **Semantic Memory**: Contextual routing and session persistence (SQLite) to track patient conditions over multiple turns.
- **Data Ingestion via RAG**: Extracts medical PDFs, uses semantic chunking to preserve exercise instructions, and enriches chunks with metadata using LLMs.
- **Clinical Safety Guardrails**: A dedicated safety screening node intercepts "red-flag" symptoms (e.g., severe swelling, numbness) and triggers an emergency medical escalation instead of returning generic advice.
- **Structured Medical Responses**: Answers follow a strict clinical format: *Condition Overview → Exercise Guidance → Precautions → Next Steps*.
- **Modern Chat UI**: Built-in vanilla HTML/JS/CSS frontend featuring glassmorphism design, async typing indicators, and markdown formatting.
- **OpenRouter Support**: Highly customizable LLM routing (currently configured for `meta-llama/llama-3-8b-instruct:free`).

## 🏗️ Architecture Design (LangGraph)

The core brain of the assistant is an event-driven state machine built in LangGraph.

**Flow:** `User Query → Analyze Query (Intent + Safety Check) → [Conditional Routing] → Retrieve Guidelines → Generate Response`

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- An API key for OpenRouter/OpenAI.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Parthpachori/AskGalore-Task-2.git
cd "AskGalore Task 2/PhysioAssistant"

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the `PhysioAssistant` root directory and add your OpenRouter or OpenAI keys:
```env
OPENROUTER_API_KEY=your_openrouter_key
# OPENAI_API_KEY=your_openai_key_here (If using OpenAI directly)
```

### 4. Step 1: Ingest Data
Put your PDF in `data/therapeutic-exercise.pdf` and run the ingestion pipeline. This process will chunk the text, semantically tag it, and store embeddings in a local `FAISS` vector database.
```bash
python src/ingestion.py
```

### 5. Step 2: Run the Application
Launch the FastAPI server (which also serves the frontend):
```bash
python src/main.py
```

### 6. Start Chatting!
Open your browser and navigate to:
**http://localhost:8000**

---

## 📂 Project Structure

```
AskGalore Task 2/
├── PhysioAssistant/
│   ├── src/
│   │   ├── main.py         # FastAPI application and routing
│   │   ├── graph.py        # LangGraph workflow constraints and AI logic
│   │   └── ingestion.py    # Document Extraction & FAISS DB loading
│   ├── static/
│   │   └── index.html      # Glassmorphism Chat UI
│   ├── data/
│   │   └── therapeutic-exercise.pdf  # Source data corpus
│   ├── db/
│   │   └── physio_index/   # Generated FAISS Vectors
│   ├── .env                # API Keys
│   └── requirements.txt
└── .gitignore
```

## ⚠️ Medical Disclaimer
This project is an experimental prototype. The AI model **does not provide medical diagnoses**. All recommendations are purely informational and drawn from indexed texts. Users should always consult a licensed physiotherapist or medical professional before beginning any new exercise or rehabilitation regimen.
