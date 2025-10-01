
# EvalLite — Lightweight LLM Comparison (GPT‑4o vs Claude Haiku)

A Streamlit web app that compares two LLMs side-by-side and scores each output for **fluency** and **factuality**.

## Features
- Prompt once → get responses from **OpenAI GPT‑4o** and **Anthropic Claude Haiku**
- **Fluency** score: textstat + spaCy features (0–100)
- **Factuality** score: entity-anchored Wikipedia checks (0–100)
- Optional **LLM‑as‑a‑Judge** ratings
- Download results as CSV

## Quickstart
1. **Python 3.11** recommended
2. Create a venv and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **API keys** (env vars):
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
4. (First run) Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
5. Run app:
   ```bash
   streamlit run app.py
   ```

## Notes
- You can toggle **Mock Mode** in the sidebar if you don't have API keys.
- Scores are illustrative; adjust weights in `eval.py` as needed.
