#DhaanyaGarapati
#G01512900
import re, requests, streamlit as st # type: ignore

OLLAMA_MODEL = "phi3"  
OLLAMA_URL = "http://localhost:11434/api/generate"

def _extract_score(text: str) -> float:
    """
    Extract a numeric 0–100 value from model output.
    Handles cases like 'I'd rate this 85 out of 100' or 'Score: 92'.
    """
    matches = re.findall(r"\b(\d{1,3})\b", text)
    if not matches:
        return 0.0
    # pick first number in valid range
    for m in matches:
        n = int(m)
        if 0 <= n <= 100:
            return float(n)
    return 0.0

def _ollama_available() -> bool:
    """Check if Ollama service is running."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def llama_fact_check(prompt: str, text: str, category: str = "factuality") -> float:
    """Use local Ollama model to rate fluency/factuality 0–100, with fallback."""
    if not _ollama_available():
        st.sidebar.warning("Ollama not running — using rule-based evaluation only.")
        return 0.0

    try:
        if category == "fluency":
            instruction = (
                "You are an expert linguistic evaluator.\n"
                "Rate how fluent, clear, and grammatically correct the following text is.\n"
                "Respond with only a single integer number between 0 and 100 (no words, no punctuation)."
            )
        else:
            instruction = (
                "You are an expert fact-checker.\n"
                "Rate how factually accurate the text is, considering the given prompt.\n"
                "Respond with only a single integer number between 0 and 100 (no words, no punctuation)."
            )

        full_prompt = (
            f"{instruction}\n\n"
            f"PROMPT:\n{prompt}\n\n"
            f"TEXT:\n{text}\n\n"
            "Your answer must be a single number between 0 and 100, with nothing else."
        )

        response = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": False},
            timeout=60,
        )

        if response.status_code != 200:
            st.sidebar.warning(f"Ollama Judge request failed ({response.status_code}). Using fallback.")
            return 0.0

        output = response.json().get("response", "")
        score = _extract_score(output)

        if score == 0.0:
            st.sidebar.warning(f"Ollama Judge returned no numeric score. Output: {output[:80]}...")
        else:
            st.sidebar.success(f"Ollama Judge active ({OLLAMA_MODEL}) — {category.capitalize()} Score: {score:.0f}/100")

        return score

    except Exception as e:
        st.sidebar.warning(f"Ollama Judge error: {str(e)[:100]}... Using fallback.")
        return 0.0
