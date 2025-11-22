#DhaanyaGarapati
#G01512900
import os
import random
import hashlib
import pandas as pd
import streamlit as st  # type: ignore
import urllib.parse
from dotenv import load_dotenv  # type: ignore

load_dotenv()

from llm_clients import LLMClients
from eval import fluency_rule_based, factuality_score
from judge_llama import llama_fact_check  # type: ignore

# - Basic configuration -
st.set_page_config(page_title="EvalLite", layout="wide")

DOMAINS = ["Biology", "Technology", "Science", "Geography"]
RESULTS_FILE = "hsr_results.csv"

QUALTRICS_SURVEY_ID = "SV_bDjypWg7Mm7GrCm"
QUALTRICS_BASE = f"https://gmu.pdx1.qualtrics.com/jfe/form/{QUALTRICS_SURVEY_ID}"

temperature = 0.2
max_tokens = 350
PROMPT_CHAR_LIMIT = 200
OUTPUT_CHAR_LIMIT = 1000


# - Helper functions -
def _mock_response(name: str) -> str:
    return f"[{name}] Mock response. Disable Mock Mode to call the real API."


@st.cache_data(show_spinner=False)
def _cached_generate(
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    mock: bool,
):
    if mock:
        return _mock_response(model_name)
    clients = LLMClients()
    if model_name == "gpt-4o":
        return clients.generate_openai(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
    elif model_name == "claude-3-haiku-20240307":
        return clients.generate_anthropic(
            prompt, max_tokens=max_tokens, temperature=temperature
        )
    else:
        return f"[unknown model] {model_name}"


def get_ab_order(key: str) -> str:
    """Consistent A/B mapping per (uid, domain, prompt_idx, prompt)."""
    if "ab_map" not in st.session_state:
        st.session_state.ab_map = {}
    if key not in st.session_state.ab_map:
        st.session_state.ab_map[key] = random.choice(
            ["A=GPT, B=Claude", "A=Claude, B=GPT"]
        )
    return st.session_state.ab_map[key]


def load_results_df() -> pd.DataFrame:
    if os.path.exists(RESULTS_FILE):
        return pd.read_csv(RESULTS_FILE)
    return pd.DataFrame(
        columns=[
            "uid",
            "email",
            "domain",
            "prompt_index",
            "prompt",
            "order",
            "A_model",
            "B_model",
            "A_text",
            "B_text",
            "A_fluency",
            "B_fluency",
            "A_factuality",
            "B_factuality",
        ]
    )


def trim_output(text: str, limit: int = OUTPUT_CHAR_LIMIT) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "... [trimmed]"


# - Admin mode (for judge options) -
st.sidebar.header("Admin")
admin_pass = st.sidebar.text_input("Admin passcode", type="password")
ADMIN_MODE = admin_pass == "Dhaanya2025" 


# - API mode for Qualtrics -
# Qualtrics Web Service will call my streamlit url here
params = st.query_params
if params.get("mode") == "api":
    uid = params.get("uid")
    if not uid:
        st.json({})
        st.stop()

    df = load_results_df()
    df_user = df[df["uid"] == uid]

    if df_user.empty:
        st.json({})
        st.stop()

    domain_order = {d: i for i, d in enumerate(DOMAINS)}
    df_user["domain_order"] = df_user["domain"].map(domain_order)
    df_user = df_user.sort_values(["domain_order", "prompt_index"]).head(8)

    payload = {}
    for i, row in enumerate(df_user.itertuples(), start=1):
        payload[f"p{i}"] = str(row.prompt)
        payload[f"A{i}"] = str(row.A_text)
        payload[f"B{i}"] = str(row.B_text)

    st.json(payload)
    st.stop()


# - UI changes -
st.title("EvalLite - Human-Subjects Study Version")

st.sidebar.header("Settings")

# Mock mode disabled for human-subjects study
mock_mode = False

# Admin-only judge toggles
if ADMIN_MODE:
    st.sidebar.subheader("Admin Options (internal use only)")
    neutral_judge = st.sidebar.toggle("Neutral Judge (Llama-as-a-Judge)", value=False)
    deep_factuality = st.sidebar.toggle(
        "Deep Factuality Mode (Llama fact check)", value=False
    )
else:
    neutral_judge = False
    deep_factuality = False

st.markdown(
    """
This version is configured for the **human-subjects study**.

- You will evaluate **2 prompts per domain** (Biology, Technology, Science, Geography).  
- For each step: enter **one prompt (≤ 200 characters)**, then click **Run Comparison**.  
- The app will generate two model outputs (A and B), show them to you, and save them.  
- After all 8 prompts, you'll be sent to a **Qualtrics survey** to rate the responses.  
- We **do not** show any automated scores here to avoid bias.
"""
)

# - Email & UID -
email = st.text_input(
    "Enter your email (used only to link your prompts to the survey):"
)
if not email:
    st.stop()

uid = hashlib.md5(email.strip().lower().encode()).hexdigest()
st.caption(f"Your anonymous participant ID (uid): `{uid}`")

# - Initiating the state machine -
if "domain_idx" not in st.session_state:
    st.session_state.domain_idx = 0 
if "prompt_idx" not in st.session_state:
    st.session_state.prompt_idx = 1 
if "finished" not in st.session_state:
    st.session_state.finished = False
if "has_outputs" not in st.session_state:
    st.session_state.has_outputs = False
if "last_result" not in st.session_state:
    st.session_state.last_result = None

domain_idx = st.session_state.domain_idx
prompt_idx = st.session_state.prompt_idx

if st.session_state.finished or domain_idx >= len(DOMAINS):
    st.success("You have completed all 8 prompts for this study.")

    qualtrics_url = QUALTRICS_BASE + "?" + urllib.parse.urlencode({"uid": uid})
    st.markdown(
        f"### Next step\n"
        f"Click the link below to open the **evaluation survey** in Qualtrics, "
        f"where you will rate the responses you saw:\n\n"
        f"[Go to EvalLite HSR Survey]({qualtrics_url})",
        unsafe_allow_html=True,
    )
    st.stop()

current_domain = DOMAINS[domain_idx]

# - outputs -
if st.session_state.has_outputs and st.session_state.last_result is not None:
    res = st.session_state.last_result

    st.header(f"{current_domain} - Prompt {prompt_idx} of 2")
    st.write("Review the two outputs below, then click **Next Prompt →**.")

    st.subheader("Model Outputs (for your reference)")
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f"### Output A ({res['a_name']})")
        st.code(res["a_text"], language="markdown")

    with colB:
        st.markdown(f"### Output B ({res['b_name']})")
        st.code(res["b_text"], language="markdown")

    if st.button("Next Prompt →"):
        # advance state machine
        if st.session_state.prompt_idx == 1:
            st.session_state.prompt_idx = 2
        else:
            st.session_state.prompt_idx = 1
            st.session_state.domain_idx += 1

            # checking if all 4 domains are done
            if st.session_state.domain_idx >= len(DOMAINS):
                st.session_state.finished = True

        st.session_state.has_outputs = False
        st.session_state.last_result = None

        st.rerun()

    st.stop()

# - Phase 1: Prompt entry -
st.header(f"{current_domain} - Prompt {prompt_idx} of 2")
st.write(
    f"Please enter a {current_domain.lower()}-related prompt "
    f"(max {PROMPT_CHAR_LIMIT} characters)."
)

widget_key = f"prompt_{domain_idx}_{prompt_idx}"

prompt = st.text_area(
    "Your prompt",
    key=widget_key,
    max_chars=PROMPT_CHAR_LIMIT,
    height=120,
    placeholder=f"Type a {current_domain.lower()} question or instruction here...",
)

st.caption(f"{len(prompt)} / {PROMPT_CHAR_LIMIT} characters")

run = st.button("Run Comparison")

if run:
    clean_prompt = prompt.strip()
    if not clean_prompt:
        st.error("Prompt cannot be empty.")
        st.stop()

    key = hashlib.md5(
        f"{uid}|{current_domain}|{prompt_idx}|{clean_prompt}".encode()
    ).hexdigest()
    order = get_ab_order(key)

    if order == "A=GPT, B=Claude":
        a_model, b_model = "gpt-4o", "claude-3-haiku-20240307"
        a_name, b_name = "GPT-4o", "Claude Haiku"
    else:
        a_model, b_model = "claude-3-haiku-20240307", "gpt-4o"
        a_name, b_name = "Claude Haiku", "GPT-4o"

    with st.spinner(
        f"Generating outputs for {current_domain} - Prompt {prompt_idx}..."
    ):
        a_text = _cached_generate(
            a_model, clean_prompt, temperature, max_tokens, mock_mode
        )
        b_text = _cached_generate(
            b_model, clean_prompt, temperature, max_tokens, mock_mode
        )

    # Trimming outputs
    a_text = trim_output(a_text, OUTPUT_CHAR_LIMIT)
    b_text = trim_output(b_text, OUTPUT_CHAR_LIMIT)

    # Optional automated evaluation (admin-mode only)
    a_flu, _ = fluency_rule_based(a_text)
    b_flu, _ = fluency_rule_based(b_text)
    a_fact, a_fact_d = factuality_score(a_text)
    b_fact, b_fact_d = factuality_score(b_text)

    if deep_factuality:
        a_fact_llama = llama_fact_check(clean_prompt, a_text)
        b_fact_llama = llama_fact_check(clean_prompt, b_text)
        a_fact = 0.6 * a_fact + 0.4 * a_fact_llama
        b_fact = 0.6 * b_fact + 0.4 * b_fact_llama

    if neutral_judge:
        a_judge = llama_fact_check(clean_prompt, a_text, category="fluency")
        b_judge = llama_fact_check(clean_prompt, b_text, category="fluency")
    else:
        a_judge = b_judge = None

    # ---- Save to CSV ----
    df_existing = load_results_df()

    new_row = pd.DataFrame(
        [
            {
                "uid": uid,
                "email": email.strip().lower(),
                "domain": current_domain,
                "prompt_index": prompt_idx,
                "prompt": clean_prompt,
                "order": order,
                "A_model": a_model,
                "B_model": b_model,
                "A_text": a_text,
                "B_text": b_text,
                "A_fluency": a_flu,
                "B_fluency": b_flu,
                "A_factuality": a_fact,
                "B_factuality": b_fact,
            }
        ]
    )

    df_all = pd.concat([df_existing, new_row], ignore_index=True)
    df_all.to_csv(RESULTS_FILE, index=False)

    st.success("This prompt and its outputs have been saved for the study dataset.")

    st.download_button(
        label="Download current hsr_results.csv",
        data=df_all.to_csv(index=False),
        file_name="hsr_results.csv",
        mime="text/csv",
    )

    st.session_state.has_outputs = True
    st.session_state.last_result = {
        "a_name": a_name,
        "b_name": b_name,
        "a_text": a_text,
        "b_text": b_text,
    }

    st.rerun()
