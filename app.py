import os, random, hashlib, pandas as pd, streamlit as st # type: ignore
from dotenv import load_dotenv # type: ignore
load_dotenv()
from llm_clients import LLMClients
from eval import fluency_rule_based, factuality_score
from judge_llama import llama_fact_check # type: ignore

# --- Streamlit config ---
st.set_page_config(page_title="EvalLite — Controlled Inputs", layout="wide")
st.title("EvalLite — Controlled LLM Evaluation")

# --- Sidebar toggles ---
st.sidebar.header("Settings")
dataset_mode = st.sidebar.toggle("Dataset mode (controlled inputs)", value=True)
mock_mode = st.sidebar.toggle("Mock mode (offline demo)", value=False)
neutral_judge = st.sidebar.toggle("Neutral Judge (Llama-as-a-Judge)", value=False)
deep_factuality = st.sidebar.toggle("Deep Factuality Mode (Llama fact check)", value=False)

# Fixed parameters for reproducibility
temperature = 0.2
max_tokens = 350

# --- Helper functions ---
def _mock_response(name: str) -> str:
    return f"[{name}] Mock response. Disable Mock Mode to call the real API."

@st.cache_data(show_spinner=False)
def _cached_generate(model_name: str, prompt: str, temperature: float, max_tokens: int, mock: bool):
    if mock:
        return _mock_response(model_name)
    clients = LLMClients()
    if model_name == "gpt-4o":
        return clients.generate_openai(prompt, max_tokens=max_tokens, temperature=temperature)
    elif model_name == "claude-3-haiku-20240307":
        return clients.generate_anthropic(prompt, max_tokens=max_tokens, temperature=temperature)
    else:
        return f"[unknown model] {model_name}"

# --- Dataset mode ---
if dataset_mode:
    st.info("Dataset Mode: choose an input from the controlled dataset CSV.")

    # Step 1: upload or fallback to local file
    upl = st.file_uploader("Upload dataset CSV", type=["csv"])

    if upl:
        try:
            df = pd.read_csv(upl, encoding="utf-8")
        except (UnicodeDecodeError, pd.errors.ParserError):
            df = pd.read_csv(upl, encoding="latin1",on_bad_lines='skip')
        st.sidebar.success(f"Loaded uploaded dataset ({len(df)} records).")
    elif os.path.exists("EvalLite_Dataset.csv"):
        df = pd.read_csv("EvalLite_Dataset.csv", encoding="utf-8")
        st.sidebar.success(f"Loaded local file EvalLite_Dataset.csv ({len(df)} records).")
    elif os.path.exists("dataset_template.csv"):
        df = pd.read_csv("dataset_template.csv", encoding="utf-8")
        st.sidebar.success(f"Loaded default dataset_template.csv ({len(df)} records).")
    else:
        st.warning("No dataset file uploaded or found locally.")
        st.stop()

    # Step 2: sanity check
    if df.empty:
        st.warning("No dataset found. Please upload one.")
        st.stop()

    if "id" not in df.columns:
        st.warning("Missing 'id' column. Using row numbers instead.")
        df["id"] = range(1, len(df) + 1)

    # Step 3: select record
    item = st.selectbox("Choose dataset item", df["id"].astype(str).tolist())
    row = df[df["id"].astype(str) == str(item)].iloc[0]

    # Step 4: show details
    st.subheader(f"Item {row['id']} — {row.get('task_type','N/A')} ({row.get('domain','N/A')})")
    st.write("**Input**")
    st.write(row.get("input_text", ""))

    if isinstance(row.get("question"), str) and row["question"].strip():
        st.write("**Question**")
        st.write(row["question"])

    if isinstance(row.get("constraints"), str) and row["constraints"].strip():
        st.write("**Constraints**")
        st.write(row["constraints"])

    # Step 5: build prompt
    prompt = row.get("input_text", "")
    if isinstance(row.get("question"), str) and row["question"].strip():
        prompt += f"\n\nQuestion: {row['question']}"
    if isinstance(row.get("constraints"), str) and row["constraints"].strip():
        prompt += f"\n\nConstraints: {row['constraints']}"

else:
    st.caption("Free Prompt Mode (not controlled).")
    prompt = st.text_area("Enter your prompt", height=150,
                          placeholder="Paste paragraph + question, or any task...")
    if not prompt.strip():
        st.stop()

# --- Comparison logic ---
go = st.button("Run Comparison")

if "ab_map" not in st.session_state:
    st.session_state.ab_map = {}

def get_ab_order(key: str):
    if key not in st.session_state.ab_map:
        st.session_state.ab_map[key] = random.choice(["A=GPT, B=Claude", "A=Claude, B=GPT"])
    return st.session_state.ab_map[key]

if go:
    key = hashlib.md5(prompt.encode()).hexdigest()
    order = get_ab_order(key)
    if order == "A=GPT, B=Claude":
        a_model, b_model = "gpt-4o", "claude-3-haiku-20240307"
        a_name, b_name = "GPT-4o", "Claude Haiku"
    else:
        a_model, b_model = "claude-3-haiku-20240307", "gpt-4o"
        a_name, b_name = "Claude Haiku", "GPT-4o"

    with st.spinner("Generating model outputs..."):
        a_text = _cached_generate(a_model, prompt, temperature, max_tokens, mock_mode)
        b_text = _cached_generate(b_model, prompt, temperature, max_tokens, mock_mode)

    # --- Evaluate fluency & factuality ---
    a_flu, _ = fluency_rule_based(a_text)
    b_flu, _ = fluency_rule_based(b_text)
    a_fact, a_fact_d = factuality_score(a_text)
    b_fact, b_fact_d = factuality_score(b_text)

    # Optional Deep Factuality check via Llama
    if deep_factuality:
        a_fact_llama = llama_fact_check(prompt, a_text)
        b_fact_llama = llama_fact_check(prompt, b_text)
        a_fact = 0.6 * a_fact + 0.4 * a_fact_llama
        b_fact = 0.6 * b_fact + 0.4 * b_fact_llama

    # Optional Neutral Judge (fluency)
    if neutral_judge:
        a_judge = llama_fact_check(prompt, a_text, category="fluency")
        b_judge = llama_fact_check(prompt, b_text, category="fluency")
    else:
        a_judge = b_judge = None

    # --- Display results ---
    colA, colB = st.columns(2)

    # ----------- OUTPUT A -----------
    with colA:
        st.subheader(f"Output A ({a_name})")
        st.code(a_text, language="markdown")

        # --- Fluency metrics ---
        st.markdown("### Fluency Evaluation")
        st.metric("Fluency (Rule-based)", f"{a_flu:.1f}/100")
        if a_judge is not None:
            st.metric("Fluency (Llama Judge)", f"{a_judge:.1f}/100")

        # --- Factuality metrics ---
        st.markdown("### Factuality Evaluation")
        col1, col2 = st.columns(2)
        col1.metric(
            label="Factuality (Rule-based)",
            value=f"{a_fact:.1f}/100",
            help="Uses entity validation and Wikipedia lookups for factual consistency."
        )

        # Deep factuality toggle output
        if deep_factuality:
            col2.metric(
                label="Factuality (Deep – Llama Judge)",
                value=f"{a_fact_llama:.1f}/100",
                help="Uses local Llama model reasoning for semantic factuality judgment."
            )
        else:
            col2.metric(
                label="Factuality (Deep – Llama Judge)",
                value="–",
                help="Enable 'Deep Factuality Mode' toggle to activate Llama-based judging."
            )

        # --- Details expander ---
        with st.expander("Entity Details & Explanation"):
            st.write("**Entities Identified:**")
            st.json(a_fact_d)
            st.caption(
                f"Rule-based factuality: {a_fact:.1f}% of entities validated. "
                + ("Llama judge score: {:.1f}/100.".format(a_fact_llama) if deep_factuality else "")
            )

    # ----------- OUTPUT B -----------
    with colB:
        st.subheader(f"Output B ({b_name})")
        st.code(b_text, language="markdown")

        # --- Fluency metrics ---
        st.markdown("### Fluency Evaluation")
        st.metric("Fluency (Rule-based)", f"{b_flu:.1f}/100")
        if b_judge is not None:
            st.metric("Fluency (Llama Judge)", f"{b_judge:.1f}/100")

        # --- Factuality metrics ---
        st.markdown("### Factuality Evaluation")
        col3, col4 = st.columns(2)
        col3.metric(
            label="Factuality (Rule-based)",
            value=f"{b_fact:.1f}/100",
            help="Uses entity validation and Wikipedia lookups for factual consistency."
        )

        if deep_factuality:
            col4.metric(
                label="Factuality (Deep – Llama Judge)",
                value=f"{b_fact_llama:.1f}/100",
                help="Uses local Llama model reasoning for semantic factuality judgment."
            )
        else:
            col4.metric(
                label="Factuality (Deep – Llama Judge)",
                value="–",
                help="Enable 'Deep Factuality Mode' toggle to activate Llama-based judging."
            )

        with st.expander("Entity Details & Explanation"):
            st.write("**Entities Identified:**")
            st.json(b_fact_d)
            st.caption(
                f"Rule-based factuality: {b_fact:.1f}% of entities validated. "
                + ("Llama judge score: {:.1f}/100.".format(b_fact_llama) if deep_factuality else "")
            )

    # --- Save results ---
    df_log = pd.DataFrame([{
        "order": order,
        "prompt": prompt,
        "A_model": a_model, "B_model": b_model,
        "A_fluency": a_flu, "B_fluency": b_flu,
        "A_factuality": a_fact, "B_factuality": b_fact,
        "A_text": a_text, "B_text": b_text
    }])

    if os.path.exists("results.csv"):
        old = pd.read_csv("results.csv")
        df_log = pd.concat([old, df_log], ignore_index=True)

    df_log.to_csv("results.csv", index=False)
    st.success(" Results appended to results.csv")

    st.download_button(
        label=" Download latest results.csv",
        data=df_log.to_csv(index=False),
        file_name="results.csv",
        mime="text/csv",
        help="Download this session’s appended results file."
    )  
