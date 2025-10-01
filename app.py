
import os
import time
import pandas as pd
import streamlit as st

from llm_clients import LLMClients
from eval import fluency_rule_based, factuality_score

st.set_page_config(page_title="EvalLite — GPT-4o vs Claude Haiku", layout="wide")

st.sidebar.title("EvalLite Settings")
mock_mode = st.sidebar.toggle("Mock mode (no API calls)", value=not (os.getenv("OPENAI_API_KEY") and os.getenv("ANTHROPIC_API_KEY")))
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.slider("Max tokens", 64, 1024, 400, 32)
use_llm_judge = st.sidebar.toggle("Use LLM-as-a-Judge (fluency only)", value=False)

st.title("EvalLite — Compare GPT-4o vs Claude Haiku")
st.caption("Enter a prompt once. See two model outputs, with fluency and factuality scores.")

prompt = st.text_area("Your prompt", height=140, placeholder="e.g., Summarize this abstract in three sentences and name any datasets.")
go = st.button("Compare")

if "history" not in st.session_state:
    st.session_state.history = []

def _mock_response(name: str) -> str:
    return f"""[{name}] This is a mock response. Replace with real API output by disabling Mock Mode in the sidebar.
It demonstrates how text will be evaluated for fluency and factuality."""

def llm_judge_score(text: str) -> float:
    # Placeholder: for now mirror rule-based score (keeps app usable in mock mode)
    val, _ = fluency_rule_based(text)
    return val

colA, colB = st.columns(2)

if go and prompt.strip():
    with st.spinner("Generating..."):
        start = time.time()
        oa_text, an_text = "", ""
        if mock_mode:
            oa_text = _mock_response("GPT-4o")
            an_text = _mock_response("Claude Haiku")
        else:
            clients = LLMClients()
            try:
                oa_text = clients.generate_openai(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                oa_text = f"[OpenAI error] {e}"
            try:
                an_text = clients.generate_anthropic(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as e:
                an_text = f"[Anthropic error] {e}"

        oa_flu, oa_feats = fluency_rule_based(oa_text)
        an_flu, an_feats = fluency_rule_based(an_text)
        oa_fact, oa_fact_d = factuality_score(oa_text)
        an_fact, an_fact_d = factuality_score(an_text)

        oa_flu_llm = llm_judge_score(oa_text) if use_llm_judge else None
        an_flu_llm = llm_judge_score(an_text) if use_llm_judge else None

        elapsed = time.time() - start

    with colA:
        st.subheader("OpenAI GPT-4o")
        st.code(oa_text, language="markdown")
        st.metric("Fluency (rule)", f"{oa_flu:.1f}/100")
        if oa_flu_llm is not None:
            st.metric("Fluency (LLM-judge)", f"{oa_flu_llm:.1f}/100")
        st.metric("Factuality", f"{oa_fact:.1f}/100")
        with st.expander("Details (fluency features)"):
            st.json(oa_feats)
        with st.expander("Details (entities)"):
            st.json(oa_fact_d)

    with colB:
        st.subheader("Anthropic Claude Haiku")
        st.code(an_text, language="markdown")
        st.metric("Fluency (rule)", f"{an_flu:.1f}/100")
        if an_flu_llm is not None:
            st.metric("Fluency (LLM-judge)", f"{an_flu_llm:.1f}/100")
        st.metric("Factuality", f"{an_fact:.1f}/100")
        with st.expander("Details (fluency features)"):
            st.json(an_feats)
        with st.expander("Details (entities)"):
            st.json(an_fact_d)

    df = pd.DataFrame([
        {"Model": "GPT-4o", "Fluency(rule)": oa_flu, "Fluency(judge)": oa_flu_llm, "Factuality": oa_fact},
        {"Model": "Claude Haiku", "Fluency(rule)": an_flu, "Fluency(judge)": an_flu_llm, "Factuality": an_fact},
    ])
    st.divider()
    st.subheader("Summary")
    st.dataframe(df, use_container_width=True)
    st.caption(f"Elapsed: {elapsed:.2f}s")

    st.download_button("Download results CSV", data=df.to_csv(index=False), file_name="evallite_results.csv", mime="text/csv")
else:
    st.info("Enter a prompt and click Compare. You can enable Mock mode if you don't have API keys.")
