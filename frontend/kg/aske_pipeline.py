"""
ASKE Pipeline page.

Extracts and enriches legal concepts from the knowledge graph. The cycle itself lives in
``legal_assistant.pipelines.aske_run``: this page collects the parameters, streams the
logs, and renders the resulting concepts.
"""
import json
import pathlib

import pandas as pd
import streamlit as st
from utils.streamlit_log_handler import stream_logs

from legal_assistant.pipelines.aske_run import AskeParams, run_aske

REPORT_PATH = pathlib.Path("results/aske_result_1.json")

st.title("ASKE Topic Extraction")
st.caption(
    "Extracts legal concepts from the knowledge graph using iterative "
    "seed-based classification and terminology enrichment."
)

# ── parameters ────────────────────────────────────────────────────────────────

defaults = AskeParams()
col_n, col_a, col_b, col_g = st.columns(4)
with col_n:
    n_generations = st.number_input("Generations", min_value=1, max_value=50, value=defaults.n_generations)
with col_a:
    alpha = st.slider("α - classification threshold", 0.0, 1.0, defaults.alpha, 0.05)
with col_b:
    beta = st.slider("β - enrichment threshold", 0.0, 1.0, defaults.beta, 0.05)
with col_g:
    gamma = st.number_input("γ - max new terms / concept", min_value=1, max_value=30, value=defaults.gamma)

if st.button("Run ASKE Pipeline", type="primary"):
    params = AskeParams(
        n_generations=int(n_generations),
        alpha=float(alpha),
        beta=float(beta),
        gamma=int(gamma),
    )

    try:
        with stream_logs(), st.spinner("Running ASKE cycle…"):
            result = run_aske(params)

        report = result.as_report()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        st.success(
            f"ASKE complete: {len(result.active)} active concepts, "
            f"{len(result.inactive)} inactive, {result.updated_paragraphs} paragraphs updated."
        )

        if report:
            st.subheader("Active concepts")
            rows = [
                {
                    "Concept": concept["label"],
                    "Terms": len(concept["terms"]),
                    "Sample terms": ", ".join(concept["terms"][:5]),
                }
                for concept in report
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    except Exception as exc:
        st.error(f"Error: {exc}")
