"""
Case Law Parser page.

Uploads a case law PDF, infers its hierarchical structure via LLM,
and renders the parsed document tree.
"""
import json
import logging
import tempfile
import re as _re
import config

import streamlit as st
from dotenv import load_dotenv

from service.graph.graph import Neo4jGraph
from utils.streamlit_log_handler import StreamlitLogHandler
from service.case_law.llm_orchestrator import parse_document, create_case_law_kg
from service.case_law.doc_parser import flatten

load_dotenv()

st.title("Case Law Parser")
st.caption(
    "Upload an EU case law PDF. The parser infers the document structure using an LLM "
    "and renders the full hierarchical tree."
)


# -- expandable tree renderer -------------------------------------------------

def _render_node(node, depth: int = 0) -> None:
    indent = "    " * depth
    label = f"{indent}{node.text}"
    with st.expander(label, expanded=(depth == 0)):
        for para in node.body:
            st.markdown(para)
        for child in node.children:
            _render_node(child, depth + 1)


# -- graphviz diagram ---------------------------------------------------------

_DEPTH_FILL = ["#1f4e79", "#2e75b6", "#9dc3e6", "#deeaf1", "#f2f2f2"]
_DEPTH_FONT = ["white",   "white",   "black",   "black",   "black"]


def _build_dot(roots: list) -> str:
    counter = [0]
    lines = [
        "digraph tree {",
        "  rankdir=TB;",
        '  node [shape=box, style=filled, fontname="Helvetica", fontsize=10, margin="0.2,0.1"];',
        '  edge [color="#555555"];',
    ]

    def _add(node, parent_id):
        nid = counter[0]
        counter[0] += 1
        label = node.text if len(node.text) <= 45 else node.text[:44] + "..."
        label = label.replace('"', '\\"').replace("\n", " ")
        d = min(node.depth, len(_DEPTH_FILL) - 1)
        lines.append(
            f'  n{nid} [label="{label}", fillcolor="{_DEPTH_FILL[d]}", fontcolor="{_DEPTH_FONT[d]}"];'
        )
        if parent_id is not None:
            lines.append(f"  n{parent_id} -> n{nid};")
        for child in node.children:
            _add(child, nid)

    for root in roots:
        _add(root, None)

    lines.append("}")
    return "\n".join(lines)


# -- upload -------------------------------------------------------------------

uploaded = st.file_uploader("Upload case law PDF", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

file_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("cl_file_key") != file_key:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(uploaded.read())
    tmp.flush()
    st.session_state["cl_tmp_path"] = tmp.name
    st.session_state["cl_file_key"] = file_key
    st.session_state.pop("cl_result", None)

pdf_path = st.session_state["cl_tmp_path"]

if st.button("Parse Document", type="primary"):
    log_area = st.empty()
    handler = StreamlitLogHandler(log_area)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        with st.spinner("Inferring document structure..."):
            parsing_rules, roots = parse_document(pdf_path)
            st.session_state["cl_result"] = (parsing_rules, roots, flatten(roots))
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        root_logger.removeHandler(handler)

# -- results ------------------------------------------------------------------

if "cl_result" not in st.session_state:
    st.stop()

parsing_rules, roots, flat = st.session_state["cl_result"]

col_info, col_main = st.columns([1, 3])

with col_info:
    st.subheader("Document info")
    st.markdown(f"**Domain:** {parsing_rules.get('domain', '-')}")
    if parsing_rules.get("notes"):
        st.markdown(f"**Notes:** {parsing_rules['notes']}")
    st.divider()
    st.subheader("Inferred rules")
    for rule in parsing_rules.get("rules", []):
        st.markdown(f"- `{rule['pattern']}` ({rule['type']}, depth {rule['depth']})")
    st.divider()
    st.download_button(
        label="Download JSON",
        data=json.dumps(flat, indent=2, ensure_ascii=False),
        file_name=f"{uploaded.name.removesuffix('.pdf')}_parsed.json",
        mime="application/json",
    )

    if st.button("Generate Summaries", type="secondary"):
        st.session_state.pop("cl_summaries", None)
        sections = [s for s in flat if s["heading"] != "Reports of Cases"]
        progress_bar = st.progress(0, text="Starting...")
        summaries: list[dict] = []
        from service.case_law.llm_orchestrator import summarize_section, summarize_document
        for i, section in enumerate(sections):
            label = section["heading"][:48] + "..." if len(section["heading"]) > 48 else section["heading"]
            progress_bar.progress((i) / len(sections), text=f"Summarising: {label}")
            result = summarize_section(section)
            if result is not None:
                summaries.append(result)
        progress_bar.progress(1.0, text="Summarising full document…")
        doc_summary = summarize_document(pdf_path)
        summaries.insert(0, {"heading": "Document Summary", "summary": doc_summary})
        progress_bar.progress(1.0, text="Done.")
        st.session_state["cl_summaries"] = summaries

    if "cl_summaries" in st.session_state:
        st.download_button(
            label="Download Summaries JSON",
            data=json.dumps(st.session_state["cl_summaries"], indent=2, ensure_ascii=False),
            file_name=f"{uploaded.name.removesuffix('.pdf')}_summaries.json",
            mime="application/json",
        )

        st.divider()
        _match = _re.search(r"CELEX_(\w+)_", uploaded.name, _re.IGNORECASE)
        _default_celex = _match.group(1) if _match else ""
        celex_input = st.text_input("CELEX ID", value=_default_celex, placeholder="e.g. 62019CJ0645")
        if st.button("Create Case Law KG", type="primary"):
            if not celex_input.strip():
                st.error("Enter a CELEX ID.")
            else:
                try:
                    with st.spinner(f"Writing {celex_input.strip()} to Neo4j…"):
                        graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
                        graph.verify_connection()
                        create_case_law_kg(
                            celex=celex_input.strip(),
                            flat=flat,
                            summaries=st.session_state["cl_summaries"],
                            graph=graph,
                        )
                        graph.close()
                    st.success(f"KG created for {celex_input.strip()}.")
                except Exception as exc:
                    st.error(f"Error: {exc}")

with col_main:
    tab_tree, tab_diagram = st.tabs(["Document Tree", "Tree Diagram"])

    with tab_tree:
        st.caption("Expand a section to read its content.")
        if not roots:
            st.warning("No structural nodes found.")
        else:
            for root in roots:
                _render_node(root, depth=0)

    with tab_diagram:
        st.caption("Hierarchical structure of the document. Darker blue = higher level.")
        if not roots:
            st.warning("No structural nodes found.")
        else:
            st.graphviz_chart(_build_dot(roots), use_container_width=True)
