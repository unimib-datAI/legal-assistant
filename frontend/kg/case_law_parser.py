"""
Case Law Parser page.

Fetches a CJEU judgment from EUR-Lex by CELEX id, reads its hierarchy straight
from the published XHTML markup, and renders the parsed document tree.
"""
import json

import streamlit as st

from legal_assistant.case_law.kg_builder import create_case_law_kg
from legal_assistant.case_law.llm_orchestrator import parse_document
from legal_assistant.case_law.tree import flatten
from legal_assistant.resources import make_graph_client
from legal_assistant.validation.gate import GraphValidationError

st.title("Case Law Parser")
st.caption(
    "Enter the CELEX id of a CJEU judgment. Its structure is read from the EUR-Lex "
    "XHTML markup, no PDF, no inference."
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


# -- input --------------------------------------------------------------------

celex = st.text_input("CELEX id", placeholder="e.g. 62012CJ0293").strip().upper()

if st.button("Parse Document", type="primary", disabled=not celex):
    st.session_state.pop("cl_summaries", None)
    try:
        with st.spinner(f"Fetching and parsing {celex}…"):
            roots = parse_document(celex)
            st.session_state["cl_result"] = (celex, roots, flatten(roots))
    except Exception as exc:
        st.session_state.pop("cl_result", None)
        st.error(f"Error: {exc}")

# -- results ------------------------------------------------------------------

if "cl_result" not in st.session_state:
    st.info("Enter a CELEX id to get started.")
    st.stop()

celex, roots, flat = st.session_state["cl_result"]

col_info, col_main = st.columns([1, 3])

with col_info:
    st.subheader("Document info")
    st.markdown(f"**CELEX:** `{celex}`")
    st.markdown(f"**Sections:** {len(flat)}")
    st.markdown(f"**Paragraphs:** {sum(len(s['body']) for s in flat)}")
    st.markdown(f"**Max depth:** {max(s['depth'] for s in flat)}")
    st.divider()
    st.download_button(
        label="Download JSON",
        data=json.dumps(flat, indent=2, ensure_ascii=False),
        file_name=f"{celex}_parsed.json",
        mime="application/json",
    )

    if st.button("Generate Summaries", type="secondary"):
        st.session_state.pop("cl_summaries", None)
        sections = [s for s in flat if s["heading"] != "Reports of Cases"]
        progress_bar = st.progress(0, text="Starting...")
        summaries: list[dict] = []
        from legal_assistant.case_law.llm_orchestrator import summarize_section, summarize_document
        for i, section in enumerate(sections):
            label = section["heading"][:48] + "..." if len(section["heading"]) > 48 else section["heading"]
            progress_bar.progress((i) / len(sections), text=f"Summarising: {label}")
            result = summarize_section(section)
            if result is not None:
                summaries.append(result)
        progress_bar.progress(1.0, text="Summarising full document…")
        doc_summary = summarize_document(roots)
        summaries.insert(0, {"heading": "Document Summary", "summary": doc_summary})
        progress_bar.progress(1.0, text="Done.")
        st.session_state["cl_summaries"] = summaries

    if "cl_summaries" in st.session_state:
        st.download_button(
            label="Download Summaries JSON",
            data=json.dumps(st.session_state["cl_summaries"], indent=2, ensure_ascii=False),
            file_name=f"{celex}_summaries.json",
            mime="application/json",
        )

        st.divider()
        if st.button("Create Case Law KG", type="primary"):
            try:
                with st.spinner(f"Writing {celex} to Neo4j…"):
                    graph = make_graph_client()
                    graph.verify_connection()
                    create_case_law_kg(
                        celex=celex,
                        flat=flat,
                        summaries=st.session_state["cl_summaries"],
                        graph=graph,
                    )
                    graph.close()
                st.success(f"KG created for {celex}.")
            except GraphValidationError as exc:
                st.error(f"Validation failed, nothing was written for {celex}:")
                st.code(exc.report(), language="text")
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
