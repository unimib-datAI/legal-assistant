"""
Case Law Parser page.

Uploads a case law PDF, infers its hierarchical structure via LLM,
and renders the parsed document tree.
"""
import json
import logging
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("Case Law Parser")
st.caption(
    "Upload an EU case law PDF. The parser infers the document structure using an LLM "
    "and renders the full hierarchical tree."
)


# -- log capture --------------------------------------------------------------

class _LogHandler(logging.Handler):
    def __init__(self, container):
        super().__init__()
        self._lines: list[str] = []
        self._container = container
        self.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )

    def emit(self, record: logging.LogRecord) -> None:
        self._lines.append(self.format(record))
        self._container.text_area("Output", "\n".join(self._lines), height=200)


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
    handler = _LogHandler(log_area)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        with st.spinner("Inferring document structure..."):
            from service.case_law.agent import parse_document
            from service.case_law.doc_parser import flatten
            cfg, roots = parse_document(pdf_path)
            st.session_state["cl_result"] = (cfg, roots, flatten(roots))
    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        root_logger.removeHandler(handler)

# -- results ------------------------------------------------------------------

if "cl_result" not in st.session_state:
    st.stop()

cfg, roots, flat = st.session_state["cl_result"]

col_info, col_main = st.columns([1, 3])

with col_info:
    st.subheader("Document info")
    st.markdown(f"**Domain:** {cfg.get('domain', '-')}")
    if cfg.get("notes"):
        st.markdown(f"**Notes:** {cfg['notes']}")
    st.divider()
    st.subheader("Inferred rules")
    for rule in cfg.get("rules", []):
        st.markdown(f"- `{rule['pattern']}` ({rule['type']}, depth {rule['depth']})")
    st.divider()
    st.download_button(
        label="Download JSON",
        data=json.dumps(flat, indent=2, ensure_ascii=False),
        file_name=f"{uploaded.name.removesuffix('.pdf')}_parsed.json",
        mime="application/json",
    )

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
