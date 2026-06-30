"""Chat page — ask legal questions and switch between RAG methods at runtime.

The sidebar lets the user pick a RAG method and tune its hyperparameters (the
controls are auto-generated from each method's ``param_specs``). Every answer is
rendered as atomic claims, each annotated with the exact source(s) it was drawn
from.
"""
import logging

import streamlit as st

from service.rag.engine import RagEngine
from service.rag.methods.base import AttributedAnswer, ParamSpec
from service.rag.methods.context import RagContext
from service.rag.methods.registry import list_methods
from utils.streamlit_log_handler import StreamlitLogHandler

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading models and connecting to Neo4j…")
def get_engine() -> RagEngine:
    """Build the shared RAG resources once, reused across reruns and methods."""
    return RagEngine(RagContext())


def _param_widget(spec: ParamSpec, key: str):
    """Render one hyperparameter control and return its current value."""
    if spec.kind == "bool":
        return st.toggle(spec.label, value=spec.default, key=key, help=spec.help or None)
    if spec.kind == "int":
        return st.slider(
            spec.label,
            min_value=int(spec.min), max_value=int(spec.max),
            value=int(spec.default), step=int(spec.step or 1),
            key=key, help=spec.help or None,
        )
    # float
    return st.slider(
        spec.label,
        min_value=float(spec.min), max_value=float(spec.max),
        value=float(spec.default), step=float(spec.step or 0.05),
        key=key, help=spec.help or None,
    )


def _render_answer(answer: AttributedAnswer) -> None:
    """Render an attributed answer: claims with source badges + a sources panel."""
    parts = []
    for seg in answer.segments:
        badges = " ".join(f":blue-badge[{m}]" for m in seg.source_markers)
        parts.append(f"{seg.text} {badges}".strip())
    if parts:
        st.markdown(" ".join(parts))

    if answer.sources:
        with st.expander(f"Fonti ({len(answer.sources)})"):
            for ref in answer.sources:
                st.markdown(f"**{ref.marker}** · {ref.act} — {ref.title}  \n`{ref.doc_id}`")


# ── page ────────────────────────────────────────────────────────────────────

st.title("💬 Legal Assistant Chat")
st.caption("Ask a question about the GDPR, AI Act, Data Act or Data Governance Act.")

methods = list_methods()

# ── sidebar: method + hyperparameters ─────────────────────────────────────────
with st.sidebar:
    st.header("RAG method")
    method = st.selectbox(
        "Method", methods, format_func=lambda m: m.name,
    )
    if method.description:
        st.caption(method.description)

    st.divider()
    st.subheader("Hyperparameters")
    config = {
        spec.name: _param_widget(spec, key=f"param_{method.id}_{spec.name}")
        for spec in method.param_specs()
    }

# ── chat history ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg["content"], AttributedAnswer):
            _render_answer(msg["content"])
        else:
            st.markdown(msg["content"])

# ── input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a legal question…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        log_box = st.expander("Retrieval log")
        handler = StreamlitLogHandler(log_box.empty())
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)
        try:
            with st.spinner("Retrieving and synthesising…"):
                engine = get_engine()
                answer = engine.answer(method.id, question, config)
        except Exception as exc:  # surface the failure in the chat instead of crashing the page
            logger.exception("Chat query failed")
            answer = AttributedAnswer(
                segments=[],
                sources=[],
                raw_answer=f"Error: {exc}",
            )
            st.error(f"Error: {exc}")
        finally:
            root_logger.removeHandler(handler)

        if answer.segments:
            _render_answer(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
