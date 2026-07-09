"""Evaluation page — compare RAG answers against the Ground Truth.

Instead of a live chat, this page loads a precomputed evaluation results CSV from
``test/rag_eval/evals/evaluations/`` and, for one question at a time, shows the RAG
answer side-by-side with the reference (Ground Truth) answer. The sources the RAG
used to support its answer are always kept and shown under the RAG answer, so a
divergence from the GT can be traced back to the retrieved passages.

No Neo4j / OpenAI calls happen here — it is a pure read of already-computed results.
Generate a fresh full (53-question) run with, e.g.::

    python -m test.rag_eval.evals_ragas --dataset golden_dataset
"""
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

import config

logger = logging.getLogger(__name__)

EVALUATIONS_DIR = config.EVALS_DIR / "evaluations"

# Columns a CSV must have to be usable as a RAG-vs-GT comparison.
REQUIRED_COLUMNS = ("question", "ground_truth", "answer")

# Numeric score columns we know how to display as metrics (whichever are present).
METRIC_COLUMNS = (
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    "factual_precision", "factual_recall", "precision", "recall", "f1",
)


@st.cache_data(show_spinner=False)
def load_results(path_str: str) -> pd.DataFrame:
    """Load a results CSV. Cached by path so switching files is instant."""
    return pd.read_csv(path_str)


def _has_required_columns(path: Path) -> bool:
    """True if the CSV header contains every required comparison column."""
    try:
        header = pd.read_csv(path, nrows=0).columns
    except (pd.errors.ParserError, OSError, UnicodeDecodeError):
        return False
    return all(col in header for col in REQUIRED_COLUMNS)


def _comparison_csvs() -> list[Path]:
    """CSV files under evaluations/ usable for comparison, newest first."""
    if not EVALUATIONS_DIR.exists():
        return []
    candidates = [p for p in EVALUATIONS_DIR.glob("*.csv") if _has_required_columns(p)]
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def _split_pipe(value: object) -> list[str]:
    """Split a ``|``-joined cell into non-empty parts (empty for NaN/blank)."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


# ── page ──────────────────────────────────────────────────────────────────────

st.title("📊 Evaluation — RAG vs Ground Truth")
st.caption(
    "Confronta la risposta del RAG con la risposta di riferimento (Ground Truth) "
    "sulle domande del golden dataset. I risultati sono letti da un CSV precalcolato."
)

csv_paths = _comparison_csvs()
if not csv_paths:
    st.warning(
        f"Nessun CSV di risultati trovato in `{EVALUATIONS_DIR}` con le colonne "
        f"richieste {REQUIRED_COLUMNS}.\n\nGenera una run con:\n\n"
        "```\npython -m test.rag_eval.evals_ragas --dataset golden_dataset\n```"
    )
    st.stop()

# ── sidebar: which results file + act filter ──────────────────────────────────
with st.sidebar:
    st.header("Risultati")
    csv_path = st.selectbox(
        "File di valutazione (CSV)",
        csv_paths,
        format_func=lambda p: p.name,
        help="I file sono ordinati dal più recente. Solo CSV con question/ground_truth/answer.",
    )

df = load_results(str(csv_path))

with st.sidebar:
    act_filter = None
    if "act" in df.columns:
        acts = ["Tutti", *sorted(df["act"].dropna().unique())]
        act_filter = st.selectbox("Filtro per atto", acts)

view = df if not act_filter or act_filter == "Tutti" else df[df["act"] == act_filter]
view = view.reset_index(drop=True)

if view.empty:
    st.info("Nessuna domanda per il filtro selezionato.")
    st.stop()

st.metric("Domande caricate", len(view))


def _question_label(i: int) -> str:
    row = view.iloc[i]
    act = f"[{row['act']}] " if "act" in view.columns and not pd.isna(row["act"]) else ""
    question = str(row["question"]).replace("\n", " ")
    return f"{i + 1}. {act}{question[:80]}{'…' if len(question) > 80 else ''}"


idx = st.selectbox(
    "Domanda", range(len(view)), format_func=_question_label,
)
row = view.iloc[idx]

st.divider()
st.markdown(f"### ❓ {row['question']}")

# ── metrics for this question (whichever score columns exist) ──────────────────
present_metrics = [m for m in METRIC_COLUMNS if m in view.columns and not pd.isna(row[m])]
if present_metrics:
    cols = st.columns(len(present_metrics))
    for col, name in zip(cols, present_metrics):
        col.metric(name.replace("_", " "), f"{float(row[name]):.3f}")

st.divider()

# ── side-by-side: RAG answer (+ supporting sources) vs Ground Truth ────────────
left, right = st.columns(2)

with left:
    st.subheader("🤖 Risposta RAG")
    st.markdown(str(row["answer"]))

    st.markdown("#### 📎 Fonti a supporto")
    if "sources" in view.columns:
        sources = _split_pipe(row["sources"])
        if sources:
            for src in sources:
                st.markdown(f"- `{src}`")
        else:
            st.caption("Nessuna fonte registrata per questa risposta.")
    else:
        st.caption("Fonti non disponibili in questo CSV.")

    if "contexts" in view.columns:
        contexts = _split_pipe(row["contexts"])
        with st.expander(f"Contesti recuperati ({len(contexts)})"):
            if contexts:
                for i, ctx in enumerate(contexts, 1):
                    st.markdown(f"**[{i}]** {ctx}")
                    st.divider()
            else:
                st.caption("Nessun contesto registrato.")

with right:
    st.subheader("✅ Ground Truth")
    st.markdown(str(row["ground_truth"]))
