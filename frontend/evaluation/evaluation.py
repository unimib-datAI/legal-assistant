"""Evaluation page: compare RAG answers against the Ground Truth.

This page loads a precomputed evaluation results CSV from ``evals/evals/evaluations/``
and, for one question at a time, shows the RAG answer side-by-side with the reference
(Ground Truth) answer. The sources the RAG used to support its answer are always kept
and shown under the RAG answer, so a divergence from the GT can be traced back to the
retrieved passages.

No Neo4j / OpenAI calls happen here: it is a pure read of already-computed results.
Generate a fresh full (53-question) run with, e.g.::

    legal-assistant eval ragas --dataset golden_dataset
"""
import json
import logging
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from legal_assistant import config
from legal_assistant.rag.acts import CELEX_TO_ACT_NAME

logger = logging.getLogger(__name__)

EVALUATIONS_DIR = config.EVALS_DIR / "evaluations"

# Article/recital ids: a CELEX id, a kind marker, then the number.
#   '32022R0868art_1' -> Article 1 | '32022R0868rct_46' -> Recital 46
_SOURCE_ID_RE = re.compile(r"^(?P<celex>\d{5}[A-Z]\d{4})(?P<kind>art|rct)_(?P<number>.+)$")

# Paragraph ids emitted by the topics retriever: zero-padded article, then paragraph.
#   '32016R0679_005.001' -> Article 5(1) | '32016R0679_004.0' -> Article 4
_PARAGRAPH_ID_RE = re.compile(
    r"^(?P<celex>\d{5}[A-Z]\d{4})_(?P<article>\d+)\.(?P<paragraph>\d+)$"
)

_KIND_LABEL = {"art": "Article", "rct": "Recital"}

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


def _format_source_id(source_id: str) -> str:
    """'32022R0868art_1' -> 'Data Governance Act, Article 1'. Unparsable ids pass through."""
    if match := _SOURCE_ID_RE.match(source_id):
        act = CELEX_TO_ACT_NAME.get(match["celex"], match["celex"])
        return f"{act}, {_KIND_LABEL[match['kind']]} {match['number']}"

    if match := _PARAGRAPH_ID_RE.match(source_id):
        act = CELEX_TO_ACT_NAME.get(match["celex"], match["celex"])
        article = int(match["article"])
        paragraph = int(match["paragraph"])
        # Paragraph 0 means the id points at the article as a whole.
        suffix = f"({paragraph})" if paragraph else ""
        return f"{act}, Article {article}{suffix}"

    return source_id


def _load_json_column(value: object) -> list[dict]:
    """Parse a JSON cell written by backfill_attribution; [] when absent or malformed."""
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        logger.warning("Unparsable JSON cell; ignoring attribution for this row.")
        return []
    return parsed if isinstance(parsed, list) else []


def _render_attributed_answer(segments: list[dict]) -> None:
    """Render the answer sentence by sentence, each followed by its [Sn] badges."""
    parts = []
    for segment in segments:
        badges = " ".join(f":blue-badge[{m}]" for m in segment.get("markers", []))
        parts.append(f"{segment.get('text', '')} {badges}".strip())
    st.markdown(" ".join(parts))


def _split_pipe(value: object) -> list[str]:
    """Split a ``|``-joined cell into non-empty parts (empty for NaN/blank)."""
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


# ── page ──────────────────────────────────────────────────────────────────────

st.title("📊 Evaluation: RAG vs Ground Truth")
st.caption(
    "Confronta la risposta del RAG con la risposta di riferimento (Ground Truth) "
    "sulle domande del golden dataset. I risultati sono letti da un CSV precalcolato."
)

csv_paths = _comparison_csvs()
if not csv_paths:
    st.warning(
        f"Nessun CSV di risultati trovato in `{EVALUATIONS_DIR}` con le colonne "
        f"richieste {REQUIRED_COLUMNS}.\n\nGenera una run con:\n\n"
        "```\nlegal-assistant eval ragas --dataset golden_dataset\n```"
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

    segments = _load_json_column(row["segments"]) if "segments" in view.columns else []
    if segments:
        _render_attributed_answer(segments)
    else:
        st.markdown(str(row["answer"]))

    st.markdown("#### 📎 Fonti a supporto")
    if "sources" not in view.columns:
        st.caption("Fonti non disponibili in questo CSV.")
    else:
        retrieved = _split_pipe(row["sources"])
        # `contexts` is positionally aligned with `sources`, so a passage can be
        # looked up by the id of the source that carries it.
        contexts = str(row["contexts"]).split("|") if "contexts" in view.columns else []
        context_by_id = dict(zip(retrieved, contexts))
        cited = _load_json_column(row["cited_sources"]) if "cited_sources" in view.columns else []

        if cited:
            # The passages the answer cites: the [Sn] badges in the text point here.
            for ref in cited:
                label = f"{ref['marker']} · {_format_source_id(ref['id'])}"
                with st.expander(label):
                    st.markdown(context_by_id.get(ref["id"], "_Chunk non disponibile._"))

            # Retrieval is high-recall, so some passages reach the synthesis prompt
            # without ever being cited. Listing them makes that noise visible.
            cited_ids = {ref["id"] for ref in cited}
            uncited = [src for src in retrieved if src not in cited_ids]
            if uncited:
                st.markdown(f"#### 🗑️ Fonti recuperate ma non utilizzate ({len(uncited)}/{len(retrieved)})")
                for src in uncited:
                    with st.expander(_format_source_id(src)):
                        st.markdown(context_by_id.get(src, "_Chunk non disponibile._"))
        elif retrieved:
            for src in retrieved:
                with st.expander(_format_source_id(src)):
                    st.markdown(context_by_id.get(src, "_Chunk non disponibile._"))
        else:
            st.caption("Nessuna fonte registrata per questa risposta.")

with right:
    st.subheader("✅ Ground Truth")
    st.markdown(str(row["ground_truth"]))
