"""
ASKE Pipeline page.

Extracts and enriches legal concepts from the knowledge graph.
"""
import logging

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
import config  # noqa: E402

st.title("ASKE Topic Extraction")
st.caption(
    "Extracts legal concepts from the knowledge graph using iterative "
    "seed-based classification and terminology enrichment."
)

# ── log capture ───────────────────────────────────────────────────────────────

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
        self._container.text_area("Output", "\n".join(self._lines), height=350)


# ── parameters ────────────────────────────────────────────────────────────────

col_n, col_a, col_b, col_g = st.columns(4)
with col_n:
    n_generations = st.number_input("Generations", min_value=1, max_value=50, value=15)
with col_a:
    alpha = st.slider("α — classification threshold", 0.0, 1.0, 0.4, 0.05)
with col_b:
    beta = st.slider("β — enrichment threshold", 0.0, 1.0, 0.4, 0.05)
with col_g:
    gamma = st.number_input("γ — max new terms / concept", min_value=1, max_value=30, value=7)

if st.button("Run ASKE Pipeline", type="primary"):
    log_area = st.empty()
    handler = _LogHandler(log_area)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        with st.spinner("Running ASKE cycle…"):
            import json
            import pathlib

            from service.graph.graph import Neo4jGraph
            from service.graph.seed import SEEDS
            from service.text.preprocessor import TextPreprocessor
            from service.topic.aske import ASKETopicExtractor

            graph = Neo4jGraph(config.NEO4J_URI, config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
            aske = ASKETopicExtractor(graph)
            preprocessor = TextPreprocessor()

            paragraphs = graph.get_paragraphs_from_kg()
            chunks = preprocessor.to_chunks(paragraphs, skip_first=True)

            concepts, final_classifications = aske.run_aske_cycle(
                chunks=chunks,
                seeds=SEEDS,
                n_generations=int(n_generations),
                alpha=float(alpha),
                beta=float(beta),
                gamma=int(gamma),
            )

            active = [c for c in concepts if c.get("active", True)]
            inactive = [c for c in concepts if not c.get("active", True)]

            paragraph_topics = aske.aggregate_topics_by_paragraph(final_classifications, top_n=3)
            updated_count = graph.update_paragraph_topics(paragraph_topics)

            report_path = pathlib.Path("results/aske_result_1.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report = [
                {
                    "label": c["label"],
                    "terms": sorted(
                        {t["label"] if isinstance(t, dict) else t for t in c.get("terms", [])}
                    ),
                }
                for c in sorted(active, key=lambda x: x["label"])
            ]
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        st.success(
            f"ASKE complete — {len(active)} active concepts, "
            f"{len(inactive)} inactive, {updated_count} paragraphs updated."
        )

        if active:
            import pandas as pd

            st.subheader("Active concepts")
            rows = [
                {
                    "Concept": c["label"],
                    "Terms": len(c.get("terms", [])),
                    "Sample terms": ", ".join(
                        sorted(
                            {t["label"] if isinstance(t, dict) else t for t in c.get("terms", [])}
                        )[:5]
                    ),
                }
                for c in sorted(active, key=lambda x: x["label"])
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    except Exception as exc:
        st.error(f"Error: {exc}")
    finally:
        root_logger.removeHandler(handler)
