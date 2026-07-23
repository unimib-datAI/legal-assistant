"""What the graph stores as text, retrieval must be able to reach."""
from __future__ import annotations

from legal_assistant.pipelines.graph_build import EMBEDDED_LABELS

# The labels ``validation.act_source.reconstructed_fragments`` counts as act text. Kept here
# as a literal rather than imported, so that adding a label there without an index fails
# loudly instead of silently agreeing with itself.
TEXT_BEARING_LABELS = ("Paragraph", "Recital", "AnnexPoint")


def test_every_text_bearing_label_is_embedded():
    """A text node with no vector index is text no dense search can ever return.

    ``RagContext._vector_store`` looks an index up by node label, so a missing index is not a
    degraded result, it is an unreachable node.
    """
    missing = [label for label in TEXT_BEARING_LABELS if label not in EMBEDDED_LABELS]
    assert missing == [], f"stored as text but never indexed: {missing}"
