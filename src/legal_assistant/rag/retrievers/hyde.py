"""HyDE query expansion (Gao et al., 2022).

Dense retrieval matches a question against passages written as *answers*. HyDE closes
that gap by having an LLM write the hypothetical passage first and embedding that
instead of the raw question.
"""
from __future__ import annotations

from typing import Any, List

from langchain_core.prompts import PromptTemplate

from legal_assistant.rag.prompts import HYDE_PROMPT


class HyDEGenerator:
    """Generates act-grounded hypothetical legal passages for dense retrieval.

    With ``iterations > 1`` it samples multiple hypothetical documents (the LLM must
    use temperature > 0 for them to differ); the retriever averages their embeddings
    to reduce the variance of any single generated passage.
    """

    def __init__(self, llm: Any, iterations: int = 1):
        self.llm = llm
        self.iterations = iterations
        self._prompt = PromptTemplate.from_template(HYDE_PROMPT)

    def generate(self, query: str, acts_context: str) -> List[str]:
        text = self._prompt.format(query=query, acts=acts_context)
        return [self.llm.invoke(text).content.strip() for _ in range(self.iterations)]
