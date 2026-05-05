import logging
from typing import Literal, List

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from service.graph.query import NodeQueries
from service.rag.prompt import QUERY_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)


class QueryClassification(BaseModel):
    intent: Literal["DEFINITIONAL", "INTERPRETIVE"] = Field(
        description="DEFINITIONAL for lookups answerable from the legislation itself; "
                    "INTERPRETIVE for questions requiring CJEU case law."
    )
    acts: List[str] = Field(
        default_factory=list,
        description="CELEX IDs of the acts the query is about. Empty list when generic or uncertain."
    )


class QueryClassifier:
    """LLM-based classifier that gates retrieval based on query intent, target acts, and recital need."""

    def __init__(self, graph, llm: ChatOpenAI):
        self.graph = graph
        self._structured_llm = llm.with_structured_output(QueryClassification)
        self._prompt = PromptTemplate.from_template(QUERY_CLASSIFICATION_PROMPT)
        self._acts_block: str | None = None

    def _format_acts(self) -> str:
        if self._acts_block is not None:
            return self._acts_block
        rows = self.graph.query(NodeQueries.GET_ALL_ACTS)
        self._acts_block = "\n".join(f"- {r['celex']}: {r['title']}" for r in rows) or "(no acts available)"
        return self._acts_block

    def classify(self, query: str) -> QueryClassification:
        prompt = self._prompt.format(query=query, acts=self._format_acts())
        result: QueryClassification = self._structured_llm.invoke(prompt)
        logger.info("[Classifier] intent=%s acts=%s", result.intent, result.acts)
        return result
