import logging
from typing import Dict, List, Literal, Optional, Tuple

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from legal_assistant import config
from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.acts import acts_mentioned_in
from legal_assistant.rag.prompts import (
    ADDRESSEE_CLASSIFICATION_PROMPT,
    QUERY_CLASSIFICATION_PROMPT,
)

logger = logging.getLogger(__name__)

# The corpus has four acts; retrieval never needs more than a handful of targets.
_MAX_ACTS = 3


class ActRelevance(BaseModel):
    celex: str = Field(
        description="CELEX id of the act being scored, exactly as listed in AVAILABLE ACTS."
    )
    relevance: float = Field(
        ge=0.0, le=1.0,
        description="0-1: how central this act's core subject matter is to the query."
    )


class ActorRelevance(BaseModel):
    actor: str = Field(description="Actor id being scored, exactly as listed in ROLES.")
    relevance: float = Field(
        ge=0.0, le=1.0, description="0-1: how central this role is to the question."
    )


class RawAddressees(BaseModel):
    """Structured output of the separate addressee classifier."""
    addressee_relevances: List[ActorRelevance] = Field(default_factory=list)


class RawClassification(BaseModel):
    """Raw structured output from the LLM: intent plus a relevance score for every act.

    The classifier post-processes this into a `QueryClassification` (thresholded act set);
    downstream retrievers only ever see the latter.
    """
    intent: Literal["DEFINITIONAL", "INTERPRETIVE"] = Field(
        description="DEFINITIONAL for lookups answerable from the legislation itself; "
                    "INTERPRETIVE for questions requiring CJEU case law."
    )
    act_relevances: List[ActRelevance] = Field(
        default_factory=list,
        description="One entry per available act, each scored 0-1 for relevance to the query."
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="[] for atomic/single-provision questions; otherwise 2-4 focused, "
                    "self-contained sub-questions, each answerable from one provision, "
                    "with pronouns resolved."
    )


class QueryClassification(BaseModel):
    """Classification consumed by the retrievers. `acts` is the thresholded target set."""
    intent: Literal["DEFINITIONAL", "INTERPRETIVE"]
    acts: List[str] = Field(
        default_factory=list,
        description="Selected CELEX ids, most relevant first (at most 3). Empty when no act "
                    "clears the threshold — the query is out of scope."
    )
    act_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw per-act relevance scores behind the selection, for logging/diagnostics."
    )
    sub_questions: List[str] = Field(
        default_factory=list,
        description="Decomposition of a compound query into focused sub-questions; empty "
                    "for atomic queries. Used by decomposition-aware retrievers."
    )
    addressees: List[str] = Field(
        default_factory=list,
        description="Actor ids the query is about, for the obligations branch. Empty unless "
                    "the classifier scores addressees; the branch stays inert when empty."
    )
    addressee_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Raw per-actor relevance scores behind the addressee selection."
    )


class QueryClassifier:
    """LLM-based classifier that gates retrieval by query intent and target acts.

    Acts are chosen *discriminatively*: the LLM scores every available act 0-1 and we keep
    those clearing `act_score_threshold` (capped at `_MAX_ACTS`, most relevant first). This
    yields 0/1/N acts uniformly, unlike a generative "name the acts" call that collapses
    toward a single act. An act named explicitly in the query is force-included regardless
    of its score. No corpus-wide fallback: if nothing clears the threshold, `acts` is empty.
    """

    def __init__(
        self,
        graph,
        llm: ChatOpenAI,
        act_score_threshold: float = config.ACT_SCORE_THRESHOLD,
        addressee_score_threshold: float = config.ADDRESSEE_SCORE_THRESHOLD,
    ):
        self.graph = graph
        self.act_score_threshold = act_score_threshold
        self.addressee_score_threshold = addressee_score_threshold
        self._structured_llm = llm.with_structured_output(RawClassification)
        self._structured_addressees = llm.with_structured_output(RawAddressees)
        self._prompt = PromptTemplate.from_template(QUERY_CLASSIFICATION_PROMPT)
        self._addressee_prompt = PromptTemplate.from_template(ADDRESSEE_CLASSIFICATION_PROMPT)
        self._acts_block: Optional[str] = None
        self._actors_block: Optional[str] = None
        self.last_classification: Optional[QueryClassification] = None

    def _format_acts(self) -> str:
        if self._acts_block is not None:
            return self._acts_block
        rows = self.graph.query(NodeQueries.GET_ALL_ACTS)
        self._acts_block = "\n".join(f"- {r['celex']}: {r['title']}" for r in rows) or "(no acts available)"
        return self._acts_block

    def _format_actors(self) -> str:
        if self._actors_block is not None:
            return self._actors_block
        rows = self.graph.query(NodeQueries.GET_ALL_ACTORS)
        self._actors_block = "\n".join(
            f"- {r['id']}: {r['label']}" for r in rows) or "(no actors available)"
        return self._actors_block

    def classify_addressees(self, query: str) -> Tuple[List[str], Dict[str, float]]:
        """Score the roles a question is about, thresholded, best first.

        A separate call from :meth:`classify`, so the shared query classifier is untouched.
        Empty when the graph holds no actors or none clears the threshold, which leaves the
        obligations branch inert.
        """
        actors_block = self._format_actors()
        if actors_block.startswith("(no actors"):
            return [], {}

        prompt = self._addressee_prompt.format(query=query, actors=actors_block)
        raw: RawAddressees = self._structured_addressees.invoke(prompt)
        scores = {ar.actor: ar.relevance for ar in raw.addressee_relevances}
        selected = sorted(
            (actor for actor, score in scores.items() if score >= self.addressee_score_threshold),
            key=lambda actor: -scores[actor],
        )
        logger.info("[Classifier] addressees=%s", selected)
        return selected, scores

    def _select_acts(self, raw: RawClassification, query: str) -> Tuple[List[str], Dict[str, float]]:
        """Threshold the per-act scores, then floor with any explicitly-named act."""
        scores = {ar.celex: ar.relevance for ar in raw.act_relevances}
        selected = sorted(
            (celex for celex, score in scores.items() if score >= self.act_score_threshold),
            key=lambda celex: -scores[celex],
        )
        # Deterministic floor: an act the user names by name/number is a strong enough
        # signal to include even if the model under-scored it.
        for celex in acts_mentioned_in(query):
            if celex not in selected:
                selected.append(celex)
        return selected[:_MAX_ACTS], scores

    def classify(self, query: str) -> QueryClassification:
        prompt = self._prompt.format(query=query, acts=self._format_acts())
        raw: RawClassification = self._structured_llm.invoke(prompt)
        acts, scores = self._select_acts(raw, query)
        result = QueryClassification(
            intent=raw.intent, acts=acts, act_scores=scores, sub_questions=raw.sub_questions,
        )
        logger.info(
            "[Classifier] intent=%s acts=%s scores=%s sub_questions=%d",
            result.intent, result.acts,
            {celex: round(score, 2) for celex, score in sorted(scores.items(), key=lambda kv: -kv[1])},
            len(result.sub_questions),
        )
        self.last_classification = result
        return result
