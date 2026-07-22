"""LLM summarisation of graph nodes.

Article summaries and chapter summaries are the same job: read the rows a query
returns, send each through a system+user prompt pair, write the result back. Only the
queries, the prompts, and the row→params mapping differ, so those are the four fields
of a :class:`SummaryTask` and the driver below is shared.

Summarising is idempotent: both fetch queries select only nodes whose ``summary`` is
still NULL, so a re-run costs nothing once the graph is fully summarised.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

from legal_assistant.graph.queries import NodeQueries
from legal_assistant.rag.prompts import (
    ARTICLE_SUMMARY_SYSTEM_PROMPT,
    ARTICLE_SUMMARY_USER_PROMPT,
    CHAPTER_SUMMARY_SYSTEM_PROMPT,
    CHAPTER_SUMMARY_USER_PROMPT,
)

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 5


@dataclass(frozen=True)
class SummaryTask:
    """One kind of node to summarise.

    ``user_params`` maps a fetched row to the user prompt's format arguments;
    ``update_params`` maps that same row plus the generated summary to the write
    query's parameters. ``label`` is used only for log lines.
    """

    label: str
    fetch_query: str
    update_query: str
    system_prompt: str
    user_prompt: str
    user_params: Callable[[Dict[str, Any]], Dict[str, Any]]
    update_params: Callable[[Dict[str, Any], str], Dict[str, Any]]
    describe: Callable[[Dict[str, Any]], str]


ARTICLE_TASK = SummaryTask(
    label="article",
    fetch_query=NodeQueries.GET_ALL_ARTICLES_WITH_PARAGRAPHS,
    update_query=NodeQueries.UPDATE_ARTICLE_SUMMARY,
    system_prompt=ARTICLE_SUMMARY_SYSTEM_PROMPT,
    user_prompt=ARTICLE_SUMMARY_USER_PROMPT,
    user_params=lambda row: {
        "celex": row["celex"],
        "act_title": row["act_title"],
        "article_title": row["article_title"],
        "body": row["body"],
    },
    update_params=lambda row, summary: {"article_id": row["article_id"], "summary": summary},
    describe=lambda row: row["article_id"],
)

CHAPTER_TASK = SummaryTask(
    label="chapter",
    fetch_query=NodeQueries.GET_ALL_CHAPTERS_WITHOUT_SUMMARY,
    update_query=NodeQueries.UPDATE_CHAPTER_SUMMARY,
    system_prompt=CHAPTER_SUMMARY_SYSTEM_PROMPT,
    user_prompt=CHAPTER_SUMMARY_USER_PROMPT,
    user_params=lambda row: {
        "celex": row["celex"],
        "act_title": row["act_title"],
        "number": row["chapter_number"],
        "title": row["chapter_title"],
        "articles": "\n".join(f"- {title}" for title in row["article_titles"]),
    },
    update_params=lambda row, summary: {
        "celex": row["celex"],
        "chapter_number": row["chapter_number"],
        "summary": summary,
    },
    describe=lambda row: f"{row['celex']} Ch.{row['chapter_number']}",
)

TASKS: Dict[str, SummaryTask] = {"articles": ARTICLE_TASK, "chapters": CHAPTER_TASK}


async def _summarise_one(
    llm: ChatOpenAI,
    task: SummaryTask,
    row: Dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> tuple[Dict[str, Any], str]:
    prompt = task.user_prompt.format(**task.user_params(row))
    async with semaphore:
        response = await llm.ainvoke([
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": prompt},
        ])
    return row, response.content


async def run_summaries(
    graph: Neo4jGraph,
    llm: ChatOpenAI,
    task: SummaryTask,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> int:
    """Summarise every unsummarised node of ``task``'s kind. Returns how many were written.

    One failed node is logged and skipped rather than aborting the run: a partial pass is
    resumable, since the next run picks up whatever still has a NULL summary.
    """
    rows = graph.query(task.fetch_query)
    total = len(rows)
    logger.info("%ss to summarise: %d", task.label.capitalize(), total)

    if not rows:
        logger.info("All %ss already have a summary.", task.label)
        return 0

    semaphore = asyncio.Semaphore(concurrency)
    pending = [_summarise_one(llm, task, row, semaphore) for row in rows]

    completed = 0
    for coro in asyncio.as_completed(pending):
        try:
            row, summary = await coro
        except Exception:
            logger.exception("Failed to summarise a %s", task.label)
            continue
        graph.query(task.update_query, params=task.update_params(row, summary))
        completed += 1
        logger.info("[%d/%d] %s", completed, total, task.describe(row))

    return completed
