import asyncio
import logging

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

import config
from service.graph.query import NodeQueries
from service.rag.prompt import CHAPTER_SUMMARY_SYSTEM_PROMPT, CHAPTER_SUMMARY_USER_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONCURRENCY = 5


async def _summarise_chapter(
    llm: ChatOpenAI,
    chapter: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, str]:
    articles_block = "\n".join(f"- {title}" for title in chapter["article_titles"])
    prompt = CHAPTER_SUMMARY_USER_PROMPT.format(
        celex=chapter["celex"],
        act_title=chapter["act_title"],
        number=chapter["chapter_number"],
        title=chapter["chapter_title"],
        articles=articles_block,
    )
    async with semaphore:
        response = await llm.ainvoke([
            {"role": "system", "content": CHAPTER_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
    return chapter["celex"], chapter["chapter_number"], response.content


async def run(graph: Neo4jGraph, llm: ChatOpenAI) -> None:
    chapters = graph.query(NodeQueries.GET_ALL_CHAPTERS_WITHOUT_SUMMARY)
    total = len(chapters)
    logger.info("Chapters to summarise: %d", total)

    if not chapters:
        logger.info("All chapters already have a summary.")
        return

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    tasks = [_summarise_chapter(llm, chapter, semaphore) for chapter in chapters]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            celex, chapter_number, summary = await coro
            graph.query(
                NodeQueries.UPDATE_CHAPTER_SUMMARY,
                params={"celex": celex, "chapter_number": chapter_number, "summary": summary},
            )
            completed += 1
            logger.info("[%d/%d] %s Ch.%s", completed, total, celex, chapter_number)
        except Exception as exc:
            logger.error("Failed to summarise chapter: %s", exc, exc_info=True)


if __name__ == "__main__":
    _graph = Neo4jGraph(
        url=config.NEO4J_URI,
        username=config.NEO4J_USERNAME,
        password=config.NEO4J_PASSWORD,
    )
    _llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )
    asyncio.run(run(_graph, _llm))
