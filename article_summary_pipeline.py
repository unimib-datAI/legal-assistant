import asyncio
import logging

from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI

import config
from service.graph.query import NodeQueries
from service.rag.prompt import ARTICLE_SUMMARY_SYSTEM_PROMPT, ARTICLE_SUMMARY_USER_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_CONCURRENCY = 5


async def _summarise_article(
    llm: ChatOpenAI,
    article: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str]:
    prompt = ARTICLE_SUMMARY_USER_PROMPT.format(
        celex=article["celex"],
        act_title=article["act_title"],
        article_title=article["article_title"],
        body=article["body"],
    )
    async with semaphore:
        response = await llm.ainvoke([
            {"role": "system", "content": ARTICLE_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
    return article["article_id"], response.content


async def run(graph: Neo4jGraph, llm: ChatOpenAI) -> None:
    articles = graph.query(NodeQueries.GET_ALL_ARTICLES_WITH_PARAGRAPHS)
    total = len(articles)
    logger.info("Articles to summarise: %d", total)

    if not articles:
        logger.info("All articles already have a summary.")
        return

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    tasks = [_summarise_article(llm, article, semaphore) for article in articles]

    completed = 0
    for coro in asyncio.as_completed(tasks):
        try:
            article_id, summary = await coro
            graph.query(
                NodeQueries.UPDATE_ARTICLE_SUMMARY,
                params={"article_id": article_id, "summary": summary},
            )
            completed += 1
            logger.info("[%d/%d] %s", completed, total, article_id)
        except Exception as exc:
            logger.error("Failed to summarise article: %s", exc, exc_info=True)


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
