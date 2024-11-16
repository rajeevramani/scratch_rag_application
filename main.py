
import asyncio
from scratch_rag_application.utils.logging_config import setup_logging
import logging
from scratch_rag_application.loader.url_loader import URLLoader

setup_logging()


async def main():
    logger = logging.getLogger(__name__)
    url_loader = URLLoader()
    docs = await url_loader.load_urls()
    logger.info(f"docs: {docs[:1]}")


if __name__ == "__main__":
    asyncio.run(main())
