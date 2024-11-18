
import asyncio
from scratch_rag_application.utils.logging_config import setup_logging
import logging
from scratch_rag_application.loader.url_loader import URLLoader
from scratch_rag_application.text_splitter.splitter_factory import TextSplitterFactory
from scratch_rag_application.config.config_handler import ConfigHandler

setup_logging()


async def main():
    logger = logging.getLogger(__name__)
    url_loader = URLLoader()
    docs = await url_loader.load_urls()
    logger.info(f"docs retrieved: {len(docs)}")
    config = ConfigHandler("config.yaml")
    t_factory = TextSplitterFactory(config)
    b_splites = t_factory.create_splitter()
    s_docs = b_splites.split_documents(docs)
    print(f"docs split: {len(s_docs)}")
    print(f"docs split: {s_docs[:3]}")

if __name__ == "__main__":
    asyncio.run(main())
