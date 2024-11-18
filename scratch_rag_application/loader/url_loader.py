from typing import List
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document
from langchain_community.document_transformers import MarkdownifyTransformer
from scratch_rag_application.config.config_handler import ConfigHandler


class URLLoader:
    """Asynchronous loader for processing URLs
    and converting content to markdown."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_handler = ConfigHandler("config.yaml")
        self.transformer = MarkdownifyTransformer()

        # Get configuration
        self.urls = self.config_handler.get(
            "pipeline.sources.website.urls", [])
        self.content_class = self.config_handler.get(
            "pipeline.sources.website.content_class", "page-content")

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> Document:
        """Fetch and process a single URL."""
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                html = await response.text()

                # Parse only the content we need
                soup = BeautifulSoup(
                    html,
                    'html.parser',
                    parse_only=SoupStrainer(class_=self.content_class)
                )

                return Document(
                    page_content=str(soup),
                    metadata={'source': url}
                )

        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return None

    async def load_urls(self) -> List[Document]:
        """Load and process URLs concurrently."""
        if not self.urls:
            self.logger.warning("No URLs configured")
            return []

        self.logger.info(f"Processing {len(self.urls)} URLs")

        async with aiohttp.ClientSession() as session:
            # Create tasks for all URLs
            tasks = [self._fetch_url(session, url) for url in self.urls]
            documents = await asyncio.gather(*tasks, return_exceptions=False)

            # Filter out failed requests and transform to markdown
            valid_docs = [doc for doc in documents if doc is not None]
            markdown_docs = self.transformer.transform_documents(valid_docs)

            self.logger.info(f"Successfully processed {
                             len(markdown_docs)} documents")
            return markdown_docs

    def load(self) -> List[Document]:
        """Synchronous wrapper for async load_urls method."""
        return asyncio.run(self.load_urls())
