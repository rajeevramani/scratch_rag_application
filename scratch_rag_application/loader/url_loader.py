from typing import List
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup, SoupStrainer
from langchain_core.documents import Document
from langchain_community.document_transformers import MarkdownifyTransformer
from scratch_rag_application.config.config_handler import ConfigHandler
from scratch_rag_application.utils.text_cleaner import TextCleaner
import re


class URLLoader:
    """Asynchronous loader for processing URLs and converting content to markdown."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_handler = ConfigHandler("config.yaml")
        self.transformer = MarkdownifyTransformer()
        self.text_cleaner = TextCleaner()

        # Get configuration
        self.urls = self.config_handler.get(
            "pipeline.sources.website.urls", [])
        self.content_class = self.config_handler.get(
            "pipeline.sources.website.content_class", "page-content")

    def _verify_cleaned_text(self, text: str) -> bool:
        """Verify text meets our cleanliness standards."""
        # Check for multiple consecutive newlines
        if re.search(r'\n{3,}', text):
            self.logger.warning("Found 3+ consecutive newlines")
            return False

        # Check for excessive spaces
        if re.search(r' {3,}', text):
            self.logger.warning("Found 3+ consecutive spaces")
            return False

        return True

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

                # Convert to markdown and clean
                markdown_text = str(soup)
                transformed_docs = self.transformer.transform_documents(
                    [Document(page_content=markdown_text,
                              metadata={'source': url})]
                )

                if transformed_docs:
                    # Clean the transformed text
                    cleaned_text = self.text_cleaner.clean(
                        transformed_docs[0].page_content)

                    # Verify cleaning was successful
                    if not self._verify_cleaned_text(cleaned_text):
                        self.logger.warning(
                            f"Text cleaning verification failed for {url}")

                    return Document(
                        page_content=cleaned_text,
                        metadata={'source': url}
                    )
                return None

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

            # Filter out failed requests
            valid_docs = [doc for doc in documents if doc is not None]

            self.logger.info(f"Successfully processed {
                             len(valid_docs)} documents")
            return valid_docs

    def load(self) -> List[Document]:
        """Synchronous wrapper for async load_urls method."""
        return asyncio.run(self.load_urls())
